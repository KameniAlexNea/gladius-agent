"""
Shared agent runner helper.

Every agent calls run_agent() which wraps claude_agent_sdk.query() and
returns (structured_output, session_id).

All messages (tool calls, text, tool results, thinking) are streamed to the
console in real-time so you can see exactly what Claude is doing.
"""

import asyncio
import json
import logging
import os
import textwrap
from typing import Any

from claude_agent_sdk import (
    AgentDefinition,
    ClaudeAgentOptions,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
    ResultMessage,
    query,
)
from claude_agent_sdk._errors import MessageParseError
from claude_agent_sdk.types import (
    AssistantMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

logger = logging.getLogger(__name__)

# ── Programmatic agent definitions ────────────────────────────────────────────
# Passed to every ClaudeAgentOptions so that:
#   1. Task subagents inherit bypassPermissions from the parent call (not
#      the filesystem .claude/agents/*.md which have acceptEdits).
#   2. Programmatic definitions always take precedence over filesystem files.
# Note: Task is intentionally omitted from subagent tool lists to prevent
# unbounded recursion.
# _model is resolved at module load; run_agent() re-reads GLADIUS_MODEL at
# call time (after load_dotenv) so the .env value is always used.
# Missing GLADIUS_MODEL is a hard error — no silent fallback to a cloud model.
_model = os.environ.get("GLADIUS_MODEL") or ""
_PLANNER_AGENT_DEF = AgentDefinition(
    description=(
        "Expert ML competition analyst. Explores data, reviews experiment history, "
        "and proposes the highest-impact next approach via planning mode (ExitPlanMode). "
        "Invoke at the start of each competition iteration when a fresh plan is needed."
    ),
    prompt=(
        "You are an expert ML competition analyst.\n\n"
        "Start every session:\n"
        "1. Read CLAUDE.md — competition state, best scores, recent experiments.\n"
        "2. Read .claude/agent-memory/planner/MEMORY.md — accumulated knowledge.\n"
        "3. Explore the data directory and existing solution files.\n\n"
        "Your job: understand what has been tried, identify the highest-impact next "
        "approach, produce a concrete ordered action plan the implementer can follow "
        "blindly. Update memory with new insights.\n\n"
        "STRICT RULES — you are in READ-ONLY planning mode:\n"
        "- You NEVER run Bash commands.\n"
        "- You NEVER write or edit any files yourself.\n"
        "- You NEVER spawn Task subagents.\n"
        "- You NEVER write implementation code.\n"
        "Use only Read, Glob, Grep, WebSearch, TodoWrite."
    ),
    # TodoWrite lets the planner track its own multi-step exploration progress.
    # Task must NOT be listed — subagents cannot spawn sub-subagents.
    # Bash is intentionally excluded — plan mode is read-only research.
    tools=["Read", "Glob", "Grep", "WebSearch", "TodoWrite"],
    model=_model,
)

_IMPLEMENTER_AGENT_DEF = AgentDefinition(
    description=(
        "Expert ML engineer. Executes a given plan end-to-end: writes code, runs it, "
        "debugs errors, measures the competition metric, and reports results. "
        "Always gets fresh context — does not retain state between invocations."
    ),
    prompt=(
        "You are an expert ML engineer executing a competition experiment.\n\n"
        "Start by reading CLAUDE.md for competition context, then the plan you received.\n"
        "Implement completely: write code, run it, fix errors, iterate until done.\n"
        "Before reporting, read .claude/skills/code-review/SKILL.md and fix every "
        "CRITICAL item (leakage, metric correctness, submission format).\n\n"
        "STRICT RULES:\n"
        "- NEVER modify or overwrite CLAUDE.md — it is managed by the orchestrator.\n"
        "- NEVER spawn Task subagents.\n"
        "Report: status, oof_score, quality_score, solution_files, submission_file, notes."
    ),
    # TodoWrite lets the implementer track steps (write code, run, fix, measure, submit).
    # Task must NOT be listed here — subagents cannot spawn sub-subagents.
    # Skill lets the implementer invoke code-review, task-review, ml-pipeline, etc.
    tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "TodoWrite", "Skill"],
    model=_model,
)

_SUMMARIZER_AGENT_DEF = AgentDefinition(
    description=(
        "Expert ML research analyst that reviews experiment results and rewrites the "
        "planner memory file. Read-only: it never edits code or data files — it only "
        "reads existing files and returns structured analysis."
    ),
    prompt=(
        "You are an expert ML research analyst maintaining a living knowledge base.\n\n"
        "You review experiment results and produce a concise, structured update for the "
        "planner's MEMORY.md file. You NEVER write files yourself — you return the "
        "full updated memory content as structured output.\n\n"
        "Always read the existing MEMORY.md before producing the update so you preserve "
        "historical entries."
    ),
    # Read-only tools only — the orchestrator writes MEMORY.md from the output.
    tools=["Read", "Grep"],
    model=_model,
)

_VALIDATION_AGENT_DEF = AgentDefinition(
    description=(
        "Validates experiment results and recommends whether to submit to the platform. "
        "Read-only: it never modifies files or state — it only observes and reports "
        "structured decisions (is_improvement, submit, reasoning)."
    ),
    prompt=(
        "You are a competition result validator.\n\n"
        "You compare new experiment scores against the current best, check submission "
        "artifact format by reading files, query platform quota via MCP tools, and return "
        "a structured JSON decision. You NEVER write files or mutate state.\n\n"
        "STRICT RULES — you are READ-ONLY:\n"
        "- NEVER run Bash commands.\n"
        "- NEVER write, edit, or delete any files.\n"
        "Use only Read, Grep, and any MCP quota tools provided."
    ),
    # Read-only tools only — MCP quota tools are injected per-call by run_validation_agent().
    tools=["Read", "Grep"],
    model=_model,
)

# Registry used by every agent call — overrides .claude/agents/*.md files
_SUBAGENT_DEFINITIONS: dict[str, AgentDefinition] = {
    "planner": _PLANNER_AGENT_DEF,
    "implementer": _IMPLEMENTER_AGENT_DEF,
    "summarizer": _SUMMARIZER_AGENT_DEF,
    "validation": _VALIDATION_AGENT_DEF,
}

# ── Console colours (degrade gracefully in non-TTY) ───────────────────────────
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BLUE = "\033[34m"
_GREY = "\033[90m"


def _c(color: str, text: str) -> str:
    """Wrap text in ANSI color codes."""
    return f"{color}{text}{_RESET}"


def _fmt_input(tool_input: dict, max_len: int = 300) -> str:
    """Compact single-line representation of tool input, capped at max_len chars."""
    try:
        s = json.dumps(tool_input, ensure_ascii=False)
    except Exception:
        s = str(tool_input)
    if len(s) > max_len:
        s = s[:max_len] + " …"
    return s


def _fmt_result(content: Any, max_len: int = 400) -> str:
    """Extract readable text from a tool result content value."""
    if content is None:
        return "(empty)"
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            else:
                parts.append(str(item))
        text = "\n".join(parts)
    else:
        text = str(content)

    text = text.strip()
    if len(text) > max_len:
        text = text[:max_len] + " …"
    return text


# Todo status icons — used when rendering TodoWrite tool calls
_TODO_ICON = {"completed": "✅", "in_progress": "🔧", "pending": "⬜"}


def _log_message(agent_name: str, message: Any) -> None:
    """Pretty-print a single SDK message to stdout."""

    if isinstance(message, SystemMessage):
        # Sessions doc: first message is subtype="init" and contains session_id.
        # Log it early so it's visible before any tool calls.
        if message.subtype == "init":
            sid = message.data.get("session_id", "?")
            print(_c(_GREY, f"  🔑 [{agent_name}] session={sid[:16]}…"))

    elif isinstance(
        message, AssistantMessage
    ):  # Emit any error before printing content blocks so it's never missed.
        if message.error:
            print(
                _c(_RED, f"  ⚠ [{agent_name}] AssistantMessage error: {message.error}")
            )  # Show a visual marker when this message originates from inside a subagent.
        sub_tag = _c(_DIM + _GREY, " ➣subagent") if message.parent_tool_use_id else ""

        for block in message.content:
            if isinstance(block, TextBlock) and block.text.strip():
                for line in textwrap.wrap(block.text.strip(), width=120):
                    print(_c(_CYAN, f"  💬 [{agent_name}]{sub_tag} {line}"))

            elif isinstance(block, ThinkingBlock) and block.thinking.strip():
                snippet = block.thinking.strip()[:200].replace("\n", " ")
                print(_c(_GREY, f"  🧠 [{agent_name}]{sub_tag} (thinking) {snippet} …"))

            elif isinstance(block, ToolUseBlock):
                if block.name == "TodoWrite":
                    # Todo doc: render the task list with status icons so progress
                    # is immediately visible in the console stream.
                    todos = block.input.get("todos", [])
                    n_done = sum(1 for t in todos if t.get("status") == "completed")
                    print(
                        _c(_BOLD + _YELLOW, f"  📋 [{agent_name}]{sub_tag} TodoWrite")
                        + _c(_DIM, f"  {n_done}/{len(todos)} done")
                    )
                    for t in todos:
                        icon = _TODO_ICON.get(t.get("status", "pending"), "⬜")
                        text = t.get("activeForm") or t.get("content", "")
                        print(f"       {icon}  {_c(_DIM, str(text)[:100])}")

                elif block.name == "ExitPlanMode":
                    # Planning mode doc: Claude finishes planning with ExitPlanMode.
                    # Show the first few lines of the plan text so progress is visible.
                    plan_preview = (
                        block.input.get("plan", "").strip().splitlines()[0][:120]
                    )
                    print(
                        _c(_BOLD + _GREEN, f"  📝 [{agent_name}]{sub_tag} ExitPlanMode")
                        + _c(_DIM, f"  {plan_preview}")
                    )

                elif block.name == "Task":
                    # Subagents doc: Task is how Claude spawns subagents.
                    # Show which subagent type and the task description clearly.
                    subagent_type = block.input.get("subagent_type", "?")
                    description = block.input.get("description", "")
                    snippet = block.input.get("prompt", "")[:80].replace("\n", " ")
                    print(
                        _c(
                            _BOLD + _BLUE,
                            f"  🤖 [{agent_name}]{sub_tag} Task → {subagent_type}",
                        )
                        + _c(_DIM, f"  [{description}]  {snippet}…")
                    )

                else:
                    inp_str = _fmt_input(block.input)
                    print(
                        _c(
                            _BOLD + _YELLOW,
                            f"  🔧 [{agent_name}]{sub_tag} {block.name}",
                        )
                        + _c(_DIM, f"  {inp_str}")
                    )

    elif isinstance(message, UserMessage):
        if isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    result_str = _fmt_result(block.content)
                    marker = _c(_RED, "  ✗") if block.is_error else _c(_GREEN, "  ✓")
                    for i, line in enumerate(result_str.splitlines()):
                        prefix = f"{marker} [{agent_name}] " if i == 0 else "      "
                        print(f"{prefix}{_c(_DIM, line)}")

    elif isinstance(message, ResultMessage):
        cost = f"  cost=${message.total_cost_usd:.4f}" if message.total_cost_usd else ""
        status = _c(_RED, "ERROR") if message.is_error else _c(_GREEN, "OK")
        print(
            _c(_BOLD, f"  ━━ [{agent_name}] done")
            + f"  status={status}"
            + f"  turns={message.num_turns}"
            + f"  {message.duration_ms / 1000:.1f}s"
            + _c(_DIM, cost)
        )


# ── Main helper ───────────────────────────────────────────────────────────────


async def run_agent(
    *,
    agent_name: str = "agent",
    prompt: str,
    system_prompt: str,
    allowed_tools: list[str],
    output_schema: dict[str, Any],
    cwd: str,
    resume: str | None = None,
    mcp_servers: dict | None = None,
    max_turns: int | None = None,
    max_retries: int = 3,
    verbose: bool = True,
    **option_kwargs: Any,
) -> tuple[dict[str, Any], str]:
    """
    Run a single Claude agent call, streaming all messages to stdout.

    Returns
    -------
    (structured_output, session_id)
    """

    def _stderr_cb(line: str) -> None:
        print(_c(_RED, f"  [CLI stderr] {line}"), flush=True)

    # Re-read model from env at call time: load_dotenv() in the orchestrator
    # runs after module import, so module-level _model may be stale.
    _runtime_model = os.environ.get("GLADIUS_MODEL")
    if not _runtime_model:
        raise RuntimeError(
            "GLADIUS_MODEL is not set. "
            "Add it to your competition's .env file, e.g.:\n"
            "  GLADIUS_MODEL=qwen3-coder"
        )
    _runtime_agents = {
        k: AgentDefinition(
            description=v.description,
            prompt=v.prompt,
            tools=v.tools,
            model=_runtime_model,
        )
        for k, v in _SUBAGENT_DEFINITIONS.items()
    }

    options = ClaudeAgentOptions(
        # Use Claude Code's built-in system prompt as the base so agents get
        # full tool-use knowledge, safety guidelines, and code-gen best
        # practices — then append the role-specific instructions.
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": system_prompt,
        },
        allowed_tools=allowed_tools,
        permission_mode="bypassPermissions",
        output_format={"type": "json_schema", "schema": output_schema},
        cwd=cwd,
        resume=resume,
        mcp_servers=mcp_servers or {},
        max_turns=max_turns,
        stderr=_stderr_cb,
        # Load CLAUDE.md and .claude/settings.json (hooks, env vars) from
        # the competition project directory.
        setting_sources=["project"],
        # Programmatic agent definitions override .claude/agents/*.md files.
        # Ensures Task subagents inherit bypassPermissions (not acceptEdits).
        # Model is resolved at call time from GLADIUS_MODEL env var.
        agents=_runtime_agents,
        model=_runtime_model,
        **option_kwargs,
    )

    for attempt in range(max_retries):
        try:
            result_msg: ResultMessage | None = None
            # Sessions doc: the SystemMessage(subtype="init") arrives before any
            # tool calls and contains the session_id.  Capture it early so we
            # have a fallback session_id even if the run crashes before ResultMessage
            # (e.g. rate-limit kill, OOM, network drop).
            early_session_id: str | None = None

            if verbose:
                resume_str = f"  resume={resume[:8]}…" if resume else ""
                print(
                    _c(_BOLD + _BLUE, f"\n▶ [{agent_name}]")
                    + f"  tools={allowed_tools}"
                    + resume_str
                )

            async for message in query(prompt=prompt, options=options):
                if verbose:
                    _log_message(agent_name, message)
                if isinstance(message, SystemMessage) and message.subtype == "init":
                    early_session_id = message.data.get("session_id")
                if isinstance(message, ResultMessage):
                    result_msg = message

            if result_msg is None:
                raise RuntimeError("No ResultMessage received from agent")
            if result_msg.is_error:
                raise RuntimeError(f"Agent returned error result: {result_msg.result}")
            if result_msg.structured_output is None:
                raise RuntimeError(
                    "Agent returned no structured_output (schema not satisfied?)"
                )

            return (
                result_msg.structured_output,
                result_msg.session_id or early_session_id or "",
            )

        except CLINotFoundError:
            raise  # Fatal — Claude Code CLI not installed

        except ProcessError as e:
            stderr = e.stderr or ""
            if "rate limit" in stderr.lower() and attempt < max_retries - 1:
                wait = 60 * (2**attempt)
                logger.warning(f"Rate-limited, waiting {wait}s (attempt {attempt + 1})")
                await asyncio.sleep(wait)
            elif attempt == max_retries - 1:
                raise
            else:
                logger.warning(f"ProcessError on attempt {attempt + 1}: {e}")

        except CLIJSONDecodeError:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"JSON decode error on attempt {attempt + 1}, retrying")

        except MessageParseError as e:
            # Can occur with Ollama models that omit Anthropic-specific fields
            # (e.g. 'signature' on thinking blocks).  Retry may help if it's
            # transient, but usually indicates a model compatibility issue.
            if attempt == max_retries - 1:
                raise
            logger.warning(f"MessageParseError on attempt {attempt + 1}: {e}, retrying")

        except RuntimeError as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"RuntimeError on attempt {attempt + 1}: {e}, retrying")

    raise RuntimeError("run_agent: max retries exceeded")


async def run_planning_agent(
    *,
    agent_name: str = "planner",
    prompt: str,
    system_prompt: str,
    allowed_tools: list[str],
    cwd: str,
    resume: str | None = None,
    mcp_servers: dict | None = None,
    max_turns: int | None = None,
    max_retries: int = 3,
    verbose: bool = True,
) -> tuple[str, str]:
    """
    Run a Claude agent in planning mode (permission_mode="plan").

    Claude uses read-only tools to research, then exits via the built-in
    ``ExitPlanMode`` tool call which carries the plan as a markdown string.

    Returns
    -------
    (plan_text, session_id)
        plan_text  — full markdown plan from ExitPlanMode.input["plan"]
        session_id — can be resumed for a follow-up non-plan call if needed
    """

    def _stderr_cb(line: str) -> None:
        print(_c(_RED, f"  [CLI stderr] {line}"), flush=True)

    _runtime_model = os.environ.get("GLADIUS_MODEL")
    if not _runtime_model:
        raise RuntimeError(
            "GLADIUS_MODEL is not set. "
            "Add it to your competition's .env file, e.g.:\n"
            "  GLADIUS_MODEL=qwen3-coder"
        )
    _runtime_agents = {
        k: AgentDefinition(
            description=v.description,
            prompt=v.prompt,
            tools=v.tools,
            model=_runtime_model,
        )
        for k, v in _SUBAGENT_DEFINITIONS.items()
    }

    options = ClaudeAgentOptions(
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": system_prompt,
        },
        allowed_tools=allowed_tools,
        permission_mode="plan",
        cwd=cwd,
        resume=resume,
        mcp_servers=mcp_servers or {},
        max_turns=max_turns,
        stderr=_stderr_cb,
        setting_sources=["project"],
        agents=_runtime_agents,
        model=_runtime_model,
    )

    for attempt in range(max_retries):
        try:
            result_msg: ResultMessage | None = None
            early_session_id: str | None = None
            captured_plan: str | None = None

            if verbose:
                resume_str = f"  resume={resume[:8]}…" if resume else ""
                print(
                    _c(_BOLD + _BLUE, f"\n▶ [{agent_name}] (plan mode)")
                    + f"  tools={allowed_tools}"
                    + resume_str
                )

            async for message in query(prompt=prompt, options=options):
                if verbose:
                    _log_message(agent_name, message)
                if isinstance(message, SystemMessage) and message.subtype == "init":
                    early_session_id = message.data.get("session_id")
                # Capture plan from ExitPlanMode tool use block
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if (
                            isinstance(block, ToolUseBlock)
                            and block.name == "ExitPlanMode"
                        ):
                            captured_plan = block.input.get("plan", "")
                if isinstance(message, ResultMessage):
                    result_msg = message

            if result_msg is None:
                raise RuntimeError("No ResultMessage received from planning agent")
            if result_msg.is_error:
                raise RuntimeError(
                    f"Planning agent returned error result: {result_msg.result}"
                )
            if not captured_plan:
                raise RuntimeError(
                    "Planning agent did not emit ExitPlanMode — no plan captured. "
                    "The model may not support planning mode."
                )

            session_id = result_msg.session_id or early_session_id or ""
            return captured_plan, session_id

        except CLINotFoundError:
            raise

        except ProcessError as e:
            stderr = e.stderr or ""
            if "rate limit" in stderr.lower() and attempt < max_retries - 1:
                wait = 60 * (2**attempt)
                logger.warning(f"Rate-limited, waiting {wait}s (attempt {attempt + 1})")
                await asyncio.sleep(wait)
            elif attempt == max_retries - 1:
                raise
            else:
                logger.warning(f"ProcessError on attempt {attempt + 1}: {e}")

        except CLIJSONDecodeError:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"JSON decode error on attempt {attempt + 1}, retrying")

        except MessageParseError as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"MessageParseError on attempt {attempt + 1}: {e}, retrying")

        except RuntimeError as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"RuntimeError on attempt {attempt + 1}: {e}, retrying")

    raise RuntimeError("run_planning_agent: max retries exceeded")
