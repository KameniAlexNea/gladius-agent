"""
Shared agent runner.

run_agent()          — structured-output agents (implementer, validation, summarizer)
run_planning_agent() — plan-mode agent (planner); exits via ExitPlanMode

Both stream all SDK messages to the console in real-time and return
(result, session_id).
"""

import asyncio
import logging
import os
from pathlib import Path
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
    PermissionResultAllow,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
)
from llm_output_parser import parse_json as _parse_json

from gladius.agents._agent_defs import SUBAGENT_DEFINITIONS as _SUBAGENT_DEFINITIONS  # re-exported for callers
from gladius.agents._console import _BLUE, _BOLD, _RED, _c, _log_message

logger = logging.getLogger(__name__)


# ── Private helpers ───────────────────────────────────────────────────────────


def _validate_runtime_invocation(
    *,
    agent_name: str,
    cwd: str,
    allowed_tools: list[str],
    max_turns: int | None,
) -> None:
    if not os.environ.get("GLADIUS_MODEL"):
        raise RuntimeError(
            "GLADIUS_MODEL is not set. Add it to your competition .env before running agents."
        )
    if not Path(cwd).exists():
        raise RuntimeError(f"Agent cwd does not exist: {cwd}")
    if not allowed_tools:
        raise RuntimeError(f"{agent_name}: allowed_tools cannot be empty")
    if max_turns is not None and max_turns <= 0:
        raise RuntimeError(f"{agent_name}: max_turns must be > 0")


def _get_runtime_model() -> str:
    """Re-read GLADIUS_MODEL at call time so load_dotenv() is always respected."""
    model = os.environ.get("GLADIUS_MODEL")
    if not model:
        raise RuntimeError(
            "GLADIUS_MODEL is not set. "
            "Add it to your competition's .env file, e.g.:\n"
            "  GLADIUS_MODEL=qwen3-coder"
        )
    return model


def _build_runtime_agents(model: str) -> dict[str, AgentDefinition]:
    """Stamp the live model name into every entry in the registry."""
    return {
        k: AgentDefinition(
            description=v.description,
            prompt=v.prompt,
            tools=v.tools,
            model=model,
        )
        for k, v in _SUBAGENT_DEFINITIONS.items()
    }


def _stderr_cb(line: str) -> None:
    print(_c(_RED, f"  [CLI stderr] {line}"), flush=True)


# ── run_agent ─────────────────────────────────────────────────────────────────


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
    permission_mode: str = "bypassPermissions",
    **option_kwargs: Any,
) -> tuple[dict[str, Any], str]:
    """
    Run a single Claude agent call, streaming all messages to stdout.

    Returns (structured_output, session_id).
    """
    _validate_runtime_invocation(
        agent_name=agent_name,
        cwd=cwd,
        allowed_tools=allowed_tools,
        max_turns=max_turns,
    )

    _runtime_model = _get_runtime_model()
    options = ClaudeAgentOptions(
        # Claude Code's built-in system prompt as the base; role instructions appended.
        system_prompt={"type": "preset", "preset": "claude_code", "append": system_prompt},
        allowed_tools=allowed_tools,
        permission_mode=permission_mode,
        output_format={"type": "json_schema", "schema": output_schema},
        cwd=cwd,
        resume=resume,
        mcp_servers=mcp_servers or {},
        max_turns=max_turns,
        stderr=_stderr_cb,
        # Load CLAUDE.md and .claude/settings.json from the competition directory.
        setting_sources=["project"],
        # Programmatic defs override .claude/agents/*.md and inherit bypassPermissions.
        agents=_build_runtime_agents(_runtime_model),
        model=_runtime_model,
        **option_kwargs,
    )

    for attempt in range(max_retries):
        try:
            result_msg: ResultMessage | None = None
            last_assistant_msg: AssistantMessage | None = None
            # Capture session_id from SystemMessage(subtype="init") early — it
            # arrives before any tool calls and serves as a fallback if the run
            # crashes before ResultMessage (rate-limit kill, OOM, network drop).
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
                if isinstance(message, AssistantMessage):
                    last_assistant_msg = message
                if isinstance(message, ResultMessage):
                    result_msg = message

            if result_msg is None:
                raise RuntimeError("No ResultMessage received from agent")

            structured = result_msg.structured_output

            # Fallback: local models (Ollama, GLM, etc.) sometimes emit the
            # required JSON as plain text instead of calling StructuredOutput.
            if structured is None and last_assistant_msg is not None:
                _full_text = "\n".join(
                    block.text.strip()
                    for block in last_assistant_msg.content
                    if isinstance(block, TextBlock) and block.text.strip()
                )
                if _full_text:
                    try:
                        structured = _parse_json(_full_text)
                        logger.warning(
                            f"[{agent_name}] structured_output was None — "
                            "extracted JSON from assistant text (llm_output_parser fallback)"
                        )
                    except Exception:
                        pass

            # Qwen-style: model calls StructuredOutput then keeps running until
            # max_turns, producing is_error=True.  Prefer the captured output.
            if result_msg.is_error and structured is None:
                raise RuntimeError(f"Agent returned error result: {result_msg.result}")
            if result_msg.is_error and structured is not None:
                logger.warning(
                    f"[{agent_name}] ResultMessage is_error=True but structured_output "
                    "was captured — using it (model ran extra turns after StructuredOutput)."
                )

            if structured is None:
                raise RuntimeError(
                    "Agent returned no structured_output (schema not satisfied?)"
                )

            return structured, result_msg.session_id or early_session_id or ""

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
            # (e.g. 'signature' on thinking blocks).
            if attempt == max_retries - 1:
                raise
            logger.warning(f"MessageParseError on attempt {attempt + 1}: {e}, retrying")

        except RuntimeError as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"RuntimeError on attempt {attempt + 1}: {e}, retrying")

    raise RuntimeError("run_agent: max retries exceeded")


# ── run_planning_agent ────────────────────────────────────────────────────────

# Tools the planner must never call — deny them even if the local model tries.
_PLAN_MODE_DENIED_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "Bash", "Task", "computer"})


async def _approve_exit_plan_mode(tool_name: str, input_data: dict, context: object):
    """Block write tools in planning mode; auto-approve everything else.

    SDK calls can_use_tool for anything needing explicit approval (write ops,
    ExitPlanMode).  Read-only tools are allowed by the permission layer before
    this callback fires.
    """
    from claude_agent_sdk.types import PermissionResultDeny

    if tool_name in _PLAN_MODE_DENIED_TOOLS:
        return PermissionResultDeny(
            message=(
                f"Tool '{tool_name}' is not permitted in planning mode. "
                "Use only Read, Glob, Grep, WebSearch, Skill, TodoWrite, or ExitPlanMode."
            )
        )
    return PermissionResultAllow(updated_input=input_data)


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

    Claude uses read-only tools to research, then exits via ExitPlanMode which
    carries the plan as a markdown string.

    Returns (plan_text, session_id).
    """
    _validate_runtime_invocation(
        agent_name=agent_name,
        cwd=cwd,
        allowed_tools=allowed_tools,
        max_turns=max_turns,
    )

    _runtime_model = _get_runtime_model()
    options = ClaudeAgentOptions(
        system_prompt={"type": "preset", "preset": "claude_code", "append": system_prompt},
        allowed_tools=allowed_tools,
        permission_mode="plan",
        can_use_tool=_approve_exit_plan_mode,
        cwd=cwd,
        resume=resume,
        mcp_servers=mcp_servers or {},
        max_turns=max_turns,
        stderr=_stderr_cb,
        setting_sources=["project"],
        agents=_build_runtime_agents(_runtime_model),
        model=_runtime_model,
    )

    for attempt in range(max_retries):
        try:
            result_msg: ResultMessage | None = None
            early_session_id: str | None = None
            captured_plan: str | None = None
            last_text_block: str = ""

            if verbose:
                resume_str = f"  resume={resume[:8]}…" if resume else ""
                print(
                    _c(_BOLD + _BLUE, f"\n▶ [{agent_name}] (plan mode)")
                    + f"  tools={allowed_tools}"
                    + resume_str
                )

            # can_use_tool requires a streaming prompt, not a plain string.
            async def _prompt_stream():
                yield {"type": "user", "message": {"role": "user", "content": prompt}}

            async for message in query(prompt=_prompt_stream(), options=options):
                if verbose:
                    _log_message(agent_name, message)
                if isinstance(message, SystemMessage) and message.subtype == "init":
                    early_session_id = message.data.get("session_id")
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        # Primary: model called ExitPlanMode.
                        if isinstance(block, ToolUseBlock) and block.name == "ExitPlanMode":
                            captured_plan = block.input.get("plan", "")
                        # Fallback: models that don't support ExitPlanMode (Ollama/Qwen)
                        # produce the plan as a final text block.
                        elif isinstance(block, TextBlock) and len(block.text.strip()) > 100:
                            last_text_block = block.text.strip()
                if isinstance(message, ResultMessage):
                    result_msg = message

            if result_msg is None:
                raise RuntimeError("No ResultMessage received from planning agent")
            if result_msg.is_error:
                raise RuntimeError(
                    f"Planning agent returned error result: {result_msg.result}"
                )
            if not captured_plan:
                if last_text_block:
                    logger.warning(
                        f"[{agent_name}] ExitPlanMode not called — using last text "
                        "block as plan (model may not support planning mode)."
                    )
                    captured_plan = last_text_block
                else:
                    raise RuntimeError(
                        "Planning agent did not emit ExitPlanMode and produced no "
                        "usable text output. The model may not support planning mode."
                    )

            return captured_plan, result_msg.session_id or early_session_id or ""

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

