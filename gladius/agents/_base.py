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
import textwrap
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
    ResultMessage,
    query,
)
from claude_agent_sdk.types import (
    AssistantMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

logger = logging.getLogger(__name__)

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


def _log_message(agent_name: str, message: Any) -> None:
    """Pretty-print a single SDK message to stdout."""

    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock) and block.text.strip():
                # Wrap long text but keep indented
                for line in textwrap.wrap(block.text.strip(), width=120):
                    print(_c(_CYAN, f"  💬 [{agent_name}] {line}"))

            elif isinstance(block, ThinkingBlock) and block.thinking.strip():
                # Only first 200 chars of thinking — it's often very long
                snippet = block.thinking.strip()[:200].replace("\n", " ")
                print(_c(_GREY, f"  🧠 [{agent_name}] (thinking) {snippet} …"))

            elif isinstance(block, ToolUseBlock):
                inp_str = _fmt_input(block.input)
                print(
                    _c(_BOLD + _YELLOW, f"  🔧 [{agent_name}] {block.name}")
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

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        permission_mode="bypassPermissions",
        output_format={"type": "json_schema", "schema": output_schema},
        cwd=cwd,
        resume=resume,
        mcp_servers=mcp_servers or {},
        max_turns=max_turns,
        stderr=_stderr_cb,
        **option_kwargs,
    )

    for attempt in range(max_retries):
        try:
            result_msg: ResultMessage | None = None

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

            return result_msg.structured_output, result_msg.session_id

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

        except RuntimeError as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"RuntimeError on attempt {attempt + 1}: {e}, retrying")

    raise RuntimeError("run_agent: max retries exceeded")
