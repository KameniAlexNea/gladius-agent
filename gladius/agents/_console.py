"""
Console output helpers.

Formats the SDK message stream into a human-readable terminal display.
Imported by _base.py; nothing outside gladius.agents should need this.
"""

import json
import textwrap
from typing import Any

from claude_agent_sdk import ResultMessage
from claude_agent_sdk.types import (
    AssistantMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

# ── ANSI colour codes (degrade gracefully in non-TTY) ─────────────────────────

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


# ── Input / result formatters ─────────────────────────────────────────────────


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


# ── Message logger ────────────────────────────────────────────────────────────

# Status icons for TodoWrite tool calls
_TODO_ICON = {"completed": "✅", "in_progress": "🔧", "pending": "⬜"}


def _log_message(agent_name: str, message: Any) -> None:
    """Pretty-print a single SDK message to stdout."""

    if isinstance(message, SystemMessage):
        if message.subtype == "init":
            sid = message.data.get("session_id", "?")
            print(_c(_GREY, f"  🔑 [{agent_name}] session={sid[:16]}…"))

    elif isinstance(message, AssistantMessage):
        # Emit errors before content blocks so they're never missed.
        if message.error:
            print(
                _c(_RED, f"  ⚠ [{agent_name}] AssistantMessage error: {message.error}")
            )
        sub_tag = _c(_DIM + _GREY, " ➣subagent") if message.parent_tool_use_id else ""

        for block in message.content:
            if isinstance(block, TextBlock) and block.text.strip():
                for line in textwrap.wrap(block.text.strip(), width=120):
                    print(_c(_CYAN, f"  💬 [{agent_name}]{sub_tag} {line}"))

            elif isinstance(block, ThinkingBlock) and block.thinking.strip():
                snippet = block.thinking.strip()[:200].replace("\n", " ")
                print(_c(_GREY, f"  🧠 [{agent_name}]{sub_tag} (thinking) {snippet} …"))

            elif isinstance(block, ToolUseBlock):
                _log_tool_use(agent_name, sub_tag, block)

    elif isinstance(message, UserMessage):
        if isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    _log_tool_result(agent_name, block)

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


def _log_tool_use(agent_name: str, sub_tag: str, block: ToolUseBlock) -> None:
    """Render a single ToolUseBlock with tool-specific formatting."""
    if block.name == "TodoWrite":
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
        plan_text = str(block.input.get("plan", "")).strip()
        lines = plan_text.splitlines()
        plan_preview = (lines[0] if lines else "(empty plan)")[:120]
        print(
            _c(_BOLD + _GREEN, f"  📝 [{agent_name}]{sub_tag} ExitPlanMode")
            + _c(_DIM, f"  {plan_preview}")
        )

    elif block.name == "Task":
        subagent_type = block.input.get("subagent_type", "?")
        description = block.input.get("description", "")
        snippet = block.input.get("prompt", "")[:80].replace("\n", " ")
        print(
            _c(_BOLD + _BLUE, f"  🤖 [{agent_name}]{sub_tag} Task → {subagent_type}")
            + _c(_DIM, f"  [{description}]  {snippet}…")
        )

    else:
        inp_str = _fmt_input(block.input)
        print(
            _c(_BOLD + _YELLOW, f"  🔧 [{agent_name}]{sub_tag} {block.name}")
            + _c(_DIM, f"  {inp_str}")
        )


def _log_tool_result(agent_name: str, block: ToolResultBlock) -> None:
    """Render a single ToolResultBlock."""
    result_str = _fmt_result(block.content)
    marker = _c(_RED, "  ✗") if block.is_error else _c(_GREEN, "  ✓")
    for i, line in enumerate(result_str.splitlines()):
        prefix = f"{marker} [{agent_name}] " if i == 0 else "      "
        print(f"{prefix}{_c(_DIM, line)}")
