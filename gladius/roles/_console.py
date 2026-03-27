"""
Console output helpers.

Formats the SDK message stream into a human-readable terminal display.
Imported by agent_runner; nothing outside src.roles should need this.
"""

import json
import os
import re
import textwrap
from typing import Any

from claude_agent_sdk import ResultMessage
from claude_agent_sdk.types import (
    AssistantMessage,
    StreamEvent,
    SystemMessage,
    TaskNotificationMessage,
    TaskProgressMessage,
    TaskStartedMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from loguru import logger

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

_SENSITIVE_KEY_RE = re.compile(
    r"(?:pass|password|secret|token|api[_-]?key|auth|authorization|cookie)",
    re.IGNORECASE,
)
_SENSITIVE_VALUE_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{10,}|Bearer\s+[A-Za-z0-9._-]{10,})"
)


def _stream_event_logging_enabled() -> bool:
    raw = os.getenv("GLADIUS_LOG_STREAM_EVENTS", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _c(color: str, text: str) -> str:
    """Wrap text in ANSI color codes."""
    return f"{color}{text}{_RESET}"


def _redact_obj(value: Any) -> Any:
    """Best-effort redaction for sensitive keys and token-like string values."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            if _SENSITIVE_KEY_RE.search(key):
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = _redact_obj(v)
        return redacted
    if isinstance(value, list):
        return [_redact_obj(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact_obj(v) for v in value)
    if isinstance(value, str):
        return _SENSITIVE_VALUE_RE.sub("***REDACTED***", value)
    return value


# ── Input / result formatters ─────────────────────────────────────────────────


def _fmt_input(tool_input: dict, max_len: int = 300) -> str:
    """Compact single-line representation of tool input, capped at max_len chars."""
    try:
        s = json.dumps(_redact_obj(tool_input), ensure_ascii=False)
    except Exception:
        s = str(_redact_obj(tool_input))
    if len(s) > max_len:
        s = s[:max_len] + " …"
    return s


def _fmt_result(content: Any, max_len: int = 400) -> str:
    """Extract readable text from a tool result content value."""
    content = _redact_obj(content)
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


# ── Status icons for TodoWrite tool calls ─────────────────────────────────────

_TODO_ICON = {"completed": "✅", "in_progress": "🔧", "pending": "⬜"}

# Maps Agent/Task tool_use_id → subagent name so sub-messages can show it.
_subagent_names: dict[str, str] = {}
# Maps task_id → subagent name for progress/notification messages.
_task_names: dict[str, str] = {}


def _log_message(agent_name: str, message: Any) -> None:
    """Pretty-print a single SDK message to stdout."""

    if isinstance(message, StreamEvent):
        if not _stream_event_logging_enabled():
            return
        event_type = str(message.event.get("type", "partial")).strip() or "partial"
        logger.debug(
            _c(_GREY, f"  📡 [{agent_name}] stream={event_type}")
            + _c(_DIM, f"  sid={message.session_id[:10]}…")
            + (
                _c(_DIM, f"  parent={message.parent_tool_use_id[:10]}…")
                if message.parent_tool_use_id
                else ""
            )
        )
        return

    if isinstance(message, TaskStartedMessage):
        task_type = (message.task_type or "unknown").strip() or "unknown"
        label = {
            "local_agent": "subagent",
            "local_bash": "bash",
            "remote_agent": "remote-agent",
        }.get(task_type, task_type)
        desc = (message.description or "").strip()
        if len(desc) > 90:
            desc = desc[:90] + "…"
        subagent_name = _subagent_names.get(message.tool_use_id or "", "")
        if subagent_name and message.task_id:
            _task_names[message.task_id] = subagent_name
        name_tag = _c(_DIM + _GREY, f" ➣{subagent_name}") if subagent_name else ""
        logger.debug(
            _c(
                _GREY,
                f"  🚀 [{agent_name}]{name_tag} task:{label} id={message.task_id[:10]}…",
            )
            + (_c(_DIM, f"  {desc}") if desc else "")
        )
        return

    if isinstance(message, TaskProgressMessage):
        usage = message.usage or {}
        total_tokens = usage.get("total_tokens")
        tool_uses = usage.get("tool_uses")
        duration_ms = usage.get("duration_ms")
        token_str = f"  tokens={total_tokens}" if isinstance(total_tokens, int) else ""
        tool_uses_str = f"  tool_uses={tool_uses}" if isinstance(tool_uses, int) else ""
        dur_str = f"  {duration_ms}ms" if isinstance(duration_ms, int) else ""
        tool = (message.last_tool_name or "").strip()
        tool_str = f"  last_tool={tool}" if tool else ""
        subagent_name = _task_names.get(message.task_id or "", "")
        name_tag = _c(_DIM + _GREY, f" ➣{subagent_name}") if subagent_name else ""
        with logger.contextualize(agent=subagent_name or agent_name):
            logger.debug(
                _c(_GREY, f"  ⏳ [{agent_name}]{name_tag} {message.description}")
                + _c(_DIM, token_str + tool_uses_str + dur_str + tool_str)
            )
        return

    if isinstance(message, TaskNotificationMessage):
        if message.status == "completed":
            status_icon = "✅"
            status_color = _GREEN
        elif message.status == "failed":
            status_icon = "❌"
            status_color = _RED
        else:
            status_icon = "⏹"
            status_color = _YELLOW
        usage = message.usage or {}
        total_tokens = usage.get("total_tokens")
        token_str = f"  tokens={total_tokens}" if isinstance(total_tokens, int) else ""
        output_file = (message.output_file or "").strip()
        logger.debug(
            _c(status_color, f"  {status_icon} [{agent_name}] task {message.status}")
            + _c(_DIM, f"  id={message.task_id[:10]}…")
            + _c(_DIM, f"  sid={message.session_id[:10]}…")
            + (
                _c(_DIM, f"  tool_use={message.tool_use_id[:10]}…")
                if message.tool_use_id
                else ""
            )
            + (_c(_DIM, f"  out={output_file}") if output_file else "")
            + _c(_DIM, token_str)
            + (_c(_DIM, f"  {message.summary}") if message.summary else "")
        )
        return

    if isinstance(message, SystemMessage):
        if message.subtype == "init" and isinstance(message.data, dict):
            sid = str(message.data.get("session_id", "?"))
            logger.debug(_c(_GREY, f"  🔑 [{agent_name}] session={sid[:16]}…"))

    elif isinstance(message, AssistantMessage):
        if message.error:
            logger.debug(
                _c(_RED, f"  ⚠ [{agent_name}] AssistantMessage error: {message.error}")
            )
        sub_name = _subagent_names.get(message.parent_tool_use_id, "") if message.parent_tool_use_id else ""
        sub_tag = _c(_DIM + _GREY, f" ➣{sub_name}") if sub_name else ""

        with logger.contextualize(agent=sub_name or agent_name):
            for block in message.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    for line in textwrap.wrap(block.text.strip(), width=120):
                        logger.debug(_c(_CYAN, f"  💬 [{agent_name}]{sub_tag} {line}"))

                elif isinstance(block, ThinkingBlock) and block.thinking.strip():
                    snippet = block.thinking.strip()[:200].replace("\n", " ")
                    logger.debug(
                        _c(_GREY, f"  🧠 [{agent_name}]{sub_tag} (thinking) {snippet} …")
                    )

                elif isinstance(block, ToolUseBlock):
                    _log_tool_use(agent_name, sub_tag, block)

    elif isinstance(message, UserMessage):
        if isinstance(message.content, list):
            sub_name = _subagent_names.get(getattr(message, "parent_tool_use_id", None) or "", "")
            with logger.contextualize(agent=sub_name or agent_name):
                for block in message.content:
                    if isinstance(block, ToolResultBlock):
                        _log_tool_result(agent_name, block)

    elif isinstance(message, ResultMessage):
        cost = f"  cost=${message.total_cost_usd:.4f}" if message.total_cost_usd else ""
        status = _c(_RED, "ERROR") if message.is_error else _c(_GREEN, "OK")
        usage = message.usage or {}
        in_tok = usage.get("input_tokens")
        out_tok = usage.get("output_tokens")
        cache_write = usage.get("cache_creation_input_tokens")
        cache_read = usage.get("cache_read_input_tokens")
        usage_str = ""
        if isinstance(in_tok, int) and isinstance(out_tok, int):
            usage_str += f"  tok(in/out)={in_tok}/{out_tok}"
        if isinstance(cache_write, int) or isinstance(cache_read, int):
            usage_str += f"  cache(w/r)={cache_write or 0}/{cache_read or 0}"
        subtype = f"  subtype={message.subtype}" if message.subtype else ""
        stop_reason = f"  stop={message.stop_reason}" if message.stop_reason else ""
        result_preview = ""
        if message.result:
            preview = str(message.result).strip().replace("\n", " ")
            if preview:
                result_preview = f"  result={preview[:120]}"
        logger.debug(
            _c(_BOLD, f"  ━━ [{agent_name}] done")
            + f"  status={status}"
            + f"  turns={message.num_turns}"
            + f"  {message.duration_ms / 1000:.1f}s"
            + f"  api={message.duration_api_ms}ms"
            + subtype
            + stop_reason
            + usage_str
            + _c(_DIM, cost)
            + _c(_DIM, result_preview)
        )


def _log_tool_use(agent_name: str, sub_tag: str, block: ToolUseBlock) -> None:
    """Render a single ToolUseBlock with tool-specific formatting."""
    if block.name == "TodoWrite":
        todos = block.input.get("todos", [])
        n_done = sum(1 for t in todos if t.get("status") == "completed")
        logger.debug(
            _c(_BOLD + _YELLOW, f"  📋 [{agent_name}]{sub_tag} TodoWrite")
            + _c(_DIM, f"  {n_done}/{len(todos)} done")
        )
        for t in todos:
            icon = _TODO_ICON.get(t.get("status", "pending"), "⬜")
            text = t.get("activeForm") or t.get("content", "")
            logger.debug(f"       {icon}  {_c(_DIM, str(text)[:100])}")

    elif block.name == "ExitPlanMode":
        plan_text = str(block.input.get("plan", "")).strip()
        lines = plan_text.splitlines()
        plan_preview = (lines[0] if lines else "(empty plan)")[:120]
        logger.debug(
            _c(_BOLD + _GREEN, f"  📝 [{agent_name}]{sub_tag} ExitPlanMode")
            + _c(_DIM, f"  {plan_preview}")
        )

    elif block.name in {"Task", "Agent"}:
        target = (
            block.input.get("subagent_type")
            or block.input.get("agent_name")
            or block.input.get("agent")
            or block.input.get("name")
            or "?"
        )
        description = str(block.input.get("description", "")).strip()
        _subagent_names[block.id] = target
        snippet = block.input.get("prompt", "")[:80].replace("\n", " ")
        logger.debug(
            _c(_BOLD + _BLUE, f"  🤖 [{agent_name}]{sub_tag} {block.name} → {target}")
            + _c(_DIM, f"  [{description}]  {snippet}…")
        )

    else:
        inp_str = _fmt_input(block.input)
        logger.debug(
            _c(_BOLD + _YELLOW, f"  🔧 [{agent_name}]{sub_tag} {block.name}")
            + _c(_DIM, f"  {inp_str}")
        )


def _log_tool_result(agent_name: str, block: ToolResultBlock) -> None:
    """Render a single ToolResultBlock."""
    result_str = _fmt_result(block.content)
    marker = _c(_RED, "  ✗") if block.is_error else _c(_GREEN, "  ✓")
    for i, line in enumerate(result_str.splitlines()):
        prefix = f"{marker} [{agent_name}] " if i == 0 else "      "
        logger.debug(f"{prefix}{_c(_DIM, line)}")
