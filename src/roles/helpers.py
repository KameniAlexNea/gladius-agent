"""Shared runtime helpers for agent execution."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

from claude_agent_sdk import AgentDefinition
from loguru import logger

from src.roles._agent_defs import SUBAGENT_DEFINITIONS as _SUBAGENT_DEFINITIONS


def validate_runtime_invocation(
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


def get_runtime_model() -> str:
    """Re-read GLADIUS_MODEL at call time so load_dotenv() is always respected."""
    model = os.environ.get("GLADIUS_MODEL")
    if not model:
        raise RuntimeError(
            "GLADIUS_MODEL is not set. "
            "Add it to your competition's .env file, e.g.:\n"
            "  GLADIUS_MODEL=qwen3-coder"
        )
    return model


_SMALL_MODEL_AGENTS = frozenset({"evaluator", "memory-keeper"})


def build_runtime_agents(model: str) -> dict[str, AgentDefinition]:
    """Stamp the live model name into every entry in the registry."""
    small_model = os.environ.get("GLADIUS_SMALL_MODEL") or model
    return {
        k: AgentDefinition(
            description=v.description,
            prompt=v.prompt,
            tools=v.tools,
            model=small_model if k in _SMALL_MODEL_AGENTS else model,
        )
        for k, v in _SUBAGENT_DEFINITIONS.items()
    }


def stderr_cb(line: str) -> None:
    logger.error(f"  [CLI stderr] {line}")


def is_tool_allowed(tool_name: str, allowed_tools: list[str]) -> bool:
    """Return True when tool_name is allowed by runtime tool policy."""
    # StructuredOutput is the terminal schema-emission channel used by
    # output_format=json_schema. It is not listed in allowed_tools.
    if tool_name == "StructuredOutput":
        return True
    if tool_name in allowed_tools:
        return True
    # Claude SDK emits the Agent() delegation tool as Task in messages.
    if tool_name == "Task" and any(t.startswith("Agent(") for t in allowed_tools):
        return True
    return False


def is_path_within_cwd(path_token: str, cwd: str) -> bool:
    """Return True when path_token resolves inside cwd."""
    try:
        base = Path(cwd).resolve()
        token = Path(path_token)
        resolved = token.resolve() if token.is_absolute() else (base / token).resolve()
        resolved.relative_to(base)
        return True
    except Exception:
        return False


def is_bash_command_scoped_to_cwd(command: str, cwd: str) -> bool:
    """Best-effort guard: reject Bash commands that reference paths outside cwd."""
    try:
        tokens = shlex.split(command)
    except Exception:
        return True

    for match in re.finditer(r"(?:^|[;&|]\s*)cd\s+([^;&|\s]+)", command):
        target = match.group(1).strip().strip("\"'")
        if target and not is_path_within_cwd(target, cwd):
            return False

    for tok in tokens:
        if tok.startswith("/"):
            if not is_path_within_cwd(tok, cwd):
                return False
            continue
        if tok.startswith("../") or tok == "..":
            if not is_path_within_cwd(tok, cwd):
                return False

    return True


PLAN_MODE_DENIED_TOOLS = frozenset(
    {"Write", "Edit", "MultiEdit", "Bash", "Task", "computer"}
)
PLAN_MODE_ALLOWED_TOOLS = frozenset(
    {"Read", "Glob", "Grep", "WebSearch", "Skill", "TodoWrite", "ExitPlanMode"}
)
