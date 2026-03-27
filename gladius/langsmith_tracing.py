"""LangSmith tracing setup for orchestrator runs."""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

_LANGSMITH_DEFAULT_ENDPOINT = "https://api.smith.langchain.com"
_LANGSMITH_DEFAULT_PROJECT = "gladius-agent"


def should_enable_langsmith_for_ollama_bridge() -> bool:
    """Auto-enable tracing when routing Anthropic SDK requests to Ollama."""
    return bool(os.getenv("ANTHROPIC_BASE_URL", "").strip()) and (
        os.getenv("ANTHROPIC_API_KEY", "").strip().upper() == "OLLAMA"
    )


def _explicit_tracing_enabled() -> bool:
    raw = os.getenv("GLADIUS_ENABLE_LANGSMITH", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def configure_langsmith_env() -> bool:
    """Populate LangSmith env defaults and validate tracing prerequisites."""
    if not (_explicit_tracing_enabled() or should_enable_langsmith_for_ollama_bridge()):
        return False

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_ENDPOINT", _LANGSMITH_DEFAULT_ENDPOINT)
    os.environ.setdefault("LANGSMITH_PROJECT", _LANGSMITH_DEFAULT_PROJECT)

    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not api_key:
        logger.warning(
            "LANGSMITH_API_KEY is not set; LangSmith tracing was requested but will remain disabled."
        )
        return False

    tracing_flag = os.getenv("LANGSMITH_TRACING", "").strip().lower()
    return tracing_flag in {"1", "true", "yes", "on"}


def init_langsmith_tracing() -> bool:
    """Configure LangSmith env and install the claude-agent-sdk integration.

    Call once at startup.  Returns True when tracing is active.
    """
    if not configure_langsmith_env():
        return False

    endpoint = os.getenv("LANGSMITH_ENDPOINT", _LANGSMITH_DEFAULT_ENDPOINT).strip()
    project_name = os.getenv("LANGSMITH_PROJECT", _LANGSMITH_DEFAULT_PROJECT).strip()

    try:
        from langsmith.integrations.claude_agent_sdk import configure_claude_agent_sdk

        configure_claude_agent_sdk()
        logger.info(
            f"LangSmith tracing enabled (project={project_name!r}, endpoint={endpoint!r})."
        )
        return True
    except Exception as exc:
        logger.warning(f"Failed to initialize LangSmith tracing: {exc}")
        return False
