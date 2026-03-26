"""LangSmith tracing setup for orchestrator runs."""

from __future__ import annotations

import contextlib
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


def configure_langsmith_env() -> bool:
    """Populate LangSmith env defaults and validate tracing prerequisites."""
    if not should_enable_langsmith_for_ollama_bridge():
        return False

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_ENDPOINT", _LANGSMITH_DEFAULT_ENDPOINT)
    os.environ.setdefault("LANGSMITH_PROJECT", _LANGSMITH_DEFAULT_PROJECT)

    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not api_key:
        logger.warning(
            "LANGSMITH_API_KEY is not set; LangSmith tracing was requested for "
            "ANTHROPIC_BASE_URL+OLLAMA mode but will remain disabled."
        )
        return False

    tracing_flag = os.getenv("LANGSMITH_TRACING", "").strip().lower()
    return tracing_flag in {"1", "true", "yes", "on"}


def init_langsmith_client() -> tuple[Any | None, str | None]:
    """Create LangSmith client from env for tracing, if enabled."""
    if not configure_langsmith_env():
        return None, None

    endpoint = os.getenv("LANGSMITH_ENDPOINT", _LANGSMITH_DEFAULT_ENDPOINT).strip()
    project_name = os.getenv("LANGSMITH_PROJECT", _LANGSMITH_DEFAULT_PROJECT).strip()
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()

    try:
        import langsmith

        client = langsmith.Client(api_key=api_key, api_url=endpoint)
        logger.info(
            f"LangSmith tracing enabled (project={project_name!r}, endpoint={endpoint!r})."
        )
        return client, project_name
    except Exception as exc:
        logger.warning(f"Failed to initialize LangSmith client: {exc}")
        return None, None


def langsmith_tracing_context(client: Any | None, project_name: str | None):
    """Return LangSmith tracing_context, or a no-op when unavailable."""
    if client is None or not project_name:
        return contextlib.nullcontext()

    try:
        from langsmith.run_helpers import tracing_context

        return tracing_context(client=client, project_name=project_name, enabled=True)
    except Exception as exc:
        logger.warning(f"Failed to create LangSmith tracing context: {exc}")
        return contextlib.nullcontext()
