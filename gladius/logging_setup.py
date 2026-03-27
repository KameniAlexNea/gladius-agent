"""Central logging configuration for Gladius runtime and trace files."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

_STDERR_CONFIGURED = False
_FILE_SINKS_CONFIGURED_FOR: Path | None = None


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def configure_logging(project_dir: str | Path | None = None) -> None:
    """Configure stderr and optional file/json sinks.

    Idempotent across repeated calls; file sinks are added once per project directory.
    """

    global _STDERR_CONFIGURED, _FILE_SINKS_CONFIGURED_FOR

    if not _STDERR_CONFIGURED:
        logger.remove()
        logger.configure(
            extra={
                "run_id": "-",
                "iteration": "-",
                "attempt": "-",
                "agent": "-",
                "session_id": "-",
                "task_id": "-",
                "tool_use_id": "-",
            }
        )
        level = os.getenv("GLADIUS_LOG_LEVEL", "DEBUG").strip().upper() or "DEBUG"
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
            "<level>{level:<8}</level> "
            "[it={extra[iteration]} at={extra[attempt]} agent={extra[agent]}] "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stderr,
            level=level,
            colorize=True,
            backtrace=False,
            diagnose=False,
            format=fmt,
        )
        _STDERR_CONFIGURED = True

    if project_dir is None:
        return

    p = Path(project_dir).resolve()
    if _FILE_SINKS_CONFIGURED_FOR == p:
        return

    logs_dir = p / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    level = os.getenv("GLADIUS_LOG_LEVEL", "DEBUG").strip().upper() or "DEBUG"
    text_fmt = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} {level:<8} "
        "it={extra[iteration]} at={extra[attempt]} agent={extra[agent]} "
        "{message}"
    )
    logger.add(
        str(logs_dir / "gladius.log"),
        level=level,
        format=text_fmt,
        colorize=False,
        backtrace=False,
        diagnose=False,
        encoding="utf-8",
        rotation="20 MB",
        retention=10,
    )

    if _flag("GLADIUS_JSON_LOG", True):
        logger.add(
            str(logs_dir / "gladius.jsonl"),
            level=level,
            serialize=True,
            backtrace=False,
            diagnose=False,
            encoding="utf-8",
            rotation="20 MB",
            retention=10,
        )

    _FILE_SINKS_CONFIGURED_FOR = p
