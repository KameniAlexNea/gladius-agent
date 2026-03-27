"""Central logging configuration for Gladius runtime and trace files."""

from __future__ import annotations

import os
import sys

from loguru import logger

_STDERR_CONFIGURED = False


def configure_logging(project_dir: str | None = None) -> None:
    """Configure stderr sink.

    Idempotent across repeated calls.  File sinks are intentionally omitted —
    redirect stderr via nohup/tee when a persistent log file is needed.
    """

    global _STDERR_CONFIGURED

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
