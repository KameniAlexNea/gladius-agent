"""Session replay helpers for postmortem diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger


def export_recent_session_diagnostics(
    *,
    project_dir: str | Path,
    output_file: str | Path,
    limit_sessions: int = 5,
    limit_messages: int = 40,
) -> Path | None:
    """Export recent Claude SDK sessions/messages for local postmortem analysis."""
    try:
        from claude_agent_sdk import get_session_messages, list_sessions
    except Exception as exc:
        logger.debug(f"Session diagnostics unavailable (SDK import failed): {exc}")
        return None

    project_dir = Path(project_dir).resolve()
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        sessions = list_sessions(directory=str(project_dir), limit=limit_sessions)
    except Exception as exc:
        logger.warning(f"Failed to list sessions for diagnostics: {exc}")
        return None

    payload: list[dict[str, Any]] = []
    for session in sessions:
        entry: dict[str, Any] = {
            "session_id": session.session_id,
            "summary": session.summary,
            "last_modified": session.last_modified,
            "cwd": session.cwd,
            "git_branch": session.git_branch,
            "messages": [],
        }
        try:
            msgs = get_session_messages(
                session_id=session.session_id,
                directory=str(project_dir),
                limit=limit_messages,
            )
            entry["messages"] = [
                {
                    "type": m.type,
                    "uuid": m.uuid,
                    "parent_tool_use_id": m.parent_tool_use_id,
                }
                for m in msgs
            ]
        except Exception as exc:
            entry["message_error"] = str(exc)
        payload.append(entry)

    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    logger.info(f"Wrote session diagnostics: {output_path}")
    return output_path
