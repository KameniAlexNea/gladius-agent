"""
Zindi platform tools exposed as an MCP server for Claude agents.

Credentials are read from environment variables:
    ZINDI_USERNAME  (or USER_NAME as fallback)
    ZINDI_PASSWORD  (or PASSWORD as fallback)
    ZINDI_CHALLENGE_ID     — immutable Zindi challenge slug/id (preferred)
    ZINDI_CHALLENGE_INDEX  — 0-based fallback index when challenge_id is not set

Usage:
    from gladius.tools.zindi_tools import zindi_server
    options = ClaudeAgentOptions(mcp_servers={"zindi": zindi_server}, ...)
"""

from __future__ import annotations

import os
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from src.tools._response import err as _err
from src.tools._response import ok as _ok
from src.tools.zindi_common import (
    create_zindi_user_from_env,
    select_zindi_challenge,
)


def _get_user():
    """Authenticate and return a connected Zindian instance."""
    user = create_zindi_user_from_env()
    select_zindi_challenge(
        user=user,
        competition_id=None,
        env_challenge_id=os.getenv("ZINDI_CHALLENGE_ID"),
        env_challenge_index=os.getenv("ZINDI_CHALLENGE_INDEX", "0"),
    )
    return user


# ── Tools ─────────────────────────────────────────────────────────────────────


@tool(
    "zindi_submit",
    (
        "Submit a prediction CSV file to the active Zindi challenge. "
        "Returns submission confirmation or an error message. "
        "Checks remaining daily submissions before proceeding."
    ),
    {"file_path": str, "comment": str},
)
async def zindi_submit(args: dict[str, Any]) -> dict[str, Any]:
    try:
        user = _get_user()

        remaining = user.remaining_subimissions
        if remaining <= 0:
            return _err(
                "quota_exceeded", "No remaining submissions today. Try again tomorrow."
            )

        user.submit(
            filepaths=[args["file_path"]],
            comments=[args.get("comment") or f"Submission: {args['file_path']}"],
        )

        remaining_after = user.remaining_subimissions
        return _ok(
            (
                f"Submission accepted.\n"
                f"File: {args['file_path']}\n"
                f"Remaining submissions today: {remaining_after}"
            ),
            data={"remaining_submissions": remaining_after},
        )
    except Exception as e:
        return _err("submission_failed", f"Submission error: {e}")


@tool(
    "zindi_leaderboard",
    (
        "Fetch the current Zindi leaderboard for the active challenge. "
        "Returns a text table of the top N entries."
    ),
    {"top_n": int},
)
async def zindi_leaderboard(args: dict[str, Any]) -> dict[str, Any]:
    try:
        user = _get_user()
        top_n = int(args.get("top_n", 20))
        lb = user.leaderboard()
        if lb is None:
            return _err("leaderboard_unavailable", "Leaderboard unavailable.")
        return _ok(lb.head(top_n).to_string(), data={"top_n": top_n})
    except Exception as e:
        return _err("leaderboard_failed", f"Error: {e}")


@tool(
    "zindi_submission_history",
    (
        "Retrieve the authenticated user's submission history for the active Zindi challenge. "
        "Returns a text table with scores and timestamps."
    ),
    {},
)
async def zindi_submission_history(args: dict[str, Any]) -> dict[str, Any]:
    try:
        user = _get_user()
        sb = user.submission_board()
        if sb is None:
            return _err("history_unavailable", "No submission history found.")
        return _ok(sb.to_string())
    except Exception as e:
        return _err("history_failed", f"Error: {e}")


@tool(
    "zindi_status",
    (
        "Check current rank and remaining daily submission quota on the active Zindi challenge."
    ),
    {},
)
async def zindi_status(args: dict[str, Any]) -> dict[str, Any]:
    try:
        user = _get_user()
        lines = [
            f"Challenge: {user.which_challenge}",
            f"Current rank: {user.my_rank}",
            f"Remaining submissions today: {user.remaining_subimissions}",
        ]
        return _ok(
            "\n".join(lines),
            data={
                "rank": user.my_rank,
                "remaining_submissions": user.remaining_subimissions,
            },
        )
    except Exception as e:
        return _err("status_failed", f"Error: {e}")


# ── MCP server instance ───────────────────────────────────────────────────────
zindi_server = create_sdk_mcp_server(
    name="zindi",
    version="1.0.0",
    tools=[zindi_submit, zindi_leaderboard, zindi_submission_history, zindi_status],
)
