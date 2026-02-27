"""
Zindi platform tools exposed as an MCP server for Claude agents.

Credentials are read from environment variables:
    ZINDI_USERNAME  (or USER_NAME as fallback)
    ZINDI_PASSWORD  (or PASSWORD as fallback)
    ZINDI_CHALLENGE_INDEX  — 0-based index of the challenge to select (default 0)

Usage:
    from gladius.tools.zindi_tools import zindi_server
    options = ClaudeAgentOptions(mcp_servers={"zindi": zindi_server}, ...)
"""

from __future__ import annotations

import os
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool


def _get_user():
    """Authenticate and return a connected Zindian instance."""
    from zindi.user import (
        Zindian,
    )  # lazy import — only available when zindi is installed

    username = os.getenv("ZINDI_USERNAME") or os.getenv("USER_NAME")
    password = os.getenv("ZINDI_PASSWORD") or os.getenv("PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Zindi credentials not found. "
            "Set ZINDI_USERNAME and ZINDI_PASSWORD environment variables."
        )
    user = Zindian(username=username, fixed_password=password)
    challenge_index = int(os.getenv("ZINDI_CHALLENGE_INDEX", "0"))
    user.select_a_challenge(fixed_index=challenge_index)
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
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "No remaining submissions today. Try again tomorrow.",
                    }
                ],
                "is_error": True,
            }

        user.submit(
            filepaths=[args["file_path"]],
            comments=[args.get("comment") or f"Submission: {args['file_path']}"],
        )

        remaining_after = user.remaining_subimissions
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Submission accepted.\n"
                        f"File: {args['file_path']}\n"
                        f"Remaining submissions today: {remaining_after}"
                    ),
                }
            ]
        }
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Submission error: {e}"}],
            "is_error": True,
        }


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
            return {"content": [{"type": "text", "text": "Leaderboard unavailable."}]}
        return {"content": [{"type": "text", "text": lb.head(top_n).to_string()}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error: {e}"}], "is_error": True}


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
            return {
                "content": [{"type": "text", "text": "No submission history found."}]
            }
        return {"content": [{"type": "text", "text": sb.to_string()}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error: {e}"}], "is_error": True}


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
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error: {e}"}], "is_error": True}


# ── MCP server instance ───────────────────────────────────────────────────────
zindi_server = create_sdk_mcp_server(
    name="zindi",
    version="1.0.0",
    tools=[zindi_submit, zindi_leaderboard, zindi_submission_history, zindi_status],
)
