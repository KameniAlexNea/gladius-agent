"""
Kaggle API tools exposed as an MCP server for Claude agents.

Usage:
    from src.tools.kaggle_tools import kaggle_server
    options = ClaudeAgentOptions(mcp_servers={"kaggle": kaggle_server}, ...)
"""

import subprocess
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from src.tools._response import err as _err
from src.tools._response import ok as _ok


@tool(
    "kaggle_submit",
    "Submit a CSV file to the Kaggle competition leaderboard. Returns submission ID and status.",
    {"competition": str, "file_path": str, "message": str},
)
async def kaggle_submit(args: dict[str, Any]) -> dict[str, Any]:
    result = subprocess.run(
        [
            "kaggle",
            "competitions",
            "submit",
            "-c",
            args["competition"],
            "-f",
            args["file_path"],
            "-m",
            args["message"],
        ],
        capture_output=True,
        text=True,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        return _err("submission_failed", output or "Kaggle submission failed")
    return _ok(output or "Kaggle submission accepted")


@tool(
    "kaggle_leaderboard",
    "Fetch the current public leaderboard top-N rows for a competition. Returns CSV text.",
    {"competition": str, "top_n": int},
)
async def kaggle_leaderboard(args: dict[str, Any]) -> dict[str, Any]:
    result = subprocess.run(
        [
            "kaggle",
            "competitions",
            "leaderboard",
            "-c",
            args["competition"],
            "--show",
            "--csv",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return _err(
            "leaderboard_unavailable",
            (result.stderr or result.stdout).strip() or "Failed to fetch leaderboard",
        )
    top_n = args.get("top_n", 20)
    lines = result.stdout.strip().split("\n")[: top_n + 1]  # +1 for header
    return _ok("\n".join(lines), data={"top_n": top_n})


@tool(
    "kaggle_submission_history",
    "Fetch the recent submission history for a competition.",
    {"competition": str},
)
async def kaggle_submission_history(args: dict[str, Any]) -> dict[str, Any]:
    result = subprocess.run(
        [
            "kaggle",
            "competitions",
            "submissions",
            "-c",
            args["competition"],
            "--csv",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return _err(
            "history_unavailable",
            (result.stderr or result.stdout).strip()
            or "Failed to fetch submission history",
        )
    return _ok(result.stdout.strip())


# ── MCP server instance ───────────────────────────────────────────────────────
kaggle_server = create_sdk_mcp_server(
    name="kaggle",
    version="1.0.0",
    tools=[kaggle_submit, kaggle_leaderboard, kaggle_submission_history],
)
