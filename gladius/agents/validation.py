"""
Validation Agent — validates a result and recommends whether to submit.

Replaces: validation_agent + submission_decider + notifier (3 nodes → 1)

IMPORTANT: This agent only REPORTS its decision. It does NOT mutate state.
The orchestrator reads .is_improvement and .submit and applies them.
This was the critical regression in the LangGraph version — the old node
was writing best_oof_score itself, breaking the submission gate permanently.

No session continuity: each validation is an independent, stateless task.
"""

from typing import TYPE_CHECKING

from gladius.agents._base import run_agent
from gladius.agents.specs.validation_spec import (
    VALIDATION_OUTPUT_SCHEMA,
    VALIDATION_SYSTEM_PROMPT,
    build_validation_prompt,
)

if TYPE_CHECKING:
    from gladius.state import CompetitionState

# Backwards-compatible aliases for existing imports/tests.
SYSTEM_PROMPT = VALIDATION_SYSTEM_PROMPT
OUTPUT_SCHEMA = VALIDATION_OUTPUT_SCHEMA


async def run_validation_agent(
    solution_path: str,
    oof_score: float | None,
    quality_score: float,
    submission_path: str,
    state: "CompetitionState",
    project_dir: str,
    platform: str = "none",
) -> dict:
    """
    Validate a new experiment result and recommend submit/hold.

    Injects the platform-specific MCP server so the agent can query live
    submission quota directly instead of relying on the state counter.
    """
    # ── Platform-specific MCP server ──────────────────────────────────────
    mcp_servers: dict = {}
    quota_tool: str = ""
    quota_instruction: str = ""

    if platform == "zindi":
        import sys

        mcp_servers = {
            "zindi": {
                "type": "stdio",
                "command": sys.executable,
                "args": [
                    "-c",
                    "from gladius.tools.zindi_tools import zindi_server; "
                    "import asyncio; asyncio.run(zindi_server.run())",
                ],
            }
        }
        quota_tool = "mcp__zindi__zindi_status"
        quota_instruction = (
            "3. Call `zindi_status` to get today's remaining submission quota.\n"
        )
    elif platform == "kaggle":
        import sys

        mcp_servers = {
            "kaggle": {
                "type": "stdio",
                "command": sys.executable,
                "args": [
                    "-c",
                    "from gladius.tools.kaggle_tools import kaggle_server; "
                    "import asyncio; asyncio.run(kaggle_server.run())",
                ],
            }
        }
        quota_tool = "mcp__kaggle__kaggle_submission_history"
        quota_instruction = (
            "3. Call `kaggle_submission_history` and count how many submissions "
            "were made today to determine remaining quota.\n"
        )
    elif platform == "fake":
        import sys

        mcp_servers = {
            "fake": {
                "type": "stdio",
                "command": sys.executable,
                "args": [
                    "-c",
                    "from gladius.tools.fake_platform_tools import fake_server; "
                    "import asyncio; asyncio.run(fake_server.run())",
                ],
            }
        }
        quota_tool = "mcp__fake__fake_status"
        quota_instruction = (
            "3. Call `fake_status` to get your current submission count and rank.\n"
        )
    # platform == "none": no MCP tools — no external platform to query

    allowed_tools = ["Read", "Grep"] + ([quota_tool] if quota_tool else [])

    prompt = build_validation_prompt(
        solution_path=solution_path,
        oof_score=oof_score,
        quality_score=quality_score,
        submission_path=submission_path,
        target_metric=state.target_metric,
        metric_direction=state.metric_direction,
        best_oof_score=state.best_oof_score,
        best_quality_score=state.best_quality_score,
        submission_count=state.submission_count,
        max_submissions_per_day=state.max_submissions_per_day,
        quota_instruction=quota_instruction,
        project_dir=project_dir,
    )
    result, _ = await run_agent(
        agent_name="validation",
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=allowed_tools,
        output_schema=OUTPUT_SCHEMA,
        cwd=project_dir,
        mcp_servers=mcp_servers,
        max_turns=25,
    )
    return result
