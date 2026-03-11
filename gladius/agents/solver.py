"""Gladius — single autonomous competition agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gladius.agents._base import run_agent
from gladius.agents.specs.gladius_spec import (
    GLADIUS_OUTPUT_SCHEMA,
    GLADIUS_SYSTEM_PROMPT,
    build_gladius_prompt,
)

if TYPE_CHECKING:
    from gladius.state import CompetitionState

OUTPUT_SCHEMA = GLADIUS_OUTPUT_SCHEMA


async def run_gladius(
    state: "CompetitionState",
    project_dir: str,
) -> dict:
    """Run Gladius for one session. Returns a result dict matching OUTPUT_SCHEMA."""
    prompt = build_gladius_prompt(target_metric=state.target_metric)
    result, _ = await run_agent(
        agent_name="gladius",
        prompt=prompt,
        system_prompt=GLADIUS_SYSTEM_PROMPT,
        allowed_tools=[
            "Read",
            "Write",
            "Edit",
            "MultiEdit",
            "Bash",
            "Glob",
            "Grep",
            "TodoWrite",
            "WebSearch",
        ],
        output_schema=OUTPUT_SCHEMA,
        cwd=project_dir,
        max_turns=500,
    )
    return result


# Backward-compatible alias
run_solver = run_gladius
