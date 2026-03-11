"""Solver — single-agent, skills-first competition solver.

Replaces the old planner + implementer-coordinator + 6-subagent pipeline
with a single agent that handles planning, implementation, evaluation,
review, and submission autonomously, guided by skills loaded via MCP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gladius.agents._base import run_agent
from gladius.agents.specs.solver_spec import (
    SOLVER_OUTPUT_SCHEMA,
    SOLVER_SYSTEM_PROMPT,
    build_solver_prompt,
)

if TYPE_CHECKING:
    from gladius.state import CompetitionState

# Backward-compatible alias so existing imports keep working.
OUTPUT_SCHEMA = SOLVER_OUTPUT_SCHEMA


async def run_solver(
    state: "CompetitionState",
    project_dir: str,
) -> dict:
    """Run the solver. Returns a result dict matching OUTPUT_SCHEMA."""
    prompt = build_solver_prompt(target_metric=state.target_metric)
    result, _ = await run_agent(
        agent_name="solver",
        prompt=prompt,
        system_prompt=SOLVER_SYSTEM_PROMPT,
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
        max_turns=300,
    )
    return result
