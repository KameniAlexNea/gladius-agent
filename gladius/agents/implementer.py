"""
Implementer — executes the planner's plan end-to-end.

Writes code, runs it, handles errors, and reports results.

Fresh session every iteration (not resumed) — it works from the plan alone.
No naming conventions. No forced output format. Claude decides everything
about file structure, libraries, and how to measure the metric.
"""

from typing import TYPE_CHECKING

from gladius.agents._base import run_agent
from gladius.agents.specs.implementer_spec import (
    IMPLEMENTER_OUTPUT_SCHEMA,
    IMPLEMENTER_SYSTEM_PROMPT,
    build_implementer_prompt,
)

if TYPE_CHECKING:
    from gladius.state import CompetitionState

# Backwards-compatible alias for existing imports/tests.
OUTPUT_SCHEMA = IMPLEMENTER_OUTPUT_SCHEMA


async def run_implementer(
    plan: dict,
    state: "CompetitionState",
    project_dir: str,
) -> dict:
    """
    Execute the plan. Return result dict matching OUTPUT_SCHEMA.
    """
    prompt = build_implementer_prompt(plan=plan, target_metric=state.target_metric)
    result, _ = await run_agent(
        agent_name="implementer",
        prompt=prompt,
        system_prompt=IMPLEMENTER_SYSTEM_PROMPT,
        allowed_tools=[
            "Read",
            "Write",
            "Edit",
            "Bash",
            "Glob",
            "Grep",
            "TodoWrite",
            "Skill",
        ],
        output_schema=OUTPUT_SCHEMA,
        cwd=project_dir,
        max_turns=80,
    )
    return result
