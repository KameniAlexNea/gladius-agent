"""
Implementer — ML experiment coordinator.

Orchestrates specialized subagents (ml-scaffolder, ml-developer, ml-scientist,
ml-evaluator, code-reviewer, submission-builder) via the Agent() tool.
Routes between phases by reading EXPERIMENT_STATE.json after each subagent completes.

Fresh session every iteration (not resumed) — it works from the plan alone.
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
    Run the coordinator. Returns a result dict matching OUTPUT_SCHEMA.
    """
    prompt = build_implementer_prompt(plan=plan, target_metric=state.target_metric)
    result, _ = await run_agent(
        agent_name="implementer",
        prompt=prompt,
        system_prompt=IMPLEMENTER_SYSTEM_PROMPT,
        allowed_tools=[
            "Agent(ml-scaffolder,ml-developer,ml-scientist,ml-evaluator,code-reviewer,submission-builder)",
            "Read",
            "Write",
            "Glob",
            "TodoWrite",
        ],
        output_schema=OUTPUT_SCHEMA,
        cwd=project_dir,
        max_turns=30,
    )
    return result
