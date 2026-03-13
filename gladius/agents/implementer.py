"""
Implementer — ML experiment coordinator.

Orchestrates specialized subagents (ml-scaffolder, ml-developer, ml-scientist,
ml-evaluator, code-reviewer, submission-builder) via the Agent() tool.
Routes between phases by reading EXPERIMENT_STATE.json after each subagent completes.

Fresh session every iteration (not resumed) — it works from the plan alone.
"""

import sys
from pathlib import Path
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
    skills_dir = str(Path(project_dir) / ".claude" / "skills")
    mcp_servers = {
        "skills-on-demand": {
            "type": "stdio",
            "command": sys.executable,
            "args": ["-m", "skills_on_demand.server"],
            "env": {"SKILLS_DIR": skills_dir},
        }
    }

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
            "mcp__skills-on-demand__search_skills",
            "mcp__skills-on-demand__list_skills",
        ],
        output_schema=OUTPUT_SCHEMA,
        cwd=project_dir,
        mcp_servers=mcp_servers,
        max_turns=30,
    )
    return result
