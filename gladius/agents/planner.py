"""
Planner — explores the competition data and decides what to try next.

This is the only persistent agent (resumed every iteration). It accumulates
understanding of the competition, the data, what has worked and what hasn't.

It is NOT told how to structure files, what to name things, or how to compute
metrics. Its only job is to produce a concrete, ordered action plan.
"""
from typing import TYPE_CHECKING

from gladius.agents._base import run_agent

if TYPE_CHECKING:
    from gladius.state import CompetitionState

# ── Output schema ─────────────────────────────────────────────────────────────
# Minimal. Claude writes the steps; orchestrator passes them to the implementer.
OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["plan", "approach_summary"],
    "properties": {
        "approach_summary": {
            "type": "string",
            "description": "One paragraph — what you decided to try and why",
        },
        "plan": {
            "type": "array",
            "description": "Ordered implementation steps for the implementer to follow",
            "items": {
                "type": "object",
                "required": ["step", "description"],
                "properties": {
                    "step":        {"type": "integer"},
                    "description": {"type": "string"},
                },
            },
        },
        "expected_metric_delta": {
            "type": "number",
            "description": "Rough estimate of OOF improvement. Can be 0 if unknown.",
        },
    },
    "additionalProperties": False,
}


async def run_planner(
    state: "CompetitionState",
    data_dir: str,
    project_dir: str,
) -> tuple[dict, str]:
    """
    Explore, analyse, plan.

    Returns (plan_dict, session_id).
    session_id is stored in state.planner_session_id to resume next iteration.
    """
    prompt = f"""\
Competition : {state.competition_id}
Metric      : {state.target_metric} ({state.metric_direction})
Current best: {state.best_oof_score:.6f}
Iteration   : {state.iteration} / {state.max_iterations}
Data dir    : {data_dir}
Project dir : {project_dir}

Completed runs ({len(state.experiments)} total):
{_format_experiments(state.experiments[-5:])}

Failed runs:
{_format_list(state.failed_runs[-3:])}

Your job:
- Explore the data and existing code however you like (EDA, profiling, reading notebooks, web search)
- Decide the highest-impact next thing to try
- Output a concrete, ordered plan the implementer can follow without further guidance

Be specific. The implementer will execute your plan blindly.
"""
    return await run_agent(
        agent_name="planner",
        prompt=prompt,
        system_prompt=(
            "You are an expert ML competition analyst. "
            "You explore first, then plan. "
            "You never implement code yourself — you produce plans for an implementer. "
            "Your plans are specific, ordered, and self-contained."
        ),
        allowed_tools=["Read", "Glob", "Grep", "Bash", "WebSearch", "Task"],
        output_schema=OUTPUT_SCHEMA,
        cwd=project_dir,
        resume=state.planner_session_id,
        max_turns=40,
    )


def _format_experiments(experiments: list) -> str:
    if not experiments:
        return "  (none yet)"
    lines = []
    for e in experiments:
        score = e.get("oof_score", "?")
        notes = e.get("notes", "")
        files = ", ".join(e.get("solution_files", []))
        lines.append(f"  oof={score}  files={files}  notes={notes}")
    return "\n".join(lines)


def _format_list(items: list) -> str:
    if not items:
        return "  (none)"
    return "\n".join(f"  - {x}" for x in items)
