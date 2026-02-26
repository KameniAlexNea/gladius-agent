"""
Implementer — executes the planner's plan end-to-end.

Writes code, runs it, handles errors, and reports results.

Fresh session every iteration (not resumed) — it works from the plan alone.
No naming conventions. No forced output format. Claude decides everything
about file structure, libraries, and how to measure the metric.
"""
from typing import TYPE_CHECKING

from gladius.agents._base import run_agent

if TYPE_CHECKING:
    from gladius.state import CompetitionState

OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["status", "oof_score"],
    "properties": {
        "status": {
            "type": "string",
            "enum": ["success", "error", "timeout", "oom"],
        },
        "oof_score": {
            "type": "number",
            "description": (
                "OOF/validation score achieved. "
                "Use the competition metric direction: higher is better for maximize, lower for minimize. "
                "Set to -1 if the run failed."
            ),
        },
        "solution_files": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Paths to all files you created or modified",
        },
        "submission_file": {
            "type": "string",
            "description": "Path to the test-set submission CSV (empty string if not produced)",
        },
        "notes": {
            "type": "string",
            "description": "Brief summary: what you built, what score you got, any issues",
        },
        "error_message": {
            "type": "string",
            "description": "Only populated on error/timeout/oom — what went wrong",
        },
    },
    "additionalProperties": False,
}


async def run_implementer(
    plan: dict,
    state: "CompetitionState",
    project_dir: str,
) -> dict:
    """
    Execute the plan. Return result dict matching OUTPUT_SCHEMA.
    """
    steps_text = "\n".join(
        f"  {s['step']}. {s['description']}" for s in plan.get("plan", [])
    )

    prompt = f"""\
Competition : {state.competition_id}
Metric      : {state.target_metric} ({state.metric_direction})
Data dir    : {state.data_dir}
Project dir : {project_dir}

Planner's approach:
{plan.get("approach_summary", "")}

Steps to execute:
{steps_text}

Existing experiments for context:
{_format_experiments(state.experiments[-3:])}

Execute the plan completely:
- Write all required code
- Run it
- Fix any errors that come up
- Measure and report the {state.target_metric} score

You decide file names, structure, libraries — there are no constraints.
Report the final {state.target_metric} score in oof_score.
"""
    result, _ = await run_agent(
        agent_name="implementer",
        prompt=prompt,
        system_prompt=(
            "You are an expert ML engineer executing a competition experiment. "
            "You implement, run, debug, and iterate until the experiment completes. "
            "You measure results yourself and report them accurately."
        ),
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        output_schema=OUTPUT_SCHEMA,
        cwd=project_dir,
        max_turns=80,
    )
    return result


def _format_experiments(experiments: list) -> str:
    if not experiments:
        return "  (none yet)"
    lines = []
    for e in experiments:
        lines.append(
            f"  oof={e.get('oof_score', '?')}  "
            f"files={', '.join(e.get('solution_files', []))}  "
            f"notes={e.get('notes', '')}"
        )
    return "\n".join(lines)
