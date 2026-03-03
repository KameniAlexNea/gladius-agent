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
    "required": ["status", "oof_score", "quality_score"],
    "properties": {
        "status": {
            "type": "string",
            "enum": ["success", "error", "timeout", "oom"],
        },
        "oof_score": {
            "type": ["number", "null"],
            "description": (
                "OOF/validation score achieved for metric-driven competitions. "
                "Use the competition metric direction: higher is better for maximize, "
                "lower for minimize. Set to -1 if the run failed. "
                "Set to null if there is no target metric (open-ended task)."
            ),
        },
        "quality_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "description": (
                "Self-assessed quality score 0–100. "
                "For metric-driven tasks: 0=failure, 50=below-baseline, 75=solid, 100=perfect. "
                "For open-ended tasks: rate completeness and correctness of the deliverable "
                "against the task description in README.md. "
                "Always required; use 0 on error or timeout."
            ),
        },
        "solution_files": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Paths to all files you created or modified",
        },
        "submission_file": {
            "type": "string",
            "description": (
                "Path to the main deliverable — CSV for ML competitions, "
                "zip/binary/URL-file for open-ended tasks. Empty string if not produced."
            ),
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
    steps_text = plan.get("plan_text") or "\n".join(
        f"  {s['step']}. {s['description']}" for s in plan.get("plan", [])
    )

    prompt = f"""\
Read CLAUDE.md first — it has competition settings, best scores, and past experiment history.

Planner's approach:
{plan.get("approach_summary", "")}

Steps to execute:
{steps_text}

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
            "You are an expert engineer executing a task. "
            "You implement, run, debug, and iterate until the task is complete. "
            "You measure results yourself and report them accurately. "
            "Always read CLAUDE.md at the start for task context. "
            "Before reporting your final result, read .claude/skills/code-review/SKILL.md "
            "and fix every CRITICAL item (leakage, metric correctness, submission format)."
        ),
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "TodoWrite"],
        output_schema=OUTPUT_SCHEMA,
        cwd=project_dir,
        max_turns=80,
    )
    return result
