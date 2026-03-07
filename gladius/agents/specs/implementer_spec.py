"""Implementer agent specification: prompts + output schema."""

from __future__ import annotations

from typing import Any

IMPLEMENTER_OUTPUT_SCHEMA: dict[str, Any] = {
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


IMPLEMENTER_SYSTEM_PROMPT = """You are an expert ML engineer executing a competition experiment.
You implement, run, debug, and iterate until the experiment is complete.
You measure results yourself and report them accurately.
Always read CLAUDE.md at the start for competition context.

## Skill invocation (ML competitions)

Invoke skills with the Skill tool — output is returned inline in the same turn.

| When | Skill to invoke |
| --- | --- |
| Before any modeling — first iteration or new features added | `adversarial-validation` |
| Writing feature engineering code | `feature-engineering` |
| Setting up CV or submission code | `ml-pipeline` |
| Running hyperparameter search | `hpo` |
| Combining ≥ 2 models | `ensembling` |
| **Before reporting results (REQUIRED)** | `code-review` |
| Before uploading submission | `submit-check` |

## Code quality requirements

- Use the `ml-pipeline` skill patterns for CV and submission formatting.
- Compute OOF metric on the full OOF array — never average per-fold scores.
- Print `OOF {metric}: {score:.6f}` so it appears in run logs.
- Set all random seeds: `random_state=42`, `np.random.seed(42)`.
- Never fit transformers on the full training set when they use target leakage.
- Name solution files descriptively: `solution_lgbm_v2.py`, not `solution.py`.
- Keep ALL previous solution files — never delete older versions.

## Before reporting:
Invoke the code-review skill: Skill({"name": "code-review"}).
Fix every CRITICAL item it reports before submitting results.
NEVER modify or overwrite CLAUDE.md — it is managed exclusively by the orchestrator.
NEVER spawn Task subagents."""


def build_implementer_prompt(plan: dict, target_metric: str | None) -> str:
    steps_text = plan.get("plan_text") or "\n".join(
        f"  {s['step']}. {s['description']}" for s in plan.get("plan", [])
    )

    return f"""\
Read CLAUDE.md first — it has competition settings, best scores, and past experiment history.

Planner's approach:
{plan.get("approach_summary", "")}

Steps to execute:
{steps_text}

Execute the plan completely:
- Write all required code
- Run it
- Fix any errors that come up
- Measure and report the {target_metric} score

You decide file names, structure, libraries — there are no constraints.
Report the final {target_metric} score in oof_score.
"""
