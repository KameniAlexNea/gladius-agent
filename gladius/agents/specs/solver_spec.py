"""Solver agent specification: prompts + output schema."""

from __future__ import annotations

from typing import Any

SOLVER_OUTPUT_SCHEMA: dict[str, Any] = {
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
                "OOF/validation score for metric-driven competitions. "
                "Higher=better for maximize, lower for minimize. "
                "Set to -1 if the run failed. "
                "Set to null if there is no target metric (open-ended task)."
            ),
        },
        "quality_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "description": (
                "Self-assessed quality score 0-100. "
                "For metric tasks: 0=failure, 50=below-baseline, 75=solid, 100=perfect. "
                "For open-ended tasks: rate completeness and correctness against README. "
                "Always required; use 0 on error."
            ),
        },
        "solution_files": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Paths to all files created or modified",
        },
        "submission_file": {
            "type": "string",
            "description": (
                "Path to the deliverable — CSV for ML competitions, "
                "zip/binary/URL-file for open-ended tasks. Empty string if not produced."
            ),
        },
        "notes": {
            "type": "string",
            "description": "Brief summary: what was built, score achieved, any issues",
        },
        "error_message": {
            "type": "string",
            "description": "Only populated on error/timeout/oom — what went wrong",
        },
        "total_turns": {
            "type": ["integer", "null"],
            "description": "Total turns used in this run for efficiency telemetry.",
        },
    },
    "additionalProperties": False,
}

SOLVER_SYSTEM_PROMPT = """\
You are a competition solver. You handle everything in a single autonomous session:
exploring data, planning, implementing, evaluating, reviewing, and submitting.
No subagents. No coordinators. Your only collaborators are skills loaded via MCP.

> MANDATORY FIRST ACTION — before anything else:
> mcp__skills-on-demand__search_skills({"query": "<task type, e.g. ml classification tabular>", "top_k": 5})
> Then read the best-match .claude/skills/<name>/SKILL.md and follow its patterns.

## Context

Read CLAUDE.md first. It contains: competition ID, metric, data_dir, best scores,
past experiments, and failed approaches. NEVER modify CLAUDE.md.

## PATH NOTE

.claude/EXPERIMENT_STATE.json is a LOCAL file inside the competition project directory
(same folder as CLAUDE.md). Use it to track iteration progress. Reset it (write `{}`)
at the start of each new experiment.

## Execution loop

1. Search skills → load SKILL.md → follow its patterns
2. Explore data: read data/train.csv head, check dtypes, target distribution
3. Plan next steps with TodoWrite
4. Implement in src/ and scripts/:
   - Install packages with `uv add <pkg>` (NEVER pip install)
   - Run with `uv run python`
   - Fix all errors until the pipeline runs clean
5. Print `OOF <metric>: <value>` in your training script (required for score extraction)
6. Save OOF predictions to artifacts/oof.npy
   - Binary classification: shape (n_samples,)
   - Multiclass: shape (n_samples, n_classes); save class order to artifacts/oof_classes.npy
7. Review your own code for data leakage, CV contamination, wrong metric, train/test mismatch
8. Build submission:
   - Load sample_submission.csv for exact format (columns, row count, ID values)
   - Save to submissions/submission.csv
9. Search for next improvement skill; reset .claude/EXPERIMENT_STATE.json; iterate
10. Call StructuredOutput when results are satisfactory or ideas are exhausted

## Continuous improvement

After each successful experiment:
- mcp__skills-on-demand__search_skills({"query": "improve <metric> <task>", "top_k": 3})
- Load the found skill, apply its technique, run the next experiment
- Continue until you are satisfied with the result

## Coding rules

- Use pathlib; no hardcoded absolute paths
- Set random_state=42 everywhere reproducibility matters
- Keep all imports at the top of each file
- Track your work with TodoWrite
- Write to .claude/EXPERIMENT_STATE.json after each major phase
"""


def build_solver_prompt(*, target_metric: str | None) -> str:
    if target_metric:
        metric_note = (
            f"Competition metric: {target_metric}. "
            f"Extract and report this as oof_score. "
            f"Print `OOF {target_metric}: <value>` in your training script."
        )
    else:
        metric_note = (
            "Open-ended task — self-assess quality 0-100. "
            "Set oof_score = null in your StructuredOutput."
        )
    return f"""\
Use your current context from CLAUDE.md (competition settings, best scores, past experiments).
{metric_note}

Search for skills first, then plan, implement, evaluate, review, submit, and iterate.
Report final results via StructuredOutput when done.
"""
