"""Validation agent specification: prompts + output schema."""

from __future__ import annotations

from typing import Any

VALIDATION_SYSTEM_PROMPT = """\
You are a brutal, impartial judge of competition results. Your job is to find
every gap, flaw, and missing requirement — not to validate the implementer's
effort. Assume the implementer is overconfident and their self-assessment is
inflated. Your score carries real consequences: stopping too early wastes the
entire competition budget on a mediocre result.

For ML competitions (metric provided):
  Given the OOF score of a new experiment:
  1. Compare it against the current best OOF score (math, no rounding).
  2. Check the submission file format — open it, verify header and row count.
  3. Decide is_improvement and submit based on strict thresholds.

For open-ended tasks (no metric, quality_score 0-100):
  The implementer gave a self-assessed score. IGNORE IT as a starting point.
  Do your own independent assessment:
  1. Read README.md. Extract EVERY explicit requirement as a checklist.
  2. Read every deliverable file. Test against each requirement.
  3. For each requirement NOT fully met, deduct points. Be specific.
  4. Ask yourself: "What specific work remains to reach 100/100?"
     If you can list anything at all — bugs, missing features, poor error
     handling, missing docs, no tests, edge cases — the score is not 95+.
  5. Your score must reflect reality, not encouragement.

  Scoring guide (enforce strictly):
  - 95-100: Genuinely polished. Every requirement met. Tested. Documented.
            No obvious improvements a senior engineer would make. RARE.
  - 80-94:  All core requirements met but polish/edge cases/docs missing.
  - 60-79:  Most requirements met; some gaps in functionality.
  - Below 60: Core functionality incomplete or broken.

In both modes:
  You do NOT write to any files. You do NOT update any state.
  You ONLY observe and report.

Stop condition:
  Set stop=True ONLY when you are certain the deliverable is production-ready
  and no further iteration would produce meaningful improvement:
  - ML: last 3 OOF scores within 0.001 of each other AND score is strong.
  - Open: quality_score >= 98 AND every README requirement is met AND the
           deliverable runs cleanly end-to-end with no edge-case failures.
  Default to stop=False. Only stop when you cannot identify a single
  concrete thing the implementer should improve.
"""


VALIDATION_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "oof_score",
        "quality_score",
        "is_improvement",
        "submit",
        "stop",
        "reasoning",
        "next_directions",
    ],
    "properties": {
        "oof_score": {"type": ["number", "null"]},
        "quality_score": {
            "type": ["number", "null"],
            "description": "0-100 quality score for open-ended tasks; null for ML tasks",
        },
        "is_improvement": {
            "type": "boolean",
            "description": "True if new score is meaningfully better than current best",
        },
        "improvement_delta": {
            "type": ["number", "null"],
            "description": "new_score - best_score (positive means improvement for maximize)",
        },
        "submit": {
            "type": "boolean",
            "description": "True if a platform submission should be made",
        },
        "submission_path": {
            "type": ["string", "null"],
            "description": "Path to the artifact to submit. Null if not submitting.",
        },
        "format_ok": {
            "type": "boolean",
            "description": "Whether the submission artifact passed format checks",
        },
        "reasoning": {"type": "string"},
        "stop": {
            "type": "boolean",
            "description": (
                "True ONLY if you cannot identify a single concrete improvement: "
                "ML score has genuinely plateaued (last 3 within 0.001), OR "
                "open-task quality >= 98 with every README requirement fully met. "
                "Default False — let the agent keep trying."
            ),
        },
        "next_directions": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Concrete, actionable improvements the implementer could still make. "
                "Must be EMPTY ([]) only when stop=True and nothing remains. "
                "If stop=False you MUST list at least one item here."
            ),
        },
    },
    "additionalProperties": False,
}


def build_validation_prompt(
    *,
    solution_path: str,
    oof_score: float | None,
    quality_score: float,
    submission_path: str,
    target_metric: str | None,
    metric_direction: str | None,
    best_oof_score: float | None,
    best_quality_score: float | None,
    submission_count: int,
    max_submissions_per_day: int,
    quota_instruction: str,
    project_dir: str,
) -> str:
    if target_metric:
        direction_word = "higher" if metric_direction == "maximize" else "lower"
        best_score_str = (
            f"{best_oof_score:.6f}" if best_oof_score is not None else "none yet"
        )
        new_oof_str = f"{oof_score:.6f}" if oof_score is not None else "n/a"
        submission_check_instruction = (
            f"Use Read to open {submission_path} and check the header + first data row (CSV format). "
            "Use CLAUDE.md to find the sample template path and respect exact filename casing "
            "(e.g., SampleSubmission.csv vs sample_submission.csv)."
            if submission_path
            else "No submission file — set format_ok=False."
        )
        score_section = f"""\
New experiment:
  Solution      : {solution_path}
  OOF score     : {new_oof_str}
  Submission CSV: {submission_path or "(none produced)"}
  Project root   : {project_dir}
  Experiment state path: {project_dir}/.claude/EXPERIMENT_STATE.json

Context:
  Metric              : {target_metric} ({metric_direction})
  Current best OOF    : {best_score_str}
  Improvement threshold: 0.0001 ({direction_word} is better)
  State submission count today: {submission_count} / {max_submissions_per_day}

## Tasks
1. Determine is_improvement: is {new_oof_str} meaningfully better than {best_score_str}?
2. {submission_check_instruction}
{quota_instruction}3. Decide stop=True only if the score has genuinely plateaued (last 3+ OOF scores
   within 0.001) and you cannot identify a concrete next improvement.
4. Populate next_directions with every concrete improvement you can still identify.
   If stop=True, next_directions MUST be empty. If stop=False, it MUST be non-empty.
5. Return the structured JSON result."""
    else:
        best_q = (
            f"{best_quality_score}/100"
            if best_quality_score is not None
            else "none yet"
        )
        score_section = f"""\
New experiment:
  Solution      : {solution_path}
  Implementer's self-score: {quality_score}/100  ← treat this as a ceiling, not a floor
  Deliverable   : {submission_path or "(none produced)"}
    Project root   : {project_dir}
    Experiment state path: {project_dir}/.claude/EXPERIMENT_STATE.json

Context:
  Task type     : open-ended (no numeric metric)
  Current best  : {best_q}
  Improvement threshold: 2 points
  Submissions today: {submission_count}

## Your job
1. Read README.md. List every explicit requirement as a numbered checklist.
2. Open and read every deliverable file listed above.
3. For each requirement: mark PASS / FAIL / PARTIAL and note the exact gap.
4. Assign your OWN quality score based only on what you verified.
   The implementer claimed {quality_score}/100 — challenge it. Find what's broken,
   missing, undocumented, untested, or incomplete. List each gap explicitly.
5. Determine is_improvement: your score > {best_q}?
{quota_instruction}6. Set stop=False unless you genuinely cannot name a single concrete improvement
   the implementer could still make. Be specific in your reasoning about what
   is preventing a score of 100/100.
7. Populate next_directions with every concrete improvement you can still identify.
   If stop=True, next_directions MUST be empty. If stop=False, it MUST be non-empty.
8. Return the structured JSON result."""

    return f"## Validation Request\n\n{score_section}\n"
