"""
Validation Agent — validates a result and recommends whether to submit.

Replaces: validation_agent + submission_decider + notifier (3 nodes → 1)

IMPORTANT: This agent only REPORTS its decision. It does NOT mutate state.
The orchestrator reads .is_improvement and .submit and applies them.
This was the critical regression in the LangGraph version — the old node
was writing best_oof_score itself, breaking the submission gate permanently.

No session continuity: each validation is an independent, stateless task.
"""
from typing import TYPE_CHECKING

from gladius.agents._base import run_agent

if TYPE_CHECKING:
    from gladius.state import CompetitionState

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are responsible for validating ML experiment results and deciding on Kaggle submissions.

Given the OOF score of a new experiment, you will:
1. Compare it against the provided current best OOF score
2. Check the submission file format (read first 3 lines)
3. Decide whether to submit, applying these rules:
   - is_improvement = True  iff  (maximize: new > best + 0.0001)
                                  OR (minimize: new < best - 0.0001)
   - submit = True  iff  is_improvement AND submissions_today < daily_limit
4. Return structured JSON with your decision and reasoning

You do NOT write to any files. You do NOT update any state.
You ONLY observe and report.
"""

# ── Output schema ─────────────────────────────────────────────────────────────
OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["oof_score", "is_improvement", "submit", "reasoning"],
    "properties": {
        "oof_score": {"type": "number"},
        "is_improvement": {
            "type": "boolean",
            "description": "True if new score is meaningfully better than current best",
        },
        "improvement_delta": {
            "type": "number",
            "description": "new_score - best_score (positive means improvement for maximize)",
        },
        "submit": {
            "type": "boolean",
            "description": "True if a Kaggle submission should be made",
        },
        "submission_path": {
            "type": ["string", "null"],
            "description": "Path to the CSV file to submit. Null if not submitting.",
        },
        "format_ok": {
            "type": "boolean",
            "description": "Whether the submission file passed format checks",
        },
        "reasoning": {"type": "string"},
    },
    "additionalProperties": False,
}


async def run_validation_agent(
    solution_path: str,
    oof_score: float,
    submission_path: str,
    state: "CompetitionState",
    project_dir: str,
) -> dict:
    """
    Validate a new experiment result and recommend submit/hold.

    The orchestrator is responsible for acting on .is_improvement and .submit.
    This function only reads; it never modifies state.
    """
    direction_word = "higher" if state.metric_direction == "maximize" else "lower"

    prompt = f"""\
## Validation Request

New experiment:
  Solution      : {solution_path}
  OOF score     : {oof_score:.6f}
  Submission CSV: {submission_path}

Context:
  Metric              : {state.target_metric} ({state.metric_direction})
  Current best OOF    : {state.best_oof_score:.6f}
  Improvement threshold: 0.0001 ({direction_word} is better)
  Submissions today   : {state.submission_count} / {state.max_submissions_per_day}

## Tasks
1. Determine is_improvement: is {oof_score:.6f} meaningfully better than {state.best_oof_score:.6f}?
2. Read the first 3 lines of {submission_path} to verify format
3. Decide submit = is_improvement AND submissions_today < {state.max_submissions_per_day}
4. Return the structured JSON result
"""
    result, _ = await run_agent(
        agent_name="validation",
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=["Read", "Bash"],
        output_schema=OUTPUT_SCHEMA,
        cwd=project_dir,
        max_turns=10,
    )
    return result
