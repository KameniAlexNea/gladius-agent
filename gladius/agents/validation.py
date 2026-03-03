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
You are responsible for validating experiment results and deciding on competition submissions.

For ML competitions (metric provided):
  Given the OOF score of a new experiment, you will:
  1. Compare it against the current best OOF score
  2. Check the submission file format (read first 3 lines)
  3. Decide whether to submit:
     - is_improvement = True  iff  (maximize: new > best + 0.0001)
                                    OR (minimize: new < best - 0.0001)
     - submit = True  iff  is_improvement AND submissions_today < daily_limit

For open-ended tasks (no metric, quality_score 0-100):
  Given the quality_score self-assessed by the implementer:
  1. Review the deliverable against the task requirements in README.md (Read it)
  2. Confirm or adjust the quality_score
  3. Decide is_improvement: quality_score > previous best + 2 points
  4. Recommend submit if is_improvement and submission artifact exists

In both modes:
  You do NOT write to any files. You do NOT update any state.
  You ONLY observe and report.

Stop condition:
  Set stop=True when further iteration is very unlikely to improve results:
  - ML: quality has plateaued (last 3 OOF scores within 0.001 of each other)
        OR the score is already excellent for the metric type.
  - Open: quality_score >= 90 and the deliverable fully meets requirements,
           OR 3+ consecutive iterations with no improvement.
  When in doubt, set stop=False to let the agent keep trying.
"""

# ── Output schema ─────────────────────────────────────────────────────────────
OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["oof_score", "quality_score", "is_improvement", "submit", "stop", "reasoning"],
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
                "True if further iteration is very unlikely to improve results: "
                "score has plateaued, task is fully complete, or quality >= 90 "
                "and all requirements are met. False to continue iterating."
            ),
        },
    },
    "additionalProperties": False,
}


async def run_validation_agent(
    solution_path: str,
    oof_score: float | None,
    quality_score: float,
    submission_path: str,
    state: "CompetitionState",
    project_dir: str,
    platform: str = "none",
) -> dict:
    """
    Validate a new experiment result and recommend submit/hold.

    Injects the platform-specific MCP server so the agent can query live
    submission quota directly instead of relying on the state counter.
    """
    # ── Platform-specific MCP server ──────────────────────────────────────
    mcp_servers: dict = {}
    quota_tool: str = ""
    quota_instruction: str = ""

    if platform == "zindi":
        from gladius.tools.zindi_tools import zindi_server

        mcp_servers = {"zindi": zindi_server}
        quota_tool = "mcp__zindi__zindi_status"
        quota_instruction = (
            "3. Call `zindi_status` to get today's remaining submission quota.\n"
        )
    elif platform == "kaggle":
        from gladius.tools.kaggle_tools import kaggle_server

        mcp_servers = {"kaggle": kaggle_server}
        quota_tool = "mcp__kaggle__kaggle_submission_history"
        quota_instruction = (
            "3. Call `kaggle_submission_history` and count how many submissions "
            "were made today to determine remaining quota.\n"
        )
    elif platform == "fake":
        import sys

        mcp_servers = {
            "fake": {
                "type": "stdio",
                "command": sys.executable,
                "args": [
                    "-c",
                    "from gladius.tools.fake_platform_tools import fake_server; "
                    "import asyncio; asyncio.run(fake_server.run())",
                ],
            }
        }
        quota_tool = "mcp__fake__fake_status"
        quota_instruction = (
            "3. Call `fake_status` to get your current submission count and rank.\n"
        )
    # platform == "none": no MCP tools — no external platform to query

    allowed_tools = ["Read", "Grep"] + ([quota_tool] if quota_tool else [])

    # Build prompt depending on task type
    if state.target_metric:
        direction_word = "higher" if state.metric_direction == "maximize" else "lower"
        best_score_str = (
            f"{state.best_oof_score:.6f}"
            if state.best_oof_score is not None
            else "none yet"
        )
        score_section = f"""\
New experiment:
  Solution      : {solution_path}
  OOF score     : {oof_score:.6f if oof_score is not None else 'n/a'}
  Submission CSV: {submission_path or "(none produced)"}

Context:
  Metric              : {state.target_metric} ({state.metric_direction})
  Current best OOF    : {best_score_str}
  Improvement threshold: 0.0001 ({direction_word} is better)
  State submission count today: {state.submission_count} / {state.max_submissions_per_day}

## Tasks
1. Determine is_improvement: is {oof_score:.6f if oof_score is not None else 'n/a'} meaningfully better than {best_score_str}?
2. {"Use Read to open " + submission_path + " and check the header + first data row (CSV format)." if submission_path else "No submission file — set format_ok=False."}
{quota_instruction}3. Decide stop=True if the score has plateaued (last 3+ OOF scores within 0.001)
   or if the score is already excellent for this metric type.
4. Return the structured JSON result."""
    else:
        best_q = (
            f"{state.best_quality_score}/100"
            if state.best_quality_score is not None
            else "none yet"
        )
        score_section = f"""\
New experiment:
  Solution      : {solution_path}
  Quality score : {quality_score}/100  (self-assessed by implementer)
  Deliverable   : {submission_path or "(none produced)"}

Context:
  Task type     : open-ended (no numeric metric — read README.md for task goal)
  Current best  : {best_q}
  Improvement threshold: 2 points (quality_score > best + 2)
  Submissions today: {state.submission_count}

## Tasks
1. Read README.md to understand the task goal and deliverable requirements.
2. Inspect the deliverable ({submission_path or "none"}) to assess completeness.
3. Confirm or adjust quality_score; determine is_improvement ({quality_score} > {best_q}?).
{quota_instruction}4. Decide stop=True if quality_score >= 90 and all requirements are met,
   or if there have been 3+ consecutive iterations with no quality improvement.
5. Return the structured JSON result."""

    prompt = f"""## Validation Request\n\n{score_section}\n"""
    result, _ = await run_agent(
        agent_name="validation",
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=allowed_tools,
        output_schema=OUTPUT_SCHEMA,
        cwd=project_dir,
        mcp_servers=mcp_servers,
        max_turns=10,
    )
    return result
