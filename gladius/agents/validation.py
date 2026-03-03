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

# ── Output schema ─────────────────────────────────────────────────────────────
OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["oof_score", "quality_score", "is_improvement", "submit", "stop", "reasoning", "next_directions"],
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
{quota_instruction}3. Decide stop=True only if the score has genuinely plateaued (last 3+ OOF scores
   within 0.001) and you cannot identify a concrete next improvement.
4. Populate next_directions with every concrete improvement you can still identify.
   If stop=True, next_directions MUST be empty. If stop=False, it MUST be non-empty.
5. Return the structured JSON result."""
    else:
        best_q = (
            f"{state.best_quality_score}/100"
            if state.best_quality_score is not None
            else "none yet"
        )
        score_section = f"""\
New experiment:
  Solution      : {solution_path}
  Implementer's self-score: {quality_score}/100  \u2190 treat this as a ceiling, not a floor
  Deliverable   : {submission_path or "(none produced)"}

Context:
  Task type     : open-ended (no numeric metric)
  Current best  : {best_q}
  Improvement threshold: 2 points
  Submissions today: {state.submission_count}

## Your job
1. Read README.md. List every explicit requirement as a numbered checklist.
2. Open and read every deliverable file listed above.
3. For each requirement: mark PASS / FAIL / PARTIAL and note the exact gap.
4. Assign your OWN quality score based only on what you verified.
   The implementer claimed {quality_score}/100 \u2014 challenge it. Find what's broken,
   missing, undocumented, untested, or incomplete. List each gap explicitly.
5. Determine is_improvement: your score > {best_q}?
{quota_instruction}6. Set stop=False unless you genuinely cannot name a single concrete improvement
   the implementer could still make. Be specific in your reasoning about what
   is preventing a score of 100/100.
7. Populate next_directions with every concrete improvement you can still identify.
   If stop=True, next_directions MUST be empty. If stop=False, it MUST be non-empty.
8. Return the structured JSON result."""

    prompt = f"""## Validation Request\n\n{score_section}\n"""
    result, _ = await run_agent(
        agent_name="validation",
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=allowed_tools,
        output_schema=OUTPUT_SCHEMA,
        cwd=project_dir,
        mcp_servers=mcp_servers,
        max_turns=25,
    )
    return result
