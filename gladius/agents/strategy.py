"""
Strategy Agent — generates the next hypothesis for the competition loop.

Replaces: strategy_agent + hypothesis_generator + lb_tracker (3 LangGraph nodes → 1)

Session continuity: resumed every iteration so the agent accumulates
competition understanding, LB history, and reasoning chain across all runs.
"""
import json
from typing import TYPE_CHECKING

from gladius.agents._base import run_agent

if TYPE_CHECKING:
    from gladius.state import CompetitionState

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an elite Kaggle Grandmaster working on a machine learning competition.
Your role is to analyse the competition, study the leaderboard, examine existing
experiments, and generate the single most promising next hypothesis to try.

You have access to:
  - The competition data directory (Read, Glob to explore)
  - The experiments log at .gladius/experiments.json
  - Web search for relevant papers and winning solutions

Always reason about:
  1. What has been tried and what worked / failed
  2. The current gap to the leaderboard top
  3. The highest-impact next improvement (data, features, model architecture, ensemble)
  4. Risk vs. reward — prefer reliable improvements over moonshots early on

Output a single, concrete, actionable hypothesis.
"""

# ── Output schema ─────────────────────────────────────────────────────────────
OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["hypothesis", "changes", "expected_improvement", "rationale", "priority"],
    "properties": {
        "hypothesis": {
            "type": "string",
            "description": "One-line hypothesis title (max 80 chars)",
        },
        "changes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete, ordered list of code/config changes to make",
        },
        "expected_improvement": {
            "type": "number",
            "description": "Estimated OOF metric delta (positive = better for the direction)",
        },
        "rationale": {"type": "string"},
        "priority": {"enum": ["critical", "high", "medium", "low"]},
        "requires_new_features": {"type": "boolean"},
        "suggested_models": {
            "type": "array",
            "items": {"type": "string"},
            "description": "e.g. ['catboost', 'lightgbm']",
        },
    },
    "additionalProperties": False,
}

# ── Knowledge-extractor subagent definition ───────────────────────────────────
# The strategy agent has access to a cheap haiku subagent for extracting
# insight from papers / notebooks via the Task tool.
KNOWLEDGE_EXTRACTOR_DEF = {
    "description": (
        "Extracts actionable ML insights from papers, Kaggle notebooks, and "
        "discussion posts. Returns a structured summary of techniques."
    ),
    "prompt": (
        "You are a knowledge extraction specialist for ML competitions. "
        "Given a document or URL, extract: model architectures, feature "
        "engineering techniques, training tricks, post-processing methods, "
        "and ensemble strategies. Focus on what is directly applicable to "
        "the current competition task."
    ),
    "tools": ["Read", "WebFetch"],
    "model": "haiku",
}


async def run_strategy_agent(
    state: "CompetitionState",
    data_dir: str,
) -> tuple[dict, str]:
    """
    Generate the next hypothesis.

    Returns (hypothesis_dict, session_id).
    The session_id should be stored in state.strategy_session_id
    so that the next iteration resumes this conversation.
    """
    completed_summary = json.dumps(state.completed_hypotheses[-5:], indent=2)
    failed_summary = json.dumps(state.failed_hypotheses[-3:], indent=2)

    prompt = f"""\
## Competition: {state.competition_id}

Metric  : {state.target_metric} ({state.metric_direction})
Best OOF: {state.best_oof_score:.6f}
Best LB : {state.best_submission_score:.6f}
Progress: iteration {state.iteration} / {state.max_iterations}

### Recent completed hypotheses (last 5)
{completed_summary}

### Failed hypotheses (last 3)
{failed_summary}

### Your tasks
1. Read the experiments log at {data_dir}/../.gladius/experiments.json (if it exists)
2. Briefly scan the data directory: {data_dir}
3. Optionally use the knowledge_extractor subagent for a relevant paper / solution
4. Propose the single most impactful next hypothesis

Return structured JSON matching the schema.
"""
    from claude_agent_sdk import AgentDefinition

    return await run_agent(
        agent_name="strategy",
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=["Read", "Glob", "Grep", "WebSearch", "Task"],
        output_schema=OUTPUT_SCHEMA,
        cwd=data_dir,
        resume=state.strategy_session_id,
        max_turns=30,
        agents={"knowledge_extractor": AgentDefinition(**KNOWLEDGE_EXTRACTOR_DEF)},
    )
