"""
Summarizer Agent — updates the planner's persistent memory after each iteration.

Reads the latest experiment result and current MEMORY.md, then rewrites MEMORY.md
with compact, structured learnings: what worked, what failed, patterns, next directions.

Called by the orchestrator at the end of every validation phase so the planner
accumulates knowledge across iterations without reading full experiment history
each time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from gladius.agents._base import run_agent

if TYPE_CHECKING:
    from gladius.state import CompetitionState

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert ML research analyst maintaining a living knowledge base for a
competition-solving agent.

Your task: read the existing MEMORY.md and the latest-iteration result, then
rewrite MEMORY.md with updated, concise learnings.

MEMORY.md format (strict — keep sections in this order):
---
# Planner Memory — {competition_id}
> Auto-updated by summarizer. Last iteration: N

## Key Data Insights
Short bullets about the dataset: shape, class balance, missing values, id/time columns,
quirks that affect modelling.

## What Works  ✅
Bullets of confirmed improvements with approximate OOF delta and iteration number.
Keep at most 10 entries; drop oldest when exceeding.

## What Fails / Dead Ends  ❌
Bullets of approaches that hurt score, timed out, or errored. Include the reason.
Keep at most 10 entries.

## Patterns & Hypotheses  💡
Open hypotheses not yet tested, or correlations observed in data.
Keep at most 8 entries; remove ones already confirmed/refuted.

## Experiment Score History
Markdown table: | iter | OOF | approach | notes |
Keep ALL entries, newest first.

## Suggested Next Directions
Ordered list: highest-expected-gain first.  Max 5 items.
Be specific (e.g. "Add lag features on user_id × day_of_week").
---

Rules:
- Be concise — every bullet should fit on one line.
- Never invent data you haven't been given.
- Keep cumulative history (don't delete good entries just to shorten).
- Write the file using the Write tool, not JSON output.
"""

# Minimal schema — the agent writes MEMORY.md directly with the Write tool.
OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["summary"],
    "properties": {
        "summary": {
            "type": "string",
            "description": "One-sentence summary of the key learning from this iteration.",
        },
    },
    "additionalProperties": False,
}


async def run_summarizer(
    state: "CompetitionState",
    project_dir: str,
    latest_experiment: dict,
    validation_notes: str = "",
) -> str:
    """
    Update .claude/agent-memory/planner/MEMORY.md with learnings from the latest
    experiment. Returns the one-sentence summary string.
    """
    memory_path = (
        Path(project_dir) / ".claude" / "agent-memory" / "planner" / "MEMORY.md"
    )

    # Recent experiments context (last 10)
    recent = list(reversed(state.experiments[-10:]))

    prompt = f"""\
## Summarizer Task

Update the planner memory file after completing iteration {state.iteration}.

### Competition context
- Competition : {state.competition_id}
- Metric      : {state.target_metric} ({state.metric_direction})
- Best OOF so far: {f'{state.best_oof_score:.6f}' if state.best_oof_score is not None else 'none yet'}
- Total experiments: {len(state.experiments)}

### Latest experiment result
```json
{json.dumps(latest_experiment, indent=2)}
```

### Validation notes (from validation agent)
{validation_notes or "_(none)_"}

### Failed runs this competition
```json
{json.dumps(state.failed_runs[-5:], indent=2)}
```

### All experiments so far (OOF + approach)
{json.dumps([{"iter": e.get("iteration"), "oof": e.get("oof_score"), "approach": e.get("approach", "")} for e in recent], indent=2)}

### Your task
1. Read the current MEMORY.md at `{memory_path}`.
2. Integrate the new result — update "What Works", "What Fails", "Patterns",
   "Experiment Score History", and "Suggested Next Directions".
3. Rewrite the entire file using the Write tool.
4. Return a one-sentence `summary` of the key learning from this iteration.

The planner reads this file at the start of every session — make it dense and actionable.
"""
    result, _ = await run_agent(
        agent_name="summarizer",
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=["Read", "Write"],
        output_schema=OUTPUT_SCHEMA,
        cwd=project_dir,
        max_turns=15,
    )
    return result.get("summary", "")
