"""Summarizer agent specification: prompts + output schema."""

from __future__ import annotations

import json
from typing import Any

SUMMARIZER_SYSTEM_PROMPT = """\
You are an expert ML research analyst maintaining a living knowledge base for a
competition-solving agent.

Your task: read the existing MEMORY.md and the latest-iteration result, then
rewrite MEMORY.md with updated, concise learnings.

MEMORY.md format (strict — keep sections in this order, start directly with the # heading,
no YAML frontmatter, no code fences, no --- separators):

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

Rules:
- Be concise — every bullet should fit on one line.
- Never invent data you haven't been given.
- Keep cumulative history (don't delete good entries just to shorten).
- Return the complete rewritten MEMORY.md text in `memory_content` — do NOT use
  the Write tool; the orchestrator writes the file on your behalf.
"""


SUMMARIZER_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["summary", "memory_content"],
    "properties": {
        "summary": {
            "type": "string",
            "description": "One-sentence summary of the key learning from this iteration.",
        },
        "memory_content": {
            "type": "string",
            "description": (
                "The complete updated MEMORY.md text (all sections, newest entries "
                "integrated). Must start directly with '# Planner Memory' — "
                "no YAML frontmatter, no code fences, no --- separators."
            ),
        },
    },
    "additionalProperties": False,
}


def build_summarizer_prompt(
    *,
    iteration: int,
    competition_id: str,
    target_metric: str | None,
    metric_direction: str | None,
    best_oof_score: float | None,
    best_quality_score: float | None,
    experiments: list[dict],
    failed_runs: list[dict],
    latest_experiment: dict,
    validation_notes: str,
    memory_path: str,
) -> str:
    recent = list(reversed(experiments[-10:]))
    if target_metric:
        score_ctx = (
            f"- Metric      : {target_metric} ({metric_direction})\n"
            f"- Best OOF so far: {f'{best_oof_score:.6f}' if best_oof_score is not None else 'none yet'}"
        )
    else:
        score_ctx = (
            f"- Task type   : open-ended (no numeric metric)\n"
            f"- Best quality so far: {f'{best_quality_score}/100' if best_quality_score is not None else 'none yet'}"
        )

    exp_list = [
        {
            "iter": e.get("iteration"),
            "oof": e.get("oof_score"),
            "approach": e.get("approach", ""),
        }
        for e in recent
    ]

    return f"""\
## Summarizer Task

Update the planner memory file after completing iteration {iteration}.

### Competition context
- Competition : {competition_id}
{score_ctx}
- Total experiments: {len(experiments)}

### Latest experiment result
```json
{json.dumps(latest_experiment, indent=2)}
```

### Validation notes (from validation agent)
{validation_notes or "_(none)_"}

### Failed runs this competition
```json
{json.dumps(failed_runs[-5:], indent=2)}
```

### All experiments so far (OOF + approach)
{json.dumps(exp_list, indent=2)}

### Your task
1. Read the current MEMORY.md at `{memory_path}`.
2. Integrate the new result — update "What Works", "What Fails", "Patterns",
   "Experiment Score History", and "Suggested Next Directions".
3. Return the **complete rewritten MEMORY.md content** in the `memory_content` field.
   Do NOT write any files — the orchestrator will write MEMORY.md for you.
4. Return a one-sentence `summary` of the key learning from this iteration.

The planner reads this file at the start of every session — make it dense and actionable.
"""
