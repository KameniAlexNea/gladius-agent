"""
Ensemble Agent — blends multiple experiment OOF/test predictions.

Replaces: ensemble_agent + knowledge_extractor (2 nodes → 1)

Triggered by the orchestrator every N iterations when >= 3 good experiments exist.
Writes a blended submission and a reproducible ensemble script.

No session continuity: reads everything it needs from .gladius/ on disk.
"""
import json
from typing import TYPE_CHECKING

from gladius.agents._base import run_agent
from gladius.tools.metric_tools import metric_server

if TYPE_CHECKING:
    from gladius.state import CompetitionState

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert at combining ML models for Kaggle competitions.

Given a list of experiment OOF/test prediction files, you will:
1. Load the OOF arrays (.gladius/oof_vN.npy)
2. Compute pairwise Pearson correlations — use the mcp__metrics__compute_oof_correlation tool
3. Select a diverse, low-correlation subset (correlation < 0.97)
4. Find optimal blend weights using scipy.optimize.minimize (Nelder-Mead):
   - Minimise/maximise the blended OOF metric (use mcp__metrics__compute_oof_metric)
   - Constrain weights to be non-negative and sum to 1.0
5. Apply the weights to the test predictions to create a blended submission
6. Save the submission to .gladius/ensemble_submission.csv
7. Save a reproducible script to src/ensemble.py that recreates this blend

Use rank-averaging as a fallback if Nelder-Mead fails to improve on simple average.
Prefer diversity over raw individual score — diverse models generalise better.
"""

# ── Output schema ─────────────────────────────────────────────────────────────
OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["ensemble_type", "components", "oof_score", "submission_path"],
    "properties": {
        "ensemble_type": {
            "enum": ["simple_average", "weighted_blend", "rank_average", "stacking"],
        },
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["path", "weight", "oof_score"],
                "properties": {
                    "path": {"type": "string"},
                    "weight": {"type": "number"},
                    "oof_score": {"type": "number"},
                },
            },
            "description": "Models included in the ensemble with their weights",
        },
        "oof_score": {
            "type": "number",
            "description": "OOF score of the blended ensemble",
        },
        "submission_path": {"type": "string"},
        "notes": {"type": "string"},
    },
    "additionalProperties": False,
}


async def run_ensemble_agent(
    state: "CompetitionState",
    project_dir: str,
) -> dict:
    """
    Build an ensemble from the best experiments and return the result dict.
    """
    # Filter to experiments within 1% of best score
    good_experiments = [
        e for e in state.experiments
        if e.get("oof_score") is not None
        and e["oof_score"] > state.best_oof_score * (
            0.99 if state.metric_direction == "maximize" else 1.01
        )
    ]

    prompt = f"""\
## Ensemble Task

Competition : {state.competition_id}
Metric      : {state.target_metric} ({state.metric_direction})
Best OOF    : {state.best_oof_score:.6f}

### Candidate experiments (within 1% of best OOF)
{json.dumps(good_experiments, indent=2)}

OOF arrays are at  : .gladius/oof_vN.npy
Test predictions at: .gladius/sub_vN.csv
Labels (if needed) : .gladius/train_labels.npy

### Steps
1. Use mcp__metrics__compute_oof_correlation to see pairwise correlations
2. Exclude components with pairwise correlation > 0.97 (keep most diverse)
3. Optimise blend weights via scipy Nelder-Mead
4. Apply weights to test CSVs → .gladius/ensemble_submission.csv
5. Save script to src/ensemble.py

Return the JSON result.
"""
    result, _ = await run_agent(
        agent_name="ensemble",
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=[
            "Read", "Write", "Edit", "Bash", "Glob",
            "mcp__metrics__compute_oof_metric",
            "mcp__metrics__compute_oof_correlation",
        ],
        output_schema=OUTPUT_SCHEMA,
        cwd=project_dir,
        mcp_servers={"metrics": metric_server},
        max_turns=40,
    )
    return result
