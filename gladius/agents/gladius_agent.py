"""Gladius — single autonomous competition agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gladius.agents.runtime.agent_runner import run_agent

if TYPE_CHECKING:
    from gladius.state import CompetitionState

# ── Output schema ─────────────────────────────────────────────────────────────

OUTPUT_SCHEMA: dict[str, Any] = {
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
                "Set to null for open-ended tasks."
            ),
        },
        "quality_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "description": "Self-assessed quality 0-100. Use 0 on error.",
        },
        "solution_files": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Paths to all files created or modified.",
        },
        "submission_file": {
            "type": "string",
            "description": "Path to submission CSV or deliverable. Empty if not produced.",
        },
        "notes": {"type": "string", "description": "Brief summary of what was built."},
        "error_message": {"type": "string", "description": "What went wrong (errors only)."},
        "total_turns": {"type": ["integer", "null"]},
    },
    "additionalProperties": False,
}

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are Gladius — an autonomous competition agent that competes against humans.
You handle everything yourself: explore, plan, implement, evaluate, improve, submit.

## First action — mandatory

Before anything else, search for a skill:

  mcp__skills-on-demand__search_skills({"query": "<task type, e.g. ml classification tabular>", "top_k": 5})

Read the best match: .claude/skills/<name>/SKILL.md
Follow its patterns from the start.

## Your context

CLAUDE.md has everything: competition ID, metric, data path, best scores, past
experiments, failed approaches. Read it first. NEVER modify CLAUDE.md.

## Memory

You own your memory. Read it at session start, update it at session end:
  .claude/agent-memory/MEMORY.md

## Execution loop

Repeat until satisfied:

1. Search skill → read SKILL.md → follow its patterns
2. Explore data: head train.csv, check dtypes, target distribution
3. Plan with TodoWrite
4. Implement in src/:
   - uv add <pkg>   (never pip install)
   - uv run python
   - Fix all errors until the pipeline runs clean
5. Print OOF <metric>: <value> in your training script
   Save predictions to artifacts/oof.npy
6. Review your own code: leakage? CV contamination? wrong metric? format mismatch?
7. Build submission matching sample_submission.csv exactly → submissions/submission.csv
8. Search for the next improvement skill
9. Update .claude/agent-memory/MEMORY.md
10. Iterate — only call StructuredOutput when genuinely done

## Available skills (170+)

Always search before starting any task. Key skills:
- ml-setup, feature-engineering, adversarial-validation, hpo, ensembling, code-review
- polars, lightgbm, xgboost, transformers, pytorch-lightning, shap, optuna
- biopython, rdkit, alphafold, esm, scanpy, clinical-decision-support
- biorxiv-database, perplexity-search, hypothesis-generation
- git-workflow, uv-venv, jupyter-mcp

mcp__skills-on-demand__search_skills({"query": "<what you need>", "top_k": 5})

## Rules

- pathlib everywhere; no hardcoded absolute paths
- random_state=42 for reproducibility
- uv add <pkg> — never pip install
- TodoWrite for progress tracking
"""


def _build_prompt(*, target_metric: str | None) -> str:
    if target_metric:
        metric_note = (
            f"Metric: {target_metric}. "
            f"Print `OOF {target_metric}: <value>` in your training script. "
            f"Report as oof_score in StructuredOutput."
        )
    else:
        metric_note = "Open-ended task. Set oof_score = null in StructuredOutput."
    return (
        f"Read CLAUDE.md and .claude/agent-memory/MEMORY.md first.\n"
        f"{metric_note}\n\n"
        "Search skills → plan → implement → evaluate → submit → iterate.\n"
        "Call StructuredOutput only when done."
    )


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_gladius(state: "CompetitionState", project_dir: str) -> dict:
    prompt = _build_prompt(target_metric=state.target_metric)
    result, _ = await run_agent(
        agent_name="gladius",
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=[
            "Read", "Write", "Edit", "MultiEdit",
            "Bash", "Glob", "Grep", "TodoWrite", "WebSearch",
        ],
        output_schema=OUTPUT_SCHEMA,
        cwd=project_dir,
        max_turns=500,
    )
    return result
