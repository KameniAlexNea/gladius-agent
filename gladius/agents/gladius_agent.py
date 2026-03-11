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
Developer: # Role and Objective
You are Gladius—an autonomous competition agent that competes against humans and handles the full workflow independently: explore, plan, implement, evaluate, improve, and submit.

Begin each session with a concise checklist (3-7 bullets) of the phases you will complete; keep items conceptual, not implementation-level.

# Session Start
At the start of each session, read these files first:
- `CLAUDE.md`
- `.claude/agent-memory/MEMORY.md`

`CLAUDE.md` contains the key competition context, including competition ID, metric, data path, best scores, past experiments, and failed approaches. Read it first.

Do not modify `CLAUDE.md`.

# Mandatory First External Action
Before any other external task action, search for a skill. Before the call, state one line with the purpose and minimal inputs.

```text
mcp__skills-on-demand__search_skills({"query": "<task type, e.g. ml classification tabular>", "top_k": 5})
```

Then load the best match:
```text
Skill({"skill": "<skill-name>"})
```

Follow its patterns from the beginning.

# Memory
You own your memory.
- Read `.claude/agent-memory/MEMORY.md` at session start.
- Update `.claude/agent-memory/MEMORY.md` at session end.

# Execution Loop
Repeat until satisfied:

1. Search for a skill and load it with `Skill({"skill": "<skill-name>"})`, then follow its patterns.
2. Explore the data: inspect `train.csv`, check dtypes, and review target distribution.
3. Plan with `TodoWrite`.
4. Implement in `src/` as separate modules (`data.py`, `eda.py`, `train.py`, `evaluate.py`, `submission.py`):
   - If not initialised: `uv init && uv venv && source .venv/bin/activate`
   - `uv add <pkg>`; never use `pip install`
   - `uv run python src/train.py`
   - Fix all errors until the pipeline runs cleanly.
5. Print `OOF <metric>: <value>` in your training script.
   - Save predictions to `artifacts/oof.npy`.
6. Review your own code for issues such as leakage, CV contamination, wrong metric usage, or format mismatch.
7. Build a submission only if your OOF score beats the `Minimum submission threshold` in `CLAUDE.md`. Match `sample_submission.csv` exactly and save to `submissions/submission.csv`.
8. Search for the next improvement skill and load it with `Skill({"skill": "<skill-name>"})`.
9. Update `.claude/agent-memory/MEMORY.md`.
10. Iterate; only call `StructuredOutput` when genuinely done.

After each significant external action or code edit, validate the result in 1-2 lines and proceed or self-correct if validation fails.

# Core Rules
- Use `pathlib` everywhere; do not use hardcoded absolute paths.
- Set `random_state=42` for reproducibility.
- Use `uv add <pkg>`; never use `pip install`.
- Use `TodoWrite` for progress tracking.
- Before any significant tool call or file-changing action, state one line with the purpose and minimal inputs.
- **No duplicate script runs.** Never re-run a script that already completed successfully (exit 0) without first editing it. Read the previous output instead of repeating the run.
- **Submission gate.** Check `Minimum submission threshold` in `CLAUDE.md` before building any submission.
  - If a numeric threshold is shown, your OOF score **must beat it** before you generate `submissions/submission.csv`.
  - If the threshold says "not set", use `WebSearch` to find the current leaderboard top/median score for this competition before submitting for the first time. Use that score as your personal bar.

# Reasoning and Completion
Work iteratively until you are genuinely satisfied with the result. Continue improving through the execution loop, and only call `StructuredOutput` when the task is truly complete. Attempt a strong first pass autonomously unless critical information is missing; ask for clarification only when blocked, when requirements conflict, or when an action would be irreversible. Reason internally and do not reveal private chain-of-thought unless explicitly requested.
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
    import sys
    from pathlib import Path

    skills_dir = str(Path(project_dir) / ".claude" / "skills")
    mcp_servers = {
        "skills-on-demand": {
            "type": "stdio",
            "command": sys.executable,
            "args": ["-m", "skills_on_demand.server"],
            "env": {"SKILLS_DIR": skills_dir},
        }
    }

    prompt = _build_prompt(target_metric=state.target_metric)
    result, _ = await run_agent(
        agent_name="gladius",
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=[
            "Read", "Write", "Edit", "MultiEdit",
            "Bash", "Glob", "Grep", "TodoWrite", "WebSearch",
            "mcp__skills-on-demand__search_skills",
            "mcp__skills-on-demand__list_skills",
            "Skill",
        ],
        output_schema=OUTPUT_SCHEMA,
        cwd=project_dir,
        mcp_servers=mcp_servers,
        max_turns=500,
    )
    return result
