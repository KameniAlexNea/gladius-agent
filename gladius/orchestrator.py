"""
Agent launcher — iterative competition loop.

Each iteration:
  1. Renders CLAUDE.md from current CompetitionState.
  2. Runs the top-level coordinator agent (fresh session per iteration).
  3. Reads .claude/EXPERIMENT_STATE.json to extract scores and update state.
  4. Stops when max_iterations reached, a stop sentinel is found, or
     3 consecutive agent errors occur.

CLAUDE.md is automatically injected into the agent context — no explicit read needed.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

import gladius.claude_md as claude_md
from gladius.project_setup import load_competition_config
from gladius.roles.agent_runner import run_agent
from gladius.state import CompetitionState

_SYSTEM_PROMPT = """\
You are a top-tier ML competition agent. Your goal for this iteration is one \
focused, high-impact experiment.

## Step 1 — Plan (read-only)
`CLAUDE.md` is already in your context — do not read it again.
**Do NOT explore data directories, run scripts, or write any files yourself.**
Your role is coordination only — delegate all implementation and exploration to specialists.

## Step 2 — Execute (delegate)
The active management topology is defined in `## Management Topology` in your context (`CLAUDE.md`).
Follow that topology's flow **exactly** — it specifies which agents to call and in what order.
Do NOT substitute a different flow.

Available specialists in `.claude/agents/`:
- `team-lead` — strategic direction and hypothesis (always first)
- `data-expert` — EDA, data loading, feature infrastructure
- `feature-engineer` — feature transforms and selection
- `ml-engineer` — model, training loop, artifacts
- `evaluator` — OOF metric verification
- `validator` — submission format and improvement gate
- `memory-keeper` — update MEMORY.md with learnings
- `full-stack-coordinator` — owns full pipeline; delegates selectively (two-pizza topology)
- `domain-expert` — domain review and leakage/CV checks (matrix topology)

**Re-dispatch rule:** before calling any specialist, read `.claude/EXPERIMENT_STATE.json`.
If that specialist's entry already has `"status": "success"`, skip them — their work is done.
Only re-dispatch a specialist if their status is missing, `"error"`, or if new upstream work requires it.

**Incomplete-agent rule:** after every Task call, check whether the result contains a line like:
`agentId: <hex> (for resuming to continue this agent's work if needed)`
This means the agent hit its turn limit and stopped **before finishing**. Its EXPERIMENT_STATE entry
will be missing or incomplete. You MUST re-dispatch that same agent immediately, passing the
`agentId` value as the `resume` parameter in the new Task call.

## Step 3 — Signal stop (if validator says so)
After memory-keeper finishes, if the validator returned `stop=True`, write the following
as the **last action** before finishing:
```json
{"done": true}
```
to `.claude/EXPERIMENT_STATE.json` (merge with existing content, do not overwrite other keys).

## Constraints
- Do not repeat any approach listed under "Failed Approaches" in your context.
- Search skills before writing new code: `mcp__skills-on-demand__search_skills`.
- Save the final submission to `submissions/submission.csv` if the validator approves."""

_TOP_LEVEL_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "Bash",
    "Glob",
    "Grep",
    "WebSearch",
    "Skill",
    "TodoWrite",
    "Task",
]

_MAX_CONSECUTIVE_ERRORS = 3


_TOPOLOGY_FIRST_STEP: dict[str, str] = {
    "functional": "data-expert → feature-engineer → ml-engineer → evaluator → validator → memory-keeper",
    "two-pizza": "full-stack-coordinator (who decides which specialists to spawn) → validator → memory-keeper",
    "matrix": "ml-engineer (full pipeline) → dual review (team-lead + domain-expert) → evaluator → validator → memory-keeper",
    "autonomous": "N parallel mini-teams each running (data-expert → feature-engineer → ml-engineer → evaluator) → validator → memory-keeper",
    "platform": "platform-layer (data-expert) → product-layer (feature-engineer → ml-engineer) → evaluator → validator → memory-keeper",
}


def _make_kickoff_prompt(state: CompetitionState) -> str:
    """Build an iteration-aware kickoff prompt."""
    flow = _TOPOLOGY_FIRST_STEP.get(
        state.topology, "follow the topology defined in CLAUDE.md"
    )

    if state.iteration == 1:
        return (
            "This is the FIRST iteration — no experiments have run yet.\n"
            "1. Delegate to `team-lead` to plan a baseline experiment "
            "(team-lead will read MEMORY.md itself).\n"
            f"2. Then follow the **{state.topology}** topology: {flow}.\n"
            "Focus on getting a clean, reproducible baseline score above all else."
        )

    best = (
        f"{state.best_oof_score:.6f}"
        if state.best_oof_score is not None
        else (
            f"{state.best_quality_score}/100"
            if state.best_quality_score is not None
            else "none yet"
        )
    )
    metric_label = state.target_metric or "quality"
    return (
        f"This is iteration {state.iteration}/{state.max_iterations}. "
        f"Current best {metric_label}: **{best}**.\n\n"
        "1. Delegate to `team-lead` to plan the next experiment "
        "(team-lead will read MEMORY.md itself).\n"
        f"2. Then follow the **{state.topology}** topology: {flow}.\n"
        '   Skip any agent whose EXPERIMENT_STATE.json entry is already `"status": "success"`.'
        "\n3. After validation, have `memory-keeper` update MEMORY.md.\n\n"
        "Avoid any approach listed under 'Failed Approaches' in your context."
    )


def _build_state(project_dir: Path, cfg: dict) -> CompetitionState:
    """Build initial CompetitionState from config + README.md via claude_md."""
    state = claude_md.write_from_project(project_dir, cfg)
    state.max_iterations = int(cfg.get("max_iterations", state.max_iterations))
    return state


def _update_state(state: CompetitionState, project_dir: Path) -> None:
    """Read EXPERIMENT_STATE.json written by agents and update CompetitionState."""
    exp_path = project_dir / ".claude" / "EXPERIMENT_STATE.json"
    if not exp_path.exists():
        return

    try:
        data = json.loads(exp_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Could not parse EXPERIMENT_STATE.json: {exc}")
        return

    # Extract OOF score — evaluator is authoritative, ml_engineer as fallback
    evaluator = data.get("evaluator", {})
    ml_eng = data.get("ml_engineer", {})
    oof_score: float | None = evaluator.get("oof_score") or ml_eng.get("oof_score")
    quality_score: float | None = ml_eng.get("quality_score")
    metric = evaluator.get("metric") or state.target_metric
    notes = ml_eng.get("notes", "")
    solution_files = ml_eng.get("solution_files", [])
    submission_file = ml_eng.get("submission_file")

    entry = {
        "iteration": state.iteration,
        "oof_score": oof_score,
        "quality_score": quality_score,
        "metric": metric,
        "notes": notes,
        "solution_files": solution_files,
        "approach": ml_eng.get("notes", ""),
    }
    state.experiments.append(entry)

    # Update best scores
    if oof_score is not None:
        if state.best_oof_score is None:
            state.best_oof_score = oof_score
        elif state.metric_direction == "maximize" and oof_score > state.best_oof_score:
            state.best_oof_score = oof_score
            state.best_submission_path = submission_file
        elif state.metric_direction == "minimize" and oof_score < state.best_oof_score:
            state.best_oof_score = oof_score
            state.best_submission_path = submission_file

    if quality_score is not None and (
        state.best_quality_score is None or quality_score > state.best_quality_score
    ):
        state.best_quality_score = quality_score

    # Stop signal written by the coordinator after validator returns stop=True
    if data.get("done"):
        logger.info("Stop signal received from agent (done=true in EXPERIMENT_STATE).")
        state.done = True


async def run_competition(
    competition_dir: str,
    max_turns: int | None = None,
    max_iterations: int | None = None,
) -> None:
    cfg = load_competition_config(competition_dir)
    project_dir = Path(competition_dir)

    state = _build_state(project_dir, cfg)
    if max_iterations is not None:
        state.max_iterations = max_iterations

    logger.info(
        f"Starting competition run: {cfg['competition_id']} "
        f"(max_iterations={state.max_iterations})"
    )

    while not state.done and state.iteration < state.max_iterations:
        state.iteration += 1

        # Refresh CLAUDE.md with current state before each iteration
        claude_md.write(state, str(project_dir))

        logger.info(
            f"Iteration {state.iteration}/{state.max_iterations} — "
            f"best={state.best_oof_score or state.best_quality_score or 'none'}"
        )

        kickoff = _make_kickoff_prompt(state)

        try:
            _, _ = await run_agent(
                agent_name="gladius",
                prompt=kickoff,
                system_prompt=_SYSTEM_PROMPT,
                allowed_tools=_TOP_LEVEL_TOOLS,
                output_schema=None,
                cwd=str(project_dir),
                max_turns=max_turns,
            )
            state.consecutive_errors = 0
        except Exception as exc:
            logger.error(f"Iteration {state.iteration} agent error: {exc}")
            state.consecutive_errors += 1
            state.error_log.append({"iteration": state.iteration, "error": str(exc)})
            if state.consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                logger.error(
                    f"{_MAX_CONSECUTIVE_ERRORS} consecutive errors — stopping run."
                )
                state.last_stop_reason = "consecutive_errors"
                state.done = True
            continue

        _update_state(state, project_dir)

    reason = state.last_stop_reason or (
        "done_signal" if state.done else "max_iterations"
    )
    logger.info(
        f"Run complete — {state.iteration} iteration(s), "
        f"best={state.best_oof_score or state.best_quality_score or 'none'}, "
        f"stop_reason={reason}"
    )
