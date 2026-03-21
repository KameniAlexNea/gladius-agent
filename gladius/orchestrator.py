"""
Agent launcher — iterative competition loop.

Each iteration:
  1. Renders CLAUDE.md from current CompetitionState.
  2. Runs the top-level coordinator agent (fresh session per iteration).
        3. Reads runtime EXPERIMENT_STATE.json to extract scores and update state.
  4. Stops when max_iterations reached, a stop sentinel is found, or
     3 consecutive agent errors occur.

CLAUDE.md is automatically injected into the agent context — no explicit read needed.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from loguru import logger

import gladius.claude_md as claude_md
from gladius import RUNTIME_DATA_BRIEFING_RELATIVE_PATH
from gladius import RUNTIME_EXPERIMENT_STATE_RELATIVE_PATH
from gladius import runtime_data_briefing_path
from gladius import runtime_experiment_state_path
from gladius._orchestrator_helper import (
    MAX_CONSECUTIVE_ERRORS as _MAX_CONSECUTIVE_ERRORS,
)
from gladius._orchestrator_helper import SYSTEM_PROMPT as _SYSTEM_PROMPT
from gladius._orchestrator_helper import TOP_LEVEL_TOOLS as _TOP_LEVEL_TOOLS
from gladius._orchestrator_helper import make_kickoff_prompt as _make_kickoff_prompt
from gladius.project_setup import load_competition_config
from gladius.roles.agent_runner import run_agent
from gladius.state import CompetitionState


def _build_state(project_dir: Path, cfg: dict) -> CompetitionState:
    """Build initial CompetitionState from config + README.md via claude_md."""
    state = claude_md.write_from_project(project_dir, cfg)
    state.max_iterations = int(cfg.get("max_iterations", state.max_iterations))
    return state


# Files in artifacts/ that are intentionally reusable across iterations.
_PERSISTENT_ARTIFACTS = {"best_params.json"}
_MAX_REDISPATCH = 3
_MAX_STATE_SNIPPET_CHARS = 12000


def _archive_stale_outputs(project_dir: Path, iteration: int) -> None:
    """Move stale artifacts/ and agent logs to iteration-stamped archives.

    Prevents agents from mistaking previous-iteration outputs (oof.npy,
    model_f*.bin, train.log) for current-iteration results.
    ``best_params.json`` is copied forward since HPO results are reusable.
    ``logs/gladius.log`` is never touched — it's the orchestrator's own log.
    """
    prev = iteration - 1
    if prev < 1:
        return

    # --- artifacts/: move the whole directory ---
    art_dir = project_dir / "artifacts"
    if art_dir.is_dir() and any(art_dir.iterdir()):
        archive_dir = project_dir / f"artifacts_iter{prev}"
        shutil.move(str(art_dir), str(archive_dir))
        art_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Archived artifacts/ → {archive_dir.name}/")

        # Copy forward persistent artifacts
        for keep in _PERSISTENT_ARTIFACTS:
            archived = archive_dir / keep
            if archived.is_file():
                shutil.copy2(str(archived), str(art_dir / keep))
                logger.debug(f"Carried forward {keep} from iter {prev}")

    # --- logs/: move only agent-produced files, leave gladius.log in place ---
    logs_dir = project_dir / "logs"
    if logs_dir.is_dir():
        agent_logs = [
            f for f in logs_dir.iterdir() if f.is_file() and f.name != "gladius.log"
        ]
        if agent_logs:
            archive_dir = project_dir / f"logs_iter{prev}"
            archive_dir.mkdir(parents=True, exist_ok=True)
            for f in agent_logs:
                shutil.move(str(f), str(archive_dir / f.name))
            logger.debug(
                f"Archived {len(agent_logs)} log file(s) → {archive_dir.name}/"
            )


def _update_state(state: CompetitionState, project_dir: Path) -> None:
    """Read EXPERIMENT_STATE.json written by agents and update CompetitionState."""
    exp_path = runtime_experiment_state_path(project_dir)
    if not exp_path.exists():
        return

    try:
        data = json.loads(exp_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Could not parse EXPERIMENT_STATE.json: {exc}")
        return

    # Extract OOF score — evaluator is authoritative, ml_engineer as fallback
    # Guard against agents writing a non-dict value (e.g. an error string) for these keys.
    _raw_evaluator = data.get("evaluator", {})
    _raw_ml_eng = data.get("ml_engineer", {})
    _raw_team_lead = data.get("team_lead", {})
    evaluator = _raw_evaluator if isinstance(_raw_evaluator, dict) else {}
    ml_eng = _raw_ml_eng if isinstance(_raw_ml_eng, dict) else {}
    team_lead = _raw_team_lead if isinstance(_raw_team_lead, dict) else {}
    if not isinstance(_raw_evaluator, dict) and _raw_evaluator:
        logger.warning(
            f"EXPERIMENT_STATE.json: 'evaluator' is not a dict ({type(_raw_evaluator).__name__!r}), ignoring."
        )
    if not isinstance(_raw_ml_eng, dict) and _raw_ml_eng:
        logger.warning(
            f"EXPERIMENT_STATE.json: 'ml_engineer' is not a dict ({type(_raw_ml_eng).__name__!r}), ignoring."
        )
    if not isinstance(_raw_team_lead, dict) and _raw_team_lead:
        logger.warning(
            f"EXPERIMENT_STATE.json: 'team_lead' is not a dict ({type(_raw_team_lead).__name__!r}), ignoring."
        )
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
        "approach_summary": team_lead.get("approach_summary", ""),
        "solution_files": solution_files,
        "approach": team_lead.get("approach_summary") or ml_eng.get("notes", ""),
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


# Agents required to have status=success for an iteration to be considered complete.
# team_lead is required so coordinators cannot skip planning and jump straight to execution.
_REQUIRED_AGENTS = ("team_lead", "ml_engineer", "evaluator", "memory_keeper")


def _incomplete_agents(exp_path: Path) -> list[str]:
    """Return names of required agents that have not yet succeeded."""
    if not exp_path.exists():
        return list(_REQUIRED_AGENTS)
    try:
        data = json.loads(exp_path.read_text(encoding="utf-8"))
    except Exception:
        return list(_REQUIRED_AGENTS)
    return [
        k
        for k in _REQUIRED_AGENTS
        if not (isinstance(data.get(k), dict) and data[k].get("status") == "success")
    ]


def _missing_scout_artifact(state: CompetitionState, project_dir: Path) -> bool:
    """Scout is mandatory in iteration 1 unless briefing already exists."""
    if state.iteration != 1:
        return False
    return not runtime_data_briefing_path(project_dir).exists()


def _read_experiment_state_snippet(exp_path: Path) -> str:
    """Return a bounded EXPERIMENT_STATE.json snippet for re-dispatch prompts."""
    if not exp_path.exists():
        return "{}"
    text = exp_path.read_text(encoding="utf-8")
    if len(text) <= _MAX_STATE_SNIPPET_CHARS:
        return text
    return text[:_MAX_STATE_SNIPPET_CHARS] + "\n...<truncated>..."


def _build_redispatch_prompt(
    state: CompetitionState,
    exp_path: Path,
    incomplete: list[str],
) -> str:
    """Build a strict continuation prompt when coordinator exits early."""
    state_text = _read_experiment_state_snippet(exp_path)
    pending = ", ".join(incomplete)
    return (
        "You returned before this iteration pipeline was complete. Continue immediately.\n\n"
        f"Iteration context: {state.iteration}/{state.max_iterations}, topology={state.topology}.\n"
        f"Pending required agents (non-success): {pending}.\n\n"
        f"Current `{RUNTIME_EXPERIMENT_STATE_RELATIVE_PATH}`:\n"
        f"```json\n{state_text}\n```\n\n"
        "Required actions:\n"
        "1. Start with a concise todo task list (3–7 bullets) and keep it updated while working.\n"
        f"2. Read `{RUNTIME_EXPERIMENT_STATE_RELATIVE_PATH}` first.\n"
        "3. Dispatch only pending/failed specialists in correct topology order.\n"
        "4. Skip any specialist already marked `status: success` unless upstream changes require rerun.\n"
        f"5. If dispatching `team-lead`, require it to read `{RUNTIME_DATA_BRIEFING_RELATIVE_PATH}`, "
        "latest EXPERIMENT_STATE_iter*.json, and current "
        f"`{RUNTIME_EXPERIMENT_STATE_RELATIVE_PATH}` "
        "before suggesting the next iteration; team-lead is non-coding and must only return planning output.\n"
        "6. For each downstream specialist, include the exact relevant section under "
        "`## Your Instructions from the Team-Lead` verbatim.\n"
        "7. Do not return until all required agents are success and memory-keeper is success."
    )


async def run_competition(
    competition_dir: str,
    max_turns: int | None = None,
    max_iterations: int | None = None,
    config_path: str | None = None,
) -> None:
    cfg = load_competition_config(competition_dir, config_path=config_path)
    project_dir = Path(competition_dir)

    # Ensure GLADIUS_MODEL / GLADIUS_SMALL_MODEL are in the environment so
    # validate_runtime_invocation() and get_runtime_model() can find them.
    # project.yaml takes precedence; existing env vars are only used as fallback.
    if cfg.get("model"):
        os.environ.setdefault("GLADIUS_MODEL", cfg["model"])
    if cfg.get("small_model") and cfg["small_model"] != "inherit":
        os.environ.setdefault("GLADIUS_SMALL_MODEL", cfg["small_model"])

    state = _build_state(project_dir, cfg)
    if max_iterations is not None:
        state.max_iterations = max_iterations

    logger.info(
        f"Starting competition run: {cfg['competition_id']} "
        f"(max_iterations={state.max_iterations})"
    )

    while not state.done and state.iteration < state.max_iterations:
        state.iteration += 1

        # Archive EXPERIMENT_STATE from the previous iteration so agents start fresh
        exp_path = runtime_experiment_state_path(project_dir)
        if exp_path.exists():
            archive = exp_path.with_name(
                f"EXPERIMENT_STATE_iter{state.iteration - 1}.json"
            )
            exp_path.rename(archive)
            logger.debug(f"Archived previous EXPERIMENT_STATE → {archive.name}")

        # Archive stale artifacts and logs so agents can't confuse old outputs
        # with current iteration results.  best_params.json is preserved
        # (intentionally reusable across iterations).
        _archive_stale_outputs(project_dir, state.iteration)

        # Refresh CLAUDE.md with current state before each iteration
        claude_md.write(state, str(project_dir))

        logger.info(
            f"Iteration {state.iteration}/{state.max_iterations} — "
            f"best={state.best_oof_score or state.best_quality_score or 'none'}"
        )

        kickoff = _make_kickoff_prompt(state)

        # Agents that must reach status=success for the iteration to count.
        # If the orchestrator returns early (text response without finishing the
        # pipeline), re-dispatch it up to _MAX_REDISPATCH times with current state.
        prompt = kickoff
        incomplete: list[str] = []
        for _attempt in range(1 + _MAX_REDISPATCH):
            try:
                _, _ = await run_agent(
                    agent_name="gladius",
                    prompt=prompt,
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
                state.error_log.append(
                    {"iteration": state.iteration, "error": str(exc)}
                )
                if state.consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    logger.error(
                        f"{_MAX_CONSECUTIVE_ERRORS} consecutive errors — stopping run."
                    )
                    state.last_stop_reason = "consecutive_errors"
                    state.done = True
                break

            # Check whether the pipeline actually completed.
            exp_path = runtime_experiment_state_path(project_dir)
            incomplete = _incomplete_agents(exp_path)
            if _missing_scout_artifact(state, project_dir):
                incomplete = ["scout", *incomplete]
            if not incomplete:
                break
            if _attempt < _MAX_REDISPATCH:
                logger.warning(
                    f"Iteration {state.iteration}: orchestrator returned early — "
                    f"agents still pending: {incomplete}. Re-dispatching (attempt {_attempt + 2}/{_MAX_REDISPATCH + 1})."
                )
                prompt = _build_redispatch_prompt(state, exp_path, incomplete)

        if incomplete:
            logger.error(
                f"Iteration {state.iteration}: required agents still incomplete after "
                f"{_MAX_REDISPATCH + 1} orchestrator attempt(s): {incomplete}"
            )
            state.consecutive_errors += 1
            state.error_log.append(
                {
                    "iteration": state.iteration,
                    "error": f"incomplete_pipeline: {incomplete}",
                }
            )
            if state.consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                logger.error(
                    f"{_MAX_CONSECUTIVE_ERRORS} consecutive errors — stopping run."
                )
                state.last_stop_reason = "consecutive_errors"
                state.done = True

        _update_state(state, project_dir)

    reason = state.last_stop_reason or (
        "done_signal" if state.done else "max_iterations"
    )
    logger.info(
        f"Run complete — {state.iteration} iteration(s), "
        f"best={state.best_oof_score or state.best_quality_score or 'none'}, "
        f"stop_reason={reason}"
    )
