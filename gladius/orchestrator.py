"""
Main competition loop.

Design rules:
  - Agents output structured JSON; the orchestrator acts on it.
  - Agents NEVER mutate state directly.
  - Two agents per iteration: planner then implementer.
  - Planner is resumed each iteration (accumulates competition understanding).
  - Implementer is fresh each iteration (focused on one plan).
  - State is saved to SQLite after every phase (crash-safe).
"""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger

from gladius.agents.planner import run_planner
from gladius.agents.solver import run_solver
from gladius.agents.summarizer import run_summarizer
from gladius.agents.validation import run_validation_agent
from gladius.phases.implementation import run_implementation_phase
from gladius.phases.planning import run_planning_phase
from gladius.phases.validation import run_validation_phase
from gladius.preflight import run_preflight_or_raise
from gladius.state import CompetitionState, StateStore
from gladius.submission import (
    score_submission_artifact,
    submit,
    update_best_submission_score,
)
from gladius.utils.competition_config import load_competition_config
from gladius.utils.project_setup import setup_project_dir, write_claude_md


def _has_iteration_result(state: CompetitionState) -> bool:
    """Return True if the current iteration already has a usable experiment artifact."""
    for exp in reversed(state.experiments):
        if exp.get("iteration") != state.iteration:
            continue
        if exp.get("submission_file") or exp.get("oof_score") is not None:
            return True
    return False


def _halt_with_reason(state: CompetitionState, *, phase: str, reason: str) -> None:
    state.last_stop_reason = reason
    state.error_log.append(
        {"phase": phase, "iteration": state.iteration, "error": reason}
    )
    state.phase = "done"
    logger.warning(f"Guardrail stop [{phase}]: {reason}")


# ── Main loop ─────────────────────────────────────────────────────────────────


async def run_competition(
    competition_dir: str,
    max_iterations: int = 20,
    resume_from_db: bool = True,
    auto_submit: bool = True,
    n_parallel: int = 1,
    mode: str = "experimental",
    max_iteration_seconds: int | None = None,
    max_agent_calls_per_iteration: int | None = None,
    max_failed_runs_total: int | None = None,
) -> CompetitionState:
    cfg = load_competition_config(competition_dir)
    competition_id = cfg["competition_id"]
    platform = cfg["platform"]
    data_dir = cfg["data_dir"]
    target_metric = cfg["metric"]
    metric_direction = cfg["direction"]

    if mode == "personal-production":
        if max_iteration_seconds is None:
            max_iteration_seconds = 1800
        if max_agent_calls_per_iteration is None:
            max_agent_calls_per_iteration = 5
        if max_failed_runs_total is None:
            max_failed_runs_total = 20
        n_parallel = 1

    run_preflight_or_raise(
        competition_dir=competition_dir,
        platform=platform,
        data_dir=data_dir,
        target_metric=target_metric,
        max_iterations=max_iterations,
        n_parallel=n_parallel,
    )

    project_dir = competition_dir
    gladius_dir = Path(project_dir) / ".gladius"
    gladius_dir.mkdir(parents=True, exist_ok=True)

    store = StateStore(str(gladius_dir / "state.db"))

    state: CompetitionState | None = store.load() if resume_from_db else None
    if state is None:
        logger.info("Initialising new competition state")
        state = CompetitionState(
            competition_id=competition_id,
            data_dir=str(Path(data_dir).resolve()),
            output_dir=str(gladius_dir.resolve()),
            target_metric=target_metric,
            metric_direction=metric_direction,
            max_iterations=max_iterations,
        )
    else:
        _best_str = (
            f"{state.best_oof_score:.6f}"
            if state.best_oof_score is not None
            else "none"
        )
        logger.info(
            f"Resuming from iteration {state.iteration}, phase={state.phase}, "
            f"best={_best_str}"
        )
        if state.max_iterations != max_iterations:
            logger.info(
                f"Updating max_iterations from CLI: {state.max_iterations} -> {max_iterations}"
            )
            state.max_iterations = max_iterations

        # Recalibrate best scores if experiments exist but bests were never persisted.
        if state.experiments:
            if target_metric and state.best_oof_score is None:
                scored = [
                    e["oof_score"]
                    for e in state.experiments
                    if e.get("oof_score") is not None
                ]
                if scored:
                    state.best_oof_score = (
                        max(scored) if metric_direction != "minimize" else min(scored)
                    )
                    logger.info(
                        f"Recalibrated best_oof_score from experiments: {state.best_oof_score:.6f}"
                    )
            elif not target_metric and state.best_quality_score is None:
                scored = [
                    e["quality_score"]
                    for e in state.experiments
                    if e.get("quality_score") is not None
                ]
                if scored:
                    state.best_quality_score = max(scored)
                    logger.info(
                        f"Recalibrated best_quality_score from experiments: {state.best_quality_score}/100"
                    )

        if state.phase == "done" and state.iteration < state.max_iterations:
            logger.info(
                f"Resuming: phase='done' but only {state.iteration}/{state.max_iterations} "
                f"iterations used — resetting to planning"
            )
            state.phase = "planning"
            state.consecutive_errors = 0

    logger.info("Setting up project directory")
    setup_project_dir(state, project_dir, platform=platform)

    while state.iteration < state.max_iterations and state.phase != "done":
        iteration_started = time.monotonic()
        agent_calls_this_iteration = 0

        def _consume_agent_call(agent_label: str) -> bool:
            return _consume_agent_calls(agent_label, 1)

        def _consume_agent_calls(agent_label: str, count: int) -> bool:
            nonlocal agent_calls_this_iteration
            agent_calls_this_iteration += count
            if (
                max_agent_calls_per_iteration is not None
                and agent_calls_this_iteration > max_agent_calls_per_iteration
            ):
                _halt_with_reason(
                    state,
                    phase="guardrail",
                    reason=(
                        "agent call budget exceeded "
                        f"({agent_calls_this_iteration}/{max_agent_calls_per_iteration}) "
                        f"before {agent_label}"
                    ),
                )
                return False
            return True

        def _check_iteration_runtime_budget() -> bool:
            if max_iteration_seconds is None:
                return True
            elapsed = time.monotonic() - iteration_started
            if elapsed > max_iteration_seconds:
                _halt_with_reason(
                    state,
                    phase="guardrail",
                    reason=(
                        "iteration runtime budget exceeded "
                        f"({elapsed:.1f}s > {max_iteration_seconds}s)"
                    ),
                )
                return False
            return True

        if (
            max_failed_runs_total is not None
            and len(state.failed_runs) >= max_failed_runs_total
        ):
            _halt_with_reason(
                state,
                phase="guardrail",
                reason=(
                    "failed run budget exceeded "
                    f"({len(state.failed_runs)}/{max_failed_runs_total})"
                ),
            )
            continue

        _score_str = (
            f"quality={f'{state.best_quality_score}/100' if state.best_quality_score is not None else 'none'}"
            if state.target_metric is None
            else f"best={f'{state.best_oof_score:.6f}' if state.best_oof_score is not None else 'none'}"
        )
        logger.info(
            f"[iter {state.iteration:02d}/{state.max_iterations}] "
            f"phase={state.phase}  {_score_str}  "
            f"experiments={len(state.experiments)}"
        )

        write_claude_md(state, project_dir)

        try:
            if state.phase == "planning":
                if await run_planning_phase(
                    state,
                    store,
                    data_dir,
                    project_dir,
                    platform,
                    n_parallel,
                    run_planner=run_planner,
                    consume_agent_call=_consume_agent_call,
                    check_budget=_check_iteration_runtime_budget,
                ):
                    continue

            elif state.phase == "implementing":
                if await run_implementation_phase(
                    state,
                    store,
                    project_dir,
                    n_parallel,
                    run_solver=run_solver,
                    consume_agent_call=_consume_agent_call,
                    consume_agent_calls=_consume_agent_calls,
                    check_budget=_check_iteration_runtime_budget,
                ):
                    continue

            elif state.phase == "validation":
                if await run_validation_phase(
                    state,
                    store,
                    project_dir,
                    platform,
                    auto_submit,
                    run_validation_agent=run_validation_agent,
                    run_summarizer=run_summarizer,
                    submit=submit,
                    score_submission_artifact=score_submission_artifact,
                    update_best_submission_score=update_best_submission_score,
                    consume_agent_call=_consume_agent_call,
                    check_budget=_check_iteration_runtime_budget,
                ):
                    continue

        except Exception as exc:
            logger.error(
                f"Unhandled error in phase={state.phase}: {exc}", exc_info=True
            )
            state.error_log.append(
                {
                    "phase": state.phase,
                    "iteration": state.iteration,
                    "error": str(exc),
                }
            )
            state.consecutive_errors += 1
            store.record_event(
                iteration=state.iteration,
                phase=state.phase,
                event="error",
                detail=f"consecutive={state.consecutive_errors} {type(exc).__name__}: {str(exc)[:200]}",
            )
            if state.consecutive_errors >= 3:
                logger.critical("3 consecutive errors — halting")
                state.last_stop_reason = "consecutive error budget exceeded (3/3)"
                state.phase = "done"
            else:
                if state.phase == "implementing" and _has_iteration_result(state):
                    logger.warning(
                        "Implementation failed after producing artifacts; "
                        "continuing with validation phase"
                    )
                    state.phase = "validation"
                else:
                    state.phase = "planning"

        finally:
            store.save(state)

    store.close()
    if state.target_metric:
        _final_score = f"best_oof={f'{state.best_oof_score:.6f}' if state.best_oof_score is not None else 'none'}"
    else:
        _final_score = f"best_quality={f'{state.best_quality_score}/100' if state.best_quality_score is not None else 'none'}"
    logger.info(
        f"Done. iterations={state.iteration}  "
        f"{_final_score}  "
        f"submissions={state.submission_count}"
    )
    return state


# ── Entry point shim ──────────────────────────────────────────────────────────
# pyproject.toml scripts entry point (gladius.orchestrator:main) stays valid
# without reinstalling the package.

from gladius.cli import main  # noqa: E402, F401
