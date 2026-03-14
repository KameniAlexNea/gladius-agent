"""
Main competition loop — topology-driven.

Design rules:
  - The orchestrator selects a management topology based on the competition config.
  - Each iteration is a single call to topology.run_iteration() which returns
    an IterationResult.  The orchestrator acts on it (update state, submit,
    check stop) but never orchestrates agent phases directly.
  - Agents output IterationResult; the orchestrator acts on it.
  - State is saved to SQLite after every iteration (crash-safe).
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

from loguru import logger

from gladius.agents.topologies import TOPOLOGY_REGISTRY, IterationResult
from gladius.preflight import run_preflight_or_raise
from gladius.state import CompetitionState, StateStore
from gladius.submission import (
    score_submission_artifact,
    submit,
    update_best_submission_score,
)
from gladius.utils.competition_config import load_competition_config
from gladius.utils.project_setup import setup_project_dir, write_claude_md


def _halt_with_reason(state: CompetitionState, *, reason: str) -> None:
    state.last_stop_reason = reason
    state.error_log.append({"iteration": state.iteration, "error": reason})
    state.done = True
    logger.warning(f"Guardrail stop: {reason}")


def _is_better(
    new_score: float,
    best_score: float | None,
    direction: str | None,
) -> bool:
    if best_score is None:
        return True
    if direction == "minimize":
        return new_score < best_score - 1e-4
    return new_score > best_score + 1e-4


def _reset_daily_submissions(state: CompetitionState) -> None:
    today = date.today().isoformat()
    if state.last_submission_date != today:
        state.submission_count = 0
        state.last_submission_date = today


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
    topology_name = cfg["topology"]

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
            topology=topology_name,
            max_iterations=max_iterations,
            submission_threshold=cfg.get("submission_threshold"),
        )
    else:
        _best_str = (
            f"{state.best_oof_score:.6f}"
            if state.best_oof_score is not None
            else "none"
        )
        logger.info(
            f"Resuming from iteration {state.iteration}, "
            f"topology={state.topology}, best={_best_str}"
        )
        if state.max_iterations != max_iterations:
            logger.info(
                f"Updating max_iterations: {state.max_iterations} -> {max_iterations}"
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
                        f"Recalibrated best_oof_score: {state.best_oof_score:.6f}"
                    )
            elif not target_metric and state.best_quality_score is None:
                scored = [
                    e["quality_score"]
                    for e in state.experiments
                    if e.get("quality_score") is not None
                ]
                if scored:
                    state.best_quality_score = max(scored)

        if state.done and state.iteration < state.max_iterations:
            logger.info(
                f"Resuming: done=True but only {state.iteration}/{state.max_iterations} "
                "iterations used — resetting"
            )
            state.done = False
            state.consecutive_errors = 0

    # Resolve topology
    if topology_name not in TOPOLOGY_REGISTRY:
        raise ValueError(
            f"Unknown topology {topology_name!r}. "
            f"Valid: {list(TOPOLOGY_REGISTRY.keys())}"
        )
    topology = TOPOLOGY_REGISTRY[topology_name]()

    logger.info(f"Topology: {topology_name}")
    setup_project_dir(state, project_dir, platform=platform)

    while state.iteration < state.max_iterations and not state.done:
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
                reason=(
                    "failed run budget exceeded "
                    f"({len(state.failed_runs)}/{max_failed_runs_total})"
                ),
            )
            continue

        _score_str = (
            f"quality={state.best_quality_score}/100"
            if state.target_metric is None
            else f"best={f'{state.best_oof_score:.6f}' if state.best_oof_score is not None else 'none'}"
        )
        logger.info(
            f"[iter {state.iteration:02d}/{state.max_iterations}] "
            f"topology={state.topology}  {_score_str}  "
            f"experiments={len(state.experiments)}"
        )

        _reset_daily_submissions(state)
        write_claude_md(state, project_dir)

        result: IterationResult | None = None
        try:
            result = await topology.run_iteration(
                state,
                project_dir,
                platform,
                n_parallel=n_parallel,
                consume_agent_call=_consume_agent_call,
                check_budget=_check_iteration_runtime_budget,
            )
        except Exception as exc:
            logger.error(f"Unhandled error in topology.run_iteration: {exc}", exc_info=True)
            state.error_log.append({"iteration": state.iteration, "error": str(exc)})
            state.consecutive_errors += 1
            store.record_event(
                iteration=state.iteration,
                topology=state.topology,
                event="error",
                detail=f"consecutive={state.consecutive_errors} {type(exc).__name__}: {str(exc)[:200]}",
            )
            if state.consecutive_errors >= 3:
                logger.critical("3 consecutive errors — halting")
                state.last_stop_reason = "consecutive error budget exceeded (3/3)"
                state.done = True
            store.save(state)
            continue

        # ── Process iteration result ──────────────────────────────────────────
        state.consecutive_errors = 0

        # Update session IDs for all roles used
        if result.team_session_ids:
            state.team_session_ids.update(result.team_session_ids)

        # Record plan
        if result.plan_text:
            store.record_plan(
                iteration=state.iteration,
                approach_summary=result.approach_summary,
                plan_text=result.plan_text,
                session_id=result.team_session_ids.get("team-lead"),
            )
            state.current_plan = {
                "approach_summary": result.approach_summary,
                "plan_text": result.plan_text,
            }

        # Record memory
        if result.memory_content:
            mem_path = (
                Path(project_dir) / ".claude" / "agent-memory" / "team-lead" / "MEMORY.md"
            )
            mem_path.parent.mkdir(parents=True, exist_ok=True)
            mem_path.write_text(result.memory_content)

        if result.status == "error":
            state.failed_runs.append(
                {
                    "iteration": state.iteration,
                    "status": "error",
                    "error": result.error_message,
                    "approach": result.approach_summary,
                }
            )
            store.record_event(
                iteration=state.iteration,
                topology=state.topology,
                event="iteration_error",
                detail=result.error_message[:200],
            )
            state.iteration += 1
            store.save(state)
            continue

        # Successful experiment
        experiment = {
            "iteration": state.iteration,
            "oof_score": result.oof_score,
            "quality_score": result.quality_score,
            "submission_file": result.submission_file,
            "notes": result.notes,
            "approach": result.approach_summary,
            "solution_files": result.solution_files,
        }
        state.experiments.append(experiment)

        # Update best scores
        if target_metric and result.oof_score is not None:
            if _is_better(result.oof_score, state.best_oof_score, metric_direction):
                state.best_oof_score = result.oof_score
                state.best_submission_path = result.submission_file or None
        elif not target_metric and result.quality_score is not None:
            if _is_better(result.quality_score, state.best_quality_score, "maximize"):
                state.best_quality_score = result.quality_score
                state.best_submission_path = result.submission_file or None

        # Submission
        _reset_daily_submissions(state)
        if (
            auto_submit
            and result.submit
            and result.format_ok
            and result.submission_file
            and state.submission_count < state.max_submissions_per_day
        ):
            submitted, sub_err = submit(
                platform=platform,
                competition_id=competition_id,
                submission_path=result.submission_file,
                message=f"iter {state.iteration}: {result.approach_summary[:60]}",
            )
            if submitted:
                state.submission_count += 1
                lb_score = score_submission_artifact(
                    platform=platform, submission_path=result.submission_file
                )
                if lb_score is not None:
                    update_best_submission_score(state, lb_score)
                    state.lb_scores.append(
                        {"score": lb_score, "timestamp": date.today().isoformat(), "public_lb": True}
                    )
                store.record_event(
                    iteration=state.iteration,
                    topology=state.topology,
                    event="submission",
                    detail=f"count={state.submission_count} lb={lb_score}",
                )
            else:
                logger.warning(f"Submission failed: {sub_err}")

        # Code snapshots
        if result.solution_files:
            store.record_code_snapshots(
                state.iteration, result.solution_files, project_dir
            )

        store.record_event(
            iteration=state.iteration,
            topology=state.topology,
            event="iteration_complete",
            detail=(
                f"status={result.status} "
                f"oof={result.oof_score} "
                f"improvement={result.is_improvement} "
                f"submit={result.submit}"
            ),
        )

        # Stop condition
        if result.stop:
            logger.info(f"Validator signalled stop at iteration {state.iteration}")
            state.last_stop_reason = "validator stop signal"
            state.done = True
            state.iteration += 1
            store.save(state)
            break

        state.iteration += 1
        store.save(state)

    store.close()

    if state.target_metric:
        _final = f"best_oof={f'{state.best_oof_score:.6f}' if state.best_oof_score is not None else 'none'}"
    else:
        _final = f"best_quality={f'{state.best_quality_score}/100' if state.best_quality_score is not None else 'none'}"
    logger.info(
        f"Done. iterations={state.iteration}  "
        f"{_final}  "
        f"submissions={state.submission_count}"
    )
    return state


# ── Entry point shim ──────────────────────────────────────────────────────────

from gladius.cli import main  # noqa: E402, F401

