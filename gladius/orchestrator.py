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

import asyncio
import logging
import time
from datetime import datetime as _dt
from datetime import timezone as _tz
from pathlib import Path

from gladius.agents.implementer import run_implementer
from gladius.agents.planner import run_planner
from gladius.agents.summarizer import run_summarizer
from gladius.agents.validation import run_validation_agent
from gladius.preflight import run_preflight_or_raise
from gladius.state import CompetitionState, StateStore
from gladius.submission import (
    score_submission_artifact,
    submit,
    update_best_submission_score,
)
from gladius.utils.competition_config import load_competition_config
from gladius.utils.project_setup import setup_project_dir, write_claude_md

logger = logging.getLogger("gladius.orchestrator")


# ── Score helpers ─────────────────────────────────────────────────────────────


def _compute_hybrid_quality_score(
    *,
    implementer_quality_score: float | None,
    validator_quality_score: float | None,
    validation: dict,
) -> float:
    """Compute conservative hybrid quality score for open-ended tasks."""
    impl = float(implementer_quality_score or 0.0)
    val = validator_quality_score
    score = impl if val is None else 0.75 * float(val) + 0.25 * impl
    if validation.get("format_ok") is False:
        score -= 15.0
    next_dirs = validation.get("next_directions") or []
    if isinstance(next_dirs, list):
        score -= min(len(next_dirs), 3) * 2.0
    return round(max(0.0, min(100.0, score)), 2)


def _is_better(
    new_score: float,
    best_score: float | None,
    direction: str | None,
    threshold: float = 1e-4,
) -> bool:
    """Deterministic improvement check.  None best_score means no prior result.

    direction=None means open-ended: higher quality score is always better,
    with a larger threshold of 2.0 points.
    """
    if best_score is None:
        return True
    if direction is None:
        return new_score > best_score + 2.0
    if direction == "maximize":
        return new_score > best_score + threshold
    return new_score < best_score - threshold


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
            # ── PLANNING ─────────────────────────────────────────────────────
            if state.phase == "planning":
                if not _consume_agent_call("planner"):
                    continue
                _t0 = time.perf_counter()
                _started_at = _dt.now(_tz.utc).isoformat()
                plan, session_id = await run_planner(
                    state,
                    data_dir,
                    project_dir,
                    platform=platform,
                    n_parallel=n_parallel,
                )
                store.record_agent_run(
                    iteration=state.iteration,
                    phase="planning",
                    agent_name="planner",
                    started_at=_started_at,
                    duration_ms=int((time.perf_counter() - _t0) * 1000),
                    session_id=session_id,
                )
                state.current_plan = plan
                state.planner_session_id = session_id
                logger.info(f"Plan ready: {plan.get('approach_summary', '')[:120]}")

                store.record_plan(
                    iteration=state.iteration,
                    approach_summary=plan.get("approach_summary", ""),
                    plan_text=plan.get("plan_text", ""),
                    session_id=session_id,
                )
                _plan_dir = Path(project_dir) / ".claude" / "plans"
                _plan_dir.mkdir(parents=True, exist_ok=True)
                (_plan_dir / f"iter-{state.iteration:02d}.md").write_text(
                    plan.get("plan_text", ""), encoding="utf-8"
                )
                store.record_event(
                    iteration=state.iteration,
                    phase="planning",
                    event="plan_ready",
                    detail=plan.get("approach_summary", "")[:200],
                )
                state.phase = "implementing"
                if not _check_iteration_runtime_budget():
                    continue

            # ── IMPLEMENTING ─────────────────────────────────────────────────
            elif state.phase == "implementing":
                if state.current_plan is None:
                    logger.warning(
                        "Resuming in 'implementing' phase with no current_plan "
                        "— falling back to planning"
                    )
                    state.phase = "planning"
                    store.save(state)
                    continue

                alt_plans: list[dict] = (
                    state.current_plan.get("plans", []) if state.current_plan else []
                )
                if n_parallel > 1 and len(alt_plans) > 1:
                    plans_to_run = alt_plans[:n_parallel]
                    if not _consume_agent_calls(
                        "parallel implementers", len(plans_to_run)
                    ):
                        continue
                    logger.info(f"Running {len(plans_to_run)} parallel implementers")
                    _t0_impl = time.perf_counter()
                    _started_impl = _dt.now(_tz.utc).isoformat()
                    results = await asyncio.gather(
                        *[run_implementer(p, state, project_dir) for p in plans_to_run],
                        return_exceptions=True,
                    )
                    _impl_dur_ms = int((time.perf_counter() - _t0_impl) * 1000)
                    successful: list[dict] = []
                    for i, r in enumerate(results):
                        if isinstance(r, Exception):
                            logger.warning(f"Parallel implementer {i} raised: {r}")
                            state.failed_runs.append(
                                {
                                    "iteration": state.iteration,
                                    "status": "error",
                                    "error": str(r),
                                    "approach": plans_to_run[i].get(
                                        "approach_summary", ""
                                    ),
                                }
                            )
                        elif isinstance(r, dict) and r.get("status") == "success":
                            successful.append(r)
                        else:
                            err = (
                                r.get("error_message", "")
                                if isinstance(r, dict)
                                else str(r)
                            )
                            state.failed_runs.append(
                                {
                                    "iteration": state.iteration,
                                    "status": (
                                        r.get("status", "error")
                                        if isinstance(r, dict)
                                        else "error"
                                    ),
                                    "error": err,
                                    "approach": plans_to_run[i].get(
                                        "approach_summary", ""
                                    ),
                                }
                            )
                    if not successful:
                        logger.warning("All parallel implementers failed")
                        state.consecutive_errors += 1
                        state.iteration += 1
                        state.phase = "planning"
                        store.record_event(
                            iteration=state.iteration - 1,
                            phase="implementing",
                            event="impl_failed",
                            detail=f"all {len(plans_to_run)} parallel implementers failed",
                        )
                        store.save(state)
                        continue
                    if state.target_metric is None:
                        result = max(
                            successful, key=lambda r: r.get("quality_score", 0) or 0
                        )
                    else:
                        direction = state.metric_direction
                        result = max(
                            successful,
                            key=lambda r: (
                                r["oof_score"]
                                if direction == "maximize"
                                else -r["oof_score"]
                            ),
                        )
                    _best_score_str = (
                        f"quality {result.get('quality_score', 0)}/100"
                        if state.target_metric is None
                        else f"OOF {result['oof_score']:.6f}"
                    )
                    logger.info(
                        f"Best parallel result: {_best_score_str} "
                        f"(from {len(successful)}/{len(plans_to_run)} successful)"
                    )
                    for i_s, r in enumerate(successful):
                        store.record_agent_run(
                            iteration=state.iteration,
                            phase="implementing",
                            agent_name="implementer",
                            started_at=_started_impl,
                            duration_ms=_impl_dur_ms,
                            notes=f"parallel {i_s + 1}/{len(successful)}",
                        )
                        if r is not result:
                            state.experiments.append(
                                {
                                    "iteration": state.iteration,
                                    "oof_score": r.get("oof_score"),
                                    "quality_score": r.get("quality_score"),
                                    "solution_files": r.get("solution_files", []),
                                    "submission_file": r.get("submission_file", ""),
                                    "notes": r.get("notes", ""),
                                    "approach": "",
                                }
                            )
                            store.record_code_snapshots(
                                state.iteration,
                                r.get("solution_files", []),
                                project_dir,
                            )
                    state.experiments.append(
                        {
                            "iteration": state.iteration,
                            "oof_score": result.get("oof_score"),
                            "quality_score": result.get("quality_score"),
                            "solution_files": result.get("solution_files", []),
                            "submission_file": result.get("submission_file", ""),
                            "notes": result.get("notes", ""),
                            "approach": "",
                        }
                    )
                    store.record_code_snapshots(
                        state.iteration, result.get("solution_files", []), project_dir
                    )
                else:
                    # ── Sequential single implementer ─────────────────────
                    if not _consume_agent_call("implementer"):
                        continue
                    _t0_impl = time.perf_counter()
                    _started_impl = _dt.now(_tz.utc).isoformat()
                    result = await run_implementer(
                        state.current_plan, state, project_dir
                    )
                    store.record_agent_run(
                        iteration=state.iteration,
                        phase="implementing",
                        agent_name="implementer",
                        started_at=_started_impl,
                        duration_ms=int((time.perf_counter() - _t0_impl) * 1000),
                        is_error=result.get("status") != "success",
                        notes=(
                            result.get("status")
                            if result.get("status") != "success"
                            else None
                        ),
                    )
                    if result["status"] != "success":
                        logger.warning(
                            f"Implementer {result['status']}: {result.get('error_message', '')}"
                        )
                        state.failed_runs.append(
                            {
                                "iteration": state.iteration,
                                "status": result["status"],
                                "error": result.get("error_message", ""),
                                "approach": (
                                    state.current_plan.get("approach_summary", "")
                                    if state.current_plan
                                    else ""
                                ),
                            }
                        )
                        state.consecutive_errors += 1
                        state.iteration += 1
                        state.phase = "planning"
                        store.record_event(
                            iteration=state.iteration - 1,
                            phase="implementing",
                            event="impl_failed",
                            detail=f"status={result['status']} error={result.get('error_message', '')[:150]}",
                        )
                        store.save(state)
                        continue
                    state.experiments.append(
                        {
                            "iteration": state.iteration,
                            "oof_score": result.get("oof_score"),
                            "quality_score": result.get("quality_score"),
                            "solution_files": result.get("solution_files", []),
                            "submission_file": result.get("submission_file", ""),
                            "notes": result.get("notes", ""),
                            "approach": (
                                state.current_plan.get("approach_summary", "")
                                if state.current_plan
                                else ""
                            ),
                        }
                    )
                    store.record_code_snapshots(
                        state.iteration, result.get("solution_files", []), project_dir
                    )

                # ── Common post-implementation logging ────────────────────────
                if state.target_metric:
                    _oof = result.get("oof_score")
                    logger.info(
                        f"Implementation done — OOF {state.target_metric}: "
                        f"{f'{_oof:.6f}' if _oof is not None else 'n/a'}"
                    )
                    store.record_event(
                        iteration=state.iteration,
                        phase="implementing",
                        event="impl_done",
                        detail=(
                            f"oof={f'{_oof:.6f}' if _oof is not None else 'n/a'} "
                            f"metric={state.target_metric} "
                            f"files={','.join(result.get('solution_files', []))}"
                        ),
                    )
                else:
                    _quality = result.get("quality_score", 0) or 0
                    logger.info(f"Implementation done — quality: {_quality}/100")
                    store.record_event(
                        iteration=state.iteration,
                        phase="implementing",
                        event="impl_done",
                        detail=f"quality={_quality}/100 files={','.join(result.get('solution_files', []))}",
                    )

                state.phase = "validation"
                if not _check_iteration_runtime_budget():
                    continue

            # ── VALIDATION ───────────────────────────────────────────────────
            elif state.phase == "validation":
                latest = state.experiments[-1]
                oof_score = latest.get("oof_score")
                quality_score = latest.get("quality_score", 0) or 0
                submission_file = latest["submission_file"]

                from datetime import date as _date

                today = _date.today().isoformat()
                if state.last_submission_date != today:
                    if state.submission_count > 0:
                        logger.info(
                            f"New day ({today}) — resetting submission_count "
                            f"from {state.submission_count} to 0"
                        )
                    state.submission_count = 0
                    state.last_submission_date = today

                if not submission_file:
                    logger.warning(
                        "No submission file produced — skipping format check, "
                        "running deterministic improvement check only"
                    )
                    validation: dict = {
                        "is_improvement": None,
                        "submit": False,
                        "reasoning": "No submission file produced",
                    }
                else:
                    if not _consume_agent_call("validation"):
                        continue
                    _t0_val = time.perf_counter()
                    _started_val = _dt.now(_tz.utc).isoformat()
                    try:
                        validation = await run_validation_agent(
                            solution_path=", ".join(latest.get("solution_files", [])),
                            oof_score=oof_score,
                            quality_score=quality_score,
                            submission_path=submission_file,
                            state=state,
                            project_dir=project_dir,
                            platform=platform,
                        )
                        store.record_agent_run(
                            iteration=state.iteration,
                            phase="validation",
                            agent_name="validation",
                            started_at=_started_val,
                            duration_ms=int((time.perf_counter() - _t0_val) * 1000),
                        )
                    except Exception as exc:
                        store.record_agent_run(
                            iteration=state.iteration,
                            phase="validation",
                            agent_name="validation",
                            started_at=_started_val,
                            duration_ms=int((time.perf_counter() - _t0_val) * 1000),
                            is_error=True,
                            notes=str(exc)[:200],
                        )
                        logger.warning(
                            f"Validation agent failed: {exc} "
                            f"— using deterministic fallback, summarizer will still run"
                        )
                        validation = {
                            "is_improvement": None,
                            "submit": False,
                            "stop": False,
                            "reasoning": f"Validation agent failed: {exc}",
                            "next_directions": ["Retry validation in next iteration"],
                        }

                validator_quality_score = validation.get("quality_score")
                if state.target_metric is None:
                    hybrid_quality_score = _compute_hybrid_quality_score(
                        implementer_quality_score=quality_score,
                        validator_quality_score=validator_quality_score,
                        validation=validation,
                    )
                    latest["quality_score"] = hybrid_quality_score
                    logger.info(
                        "Open-task quality scoring: "
                        f"implementer={quality_score}/100, "
                        f"validator={validator_quality_score if validator_quality_score is not None else 'n/a'}/100, "
                        f"hybrid={hybrid_quality_score}/100"
                    )

                primary_score = (
                    hybrid_quality_score if state.target_metric is None else oof_score
                )
                best_primary = (
                    state.best_quality_score
                    if state.target_metric is None
                    else state.best_oof_score
                )
                deterministic_improvement = _is_better(
                    primary_score, best_primary, state.metric_direction
                )

                if validation.get("is_improvement") != deterministic_improvement:
                    _score_label = (
                        f"quality {primary_score}/100 vs best "
                        f"{f'{best_primary}/100' if best_primary is not None else 'none'}"
                        if state.target_metric is None
                        else f"OOF {oof_score:.6f} vs best "
                        f"{f'{state.best_oof_score:.6f}' if state.best_oof_score is not None else 'none'}"
                    )
                    logger.warning(
                        f"Validation agent is_improvement={validation.get('is_improvement')} "
                        f"overridden by deterministic check -> {deterministic_improvement} "
                        f"({_score_label})"
                    )

                if deterministic_improvement:
                    if state.target_metric is None:
                        state.best_quality_score = primary_score
                        logger.info(f"New best quality score: {primary_score}/100")
                        store.record_event(
                            iteration=state.iteration,
                            phase="validation",
                            event="new_best",
                            detail=f"quality={primary_score}/100 (prev={best_primary}/100 if best_primary is not None else 'none')",
                        )
                    else:
                        state.best_oof_score = oof_score
                        logger.info(f"New best OOF: {oof_score:.6f}")
                        store.record_event(
                            iteration=state.iteration,
                            phase="validation",
                            event="new_best",
                            detail=(
                                f"oof={oof_score:.6f} "
                                f"prev={f'{best_primary:.6f}' if best_primary is not None else 'none'} "
                                f"metric={state.target_metric}"
                            ),
                        )
                else:
                    store.record_event(
                        iteration=state.iteration,
                        phase="validation",
                        event="no_improvement",
                        detail=(
                            f"score={f'{primary_score:.6f}' if primary_score is not None else 'n/a'} "
                            f"best={f'{best_primary:.6f}' if best_primary is not None else 'none'}"
                        ),
                    )

                if (
                    deterministic_improvement
                    and submission_file
                    and (
                        platform == "none"
                        or state.submission_count < state.max_submissions_per_day
                    )
                    and auto_submit
                ):
                    logger.info(f"Submitting [{platform}]: {submission_file}")
                    _submit_msg = (
                        f"iter-{state.iteration} quality={primary_score}/100"
                        if state.target_metric is None
                        else f"iter-{state.iteration} oof={oof_score:.6f}"
                    )
                    submit_ok, submit_error_type = submit(
                        platform=platform,
                        competition_id=state.competition_id,
                        submission_path=submission_file,
                        message=_submit_msg,
                    )
                    if submit_ok:
                        state.best_submission_path = submission_file
                        if platform != "none":
                            state.submission_count += 1
                        scored_lb = score_submission_artifact(
                            platform=platform, submission_path=submission_file
                        )
                        if scored_lb is not None:
                            from datetime import datetime as _datetime

                            state.lb_scores.append(
                                {
                                    "score": scored_lb,
                                    "timestamp": _datetime.now().isoformat(),
                                    "public_lb": True,
                                }
                            )
                            update_best_submission_score(
                                state=state, new_score=scored_lb
                            )
                            logger.info(
                                f"Recorded leaderboard score: {scored_lb:.6f} "
                                f"(best={state.best_submission_score:.6f})"
                            )
                        store.record_event(
                            iteration=state.iteration,
                            phase="validation",
                            event="submitted",
                            detail=(
                                f"file={submission_file} platform={platform}"
                                + (
                                    f" lb={scored_lb:.6f}"
                                    if scored_lb is not None
                                    else ""
                                )
                            ),
                        )
                    else:
                        state.error_log.append(
                            {
                                "phase": "submission",
                                "iteration": state.iteration,
                                "error": f"submit_failed:{submit_error_type or 'unknown'}",
                            }
                        )
                        store.record_event(
                            iteration=state.iteration,
                            phase="validation",
                            event="submit_failed",
                            detail=f"platform={platform} reason={submit_error_type or 'unknown'}",
                        )

                try:
                    if not _consume_agent_call("summarizer"):
                        continue
                    _t0_sum = time.perf_counter()
                    _started_sum = _dt.now(_tz.utc).isoformat()
                    summary = await run_summarizer(
                        state,
                        project_dir,
                        latest_experiment=latest,
                        validation_notes=validation.get("reasoning", ""),
                    )
                    store.record_agent_run(
                        iteration=state.iteration,
                        phase="summarizing",
                        agent_name="summarizer",
                        started_at=_started_sum,
                        duration_ms=int((time.perf_counter() - _t0_sum) * 1000),
                    )
                    if summary:
                        logger.info(f"Summarizer: {summary}")
                except Exception as exc:
                    logger.warning(f"Summarizer failed (non-fatal): {exc}")

                if not _check_iteration_runtime_budget():
                    continue

                state.consecutive_errors = 0
                state.iteration += 1

                # ── Deterministic stop check ──────────────────────────────────
                _plateau_key = (
                    "quality_score" if state.target_metric is None else "oof_score"
                )
                _scored = [
                    e[_plateau_key]
                    for e in state.experiments
                    if e.get(_plateau_key) is not None
                ]
                _plateau_threshold = 3.0 if state.target_metric is None else 0.001
                deterministic_stop = False
                if len(_scored) >= 3:
                    _span = max(_scored[-3:]) - min(_scored[-3:])
                    if _span < _plateau_threshold:
                        deterministic_stop = True
                        logger.info(
                            f"Deterministic stop: last 3 scores span {_span:.4f} "
                            f"(< {_plateau_threshold}) — plateau detected."
                        )

                agent_stop = bool(validation.get("stop"))
                no_next_directions = not bool(validation.get("next_directions"))
                should_stop = deterministic_stop and agent_stop and no_next_directions
                if should_stop:
                    logger.info(
                        "Stopping: plateau detected + agent stop=True + no next directions."
                    )
                    store.record_event(
                        iteration=state.iteration,
                        phase="validation",
                        event="stop_plateau",
                        detail="plateau + agent stop=True + no next_directions",
                    )
                elif agent_stop and not should_stop:
                    _why = []
                    if not deterministic_stop:
                        _why.append("no plateau")
                    if not no_next_directions:
                        _why.append(
                            f"next_directions={validation.get('next_directions')}"
                        )
                    logger.info(
                        f"Agent requested stop=True but continuing — {', '.join(_why)}."
                    )
                state.phase = "done" if should_stop else "planning"

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
