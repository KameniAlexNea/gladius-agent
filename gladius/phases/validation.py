"""Validation phase: validates results, handles submission, summarises, checks stop conditions."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import date as _date
from datetime import datetime as _dt
from datetime import timezone as _tz

from gladius.db.store import StateStore
from gladius.state import CompetitionState

logger = logging.getLogger(__name__)


# ── Score helpers (used only in this phase) ───────────────────────────────────


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
    """Deterministic improvement check.  ``None`` best_score means no prior result.

    ``direction=None`` means open-ended: higher quality score is always better,
    with a larger threshold of 2.0 points.
    """
    if best_score is None:
        return True
    if direction is None:
        return new_score > best_score + 2.0
    if direction == "maximize":
        return new_score > best_score + threshold
    return new_score < best_score - threshold


# ── Phase entry-point ─────────────────────────────────────────────────────────


async def run_validation_phase(
    state: CompetitionState,
    store: StateStore,
    project_dir: str,
    platform: str,
    auto_submit: bool,
    *,
    run_validation_agent: Callable,
    run_summarizer: Callable,
    submit: Callable,
    score_submission_artifact: Callable,
    update_best_submission_score: Callable,
    consume_agent_call: Callable[[str], bool],
    check_budget: Callable[[], bool],
) -> bool:
    """Run the validation phase.

    Returns ``True`` if the outer loop should ``continue``, ``False`` on normal
    completion.
    """
    latest = state.experiments[-1]
    oof_score = latest.get("oof_score")
    quality_score = latest.get("quality_score", 0) or 0
    submission_file = latest["submission_file"]

    today = _date.today().isoformat()
    if state.last_submission_date != today:
        if state.submission_count > 0:
            logger.info(
                f"New day ({today}) — resetting submission_count "
                f"from {state.submission_count} to 0"
            )
        state.submission_count = 0
        state.last_submission_date = today

    validation = await _run_validation_agent(
        state,
        store,
        project_dir,
        platform,
        latest,
        oof_score=oof_score,
        quality_score=quality_score,
        submission_file=submission_file,
        run_validation_agent=run_validation_agent,
        consume_agent_call=consume_agent_call,
    )
    if validation is None:
        return True  # budget fired inside agent call

    # ── Hybrid quality scoring for open-ended tasks ───────────────────────────
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

    primary_score = hybrid_quality_score if state.target_metric is None else oof_score
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

    _record_improvement(
        state,
        store,
        deterministic_improvement=deterministic_improvement,
        primary_score=primary_score,
        best_primary=best_primary,
        oof_score=oof_score,
    )

    if (
        deterministic_improvement
        and submission_file
        and validation.get("submit", True)
        and (
            platform == "none" or state.submission_count < state.max_submissions_per_day
        )
        and auto_submit
    ):
        _handle_submission(
            state,
            store,
            platform,
            submission_file,
            primary_score=primary_score,
            oof_score=oof_score,
            submit=submit,
            score_submission_artifact=score_submission_artifact,
            update_best_submission_score=update_best_submission_score,
        )
    elif (
        deterministic_improvement
        and submission_file
        and not validation.get("submit", True)
    ):
        logger.info(
            "Validation agent requested no submission (submit=false) "
            "despite improvement; respecting agent decision"
        )

    try:
        if not consume_agent_call("summarizer"):
            return True
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

    if not check_budget():
        return True

    state.consecutive_errors = 0
    state.iteration += 1

    _check_plateau_and_set_phase(state, store, validation)
    return False


# ── Private helpers ───────────────────────────────────────────────────────────


async def _run_validation_agent(
    state: CompetitionState,
    store: StateStore,
    project_dir: str,
    platform: str,
    latest: dict,
    *,
    oof_score,
    quality_score,
    submission_file: str,
    run_validation_agent: Callable,
    consume_agent_call: Callable[[str], bool],
) -> dict | None:
    """Call the validation agent; return its dict or None if budget fired."""
    if not submission_file:
        logger.warning(
            "No submission file produced — skipping format check, "
            "running deterministic improvement check only"
        )
        return {
            "is_improvement": None,
            "submit": False,
            "reasoning": "No submission file produced",
        }

    if not consume_agent_call("validation"):
        return None

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
    return validation


def _record_improvement(
    state: CompetitionState,
    store: StateStore,
    *,
    deterministic_improvement: bool,
    primary_score,
    best_primary,
    oof_score,
) -> None:
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


def _handle_submission(
    state: CompetitionState,
    store: StateStore,
    platform: str,
    submission_file: str,
    *,
    primary_score,
    oof_score,
    submit: Callable,
    score_submission_artifact: Callable,
    update_best_submission_score: Callable,
) -> None:
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
            state.lb_scores.append(
                {
                    "score": scored_lb,
                    "timestamp": _dt.now(_tz.utc).isoformat(),
                    "public_lb": True,
                }
            )
            update_best_submission_score(state=state, new_score=scored_lb)
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
                + (f" lb={scored_lb:.6f}" if scored_lb is not None else "")
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


def _check_plateau_and_set_phase(
    state: CompetitionState,
    store: StateStore,
    validation: dict,
) -> None:
    _plateau_key = "quality_score" if state.target_metric is None else "oof_score"
    _scored = [
        e[_plateau_key] for e in state.experiments if e.get(_plateau_key) is not None
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
            _why.append(f"next_directions={validation.get('next_directions')}")
        logger.info(f"Agent requested stop=True but continuing — {', '.join(_why)}.")

    state.phase = "done" if should_stop else "planning"
