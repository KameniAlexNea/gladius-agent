"""
Competition loop — single Gladius agent architecture.

One agent per iteration. Gladius handles everything: explore, plan, implement,
evaluate, review, submit. No phases, no sub-coordinators.
State lives in memory; all observability comes from gladius.log.
"""

from __future__ import annotations

import time
from datetime import date as _date
from pathlib import Path

from loguru import logger

from gladius.agents.gladius_agent import run_gladius
from gladius.preflight import run_preflight_or_raise
from gladius.state import CompetitionState
from gladius.submission import (
    score_submission_artifact,
    submit,
    update_best_submission_score,
)
from gladius.utils.competition_config import load_competition_config
from gladius.utils.project_setup import setup_project_dir, write_claude_md


def _is_better(
    new_score: float,
    best_score: float | None,
    direction: str | None,
    threshold: float = 1e-4,
) -> bool:
    if best_score is None:
        return True
    if direction is None:
        return new_score > best_score + 2.0
    if direction == "maximize":
        return new_score > best_score + threshold
    return new_score < best_score - threshold


def _reset_experiment_state(project_dir: str, iteration: int) -> None:
    claude_dir = Path(project_dir) / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    state_path = claude_dir / "EXPERIMENT_STATE.json"
    if state_path.exists():
        archive = claude_dir / f"EXPERIMENT_STATE.iter-{max(iteration - 1, 0):02d}.json"
        n = 1
        while archive.exists():
            archive = (
                claude_dir
                / f"EXPERIMENT_STATE.iter-{max(iteration - 1, 0):02d}.{n}.json"
            )
            n += 1
        state_path.replace(archive)
    state_path.write_text("{}\n", encoding="utf-8")


async def run_competition(
    competition_dir: str,
    max_iterations: int = 20,
    auto_submit: bool = True,
    mode: str = "experimental",
    max_iteration_seconds: int | None = None,
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
        if max_failed_runs_total is None:
            max_failed_runs_total = 20

    run_preflight_or_raise(
        competition_dir=competition_dir,
        platform=platform,
        data_dir=data_dir,
        target_metric=target_metric,
        max_iterations=max_iterations,
    )

    logger.info("Initialising new competition state")
    state = CompetitionState(
        competition_id=competition_id,
        data_dir=str(Path(data_dir).resolve()),
        output_dir=str((Path(competition_dir) / ".gladius").resolve()),
        target_metric=target_metric,
        metric_direction=metric_direction,
        max_iterations=max_iterations,
        submission_threshold=cfg.get("submission_threshold"),
    )

    setup_project_dir(state, competition_dir, platform=platform)

    while state.iteration < state.max_iterations and state.phase != "done":
        t_iter_start = time.monotonic()

        if (
            max_failed_runs_total is not None
            and len(state.failed_runs) >= max_failed_runs_total
        ):
            state.last_stop_reason = f"failed run budget exceeded ({len(state.failed_runs)}/{max_failed_runs_total})"
            state.phase = "done"
            logger.warning(state.last_stop_reason)
            break

        _score_str = (
            f"quality={state.best_quality_score}/100"
            if state.target_metric is None
            else f"best={f'{state.best_oof_score:.6f}' if state.best_oof_score is not None else 'none'}"
        )
        logger.info(
            f"[iter {state.iteration:02d}/{state.max_iterations}] {_score_str}  "
            f"experiments={len(state.experiments)}"
        )

        write_claude_md(state, competition_dir)
        _reset_experiment_state(competition_dir, state.iteration)

        try:
            t0 = time.perf_counter()
            result = await run_gladius(state, competition_dir)
            dur_s = time.perf_counter() - t0

            if result["status"] != "success":
                logger.warning(
                    f"Gladius {result['status']}: {result.get('error_message', '')}"
                )
                state.failed_runs.append(
                    {
                        "iteration": state.iteration,
                        "status": result["status"],
                        "error": result.get("error_message", ""),
                    }
                )
                state.consecutive_errors += 1
                state.iteration += 1
                continue

            logger.info(f"Gladius finished in {dur_s:.1f}s")
            state.consecutive_errors = 0
            oof_score = result.get("oof_score")
            quality_score = result.get("quality_score", 0) or 0
            submission_file = result.get("submission_file", "")

            state.experiments.append(
                {
                    "iteration": state.iteration,
                    "oof_score": oof_score,
                    "quality_score": quality_score,
                    "solution_files": result.get("solution_files", []),
                    "submission_file": submission_file,
                    "notes": result.get("notes", ""),
                }
            )

            primary_score = quality_score if target_metric is None else oof_score
            best_primary = (
                state.best_quality_score
                if target_metric is None
                else state.best_oof_score
            )

            if primary_score is not None and _is_better(
                primary_score, best_primary, metric_direction
            ):
                if target_metric:
                    logger.info(f"New best OOF {target_metric}: {oof_score:.6f}")
                    state.best_oof_score = oof_score
                else:
                    logger.info(f"New best quality: {quality_score}/100")
                    state.best_quality_score = quality_score

            # Auto-submit if we have a submission file and it improved
            today = _date.today().isoformat()
            if state.last_submission_date != today:
                state.submission_count = 0
                state.last_submission_date = today

            if (
                auto_submit
                and submission_file
                and primary_score is not None
                and _is_better(primary_score, best_primary, metric_direction)
                and (
                    platform == "none"
                    or state.submission_count < state.max_submissions_per_day
                )
            ):
                try:
                    await submit(
                        submission_file=submission_file,
                        competition_id=competition_id,
                        platform=platform,
                        project_dir=competition_dir,
                    )
                    state.submission_count += 1
                    lb_score = await score_submission_artifact(
                        submission_file=submission_file,
                        competition_id=competition_id,
                        platform=platform,
                    )
                    if lb_score is not None:
                        update_best_submission_score(state, lb_score, metric_direction)
                        logger.info(f"LB score: {lb_score}")
                except Exception as sub_exc:
                    logger.warning(f"Submission failed (non-fatal): {sub_exc}")

            # Runtime budget check
            if max_iteration_seconds is not None:
                elapsed = time.monotonic() - t_iter_start
                if elapsed > max_iteration_seconds:
                    logger.warning(
                        f"Iteration runtime budget exceeded ({elapsed:.1f}s > {max_iteration_seconds}s)"
                    )

        except Exception as exc:
            logger.error(
                f"Unhandled error in iteration {state.iteration}: {exc}", exc_info=True
            )
            state.error_log.append(
                {"phase": "running", "iteration": state.iteration, "error": str(exc)}
            )
            state.consecutive_errors += 1
            if state.consecutive_errors >= 3:
                logger.critical("3 consecutive errors — halting")
                state.last_stop_reason = "consecutive error budget exceeded (3/3)"
                state.phase = "done"

        finally:
            state.iteration += 1

    _final = (
        f"best_oof={f'{state.best_oof_score:.6f}' if state.best_oof_score is not None else 'none'}"
        if state.target_metric
        else f"best_quality={state.best_quality_score}/100"
    )
    logger.info(
        f"Done. iterations={state.iteration}  {_final}  submissions={state.submission_count}"
    )
    return state


# Entry-point shim for pyproject.toml
from gladius.cli import main  # noqa: E402, F401
