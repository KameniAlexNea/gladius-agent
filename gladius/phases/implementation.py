"""Implementation phase: runs one or more implementer agents and picks the best result."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import datetime as _dt
from datetime import timezone as _tz
from pathlib import Path

from loguru import logger

from gladius.db.store import StateStore
from gladius.state import CompetitionState


def _reset_iteration_experiment_state(project_dir: str, iteration: int) -> None:
    """Archive prior EXPERIMENT_STATE.json and initialize a fresh file for this iteration."""
    claude_dir = Path(project_dir) / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    state_path = claude_dir / "EXPERIMENT_STATE.json"

    if state_path.exists():
        archive_base = (
            claude_dir / f"EXPERIMENT_STATE.iter-{max(iteration - 1, 0):02d}.json"
        )
        archive_path = archive_base
        suffix = 1
        while archive_path.exists():
            archive_path = claude_dir / (
                f"EXPERIMENT_STATE.iter-{max(iteration - 1, 0):02d}.{suffix}.json"
            )
            suffix += 1
        state_path.replace(archive_path)
        logger.info(f"Archived prior experiment state to {archive_path}")

    state_path.write_text("{}\n", encoding="utf-8")


async def run_implementation_phase(
    state: CompetitionState,
    store: StateStore,
    project_dir: str,
    n_parallel: int,
    *,
    run_implementer: Callable,
    consume_agent_call: Callable[[str], bool],
    consume_agent_calls: Callable[[str, int], bool],
    check_budget: Callable[[], bool],
) -> bool:
    """Run the implementation phase.

    Returns ``True`` if the outer loop should ``continue``, ``False`` on normal
    completion.
    """
    if state.current_plan is None:
        logger.warning(
            "Resuming in 'implementing' phase with no current_plan "
            "— falling back to planning"
        )
        state.phase = "planning"
        return True

    alt_plans: list[dict] = (
        state.current_plan.get("plans", []) if state.current_plan else []
    )

    _reset_iteration_experiment_state(project_dir, state.iteration)

    if n_parallel > 1 and len(alt_plans) > 1:
        result = await _run_parallel(
            state,
            store,
            project_dir,
            alt_plans,
            n_parallel,
            run_implementer=run_implementer,
            consume_agent_calls=consume_agent_calls,
        )
        if result is None:
            return True  # all failed — phase already reset to planning
    else:
        result = await _run_sequential(
            state,
            store,
            project_dir,
            run_implementer=run_implementer,
            consume_agent_call=consume_agent_call,
        )
        if result is None:
            return True  # implementer failed or budget fired

    # ── Common post-implementation logging ────────────────────────────────────
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
    if not check_budget():
        return True
    return False


async def _run_parallel(
    state: CompetitionState,
    store: StateStore,
    project_dir: str,
    alt_plans: list[dict],
    n_parallel: int,
    *,
    run_implementer: Callable,
    consume_agent_calls: Callable[[str, int], bool],
) -> dict | None:
    """Run multiple implementers in parallel; return the best result or None on total failure."""
    plans_to_run = alt_plans[:n_parallel]
    if not consume_agent_calls("parallel implementers", len(plans_to_run)):
        return None

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
                    "approach": plans_to_run[i].get("approach_summary", ""),
                }
            )
        elif isinstance(r, dict) and r.get("status") == "success":
            successful.append(r)
        else:
            err = r.get("error_message", "") if isinstance(r, dict) else str(r)
            state.failed_runs.append(
                {
                    "iteration": state.iteration,
                    "status": (
                        r.get("status", "error") if isinstance(r, dict) else "error"
                    ),
                    "error": err,
                    "approach": plans_to_run[i].get("approach_summary", ""),
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
        return None

    if state.target_metric is None:
        result = max(successful, key=lambda r: r.get("quality_score", 0) or 0)
    else:
        direction = state.metric_direction
        result = max(
            successful,
            key=lambda r: (
                r["oof_score"] if direction == "maximize" else -r["oof_score"]
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
                state.iteration, r.get("solution_files", []), project_dir
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
    return result


async def _run_sequential(
    state: CompetitionState,
    store: StateStore,
    project_dir: str,
    *,
    run_implementer: Callable,
    consume_agent_call: Callable[[str], bool],
) -> dict | None:
    """Run a single implementer; return the result dict or None on failure."""
    if not consume_agent_call("implementer"):
        return None

    _t0_impl = time.perf_counter()
    _started_impl = _dt.now(_tz.utc).isoformat()
    result = await run_implementer(state.current_plan, state, project_dir)
    store.record_agent_run(
        iteration=state.iteration,
        phase="implementing",
        agent_name="implementer",
        started_at=_started_impl,
        duration_ms=int((time.perf_counter() - _t0_impl) * 1000),
        is_error=result.get("status") != "success",
        notes=(result.get("status") if result.get("status") != "success" else None),
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
        return None

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
    return result
