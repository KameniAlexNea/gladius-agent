"""Implementation phase: runs the solver agent and records the result."""

from __future__ import annotations

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
    run_solver: Callable,
    consume_agent_call: Callable[[str], bool],
    consume_agent_calls: Callable[[str, int], bool],
    check_budget: Callable[[], bool],
) -> bool:
    """Run the solver agent for one iteration.

    Returns ``True`` if the outer loop should ``continue`` (budget exceeded or
    guardrail fired), ``False`` when the phase completed normally.
    """
    _reset_iteration_experiment_state(project_dir, state.iteration)

    if not consume_agent_call("solver"):
        return True

    _t0 = time.perf_counter()
    _started_at = _dt.now(_tz.utc).isoformat()
    result = await run_solver(state, project_dir)
    _dur_ms = int((time.perf_counter() - _t0) * 1000)

    store.record_agent_run(
        iteration=state.iteration,
        phase="implementing",
        agent_name="solver",
        started_at=_started_at,
        duration_ms=_dur_ms,
        is_error=result.get("status") != "success",
        notes=(result.get("status") if result.get("status") != "success" else None),
    )

    if result["status"] != "success":
        logger.warning(f"Solver {result['status']}: {result.get('error_message', '')}")
        state.failed_runs.append(
            {
                "iteration": state.iteration,
                "status": result["status"],
                "error": result.get("error_message", ""),
                "approach": "",
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
        return False

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

    if state.target_metric:
        _oof = result.get("oof_score")
        logger.info(
            f"Solver done — OOF {state.target_metric}: "
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
        logger.info(f"Solver done — quality: {_quality}/100")
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
