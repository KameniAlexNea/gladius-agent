"""Planning phase: runs the planner agent and persists the resulting plan."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime as _dt
from datetime import timezone as _tz
from pathlib import Path

from gladius.db.store import StateStore
from gladius.state import CompetitionState

logger = logging.getLogger(__name__)


async def run_planning_phase(
    state: CompetitionState,
    store: StateStore,
    data_dir: str,
    project_dir: str,
    platform: str,
    n_parallel: int,
    *,
    run_planner: Callable,
    consume_agent_call: Callable[[str], bool],
    check_budget: Callable[[], bool],
) -> bool:
    """Run the planning phase.

    Returns ``True`` if the outer loop should ``continue`` (budget exceeded or
    guardrail fired), ``False`` when the phase completed normally.
    """
    if not consume_agent_call("planner"):
        return True

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
    if not check_budget():
        return True
    return False
