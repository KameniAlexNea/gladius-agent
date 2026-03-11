"""Planning phase: no-op — the solver agent handles planning internally."""

from __future__ import annotations

from collections.abc import Callable

from gladius.db.store import StateStore
from gladius.state import CompetitionState


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
    """Immediately advance to implementing.

    The solver agent now handles its own internal planning via TodoWrite.
    No planning agent is spawned; this phase is a no-op transition.

    Returns False (outer loop should NOT ``continue``).
    """
    state.phase = "implementing"
    store.save(state)
    return False
