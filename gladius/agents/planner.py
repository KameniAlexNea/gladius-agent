"""
Planner — no-op shim.

Planning is now handled inside the solver agent via TodoWrite.
This module is kept for backward compatibility (orchestrator + tests import it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gladius.state import CompetitionState


async def run_planner(
    state: "CompetitionState",
    data_dir: str,
    project_dir: str,
    platform: str = "kaggle",
    n_parallel: int = 1,
) -> tuple[dict, str | None]:
    """No-op — the solver agent now handles planning internally via TodoWrite."""
    plan: dict = {
        "approach_summary": "",
        "plan_text": "",
        "plan": [],
        "plans": [],
    }
    return plan, None
