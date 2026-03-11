"""Backward-compat shim — delegates to run_solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gladius.agents.solver import OUTPUT_SCHEMA as SOLVER_OUTPUT_SCHEMA
from gladius.agents.solver import run_solver

if TYPE_CHECKING:
    from gladius.state import CompetitionState

# Backward-compatible alias so existing imports/tests keep working.
OUTPUT_SCHEMA = SOLVER_OUTPUT_SCHEMA
IMPLEMENTER_OUTPUT_SCHEMA = SOLVER_OUTPUT_SCHEMA


async def run_implementer(
    plan: dict,
    state: "CompetitionState",
    project_dir: str,
) -> dict:
    """Backward-compat wrapper. Plan context comes from CLAUDE.md, not plan dict."""
    return await run_solver(state, project_dir)
