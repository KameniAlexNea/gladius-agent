"""
SQLite-backed persistence for CompetitionState.

Stub — full implementation pending (Phase: db).
The orchestrator imports this module; StateStore raises NotImplementedError
at construction time until the full implementation is added.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.state import CompetitionState


class StateStore:
    """
    SQLite-backed persistence for CompetitionState.

    Schema (11 tables):
      competition    — static settings, one row
      current_state  — mutable scalars + team_session_ids + current_plan, one row
      experiments    — one row per experiment
      failed_runs    — one row per failed run
      error_log      — one row per error
      lb_scores      — one row per LB entry
      state_history  — one row per save (audit log)
      agent_runs     — per-agent call stats
      code_snapshots — file hashes per iteration
      plans          — full plan text per iteration
      event_log      — chronological iteration-transition log
    """

    def __init__(self, db_path: str = ".gladius/state.db") -> None:
        raise NotImplementedError(
            "src.db.store.StateStore is not yet implemented. "
            "Port gladius/db/store.py → src/db/store.py to enable persistence."
        )

    def save(self, state: "CompetitionState") -> None:
        """Persist the full CompetitionState to SQLite."""
        ...

    def load(self) -> "CompetitionState | None":
        """Load the most recent CompetitionState, or None if no state exists."""
        ...

    def close(self) -> None:
        """Close the database connection."""
        ...

    def record_event(
        self,
        *,
        iteration: int,
        topology: str,
        event: str,
        detail: str = "",
    ) -> None:
        """Append a row to event_log."""
        ...

    def record_plan(
        self,
        *,
        iteration: int,
        approach_summary: str,
        plan_text: str,
        session_id: str | None,
    ) -> None:
        """Append a row to the plans table."""
        ...

    def record_code_snapshots(
        self,
        iteration: int,
        solution_files: list[str],
        project_dir: str,
    ) -> None:
        """Hash and record source files in the code_snapshots table."""
        ...

    def record_agent_run(self, **kwargs) -> None:
        """Record per-agent call statistics."""
        ...
