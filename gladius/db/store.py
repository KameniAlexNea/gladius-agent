"""
SQLite-backed persistence for CompetitionState.

Currently a no-op in-memory stub — state is not persisted between runs.
Full SQLite implementation pending.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gladius import STATE_DB_RELATIVE_PATH

if TYPE_CHECKING:
    from gladius.state import CompetitionState


class StateStore:
    """No-op store — all methods are safe no-ops until the DB is implemented."""

    def __init__(self, db_path: str = STATE_DB_RELATIVE_PATH) -> None:
        self._state: "CompetitionState | None" = None

    def save(self, state: "CompetitionState") -> None:
        self._state = state

    def load(self) -> "CompetitionState | None":
        return self._state

    def close(self) -> None:
        pass

    def record_event(
        self, *, iteration: int, topology: str, event: str, detail: str = ""
    ) -> None:
        pass

    def record_plan(
        self,
        *,
        iteration: int,
        approach_summary: str,
        plan_text: str,
        session_id: str | None,
    ) -> None:
        pass

    def record_code_snapshots(
        self, iteration: int, solution_files: list[str], project_dir: str
    ) -> None:
        pass

    def record_agent_run(self, **kwargs) -> None:
        pass
