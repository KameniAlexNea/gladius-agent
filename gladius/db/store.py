"""
SQLite-backed persistence for CompetitionState.

State is persisted in a lightweight local database under .gladius/state.db.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from gladius.config import LAYOUT as _LAYOUT

if TYPE_CHECKING:
    from gladius.state import CompetitionState


class StateStore:
    """SQLite-backed state and execution trace store."""

    def __init__(self, db_path: str = _LAYOUT.state_db_relative_path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._closed = False
        self._init_schema()

    def _is_open(self) -> bool:
        return not self._closed

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS kv_state (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                iteration INTEGER,
                topology TEXT,
                event TEXT NOT NULL,
                detail_json TEXT
            );

            CREATE TABLE IF NOT EXISTS agent_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def save(self, state: "CompetitionState") -> None:
        if not self._is_open():
            return
        payload = asdict(state) if is_dataclass(state) else state
        self._conn.execute(
            """
            INSERT INTO kv_state(key, value_json, updated_at)
            VALUES(?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value_json=excluded.value_json,
                updated_at=excluded.updated_at
            """,
            ("competition_state", json.dumps(payload), self._utc_now()),
        )
        self._conn.commit()

    def load(self) -> "CompetitionState | None":
        if not self._is_open():
            return None
        row = self._conn.execute(
            "SELECT value_json FROM kv_state WHERE key=?", ("competition_state",)
        ).fetchone()
        if row is None:
            return None
        data = json.loads(row["value_json"])
        if not isinstance(data, dict):
            return data

        required = {"competition_id", "data_dir", "output_dir"}
        if required.issubset(set(data)):
            from gladius.state import CompetitionState

            return CompetitionState(**data)
        return data

    def close(self) -> None:
        if not self._is_open():
            return
        self._closed = True
        self._conn.close()

    def record_event(
        self, *, iteration: int, topology: str, event: str, detail: str = ""
    ) -> None:
        if not self._is_open():
            return
        self._conn.execute(
            """
            INSERT INTO events(created_at, iteration, topology, event, detail_json)
            VALUES(?, ?, ?, ?, ?)
            """,
            (
                self._utc_now(),
                iteration,
                topology,
                event,
                json.dumps({"detail": detail}),
            ),
        )
        self._conn.commit()

    def record_agent_run(self, **kwargs) -> None:
        if not self._is_open():
            return
        self._conn.execute(
            "INSERT INTO agent_runs(created_at, payload_json) VALUES(?, ?)",
            (self._utc_now(), json.dumps(kwargs, ensure_ascii=True)),
        )
        self._conn.commit()
