"""SQLite-backed persistence for CompetitionState."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from gladius.db.queries import (
    INSERT_AGENT_RUN,
    INSERT_CODE_SNAPSHOT,
    INSERT_COMPETITION,
    INSERT_ERROR_LOG,
    INSERT_EVENT,
    INSERT_EXPERIMENT,
    INSERT_FAILED_RUN,
    INSERT_LB_SCORE,
    INSERT_PLAN,
    INSERT_STATE_HISTORY,
    SELECT_COMPETITION,
    SELECT_CURRENT_STATE,
    SELECT_ERROR_LOG,
    SELECT_EXPERIMENTS,
    SELECT_FAILED_RUNS,
    SELECT_LB_SCORES,
    UPSERT_CURRENT_STATE,
)
from gladius.db.schema import CREATE_TABLES, MIGRATION_COLUMNS, UNIQUE_INDEXES

if TYPE_CHECKING:
    from gladius.state import CompetitionState


class StateStore:
    """
    SQLite-backed persistence for CompetitionState.

    Schema (clean break — no migration columns):
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

    def __init__(self, db_path: str = ".gladius/state.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        self.conn.executescript(CREATE_TABLES)
        self.conn.commit()

        # MIGRATION_COLUMNS is empty for the new schema (clean break).
        for _tbl, _col, _col_def in MIGRATION_COLUMNS:
            try:
                self.conn.execute(f"SELECT {_col} FROM {_tbl} LIMIT 1")
            except sqlite3.OperationalError:
                self.conn.execute(f"ALTER TABLE {_tbl} ADD COLUMN {_col} {_col_def}")

        for idx_name, tbl, cols in UNIQUE_INDEXES:
            try:
                self.conn.execute(
                    f"CREATE UNIQUE INDEX IF NOT EXISTS {idx_name} ON {tbl}({cols})"
                )
            except sqlite3.OperationalError:
                pass
        self.conn.commit()

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(self, state: CompetitionState) -> None:
        with self.conn:
            self.conn.execute(
                INSERT_COMPETITION,
                (
                    state.competition_id,
                    state.data_dir,
                    state.output_dir,
                    state.target_metric,
                    state.metric_direction,
                    state.topology,
                    state.max_iterations,
                    state.max_submissions_per_day,
                ),
            )

            self.conn.execute(
                UPSERT_CURRENT_STATE,
                (
                    state.iteration,
                    1 if state.done else 0,
                    state.best_oof_score,
                    state.best_submission_score,
                    state.best_quality_score,
                    state.best_submission_path,
                    state.submission_count,
                    state.consecutive_errors,
                    json.dumps(state.team_session_ids) if state.team_session_ids else None,
                    json.dumps(state.current_plan) if state.current_plan else None,
                    state.last_submission_date,
                    state.last_stop_reason,
                ),
            )

            def _has_index(name: str) -> bool:
                return bool(
                    self.conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
                        (name,),
                    ).fetchone()
                )

            def _count(tbl: str) -> int:
                return self.conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]

            _exp_offset = 0 if _has_index("uniq_experiments") else _count("experiments")
            for e in state.experiments[_exp_offset:]:
                files = ",".join(e.get("solution_files") or [])
                self.conn.execute(
                    INSERT_EXPERIMENT,
                    (
                        e.get("iteration"),
                        e.get("oof_score"),
                        e.get("quality_score"),
                        e.get("submission_file"),
                        e.get("notes"),
                        e.get("approach"),
                        files,
                    ),
                )

            _fr_offset = 0 if _has_index("uniq_failed_runs") else _count("failed_runs")
            for f in state.failed_runs[_fr_offset:]:
                self.conn.execute(
                    INSERT_FAILED_RUN,
                    (
                        f.get("iteration"),
                        f.get("status"),
                        f.get("error"),
                        f.get("approach"),
                    ),
                )

            _el_offset = 0 if _has_index("uniq_error_log") else _count("error_log")
            for e in state.error_log[_el_offset:]:
                self.conn.execute(
                    INSERT_ERROR_LOG,
                    (e.get("iteration"), e.get("error")),
                )

            existing_lb = self.conn.execute(
                "SELECT COUNT(*) FROM lb_scores"
            ).fetchone()[0]
            for lb in state.lb_scores[existing_lb:]:
                self.conn.execute(
                    INSERT_LB_SCORE,
                    (
                        lb.get("score"),
                        lb.get("timestamp"),
                        1 if lb.get("public_lb", True) else 0,
                    ),
                )

            self.conn.execute(
                INSERT_STATE_HISTORY,
                (
                    state.iteration,
                    1 if state.done else 0,
                    state.best_oof_score,
                    state.best_submission_score,
                    state.best_quality_score,
                    state.best_submission_path,
                    state.submission_count,
                    state.consecutive_errors,
                    len(state.experiments),
                    len(state.failed_runs),
                    state.last_stop_reason,
                ),
            )

    # ── Agent run stats ───────────────────────────────────────────────────────

    def record_agent_run(
        self,
        *,
        iteration: int,
        topology: str = "unknown",
        agent_name: str,
        started_at: str,
        duration_ms: int,
        is_error: bool = False,
        num_turns: int | None = None,
        cost_usd: float | None = None,
        session_id: str | None = None,
        notes: str | None = None,
    ) -> None:
        """Append one row to agent_runs (best-effort, never raises)."""
        try:
            with self.conn:
                self.conn.execute(
                    INSERT_AGENT_RUN,
                    (
                        iteration,
                        topology,
                        agent_name,
                        started_at,
                        duration_ms,
                        num_turns,
                        cost_usd,
                        session_id,
                        1 if is_error else 0,
                        notes,
                    ),
                )
        except Exception:
            pass

    # ── Code snapshots ────────────────────────────────────────────────────────

    def record_code_snapshots(
        self,
        iteration: int,
        file_paths: list[str],
        project_dir: str,
    ) -> None:
        """Hash each solution file and insert an (iteration, path, hash) row."""
        rows: list[tuple] = []
        for fp in file_paths:
            p = Path(fp) if Path(fp).is_absolute() else Path(project_dir) / fp
            try:
                data = p.read_bytes()
                rows.append(
                    (
                        iteration,
                        str(fp),
                        hashlib.sha256(data).hexdigest(),
                        len(data),
                    )
                )
            except OSError:
                pass

        if rows:
            try:
                with self.conn:
                    self.conn.executemany(INSERT_CODE_SNAPSHOT, rows)
            except Exception:
                pass

    # ── Plan recording ────────────────────────────────────────────────────────

    def record_plan(
        self,
        *,
        iteration: int,
        approach_summary: str,
        plan_text: str,
        session_id: str | None = None,
    ) -> None:
        """Insert or replace the plan for this iteration."""
        try:
            with self.conn:
                self.conn.execute(
                    INSERT_PLAN,
                    (iteration, approach_summary, plan_text, session_id),
                )
        except Exception:
            pass

    # ── Event log ─────────────────────────────────────────────────────────────

    def record_event(
        self,
        *,
        iteration: int,
        topology: str = "unknown",
        event: str,
        detail: str | None = None,
    ) -> None:
        """Append one event row (best-effort, never raises)."""
        try:
            with self.conn:
                self.conn.execute(
                    INSERT_EVENT,
                    (iteration, topology, event, detail),
                )
        except Exception:
            pass

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self) -> Optional[CompetitionState]:
        from gladius.state import CompetitionState

        comp = self.conn.execute(SELECT_COMPETITION).fetchone()
        curr = self.conn.execute(SELECT_CURRENT_STATE).fetchone()
        if not comp or not curr:
            return None

        experiments = [
            {
                "iteration": row["iteration"],
                "oof_score": row["oof_score"],
                "quality_score": row["quality_score"],
                "submission_file": row["submission_file"],
                "notes": row["notes"],
                "approach": row["approach"],
                "solution_files": [
                    f for f in (row["solution_files"] or "").split(",") if f
                ],
            }
            for row in self.conn.execute(SELECT_EXPERIMENTS)
        ]

        failed_runs = [
            {
                "iteration": r["iteration"],
                "status": r["status"],
                "error": r["error"],
                "approach": r["approach"],
            }
            for r in self.conn.execute(SELECT_FAILED_RUNS)
        ]

        error_log = [
            {"iteration": r["iteration"], "error": r["error"]}
            for r in self.conn.execute(SELECT_ERROR_LOG)
        ]

        lb_scores = [
            {
                "score": r["score"],
                "timestamp": r["timestamp"],
                "public_lb": bool(r["public_lb"]),
            }
            for r in self.conn.execute(SELECT_LB_SCORES)
        ]

        raw_session_ids = curr["team_session_ids"]
        team_session_ids = (
            json.loads(raw_session_ids) if raw_session_ids else {}
        )

        return CompetitionState(
            competition_id=comp["competition_id"],
            data_dir=comp["data_dir"],
            output_dir=comp["output_dir"],
            target_metric=comp["target_metric"],
            metric_direction=comp["metric_direction"],
            topology=comp["topology"],
            max_iterations=comp["max_iterations"],
            max_submissions_per_day=comp["max_submissions_per_day"],
            iteration=curr["iteration"],
            done=bool(curr["done"]),
            best_oof_score=curr["best_oof_score"],
            best_submission_score=curr["best_submission_score"],
            best_quality_score=curr["best_quality_score"],
            best_submission_path=curr["best_submission_path"],
            submission_count=curr["submission_count"],
            consecutive_errors=curr["consecutive_errors"],
            team_session_ids=team_session_ids,
            current_plan=(
                json.loads(curr["current_plan"]) if curr["current_plan"] else None
            ),
            last_submission_date=curr["last_submission_date"],
            last_stop_reason=curr["last_stop_reason"],
            experiments=experiments,
            failed_runs=failed_runs,
            error_log=error_log,
            lb_scores=lb_scores,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __del__(self) -> None:
        self.close()

