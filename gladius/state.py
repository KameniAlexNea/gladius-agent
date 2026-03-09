"""
Competition state and persistence.
Replaces: GraphState TypedDict + LangGraph SqliteSaver
"""

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CompetitionState:
    # Competition context
    competition_id: str
    data_dir: str
    output_dir: str
    # None for open-ended / app-building tasks where the agent self-assesses quality.
    target_metric: str | None = None  # "auc_roc" | "rmse" | "logloss" | None
    metric_direction: str | None = None  # "maximize" | "minimize" | None

    # Loop control
    iteration: int = 0
    max_iterations: int = 20
    phase: str = "planning"  # planning | implementing | validation | done

    # Best known performance (None = no result yet)
    best_oof_score: float | None = None
    best_submission_score: float | None = None
    # Quality score for open-ended tasks (0-100, agent self-assessed; None = no result yet)
    best_quality_score: float | None = None
    best_submission_path: Optional[str] = None
    submission_count: int = 0
    max_submissions_per_day: int = 5
    last_submission_date: Optional[str] = (
        None  # ISO date (YYYY-MM-DD); None = never submitted
    )

    # Current plan from planner agent — kept as dict; stored as JSON in DB
    # because it's a nested structure (list of step dicts) with no clean columns
    current_plan: Optional[dict] = None

    # Experiment registry
    # Each entry: {iteration, oof_score, submission_file, notes, approach, solution_files}
    experiments: list = field(default_factory=list)

    # Failed run summaries
    # Each entry: {iteration, status, error, approach}
    failed_runs: list = field(default_factory=list)

    # Session ID for the planner (resumed every iteration)
    planner_session_id: Optional[str] = None

    # Error tracking
    consecutive_errors: int = 0
    error_log: list = field(default_factory=list)  # {iteration, phase, error}
    last_stop_reason: Optional[str] = None

    # Leaderboard score tracking
    lb_scores: list = field(default_factory=list)  # {score, timestamp, public_lb}


class StateStore:
    """
    SQLite-backed persistence for CompetitionState.

    Schema (fully normalised — no JSON blobs except current_plan):
      competition    — static settings, one row
      current_state  — mutable scalars + current_plan, one row (upserted)
      experiments    — one row per experiment
      failed_runs    — one row per failed run
      error_log      — one row per error
      lb_scores      — one row per LB entry
      state_history  — one row per save, scalar columns only (audit log)
    """

    def __init__(self, db_path: str = ".gladius/state.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS competition (
                competition_id      TEXT PRIMARY KEY,
                data_dir            TEXT NOT NULL,
                output_dir          TEXT NOT NULL,
                target_metric       TEXT,           -- NULL for open-ended tasks
                metric_direction    TEXT,           -- NULL for open-ended tasks
                max_iterations      INTEGER NOT NULL,
                max_submissions_per_day INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS current_state (
                id                  INTEGER PRIMARY KEY CHECK (id = 1),
                iteration           INTEGER NOT NULL,
                phase               TEXT NOT NULL,
                best_oof_score      REAL,
                best_submission_score REAL,
                best_quality_score  REAL,       -- NULL for ML tasks
                best_submission_path TEXT,
                submission_count    INTEGER NOT NULL,
                consecutive_errors  INTEGER NOT NULL,
                planner_session_id  TEXT,
                current_plan        TEXT,       -- JSON: nested plan dict
                last_submission_date TEXT,
                last_stop_reason    TEXT
            );

            CREATE TABLE IF NOT EXISTS experiments (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration       INTEGER NOT NULL,
                oof_score       REAL,
                quality_score   REAL,           -- 0-100 for open-ended tasks
                submission_file TEXT,
                notes           TEXT,
                approach        TEXT,
                solution_files  TEXT            -- comma-separated paths
            );

            CREATE TABLE IF NOT EXISTS failed_runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration   INTEGER NOT NULL,
                status      TEXT,
                error       TEXT,
                approach    TEXT
            );

            CREATE TABLE IF NOT EXISTS error_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration   INTEGER,
                phase       TEXT,
                error       TEXT
            );

            CREATE TABLE IF NOT EXISTS lb_scores (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                score       REAL NOT NULL,
                timestamp   TEXT,
                public_lb   INTEGER NOT NULL DEFAULT 1  -- boolean
            );

            CREATE TABLE IF NOT EXISTS state_history (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                saved_at                TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                iteration               INTEGER NOT NULL,
                phase                   TEXT NOT NULL,
                best_oof_score          REAL,
                best_submission_score   REAL,
                best_quality_score      REAL,
                best_submission_path    TEXT,
                submission_count        INTEGER NOT NULL,
                consecutive_errors      INTEGER NOT NULL,
                experiments_count       INTEGER NOT NULL,
                failed_runs_count       INTEGER NOT NULL,
                stop_reason             TEXT
            );

            -- Per-agent-call stats: timing, turns, cost, errors.
            CREATE TABLE IF NOT EXISTS agent_runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration       INTEGER NOT NULL,
                phase           TEXT NOT NULL,
                agent_name      TEXT NOT NULL,
                started_at      TEXT,               -- ISO-8601 UTC timestamp
                duration_ms     INTEGER,            -- wall-clock ms in orchestrator
                num_turns       INTEGER,            -- SDK-reported turns (when available)
                cost_usd        REAL,               -- SDK-reported cost (when available)
                session_id      TEXT,
                is_error        INTEGER NOT NULL DEFAULT 0,
                notes           TEXT                -- e.g. "resumed", "json_fallback"
            );

            -- Code file evolution: one row per (iteration, file path).
            -- content_hash lets you detect changes across iterations cheaply.
            CREATE TABLE IF NOT EXISTS code_snapshots (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration       INTEGER NOT NULL,
                file_path       TEXT NOT NULL,
                content_hash    TEXT NOT NULL,      -- sha256 hex
                size_bytes      INTEGER NOT NULL,
                saved_at        TEXT DEFAULT (datetime('now')),
                UNIQUE(iteration, file_path)
            );

            -- Full plan text produced by the planner each iteration.
            CREATE TABLE IF NOT EXISTS plans (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration       INTEGER NOT NULL,
                approach_summary TEXT,
                plan_text       TEXT NOT NULL,
                session_id      TEXT,
                saved_at        TEXT DEFAULT (datetime('now')),
                UNIQUE(iteration)                   -- one plan per iteration
            );

            -- Chronological event stream: every phase transition + key decisions.
            -- Provides a human-readable audit trail without parsing agent logs.
            CREATE TABLE IF NOT EXISTS event_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              TEXT DEFAULT (datetime('now')),
                iteration       INTEGER NOT NULL,
                phase           TEXT NOT NULL,      -- planning|implementing|validation|done|error
                event           TEXT NOT NULL,      -- short machine-readable tag
                detail          TEXT                -- human-readable explanation
            );
        """
        )
        self.conn.commit()

        # Schema migrations: add columns introduced after the initial release.
        # ALTER TABLE is safe to retry — the SELECT probe catches existing columns.
        for _tbl, _col, _col_def in [
            ("current_state", "last_submission_date", "TEXT"),
            ("current_state", "best_quality_score", "REAL"),
            ("experiments", "quality_score", "REAL"),
            ("state_history", "best_quality_score", "REAL"),
            ("current_state", "last_stop_reason", "TEXT"),
            ("state_history", "stop_reason", "TEXT"),
        ]:
            try:
                self.conn.execute(f"SELECT {_col} FROM {_tbl} LIMIT 1")
            except sqlite3.OperationalError:
                self.conn.execute(f"ALTER TABLE {_tbl} ADD COLUMN {_col} {_col_def}")

        # UNIQUE indexes on list tables so INSERT OR IGNORE is idempotent.
        # Wrapped in try/except — creation fails gracefully if existing rows
        # already violate uniqueness (stale DB from a previous bug).
        _unique_indexes = [
            (
                "uniq_experiments",
                "experiments",
                "iteration, COALESCE(oof_score,-999.0), COALESCE(submission_file,'')",
            ),
            (
                "uniq_failed_runs",
                "failed_runs",
                "iteration, COALESCE(status,''), COALESCE(error,'')",
            ),
            (
                "uniq_error_log",
                "error_log",
                "iteration, COALESCE(phase,''), COALESCE(error,'')",
            ),
        ]
        for idx_name, tbl, cols in _unique_indexes:
            try:
                self.conn.execute(
                    f"CREATE UNIQUE INDEX IF NOT EXISTS {idx_name} ON {tbl}({cols})"
                )
            except sqlite3.OperationalError:
                pass  # existing duplicate rows — fall back to count-based save
        self.conn.commit()

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(self, state: CompetitionState) -> None:
        with self.conn:
            # Static settings (INSERT OR IGNORE — written once)
            self.conn.execute(
                """
                INSERT OR IGNORE INTO competition
                    (competition_id, data_dir, output_dir, target_metric,
                     metric_direction, max_iterations, max_submissions_per_day)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    state.competition_id,
                    state.data_dir,
                    state.output_dir,
                    state.target_metric,
                    state.metric_direction,
                    state.max_iterations,
                    state.max_submissions_per_day,
                ),
            )

            # Mutable scalars (single row, always id=1)
            self.conn.execute(
                """
                INSERT OR REPLACE INTO current_state
                    (id, iteration, phase, best_oof_score, best_submission_score,
                     best_quality_score, best_submission_path, submission_count,
                     consecutive_errors, planner_session_id, current_plan,
                     last_submission_date, last_stop_reason)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    state.iteration,
                    state.phase,
                    state.best_oof_score,
                    state.best_submission_score,
                    state.best_quality_score,
                    state.best_submission_path,
                    state.submission_count,
                    state.consecutive_errors,
                    state.planner_session_id,
                    json.dumps(state.current_plan) if state.current_plan else None,
                    state.last_submission_date,
                    state.last_stop_reason,
                ),
            )

            # List tables: INSERT OR IGNORE so double-saves are harmless.
            # If the UNIQUE index exists (new DBs), scan the full list — the
            # index prevents duplicates.  If the index is absent (old DB with
            # duplicate rows), fall back to the count-based slice.
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
                    """
                    INSERT OR IGNORE INTO experiments
                        (iteration, oof_score, quality_score, submission_file,
                         notes, approach, solution_files)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
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
                    """
                    INSERT OR IGNORE INTO failed_runs (iteration, status, error, approach)
                    VALUES (?, ?, ?, ?)
                """,
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
                    """
                    INSERT OR IGNORE INTO error_log (iteration, phase, error)
                    VALUES (?, ?, ?)
                """,
                    (e.get("iteration"), e.get("phase"), e.get("error")),
                )

            existing_lb = self.conn.execute(
                "SELECT COUNT(*) FROM lb_scores"
            ).fetchone()[0]
            for lb in state.lb_scores[existing_lb:]:
                self.conn.execute(
                    """
                    INSERT INTO lb_scores (score, timestamp, public_lb)
                    VALUES (?, ?, ?)
                """,
                    (
                        lb.get("score"),
                        lb.get("timestamp"),
                        1 if lb.get("public_lb", True) else 0,
                    ),
                )

            # Append to audit log
            self.conn.execute(
                """
                INSERT INTO state_history
                    (iteration, phase, best_oof_score, best_submission_score,
                     best_quality_score, best_submission_path, submission_count,
                     consecutive_errors, experiments_count, failed_runs_count,
                     stop_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    state.iteration,
                    state.phase,
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
        phase: str,
        agent_name: str,
        started_at: str,
        duration_ms: int,
        is_error: bool = False,
        num_turns: int | None = None,
        cost_usd: float | None = None,
        session_id: str | None = None,
        notes: str | None = None,
    ) -> None:
        """Append one row to agent_runs (non-transactional, best-effort)."""
        try:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT INTO agent_runs
                        (iteration, phase, agent_name, started_at, duration_ms,
                         num_turns, cost_usd, session_id, is_error, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        iteration,
                        phase,
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
            pass  # stats are best-effort; never crash the main loop

    # ── Code snapshots ────────────────────────────────────────────────────────

    def record_code_snapshots(
        self,
        iteration: int,
        file_paths: list[str],
        project_dir: str,
    ) -> None:
        """Hash each solution file and insert an (iteration, path, hash) row.

        UNIQUE(iteration, file_path) prevents duplicates on re-save.
        Missing or unreadable files are silently skipped.
        """
        import hashlib
        from pathlib import Path

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
                    self.conn.executemany(
                        """
                        INSERT OR IGNORE INTO code_snapshots
                            (iteration, file_path, content_hash, size_bytes)
                        VALUES (?, ?, ?, ?)
                        """,
                        rows,
                    )
            except Exception:
                pass  # best-effort

    # ── Plan recording ────────────────────────────────────────────────────────

    def record_plan(
        self,
        *,
        iteration: int,
        approach_summary: str,
        plan_text: str,
        session_id: str | None = None,
    ) -> None:
        """Insert or replace the plan for this iteration (one plan per iteration)."""
        try:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO plans
                        (iteration, approach_summary, plan_text, session_id)
                    VALUES (?, ?, ?, ?)
                    """,
                    (iteration, approach_summary, plan_text, session_id),
                )
        except Exception:
            pass  # best-effort

    # ── Event log ─────────────────────────────────────────────────────────────

    def record_event(
        self,
        *,
        iteration: int,
        phase: str,
        event: str,
        detail: str | None = None,
    ) -> None:
        """Append one event row (best-effort, never raises)."""
        try:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT INTO event_log (iteration, phase, event, detail)
                    VALUES (?, ?, ?, ?)
                    """,
                    (iteration, phase, event, detail),
                )
        except Exception:
            pass

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self) -> Optional[CompetitionState]:
        comp = self.conn.execute("SELECT * FROM competition LIMIT 1").fetchone()
        curr = self.conn.execute("SELECT * FROM current_state WHERE id = 1").fetchone()
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
            for row in self.conn.execute("SELECT * FROM experiments ORDER BY id")
        ]

        failed_runs = [
            {
                "iteration": r["iteration"],
                "status": r["status"],
                "error": r["error"],
                "approach": r["approach"],
            }
            for r in self.conn.execute("SELECT * FROM failed_runs ORDER BY id")
        ]

        error_log = [
            {"iteration": r["iteration"], "phase": r["phase"], "error": r["error"]}
            for r in self.conn.execute("SELECT * FROM error_log ORDER BY id")
        ]

        lb_scores = [
            {
                "score": r["score"],
                "timestamp": r["timestamp"],
                "public_lb": bool(r["public_lb"]),
            }
            for r in self.conn.execute("SELECT * FROM lb_scores ORDER BY id")
        ]

        return CompetitionState(
            competition_id=comp["competition_id"],
            data_dir=comp["data_dir"],
            output_dir=comp["output_dir"],
            target_metric=comp["target_metric"],  # may be None for open tasks
            metric_direction=comp["metric_direction"],  # may be None for open tasks
            max_iterations=comp["max_iterations"],
            max_submissions_per_day=comp["max_submissions_per_day"],
            iteration=curr["iteration"],
            phase=curr["phase"],
            best_oof_score=curr["best_oof_score"],
            best_submission_score=curr["best_submission_score"],
            best_quality_score=curr["best_quality_score"],
            best_submission_path=curr["best_submission_path"],
            submission_count=curr["submission_count"],
            consecutive_errors=curr["consecutive_errors"],
            planner_session_id=curr["planner_session_id"],
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

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None  # guard against double-close

    def __del__(self) -> None:
        """Safety net: release the connection on garbage-collection / abnormal exit."""
        self.close()
