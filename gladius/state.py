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
    target_metric: str  # "auc_roc" | "rmse" | "logloss" | etc.
    metric_direction: str  # "maximize" | "minimize"

    # Loop control
    iteration: int = 0
    max_iterations: int = 20
    phase: str = "planning"  # planning | implementing | validation | done

    # Best known performance
    best_oof_score: float = -1.0
    best_submission_score: float = -1.0
    best_submission_path: Optional[str] = None
    submission_count: int = 0
    max_submissions_per_day: int = 5

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
                target_metric       TEXT NOT NULL,
                metric_direction    TEXT NOT NULL,
                max_iterations      INTEGER NOT NULL,
                max_submissions_per_day INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS current_state (
                id                  INTEGER PRIMARY KEY CHECK (id = 1),
                iteration           INTEGER NOT NULL,
                phase               TEXT NOT NULL,
                best_oof_score      REAL NOT NULL,
                best_submission_score REAL NOT NULL,
                best_submission_path TEXT,
                submission_count    INTEGER NOT NULL,
                consecutive_errors  INTEGER NOT NULL,
                planner_session_id  TEXT,
                current_plan        TEXT        -- JSON: nested plan dict
            );

            CREATE TABLE IF NOT EXISTS experiments (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration       INTEGER NOT NULL,
                oof_score       REAL,
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
                best_oof_score          REAL NOT NULL,
                best_submission_score   REAL NOT NULL,
                best_submission_path    TEXT,
                submission_count        INTEGER NOT NULL,
                consecutive_errors      INTEGER NOT NULL,
                experiments_count       INTEGER NOT NULL,
                failed_runs_count       INTEGER NOT NULL
            );
        """
        )
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
                     best_submission_path, submission_count, consecutive_errors,
                     planner_session_id, current_plan)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    state.iteration,
                    state.phase,
                    state.best_oof_score,
                    state.best_submission_score,
                    state.best_submission_path,
                    state.submission_count,
                    state.consecutive_errors,
                    state.planner_session_id,
                    json.dumps(state.current_plan) if state.current_plan else None,
                ),
            )

            # List tables: clear and re-insert (runs stay small during a competition)
            self.conn.execute("DELETE FROM experiments")
            for e in state.experiments:
                files = ",".join(e.get("solution_files") or [])
                self.conn.execute(
                    """
                    INSERT INTO experiments
                        (iteration, oof_score, submission_file, notes, approach, solution_files)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        e.get("iteration"),
                        e.get("oof_score"),
                        e.get("submission_file"),
                        e.get("notes"),
                        e.get("approach"),
                        files,
                    ),
                )

            self.conn.execute("DELETE FROM failed_runs")
            for f in state.failed_runs:
                self.conn.execute(
                    """
                    INSERT INTO failed_runs (iteration, status, error, approach)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        f.get("iteration"),
                        f.get("status"),
                        f.get("error"),
                        f.get("approach"),
                    ),
                )

            self.conn.execute("DELETE FROM error_log")
            for e in state.error_log:
                self.conn.execute(
                    """
                    INSERT INTO error_log (iteration, phase, error)
                    VALUES (?, ?, ?)
                """,
                    (e.get("iteration"), e.get("phase"), e.get("error")),
                )

            self.conn.execute("DELETE FROM lb_scores")
            for lb in state.lb_scores:
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
                     best_submission_path, submission_count, consecutive_errors,
                     experiments_count, failed_runs_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    state.iteration,
                    state.phase,
                    state.best_oof_score,
                    state.best_submission_score,
                    state.best_submission_path,
                    state.submission_count,
                    state.consecutive_errors,
                    len(state.experiments),
                    len(state.failed_runs),
                ),
            )

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
            target_metric=comp["target_metric"],
            metric_direction=comp["metric_direction"],
            max_iterations=comp["max_iterations"],
            max_submissions_per_day=comp["max_submissions_per_day"],
            iteration=curr["iteration"],
            phase=curr["phase"],
            best_oof_score=curr["best_oof_score"],
            best_submission_score=curr["best_submission_score"],
            best_submission_path=curr["best_submission_path"],
            submission_count=curr["submission_count"],
            consecutive_errors=curr["consecutive_errors"],
            planner_session_id=curr["planner_session_id"],
            current_plan=(
                json.loads(curr["current_plan"]) if curr["current_plan"] else None
            ),
            experiments=experiments,
            failed_runs=failed_runs,
            error_log=error_log,
            lb_scores=lb_scores,
        )

    def close(self) -> None:
        self.conn.close()
