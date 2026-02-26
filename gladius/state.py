"""
Competition state and persistence.
Replaces: GraphState TypedDict + LangGraph SqliteSaver
"""
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CompetitionState:
    # Competition context
    competition_id: str
    data_dir: str
    output_dir: str
    target_metric: str          # "auc_roc" | "rmse" | "logloss" | etc.
    metric_direction: str       # "maximize" | "minimize"

    # Loop control
    iteration: int = 0
    max_iterations: int = 20
    phase: str = "planning"     # planning | implementing | validation | done

    # Best known performance
    best_oof_score: float = -1.0
    best_submission_score: float = -1.0
    best_submission_path: Optional[str] = None
    submission_count: int = 0
    max_submissions_per_day: int = 5

    # Current plan from planner agent
    current_plan: Optional[dict] = None   # {plan: [...], approach_summary: str}

    # Experiment registry — each entry is whatever the implementer reported
    # {oof_score, solution_files, submission_file, notes, iteration}
    experiments: list = field(default_factory=list)

    # Failed run summaries (error_message, iteration, approach_summary)
    failed_runs: list = field(default_factory=list)

    # Session ID for the planner (resumed every iteration)
    planner_session_id: Optional[str] = None

    # Error tracking
    consecutive_errors: int = 0
    error_log: list = field(default_factory=list)

    # LB tracking
    lb_scores: list = field(default_factory=list)  # [{score, timestamp, public_lb}]


class StateStore:
    """SQLite-backed persistence for CompetitionState."""

    def __init__(self, db_path: str = ".gladius/state.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS state_history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration  INTEGER,
                phase      TEXT,
                snapshot   TEXT NOT NULL,
                saved_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def save(self, state: CompetitionState) -> None:
        data = json.dumps(asdict(state))
        self.conn.execute(
            "INSERT OR REPLACE INTO state(key, value) VALUES ('current', ?)", (data,)
        )
        self.conn.execute(
            "INSERT INTO state_history(iteration, phase, snapshot) VALUES (?, ?, ?)",
            (state.iteration, state.phase, data),
        )
        self.conn.commit()

    def load(self) -> Optional[CompetitionState]:
        row = self.conn.execute(
            "SELECT value FROM state WHERE key='current'"
        ).fetchone()
        if row:
            data = json.loads(row[0])
            return CompetitionState(**data)
        return None

    def close(self) -> None:
        self.conn.close()
