"""SQL DDL: table definitions for the topology-driven schema (clean break)."""

# ---------------------------------------------------------------------------
# Table definitions
# ---------------------------------------------------------------------------

CREATE_TABLES = """
    CREATE TABLE IF NOT EXISTS competition (
        competition_id          TEXT PRIMARY KEY,
        data_dir                TEXT NOT NULL,
        output_dir              TEXT NOT NULL,
        target_metric           TEXT,           -- NULL for open-ended tasks
        metric_direction        TEXT,           -- NULL for open-ended tasks
        topology                TEXT NOT NULL DEFAULT 'functional',
        max_iterations          INTEGER NOT NULL,
        max_submissions_per_day INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS current_state (
        id                      INTEGER PRIMARY KEY CHECK (id = 1),
        iteration               INTEGER NOT NULL,
        done                    INTEGER NOT NULL DEFAULT 0,   -- boolean
        best_oof_score          REAL,
        best_submission_score   REAL,
        best_quality_score      REAL,           -- NULL for ML tasks
        best_submission_path    TEXT,
        submission_count        INTEGER NOT NULL,
        consecutive_errors      INTEGER NOT NULL,
        team_session_ids        TEXT,           -- JSON: {role_name: session_id}
        current_plan            TEXT,           -- JSON: nested plan dict
        last_submission_date    TEXT,
        last_stop_reason        TEXT
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
        done                    INTEGER NOT NULL DEFAULT 0,
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
        topology        TEXT NOT NULL DEFAULT 'unknown',
        agent_name      TEXT NOT NULL,
        started_at      TEXT,               -- ISO-8601 UTC timestamp
        duration_ms     INTEGER,            -- wall-clock ms
        num_turns       INTEGER,            -- SDK-reported turns (when available)
        cost_usd        REAL,               -- SDK-reported cost (when available)
        session_id      TEXT,
        is_error        INTEGER NOT NULL DEFAULT 0,
        notes           TEXT
    );

    -- Code file evolution: one row per (iteration, file path).
    CREATE TABLE IF NOT EXISTS code_snapshots (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        iteration       INTEGER NOT NULL,
        file_path       TEXT NOT NULL,
        content_hash    TEXT NOT NULL,
        size_bytes      INTEGER NOT NULL,
        saved_at        TEXT DEFAULT (datetime('now')),
        UNIQUE(iteration, file_path)
    );

    -- Full plan text produced by team-lead each iteration.
    CREATE TABLE IF NOT EXISTS plans (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        iteration       INTEGER NOT NULL,
        approach_summary TEXT,
        plan_text       TEXT NOT NULL,
        session_id      TEXT,
        saved_at        TEXT DEFAULT (datetime('now')),
        UNIQUE(iteration)
    );

    CREATE TABLE IF NOT EXISTS event_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        iteration   INTEGER NOT NULL,
        topology    TEXT,
        event       TEXT NOT NULL,
        detail      TEXT,
        _ts         TEXT DEFAULT (datetime('now'))
    );
"""

# No migration columns — clean break schema.
MIGRATION_COLUMNS: list = []

UNIQUE_INDEXES: list[tuple[str, str, str]] = [
    ("uniq_experiments", "experiments", "iteration, submission_file"),
    ("uniq_failed_runs", "failed_runs", "iteration, error"),
    ("uniq_error_log", "error_log", "iteration, error"),
    ("uniq_plans", "plans", "iteration"),
]

