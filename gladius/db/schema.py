"""SQL DDL: table definitions, migration columns, and unique indexes."""

# ---------------------------------------------------------------------------
# Table definitions
# ---------------------------------------------------------------------------

CREATE_TABLES = """
    CREATE TABLE IF NOT EXISTS competition (
        competition_id      TEXT PRIMARY KEY,
        data_dir            TEXT NOT NULL,
        output_dir          TEXT NOT NULL,
        target_metric       TEXT,           -- NULL for open-ended tasks
        metric_direction    TEXT,           -- NULL for open-ended tasks
        max_iterations      INTEGER NOT NULL,
        max_submissions_per_day INTEGER NOT NULL,
        submission_threshold REAL            -- minimum OOF before submitting; NULL = no gate
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

# ---------------------------------------------------------------------------
# Schema migrations
# ---------------------------------------------------------------------------
# Each tuple: (table, column, column_def).  The migration runner SELECTs the
# column first; if that raises OperationalError the column is missing and
# ALTER TABLE is executed.  Safe to run on every startup.

MIGRATION_COLUMNS: list[tuple[str, str, str]] = [
    ("current_state", "last_submission_date", "TEXT"),
    ("current_state", "best_quality_score", "REAL"),
    ("experiments", "quality_score", "REAL"),
    ("state_history", "best_quality_score", "REAL"),
    ("current_state", "last_stop_reason", "TEXT"),
    ("state_history", "stop_reason", "TEXT"),
    ("competition", "submission_threshold", "REAL"),
]

# ---------------------------------------------------------------------------
# Unique indexes on list tables
# ---------------------------------------------------------------------------
# Wrapped in try/except on creation — creation fails gracefully if existing
# rows already violate uniqueness (stale DB from a previous bug).
# Format: (index_name, table_name, column_expression)

UNIQUE_INDEXES: list[tuple[str, str, str]] = [
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
