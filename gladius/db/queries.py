"""SQL DML: parameterised query strings for StateStore operations."""

# ---------------------------------------------------------------------------
# Save — competition (written once)
# ---------------------------------------------------------------------------

INSERT_COMPETITION = """
    INSERT OR IGNORE INTO competition
        (competition_id, data_dir, output_dir, target_metric,
         metric_direction, max_iterations, max_submissions_per_day,
         submission_threshold)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

# ---------------------------------------------------------------------------
# Save — current mutable state (single row, id always = 1)
# ---------------------------------------------------------------------------

UPSERT_CURRENT_STATE = """
    INSERT OR REPLACE INTO current_state
        (id, iteration, phase, best_oof_score, best_submission_score,
         best_quality_score, best_submission_path, submission_count,
         consecutive_errors, last_submission_date, last_stop_reason)
    VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# ---------------------------------------------------------------------------
# Save — list tables
# ---------------------------------------------------------------------------

INSERT_EXPERIMENT = """
    INSERT OR IGNORE INTO experiments
        (iteration, oof_score, quality_score, submission_file,
         notes, approach, solution_files)
    VALUES (?, ?, ?, ?, ?, ?, ?)
"""

INSERT_FAILED_RUN = """
    INSERT OR IGNORE INTO failed_runs (iteration, status, error, approach)
    VALUES (?, ?, ?, ?)
"""

INSERT_ERROR_LOG = """
    INSERT OR IGNORE INTO error_log (iteration, phase, error)
    VALUES (?, ?, ?)
"""

INSERT_LB_SCORE = """
    INSERT INTO lb_scores (score, timestamp, public_lb)
    VALUES (?, ?, ?)
"""

INSERT_STATE_HISTORY = """
    INSERT INTO state_history
        (iteration, phase, best_oof_score, best_submission_score,
         best_quality_score, best_submission_path, submission_count,
         consecutive_errors, experiments_count, failed_runs_count,
         stop_reason)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

SELECT_COMPETITION = "SELECT * FROM competition LIMIT 1"
SELECT_CURRENT_STATE = "SELECT * FROM current_state WHERE id = 1"
SELECT_EXPERIMENTS = "SELECT * FROM experiments ORDER BY id"
SELECT_FAILED_RUNS = "SELECT * FROM failed_runs ORDER BY id"
SELECT_ERROR_LOG = "SELECT * FROM error_log ORDER BY id"
SELECT_LB_SCORES = "SELECT * FROM lb_scores ORDER BY id"

# ---------------------------------------------------------------------------
# Agent run stats
# ---------------------------------------------------------------------------

INSERT_AGENT_RUN = """
    INSERT INTO agent_runs
        (iteration, phase, agent_name, started_at, duration_ms,
         num_turns, cost_usd, session_id, is_error, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# ---------------------------------------------------------------------------
# Code snapshots
# ---------------------------------------------------------------------------

INSERT_CODE_SNAPSHOT = """
    INSERT OR IGNORE INTO code_snapshots
        (iteration, file_path, content_hash, size_bytes)
    VALUES (?, ?, ?, ?)
"""

# ---------------------------------------------------------------------------
# Plan recording
# ---------------------------------------------------------------------------

INSERT_PLAN = """
    INSERT OR REPLACE INTO plans
        (iteration, approach_summary, plan_text, session_id)
    VALUES (?, ?, ?, ?)
"""

# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------

INSERT_EVENT = """
    INSERT INTO event_log (iteration, phase, event, detail)
    VALUES (?, ?, ?, ?)
"""
