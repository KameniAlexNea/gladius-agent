"""
Competition state data model.

CompetitionState is the single source of truth passed between the orchestrator
and all agents.  Persistence is handled by StateStore (gladius.db.store).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Re-export so existing imports (`from gladius.state import StateStore`) keep working.
from gladius.db.store import StateStore  # noqa: F401

__all__ = ["CompetitionState", "StateStore"]


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
    last_submission_date: Optional[str] = None  # ISO date (YYYY-MM-DD)

    # Experiment registry — each entry: {iteration, oof_score, submission_file, notes, approach, solution_files}
    experiments: list = field(default_factory=list)

    # Failed run summaries — each entry: {iteration, status, error, approach}
    failed_runs: list = field(default_factory=list)

    # Error tracking
    consecutive_errors: int = 0
    error_log: list = field(default_factory=list)  # {iteration, phase, error}
    last_stop_reason: Optional[str] = None

    # Minimum score the OOF must reach before a submission is built.
    # Set via README.md frontmatter: submission_threshold: 0.85
    # None = no gate (agent may submit any result)
    submission_threshold: float | None = None

    # Leaderboard score tracking — each entry: {score, timestamp, public_lb}
    lb_scores: list = field(default_factory=list)
