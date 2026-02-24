from typing import TypedDict, Optional
from dataclasses import dataclass, field
from enum import Enum


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    KILLED = "killed"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    HELD = "held"
    SCORE_TIMEOUT = "score_timeout"
    COMPLETE = "complete"


@dataclass
class CompetitionConfig:
    name: str
    metric: str
    target: str
    deadline: str
    days_remaining: float
    submission_limit: int = 5


@dataclass
class ExperimentSpec:
    parent_version: str
    changes: list
    estimated_runtime_multiplier: float
    rationale: str


@dataclass
class DirectiveJSON:
    directive_type: str  # tune_existing | new_features | new_model_type | ensemble | seed_average
    target_model: str    # catboost | lgbm | xgboost | nn | blend
    rationale: str
    exploration_flag: bool
    priority: int        # 1-5


class GraphState(TypedDict):
    # Competition
    competition: dict  # CompetitionConfig as dict

    # Experiment lifecycle
    current_experiment: Optional[dict]  # ExperimentSpec as dict
    experiment_status: str  # ExperimentStatus value
    running_pid: Optional[int]
    run_id: Optional[str]

    # Scores
    oof_score: Optional[float]
    lb_score: Optional[float]
    gap_history: list  # OOF-LB gaps for last N scored submissions

    # Budget
    submissions_today: int
    last_submission_time: Optional[float]

    # Strategy context
    directive: Optional[dict]  # DirectiveJSON as dict
    exploration_flag: bool
    consecutive_same_directive: int
    session_summary: Optional[str]

    # Code
    generated_script_path: Optional[str]
    reviewer_feedback: Optional[str]
    code_retry_count: int

    # Routing
    next_node: str
    error_message: Optional[str]

    # Error handling
    node_retry_counts: dict
    next_node_before_error: Optional[str]
