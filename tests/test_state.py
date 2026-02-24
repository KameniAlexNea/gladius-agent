"""Tests for gladius.state module."""
import pytest
from gladius.state import (
    GraphState,
    ExperimentStatus,
    CompetitionConfig,
    ExperimentSpec,
    DirectiveJSON,
)


def make_state(**kwargs) -> dict:
    defaults = dict(
        competition={"name": "test-comp", "metric": "auc", "target": "label",
                     "deadline": "2026-01-01", "days_remaining": 30, "submission_limit": 5},
        current_experiment=None,
        experiment_status="pending",
        running_pid=None,
        run_id=None,
        oof_score=None,
        lb_score=None,
        gap_history=[],
        submissions_today=0,
        last_submission_time=None,
        directive=None,
        exploration_flag=True,
        consecutive_same_directive=0,
        session_summary=None,
        generated_script_path=None,
        reviewer_feedback=None,
        code_retry_count=0,
        next_node="strategy",
        error_message=None,
        node_retry_counts={},
        next_node_before_error=None,
    )
    defaults.update(kwargs)
    return defaults


def test_graph_state_fields():
    state = make_state()
    assert state["experiment_status"] == "pending"
    assert state["exploration_flag"] is True
    assert state["gap_history"] == []
    assert state["submissions_today"] == 0
    assert state["code_retry_count"] == 0
    assert state["node_retry_counts"] == {}


def test_experiment_status_enum():
    assert ExperimentStatus.PENDING == "pending"
    assert ExperimentStatus.RUNNING == "running"
    assert ExperimentStatus.DONE == "done"
    assert ExperimentStatus.FAILED == "failed"
    assert ExperimentStatus.VALIDATED == "validated"
    assert ExperimentStatus.SUBMITTED == "submitted"
    assert ExperimentStatus.HELD == "held"
    assert ExperimentStatus.SCORE_TIMEOUT == "score_timeout"
    assert ExperimentStatus.COMPLETE == "complete"
    assert ExperimentStatus.KILLED == "killed"
    assert ExperimentStatus.QUEUED == "queued"


def test_competition_config_dataclass():
    config = CompetitionConfig(
        name="titanic",
        metric="accuracy",
        target="Survived",
        deadline="2026-12-31",
        days_remaining=90,
        submission_limit=10,
    )
    assert config.name == "titanic"
    assert config.submission_limit == 10
    assert config.days_remaining == 90


def test_experiment_spec_dataclass():
    spec = ExperimentSpec(
        parent_version="v5",
        changes=[{"type": "param_change", "param": "lr", "old": 0.01, "new": 0.001}],
        estimated_runtime_multiplier=1.2,
        rationale="Lower learning rate to reduce overfitting",
    )
    assert spec.parent_version == "v5"
    assert len(spec.changes) == 1
    assert spec.estimated_runtime_multiplier == 1.2


def test_directive_json_dataclass():
    d = DirectiveJSON(
        directive_type="tune_existing",
        target_model="catboost",
        rationale="Tune hyperparameters",
        exploration_flag=False,
        priority=3,
    )
    assert d.directive_type == "tune_existing"
    assert d.priority == 3


def test_graph_state_optional_fields():
    state = make_state(run_id="v42", oof_score=0.85, lb_score=0.82)
    assert state["run_id"] == "v42"
    assert state["oof_score"] == 0.85
    assert state["lb_score"] == 0.82


def test_graph_state_competition_dict():
    comp = {"name": "house-prices", "metric": "rmse", "target": "SalePrice",
            "deadline": "2026-06-01", "days_remaining": 60, "submission_limit": 5}
    state = make_state(competition=comp)
    assert state["competition"]["name"] == "house-prices"
    assert state["competition"]["submission_limit"] == 5
