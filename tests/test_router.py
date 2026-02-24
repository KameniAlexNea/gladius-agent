"""Tests for gladius.nodes.router module."""

from gladius.nodes.router import _resolve_next_node, router_node


def make_state(**kwargs) -> dict:
    defaults = dict(
        competition={"name": "test-comp", "submission_limit": 5},
        experiment_status="pending",
        next_node="strategy",
        submissions_today=0,
        current_experiment=None,
        running_pid=None,
        run_id=None,
        oof_score=None,
        lb_score=None,
        best_oof=None,
        gap_history=[],
        last_submission_time=None,
        directive=None,
        exploration_flag=True,
        consecutive_same_directive=0,
        session_summary=None,
        generated_script_path=None,
        reviewer_feedback=None,
        code_retry_count=0,
        error_message=None,
        node_retry_counts={},
        next_node_before_error=None,
    )
    defaults.update(kwargs)
    return defaults


def test_router_running_goes_to_watchdog():
    state = make_state(experiment_status="running")
    assert _resolve_next_node(state) == "watchdog"
    assert router_node(state)["next_node"] == "watchdog"


def test_router_done_goes_to_validation():
    state = make_state(experiment_status="done")
    assert _resolve_next_node(state) == "validation_agent"


def test_router_validated_goes_to_submission_decider():
    state = make_state(experiment_status="validated")
    assert _resolve_next_node(state) == "submission_decider"


def test_router_submitted_goes_to_lb_tracker():
    state = make_state(experiment_status="submitted")
    assert _resolve_next_node(state) == "lb_tracker"


def test_router_score_timeout_goes_to_notifier():
    state = make_state(experiment_status="score_timeout")
    assert _resolve_next_node(state) == "notifier"


def test_router_default_goes_to_strategy():
    state = make_state(experiment_status="pending", next_node="")
    assert _resolve_next_node(state) == "strategy"


def test_router_next_node_respected():
    state = make_state(experiment_status="pending", next_node="hypothesis")
    assert _resolve_next_node(state) == "hypothesis"


def test_router_budget_exceeded_blocks_submission_decider():
    state = make_state(
        experiment_status="pending",
        next_node="submission_decider",
        submissions_today=5,
        competition={"submission_limit": 5},
    )
    assert _resolve_next_node(state) == "strategy"


def test_router_budget_not_exceeded_allows_submission_decider():
    state = make_state(
        experiment_status="pending",
        next_node="submission_decider",
        submissions_today=3,
        competition={"submission_limit": 5},
    )
    assert _resolve_next_node(state) == "submission_decider"


def test_router_status_takes_priority_over_next_node():
    state = make_state(experiment_status="running", next_node="strategy")
    assert _resolve_next_node(state) == "watchdog"


def test_router_node_returns_dict():
    state = make_state(experiment_status="pending", next_node="hypothesis")
    result = router_node(state)
    assert isinstance(result, dict)
    assert result["next_node"] == "hypothesis"
