"""Tests for the no-op planner shim and solver iteration logic."""

import asyncio

from gladius.agents import planner as planner_module
from gladius.state import CompetitionState


def _state() -> CompetitionState:
    return CompetitionState(
        competition_id="comp-1",
        data_dir="/tmp/data",
        output_dir="/tmp/out",
        target_metric="auc_roc",
        metric_direction="maximize",
    )


def test_run_planner_returns_empty_plan_without_calling_agent():
    """Planner is now a no-op: returns empty plan, no agent call made."""
    plan_dict, session_id = asyncio.run(
        planner_module.run_planner(
            state=_state(),
            data_dir="/tmp/data",
            project_dir="/tmp/project",
            platform="kaggle",
            n_parallel=1,
        )
    )

    assert plan_dict["approach_summary"] == ""
    assert plan_dict["plan_text"] == ""
    assert plan_dict["plan"] == []
    assert plan_dict["plans"] == []
    assert session_id is None


def test_run_planner_no_op_regardless_of_n_parallel():
    """No parallel planning anymore — always returns empty plan."""
    plan_dict, session_id = asyncio.run(
        planner_module.run_planner(
            state=_state(),
            data_dir="/tmp/data",
            project_dir="/tmp/project",
            platform="kaggle",
            n_parallel=4,
        )
    )

    assert plan_dict["plans"] == []
    assert session_id is None


def test_run_planner_preserves_existing_session_id_in_state():
    """Planner no longer creates/resumes sessions; state.planner_session_id is unmodified."""
    state = _state()
    state.planner_session_id = "old-session-abc"

    plan_dict, session_id = asyncio.run(
        planner_module.run_planner(
            state=state,
            data_dir="/tmp/data",
            project_dir="/tmp/project",
        )
    )

    assert session_id is None
    # Orchestrator won't overwrite state.planner_session_id with None — that's its job.
    assert state.planner_session_id == "old-session-abc"
