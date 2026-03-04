"""Tests for CompetitionState / StateStore round-trip."""

import pytest

from gladius.state import CompetitionState, StateStore


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "state.db")


def make_state(**kwargs) -> CompetitionState:
    defaults = dict(
        competition_id="test-comp",
        data_dir="/data",
        output_dir="/output",
        target_metric="auc_roc",
        metric_direction="maximize",
    )
    defaults.update(kwargs)
    return CompetitionState(**defaults)


# ── Basic round-trip ───────────────────────────────────────────────────────────


def test_save_and_load_scalars(db_path):
    store = StateStore(db_path)
    state = make_state()
    state.iteration = 3
    state.phase = "implementing"
    state.best_oof_score = 0.87654
    state.submission_count = 2
    state.planner_session_id = "sess-abc-123"
    store.save(state)
    loaded = store.load()
    store.close()

    assert loaded is not None
    assert loaded.competition_id == "test-comp"
    assert loaded.iteration == 3
    assert loaded.phase == "implementing"
    assert abs(loaded.best_oof_score - 0.87654) < 1e-9
    assert loaded.submission_count == 2
    assert loaded.planner_session_id == "sess-abc-123"
    assert loaded.target_metric == "auc_roc"
    assert loaded.metric_direction == "maximize"


def test_save_and_load_experiments(db_path):
    store = StateStore(db_path)
    state = make_state()
    state.experiments = [
        {
            "iteration": 0,
            "oof_score": 0.75,
            "submission_file": "sub0.csv",
            "notes": "baseline",
            "approach": "LR",
            "solution_files": ["train.py"],
        },
        {
            "iteration": 1,
            "oof_score": 0.82,
            "submission_file": "sub1.csv",
            "notes": "xgb",
            "approach": "XGBoost",
            "solution_files": ["train.py", "features.py"],
        },
    ]
    store.save(state)
    loaded = store.load()
    store.close()

    assert len(loaded.experiments) == 2
    assert loaded.experiments[0]["oof_score"] == 0.75
    assert loaded.experiments[1]["oof_score"] == 0.82
    assert loaded.experiments[1]["solution_files"] == ["train.py", "features.py"]


def test_save_and_load_failed_runs(db_path):
    store = StateStore(db_path)
    state = make_state()
    state.failed_runs = [
        {"iteration": 0, "status": "error", "error": "OOM", "approach": "LGBM"}
    ]
    store.save(state)
    loaded = store.load()
    store.close()

    assert len(loaded.failed_runs) == 1
    assert loaded.failed_runs[0]["error"] == "OOM"


def test_load_returns_none_when_empty(db_path):
    store = StateStore(db_path)
    result = store.load()
    store.close()
    assert result is None


def test_current_plan_roundtrip(db_path):
    store = StateStore(db_path)
    state = make_state()
    state.current_plan = {
        "approach_summary": "Use XGBoost with target encoding",
        "plan": [
            {"step": 1, "description": "Load data"},
            {"step": 2, "description": "Fit model"},
        ],
        "expected_metric_delta": 0.02,
    }
    store.save(state)
    loaded = store.load()
    store.close()

    assert loaded.current_plan["approach_summary"] == "Use XGBoost with target encoding"
    assert len(loaded.current_plan["plan"]) == 2


# ── Append-only behaviour (issue #10 fix) ────────────────────────────────────


def test_append_only_experiments(db_path):
    """Second save should only insert new rows, not re-insert existing ones."""
    store = StateStore(db_path)
    state = make_state()
    state.experiments = [
        {
            "iteration": 0,
            "oof_score": 0.75,
            "submission_file": "",
            "notes": "",
            "approach": "",
            "solution_files": [],
        }
    ]
    store.save(state)

    # Add a second experiment and save again
    state.experiments.append(
        {
            "iteration": 1,
            "oof_score": 0.80,
            "submission_file": "",
            "notes": "",
            "approach": "",
            "solution_files": [],
        }
    )
    store.save(state)

    loaded = store.load()
    store.close()

    # Should have exactly 2 experiments, not 3 or 4
    assert len(loaded.experiments) == 2
    assert loaded.experiments[0]["oof_score"] == 0.75
    assert loaded.experiments[1]["oof_score"] == 0.80


def test_resume_after_save(db_path):
    """Load after save should produce identical state."""
    store = StateStore(db_path)
    state = make_state()
    state.iteration = 5
    state.best_oof_score = 0.91
    state.experiments = [
        {
            "iteration": i,
            "oof_score": 0.7 + i * 0.01,
            "submission_file": f"s{i}.csv",
            "notes": "",
            "approach": "",
            "solution_files": [],
        }
        for i in range(5)
    ]
    store.save(state)
    store.close()

    # Simulate new process opening the same DB
    store2 = StateStore(db_path)
    loaded = store2.load()
    store2.close()

    assert loaded.iteration == 5
    assert len(loaded.experiments) == 5
    assert abs(loaded.experiments[4]["oof_score"] - 0.74) < 1e-9


def test_last_stop_reason_roundtrip(db_path):
    store = StateStore(db_path)
    state = make_state()
    state.phase = "done"
    state.last_stop_reason = "agent call budget exceeded (6/5)"
    store.save(state)
    loaded = store.load()
    store.close()

    assert loaded is not None
    assert loaded.last_stop_reason == "agent call budget exceeded (6/5)"
