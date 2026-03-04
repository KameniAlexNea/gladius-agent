import asyncio
from pathlib import Path

from gladius import orchestrator
from gladius.agents import validation as validation_module
from gladius.state import CompetitionState


def _make_state(competition_dir: Path) -> CompetitionState:
    return CompetitionState(
        competition_id="comp-1",
        data_dir=str((competition_dir / "data").resolve()),
        output_dir=str((competition_dir / ".gladius").resolve()),
        target_metric="auc_roc",
        metric_direction="maximize",
        max_iterations=1,
    )


def test_validation_prompt_handles_none_score_without_format_error(
    monkeypatch, tmp_path
):
    captured = {}

    async def fake_run_agent(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        return (
            {
                "oof_score": None,
                "quality_score": None,
                "is_improvement": False,
                "submit": False,
                "stop": False,
                "reasoning": "ok",
                "next_directions": ["try again"],
            },
            "session-1",
        )

    monkeypatch.setattr(validation_module, "run_agent", fake_run_agent)

    state = _make_state(tmp_path)
    result = asyncio.run(
        validation_module.run_validation_agent(
            solution_path="solution.py",
            oof_score=None,
            quality_score=0,
            submission_path="",
            state=state,
            project_dir=str(tmp_path),
            platform="none",
        )
    )

    assert result["reasoning"] == "ok"
    assert "OOF score     : n/a" in captured["prompt"]
    assert "No submission file — set format_ok=False." in captured["prompt"]


def test_submission_counter_not_incremented_on_failed_submit(monkeypatch, tmp_path):
    competition_dir = tmp_path / "competition"
    data_dir = competition_dir / "data"
    data_dir.mkdir(parents=True)

    # Minimal files the setup/generation flow expects to exist
    (competition_dir / "README.md").write_text("# test\n", encoding="utf-8")

    monkeypatch.setattr(
        orchestrator,
        "load_competition_config",
        lambda _: {
            "competition_id": "comp-1",
            "platform": "fake",
            "data_dir": str(data_dir.resolve()),
            "metric": "auc_roc",
            "direction": "maximize",
        },
    )

    async def fake_planner(*args, **kwargs):
        return (
            {
                "approach_summary": "baseline",
                "plan_text": "do baseline",
                "plan": [{"step": 1, "description": "run"}],
                "plans": [],
            },
            "planner-session",
        )

    async def fake_implementer(*args, **kwargs):
        return {
            "status": "success",
            "oof_score": 0.8,
            "quality_score": 75,
            "solution_files": ["solution.py"],
            "submission_file": "submission.csv",
            "notes": "ok",
        }

    async def fake_validation(*args, **kwargs):
        return {
            "oof_score": 0.8,
            "quality_score": None,
            "is_improvement": True,
            "submit": True,
            "stop": False,
            "reasoning": "good",
            "next_directions": ["try features"],
        }

    monkeypatch.setattr(orchestrator, "run_planner", fake_planner)
    monkeypatch.setattr(orchestrator, "run_implementer", fake_implementer)
    monkeypatch.setattr(orchestrator, "run_validation_agent", fake_validation)
    monkeypatch.setattr(orchestrator, "_submit", lambda **kwargs: False)

    state = asyncio.run(
        orchestrator.run_competition(
            competition_dir=str(competition_dir),
            max_iterations=1,
            resume_from_db=False,
            auto_submit=True,
            n_parallel=1,
        )
    )

    assert state.submission_count == 0
    assert state.best_submission_path is None


def test_submission_counter_incremented_on_successful_submit(monkeypatch, tmp_path):
    competition_dir = tmp_path / "competition"
    data_dir = competition_dir / "data"
    data_dir.mkdir(parents=True)

    (competition_dir / "README.md").write_text("# test\n", encoding="utf-8")

    monkeypatch.setattr(
        orchestrator,
        "load_competition_config",
        lambda _: {
            "competition_id": "comp-1",
            "platform": "fake",
            "data_dir": str(data_dir.resolve()),
            "metric": "auc_roc",
            "direction": "maximize",
        },
    )

    async def fake_planner(*args, **kwargs):
        return (
            {
                "approach_summary": "baseline",
                "plan_text": "do baseline",
                "plan": [{"step": 1, "description": "run"}],
                "plans": [],
            },
            "planner-session",
        )

    async def fake_implementer(*args, **kwargs):
        return {
            "status": "success",
            "oof_score": 0.8,
            "quality_score": 75,
            "solution_files": ["solution.py"],
            "submission_file": "submission.csv",
            "notes": "ok",
        }

    async def fake_validation(*args, **kwargs):
        return {
            "oof_score": 0.8,
            "quality_score": None,
            "is_improvement": True,
            "submit": True,
            "stop": False,
            "reasoning": "good",
            "next_directions": ["try features"],
        }

    monkeypatch.setattr(orchestrator, "run_planner", fake_planner)
    monkeypatch.setattr(orchestrator, "run_implementer", fake_implementer)
    monkeypatch.setattr(orchestrator, "run_validation_agent", fake_validation)
    monkeypatch.setattr(orchestrator, "_submit", lambda **kwargs: True)

    state = asyncio.run(
        orchestrator.run_competition(
            competition_dir=str(competition_dir),
            max_iterations=1,
            resume_from_db=False,
            auto_submit=True,
            n_parallel=1,
        )
    )

    assert state.submission_count == 1
    assert state.best_submission_path == "submission.csv"
