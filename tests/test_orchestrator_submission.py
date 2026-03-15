"""Tests for orchestrator submission counting and topology integration."""

import asyncio
from pathlib import Path

from gladius import orchestrator
from gladius.agents.topologies import TOPOLOGY_REGISTRY
from gladius.agents.topologies.base import IterationResult
from gladius.agents.roles.specs import build_validator_prompt
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


def _make_cfg(data_dir: Path) -> dict:
    return {
        "competition_id": "comp-1",
        "platform": "fake",
        "data_dir": str(data_dir.resolve()),
        "metric": "auc_roc",
        "direction": "maximize",
        "topology": "functional",
    }


# ── Validator prompt helpers ──────────────────────────────────────────────────


def test_validator_prompt_handles_none_score():
    prompt = build_validator_prompt(
        oof_score=None,
        quality_score=None,
        best_oof_score=None,
        best_quality_score=None,
        submission_path=None,
        target_metric="auc_roc",
        metric_direction="maximize",
        submission_quota_remaining=5,
        project_dir="/tmp/project",
    )
    assert "None" in prompt or "auc_roc" in prompt


# ── Submission counter tests ──────────────────────────────────────────────────


def _fake_topology_factory(
    *,
    submit: bool = True,
    is_improvement: bool = True,
    stop: bool = False,
    oof_score: float = 0.8,
    submission_file: str = "submission.csv",
    format_ok: bool = True,
):
    class _FakeTopology:
        async def run_iteration(
            self, state, project_dir, platform, *, n_parallel=1,
            consume_agent_call=None, check_budget=None,
        ):
            return IterationResult(
                status="success",
                oof_score=oof_score,
                quality_score=75,
                solution_files=["solution.py"],
                submission_file=submission_file,
                notes="ok",
                is_improvement=is_improvement,
                submit=submit,
                format_ok=format_ok,
                stop=stop,
            )

    return _FakeTopology


def test_submission_counter_not_incremented_on_failed_submit(monkeypatch, tmp_path):
    monkeypatch.setenv("GLADIUS_MODEL", "test-model")

    competition_dir = tmp_path / "competition"
    data_dir = competition_dir / "data"
    data_dir.mkdir(parents=True)
    (competition_dir / "README.md").write_text("# test\n", encoding="utf-8")

    monkeypatch.setattr(
        orchestrator, "load_competition_config", lambda _: _make_cfg(data_dir)
    )
    monkeypatch.setitem(
        TOPOLOGY_REGISTRY, "functional",
        _fake_topology_factory(submit=True, is_improvement=True),
    )
    monkeypatch.setattr(
        orchestrator, "submit", lambda **kwargs: (False, "submission_failed")
    )

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
    # best_submission_path is set when OOF improves locally (independent of submission success)
    assert state.best_oof_score is not None


def test_submission_counter_incremented_on_successful_submit(monkeypatch, tmp_path):
    monkeypatch.setenv("GLADIUS_MODEL", "test-model")

    competition_dir = tmp_path / "competition"
    data_dir = competition_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "SampleSubmission.csv").write_text(
        "ID,Target\n1,Low\n2,High\n", encoding="utf-8"
    )
    (competition_dir / "submission.csv").write_text(
        "ID,Target\n1,Low\n2,High\n", encoding="utf-8"
    )
    (competition_dir / "README.md").write_text("# test\n", encoding="utf-8")

    monkeypatch.setattr(
        orchestrator, "load_competition_config", lambda _: _make_cfg(data_dir)
    )
    monkeypatch.setitem(
        TOPOLOGY_REGISTRY, "functional",
        _fake_topology_factory(
            submit=True, is_improvement=True,
            submission_file=str(competition_dir / "submission.csv"),
        ),
    )
    monkeypatch.setattr(
        orchestrator, "submit", lambda **kwargs: (True, None)
    )

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


def test_submit_false_blocks_submission_even_when_improved(monkeypatch, tmp_path):
    monkeypatch.setenv("GLADIUS_MODEL", "test-model")

    competition_dir = tmp_path / "competition"
    data_dir = competition_dir / "data"
    data_dir.mkdir(parents=True)
    (competition_dir / "README.md").write_text("# test\n", encoding="utf-8")

    calls: list[dict] = []

    def fake_submit(**kwargs):
        calls.append(kwargs)
        return True, None

    monkeypatch.setattr(
        orchestrator, "load_competition_config", lambda _: _make_cfg(data_dir)
    )
    monkeypatch.setitem(
        TOPOLOGY_REGISTRY, "functional",
        _fake_topology_factory(submit=False, is_improvement=True),
    )
    monkeypatch.setattr(orchestrator, "submit", fake_submit)

    state = asyncio.run(
        orchestrator.run_competition(
            competition_dir=str(competition_dir),
            max_iterations=1,
            resume_from_db=False,
            auto_submit=True,
            n_parallel=1,
        )
    )

    assert len(calls) == 0, "submit() must not be called when result.submit=False"
    assert state.submission_count == 0
