"""Tests for orchestrator iteration start/resume helpers."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from gladius.orchestrator import (
    _archive_stale_outputs,
    _incomplete_agents,
    _missing_scout_artifact,
    _read_experiment_state_snippet,
    _resolve_start_iteration,
    _update_state,
    run_competition,
)
from gladius.state import CompetitionState
from gladius.utilities.langsmith_tracing import configure_langsmith_env


def test_resolve_start_iteration_default_when_env_missing(monkeypatch):
    monkeypatch.delenv("GLADIUS_START_ITERATION", raising=False)
    assert _resolve_start_iteration(max_iterations=20) == 1


def test_resolve_start_iteration_uses_env_value(monkeypatch):
    monkeypatch.setenv("GLADIUS_START_ITERATION", "4")
    assert _resolve_start_iteration(max_iterations=20) == 4


def test_resolve_start_iteration_rejects_non_integer(monkeypatch):
    monkeypatch.setenv("GLADIUS_START_ITERATION", "abc")
    assert _resolve_start_iteration(max_iterations=20) == 1


def test_resolve_start_iteration_rejects_values_below_one(monkeypatch):
    monkeypatch.setenv("GLADIUS_START_ITERATION", "0")
    assert _resolve_start_iteration(max_iterations=20) == 1


def test_resolve_start_iteration_clamps_to_max_iterations(monkeypatch):
    monkeypatch.setenv("GLADIUS_START_ITERATION", "99")
    assert _resolve_start_iteration(max_iterations=7) == 7


def test_langsmith_env_not_enabled_without_ollama_bridge(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    assert configure_langsmith_env() is False
    assert "LANGSMITH_TRACING" not in os.environ


def test_langsmith_env_defaults_are_set_for_ollama_bridge(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "OLLAMA")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)

    assert configure_langsmith_env() is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://api.smith.langchain.com"
    assert os.environ["LANGSMITH_PROJECT"] == "gladius-agent"


def test_langsmith_env_requires_langsmith_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "OLLAMA")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    assert configure_langsmith_env() is False


def test_update_state_updates_submission_path_on_tie_maximize(tmp_path: Path):
    state = CompetitionState(
        competition_id="c",
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path),
        target_metric="auc",
        metric_direction="maximize",
    )
    state.iteration = 1
    state.best_oof_score = 0.91
    state.best_submission_path = "submissions/old.csv"

    runtime_dir = tmp_path / ".gladius" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    exp = runtime_dir / "EXPERIMENT_STATE.json"
    exp.write_text(
        json.dumps(
            {
                "evaluator": {"oof_score": 0.91, "status": "success"},
                "ml_engineer": {
                    "status": "success",
                    "submission_file": "submissions/new.csv",
                },
                "team_lead": {"status": "success"},
            }
        ),
        encoding="utf-8",
    )

    _update_state(state, tmp_path)
    assert state.best_submission_path == "submissions/new.csv"


def test_read_experiment_state_snippet_prefers_compact_summary(tmp_path: Path):
    exp = tmp_path / "EXPERIMENT_STATE.json"
    payload = {
        "team_lead": {"status": "success", "plan": "x" * 5000},
        "ml_engineer": {"status": "error", "reason": "failed", "notes": "y" * 3000},
        "evaluator": {"status": "pending", "notes": "z" * 3000},
        "done": False,
    }
    exp.write_text(json.dumps(payload), encoding="utf-8")

    snippet = _read_experiment_state_snippet(exp)
    summary = json.loads(snippet)
    assert "pending_agents" in summary
    assert "ml_engineer" in summary["pending_agents"]


def test_incomplete_agents_handles_missing_and_invalid_json(tmp_path: Path):
    missing = tmp_path / "missing.json"
    assert _incomplete_agents(missing) == ["team_lead", "ml_engineer", "evaluator"]

    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert _incomplete_agents(bad) == ["team_lead", "ml_engineer", "evaluator"]


def test_incomplete_agents_returns_only_non_success(tmp_path: Path):
    exp = tmp_path / "EXPERIMENT_STATE.json"
    exp.write_text(
        json.dumps(
            {
                "team_lead": {"status": "success"},
                "ml_engineer": {"status": "error"},
                "evaluator": {"status": "success"},
            }
        ),
        encoding="utf-8",
    )
    assert _incomplete_agents(exp) == ["ml_engineer"]


def test_missing_scout_artifact_only_first_iteration(tmp_path: Path):
    state = CompetitionState(
        competition_id="c",
        data_dir=str(tmp_path),
        output_dir=str(tmp_path),
        target_metric="auc",
        metric_direction="maximize",
    )
    state.iteration = 1
    assert _missing_scout_artifact(state, tmp_path) is True

    runtime_dir = tmp_path / ".gladius" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "DATA_BRIEFING.md").write_text("ok", encoding="utf-8")
    assert _missing_scout_artifact(state, tmp_path) is False

    state.iteration = 2
    assert _missing_scout_artifact(state, tmp_path) is False


def test_archive_stale_outputs_moves_artifacts_and_logs(tmp_path: Path):
    art = tmp_path / "artifacts"
    logs = tmp_path / "logs"
    art.mkdir()
    logs.mkdir()
    (art / "best_params.json").write_text("{}", encoding="utf-8")
    (art / "junk.bin").write_text("x", encoding="utf-8")
    (logs / "train.log").write_text("l", encoding="utf-8")
    (logs / "gladius.log").write_text("keep", encoding="utf-8")

    _archive_stale_outputs(tmp_path, iteration=2)

    assert (tmp_path / "artifacts_iter1" / "junk.bin").exists()
    assert (tmp_path / "artifacts" / "best_params.json").exists()
    assert (tmp_path / "logs_iter1" / "train.log").exists()
    assert (tmp_path / "logs" / "gladius.log").exists()


def test_update_state_updates_submission_path_on_tie_minimize(tmp_path: Path):
    state = CompetitionState(
        competition_id="c",
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path),
        target_metric="rmse",
        metric_direction="minimize",
    )
    state.iteration = 1
    state.best_oof_score = 0.123
    state.best_submission_path = "submissions/old.csv"

    runtime_dir = tmp_path / ".gladius" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    exp = runtime_dir / "EXPERIMENT_STATE.json"
    exp.write_text(
        json.dumps(
            {
                "evaluator": {"oof_score": 0.123, "status": "success"},
                "ml_engineer": {
                    "status": "success",
                    "submission_file": "submissions/new_min.csv",
                },
                "team_lead": {"status": "success"},
            }
        ),
        encoding="utf-8",
    )

    _update_state(state, tmp_path)
    assert state.best_submission_path == "submissions/new_min.csv"


def test_run_competition_single_iteration_success(monkeypatch, tmp_path: Path):
    calls = {"run_agent": 0, "cleanup": 0}
    runtime_dir = tmp_path / ".gladius" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "DATA_BRIEFING.md").write_text("briefing", encoding="utf-8")

    monkeypatch.setattr(
        "gladius.orchestrator.load_competition_config",
        lambda *args, **kwargs: {"competition_id": "comp-x", "max_iterations": 1},
    )
    monkeypatch.setattr(
        "gladius.orchestrator._build_state",
        lambda project_dir, cfg: CompetitionState(
            competition_id="comp-x",
            data_dir=str(project_dir / "data"),
            output_dir=str(project_dir),
            target_metric="auc",
            metric_direction="maximize",
            max_iterations=1,
        ),
    )
    monkeypatch.setattr(
        "gladius.orchestrator.claude_md.write", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "gladius.orchestrator._make_kickoff_prompt", lambda state: "kickoff"
    )
    monkeypatch.setattr("gladius.orchestrator.init_langsmith_tracing", lambda: False)
    monkeypatch.setattr(
        "gladius.orchestrator.should_cleanup_orphan_processes", lambda: True
    )
    monkeypatch.setattr(
        "gladius.orchestrator.cleanup_orphan_processes",
        lambda _project_dir: calls.__setitem__("cleanup", calls["cleanup"] + 1),
    )

    async def _fake_run_agent(**kwargs):
        calls["run_agent"] += 1
        exp = tmp_path / ".gladius" / "runtime" / "EXPERIMENT_STATE.json"
        exp.parent.mkdir(parents=True, exist_ok=True)
        exp.write_text(
            json.dumps(
                {
                    "team_lead": {"status": "success"},
                    "ml_engineer": {
                        "status": "success",
                        "oof_score": 0.81,
                        "quality_score": 0.9,
                        "submission_file": "submissions/s1.csv",
                    },
                    "evaluator": {
                        "status": "success",
                        "oof_score": 0.81,
                        "metric": "auc",
                    },
                    "done": True,
                }
            ),
            encoding="utf-8",
        )
        return "ok", {"status": "ok"}

    monkeypatch.setattr("gladius.orchestrator.run_agent", _fake_run_agent)

    asyncio.run(run_competition(str(tmp_path), max_iterations=1))

    assert calls["run_agent"] == 1
    assert calls["cleanup"] == 1


def test_run_competition_redispatches_then_succeeds(monkeypatch, tmp_path: Path):
    calls = {"run_agent": 0}
    runtime_dir = tmp_path / ".gladius" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "DATA_BRIEFING.md").write_text("briefing", encoding="utf-8")

    monkeypatch.setattr(
        "gladius.orchestrator.load_competition_config",
        lambda *args, **kwargs: {"competition_id": "comp-x", "max_iterations": 1},
    )
    monkeypatch.setattr(
        "gladius.orchestrator._build_state",
        lambda project_dir, cfg: CompetitionState(
            competition_id="comp-x",
            data_dir=str(project_dir / "data"),
            output_dir=str(project_dir),
            target_metric="auc",
            metric_direction="maximize",
            max_iterations=1,
        ),
    )
    monkeypatch.setattr(
        "gladius.orchestrator.claude_md.write", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "gladius.orchestrator._make_kickoff_prompt", lambda state: "kickoff"
    )
    monkeypatch.setattr("gladius.orchestrator.init_langsmith_tracing", lambda: False)
    monkeypatch.setattr(
        "gladius.orchestrator.should_cleanup_orphan_processes", lambda: False
    )

    async def _fake_run_agent(**kwargs):
        calls["run_agent"] += 1
        exp = tmp_path / ".gladius" / "runtime" / "EXPERIMENT_STATE.json"
        exp.parent.mkdir(parents=True, exist_ok=True)
        if calls["run_agent"] == 1:
            exp.write_text(
                json.dumps({"team_lead": {"status": "success"}, "done": False}),
                encoding="utf-8",
            )
        else:
            exp.write_text(
                json.dumps(
                    {
                        "team_lead": {"status": "success"},
                        "ml_engineer": {"status": "success", "oof_score": 0.77},
                        "evaluator": {"status": "success", "oof_score": 0.77},
                        "done": True,
                    }
                ),
                encoding="utf-8",
            )
        return "ok", {"status": "ok"}

    monkeypatch.setattr("gladius.orchestrator.run_agent", _fake_run_agent)

    asyncio.run(run_competition(str(tmp_path), max_iterations=1))

    assert calls["run_agent"] == 2


def test_run_competition_stops_on_consecutive_errors(monkeypatch, tmp_path: Path):
    calls = {"run_agent": 0}

    monkeypatch.setattr(
        "gladius.orchestrator.load_competition_config",
        lambda *args, **kwargs: {"competition_id": "comp-x", "max_iterations": 1},
    )
    monkeypatch.setattr(
        "gladius.orchestrator._build_state",
        lambda project_dir, cfg: CompetitionState(
            competition_id="comp-x",
            data_dir=str(project_dir / "data"),
            output_dir=str(project_dir),
            target_metric="auc",
            metric_direction="maximize",
            max_iterations=1,
        ),
    )
    monkeypatch.setattr(
        "gladius.orchestrator.claude_md.write", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "gladius.orchestrator._make_kickoff_prompt", lambda state: "kickoff"
    )
    monkeypatch.setattr("gladius.orchestrator.init_langsmith_tracing", lambda: False)
    monkeypatch.setattr(
        "gladius.orchestrator.should_cleanup_orphan_processes", lambda: False
    )
    import dataclasses
    import gladius.orchestrator as _orch_mod

    fast_settings = dataclasses.replace(
        _orch_mod.SETTINGS, max_redispatch=1, max_consecutive_errors=1
    )
    monkeypatch.setattr(_orch_mod, "SETTINGS", fast_settings)

    async def _always_fail(**kwargs):
        calls["run_agent"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr("gladius.orchestrator.run_agent", _always_fail)

    asyncio.run(run_competition(str(tmp_path), max_iterations=1))

    assert calls["run_agent"] == 2
