"""Tests for orchestrator iteration start/resume helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

from gladius.langsmith_tracing import configure_langsmith_env
from gladius.orchestrator import (
    _read_experiment_state_snippet,
    _resolve_start_iteration,
    _update_state,
)
from gladius.state import CompetitionState


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
