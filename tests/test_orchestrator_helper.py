from __future__ import annotations

from pathlib import Path

import pytest

from gladius.state import CompetitionState
from gladius.utilities._orchestrator_helper import (
    _load_system_prompt,
    make_kickoff_prompt,
)


def _state(iteration: int, topology: str = "functional") -> CompetitionState:
    s = CompetitionState(
        competition_id="c",
        data_dir="/tmp/data",
        output_dir="/tmp/out",
        target_metric="auc",
        metric_direction="maximize",
        topology=topology,
    )
    s.iteration = iteration
    s.max_iterations = 10
    s.best_oof_score = 0.812345
    return s


def test_load_system_prompt_replaces_placeholders():
    text = _load_system_prompt()
    assert "{{RUNTIME_EXPERIMENT_STATE_RELATIVE_PATH}}" not in text
    assert "{{RUNTIME_DATA_BRIEFING_RELATIVE_PATH}}" not in text
    assert "{{TEAM_LEAD_MEMORY_RELATIVE_PATH}}" not in text


def test_load_system_prompt_raises_for_too_short(monkeypatch, tmp_path: Path):
    p = tmp_path / "prompt.md"
    p.write_text("short", encoding="utf-8")
    monkeypatch.setattr("gladius.config.SYSTEM_PROMPT_PATH", p)
    monkeypatch.setattr("gladius.utilities._orchestrator_helper._SYSTEM_PROMPT_PATH", p)
    with pytest.raises(RuntimeError, match="unexpectedly short"):
        _load_system_prompt()


def test_make_kickoff_prompt_first_iteration_includes_scout_and_team_lead():
    text = make_kickoff_prompt(_state(iteration=1))
    assert "FIRST iteration" in text
    assert "delegate to `scout`" in text
    assert "Delegate to `team-lead`" in text


def test_make_kickoff_prompt_later_iteration_has_best_metric_and_skip_scout():
    s = _state(iteration=3, topology="matrix")
    text = make_kickoff_prompt(s)
    assert "iteration 3/10" in text
    assert "Current best auc" in text
    assert "Skip `scout`" in text
    assert "matrix" in text
