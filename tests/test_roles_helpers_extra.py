from __future__ import annotations

from pathlib import Path

import pytest

from gladius.roles import helpers


def test_get_runtime_model_returns_env_value(monkeypatch):
    monkeypatch.setenv("GLADIUS_MODEL", "qwen3-coder")
    assert helpers.get_runtime_model() == "qwen3-coder"


def test_get_runtime_model_raises_when_missing(monkeypatch):
    monkeypatch.delenv("GLADIUS_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="GLADIUS_MODEL"):
        helpers.get_runtime_model()


def test_stderr_cb_logs_line(monkeypatch):
    seen = []
    monkeypatch.setattr(helpers.logger, "error", lambda msg: seen.append(msg))
    helpers.stderr_cb("oops")
    assert seen and "CLI stderr" in seen[0]


def test_is_tool_allowed_structured_output_and_task_delegation():
    assert helpers.is_tool_allowed("StructuredOutput", ["Read"]) is True
    assert helpers.is_tool_allowed("Task", ["Agent"]) is True
    assert helpers.is_tool_allowed("Task", ["Agent(data-scientist)"]) is True
    assert helpers.is_tool_allowed("Read", ["Read"]) is True
    assert helpers.is_tool_allowed("Write", ["Read"]) is False


def test_is_path_within_cwd_relative_and_absolute(tmp_path: Path):
    base = tmp_path
    inside_rel = "sub/file.txt"
    (base / "sub").mkdir(parents=True)
    (base / "sub" / "file.txt").write_text("x", encoding="utf-8")

    assert helpers.is_path_within_cwd(inside_rel, str(base)) is True
    assert helpers.is_path_within_cwd(str(base / "sub" / "file.txt"), str(base)) is True
    assert helpers.is_path_within_cwd("../outside.txt", str(base)) is False
