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


def test_stderr_cb_logs_error_line(monkeypatch):
    seen = []
    monkeypatch.setattr(helpers.logger, "error", lambda msg: seen.append(msg))
    helpers.stderr_cb("fatal error: boom")
    assert seen and "CLI stderr" in seen[0]


def test_stderr_cb_logs_info_for_non_error(monkeypatch):
    seen = []
    monkeypatch.setattr(helpers.logger, "info", lambda msg: seen.append(msg))
    helpers.stderr_cb("normal status line")
    assert seen and "CLI stderr" in seen[0]


def test_stderr_cb_summarizes_hook_callback_noise(monkeypatch):
    seen = []
    monkeypatch.setattr(helpers.logger, "warning", lambda msg: seen.append(msg))
    helpers.stderr_cb("Error in hook callback hook_2: giant minified blob")
    assert seen and "hook callback failure" in seen[0]


def test_stderr_cb_suppresses_stack_source_lines(monkeypatch):
    seen = []
    monkeypatch.setattr(helpers.logger, "debug", lambda msg: seen.append(msg))
    helpers.stderr_cb("7548 | minified source line")
    helpers.stderr_cb("at sendRequest (/$bunfs/root/src/entrypoints/cli.js:7552:133)")
    assert len(seen) == 2
    assert all("suppressed" in s for s in seen)


def test_stderr_cb_stream_closed_is_warning(monkeypatch):
    seen = []
    monkeypatch.setattr(helpers.logger, "warning", lambda msg: seen.append(msg))
    helpers.stderr_cb("error: Stream closed")
    assert seen and "stream closed during hook/control processing" in seen[0]


def test_stderr_cb_dedupes_repeated_hook_callback_warning(monkeypatch):
    warnings = []
    debugs = []
    t = [1000.0]

    helpers._LAST_HOOK_WARNING = ("", 0.0)
    monkeypatch.setattr(helpers.time, "monotonic", lambda: t[0])
    monkeypatch.setattr(helpers.logger, "warning", lambda msg: warnings.append(msg))
    monkeypatch.setattr(helpers.logger, "debug", lambda msg: debugs.append(msg))

    helpers.stderr_cb("Error in hook callback hook_0: blob")
    helpers.stderr_cb("Error in hook callback hook_4: blob")

    assert len(warnings) == 1
    assert any("repeated hook callback failure suppressed" in d for d in debugs)


def test_stderr_cb_allows_hook_warning_after_dedupe_window(monkeypatch):
    warnings = []
    t = [1000.0]

    helpers._LAST_HOOK_WARNING = ("", 0.0)
    monkeypatch.setattr(helpers.time, "monotonic", lambda: t[0])
    monkeypatch.setattr(helpers.logger, "warning", lambda msg: warnings.append(msg))

    helpers.stderr_cb("Error in hook callback hook_0: blob")
    t[0] += helpers._HOOK_WARN_DEDUPE_WINDOW_S + 0.1
    helpers.stderr_cb("Error in hook callback hook_9: blob")

    assert len(warnings) == 2


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
