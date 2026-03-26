from __future__ import annotations

from pathlib import Path

from gladius.process_cleanup import _is_under_project, should_cleanup_orphan_processes


def test_should_cleanup_orphan_processes_truthy(monkeypatch):
    monkeypatch.setenv("GLADIUS_KILL_ORPHAN_PROCESSES", "true")
    assert should_cleanup_orphan_processes() is True


def test_should_cleanup_orphan_processes_falsey(monkeypatch):
    monkeypatch.delenv("GLADIUS_KILL_ORPHAN_PROCESSES", raising=False)
    assert should_cleanup_orphan_processes() is False


def test_is_under_project_for_child_path(tmp_path: Path):
    project = tmp_path / "proj"
    child = project / "subdir"
    child.mkdir(parents=True)
    assert _is_under_project(str(child), project) is True


def test_is_under_project_for_outside_path(tmp_path: Path):
    project = tmp_path / "proj"
    outside = tmp_path / "other"
    project.mkdir(parents=True)
    outside.mkdir(parents=True)
    assert _is_under_project(str(outside), project) is False
