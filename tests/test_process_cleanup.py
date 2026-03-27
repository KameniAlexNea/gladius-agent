from __future__ import annotations

from pathlib import Path

import psutil

from gladius.utilities.process_cleanup import (
    _is_under_project,
    cleanup_orphan_processes,
    should_cleanup_orphan_processes,
)


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


class _FakeProc:
    def __init__(self, pid: int, ppid: int, cwd: str):
        self.info = {"pid": pid, "ppid": ppid, "cwd": cwd, "name": "python"}
        self.pid = pid
        self.terminated = False
        self.killed = False

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def test_cleanup_orphan_processes_returns_empty_when_no_candidates(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr("gladius.utilities.process_cleanup.os.getpid", lambda: 999)
    monkeypatch.setattr("gladius.utilities.process_cleanup.psutil.process_iter", lambda attrs: [])
    assert cleanup_orphan_processes(tmp_path) == []


def test_cleanup_orphan_processes_terminates_and_kills(monkeypatch, tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir()
    p1 = _FakeProc(101, 1, str(project))
    p2 = _FakeProc(102, 222, str(project))

    monkeypatch.setattr("gladius.utilities.process_cleanup.os.getpid", lambda: 999)
    monkeypatch.setattr(
        "gladius.utilities.process_cleanup.psutil.process_iter", lambda attrs: [p1, p2]
    )
    monkeypatch.setattr("gladius.utilities.process_cleanup.psutil.pid_exists", lambda pid: False)
    monkeypatch.setattr(
        "gladius.utilities.process_cleanup.psutil.wait_procs", lambda procs, timeout: ([p1], [p2])
    )

    killed = cleanup_orphan_processes(project)
    assert sorted(killed) == [101, 102]
    assert p1.terminated is True
    assert p2.terminated is True
    assert p2.killed is True


def test_cleanup_orphan_processes_skips_access_denied(monkeypatch, tmp_path: Path):
    class _DeniedProc:
        @property
        def info(self):
            raise psutil.AccessDenied(pid=123)

    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.setattr("gladius.utilities.process_cleanup.os.getpid", lambda: 999)
    monkeypatch.setattr(
        "gladius.utilities.process_cleanup.psutil.process_iter", lambda attrs: [_DeniedProc()]
    )

    assert cleanup_orphan_processes(project) == []
