"""Utilities to cleanup orphaned background processes for a project workspace."""

from __future__ import annotations

import os
from pathlib import Path

import psutil
from loguru import logger

_CLEANUP_ENV_VAR = "GLADIUS_KILL_ORPHAN_PROCESSES"


def _is_truthy_env(var_name: str) -> bool:
    value = os.getenv(var_name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def should_cleanup_orphan_processes() -> bool:
    """Return whether orphan-process cleanup is enabled via env var."""
    return _is_truthy_env(_CLEANUP_ENV_VAR)


def _is_under_project(cwd: str | None, project_dir: Path) -> bool:
    if not cwd:
        return False
    try:
        cwd_path = Path(cwd).resolve()
        project_path = project_dir.resolve()
    except Exception:
        return False
    return cwd_path == project_path or project_path in cwd_path.parents


def cleanup_orphan_processes(project_dir: Path) -> list[int]:
    """Terminate orphaned processes still running from previous iterations.

    Orphan criteria: process has cwd under project_dir and parent pid is 0/1
    (or parent does not exist anymore). Current process is never targeted.
    """
    current_pid = os.getpid()
    killed: list[int] = []
    candidates: list[psutil.Process] = []

    for proc in psutil.process_iter(attrs=["pid", "ppid", "cwd", "name"]):
        try:
            pid = int(proc.info.get("pid") or 0)
            if pid <= 0 or pid == current_pid:
                continue

            cwd = proc.info.get("cwd")
            if not _is_under_project(cwd, project_dir):
                continue

            ppid = int(proc.info.get("ppid") or 0)
            parent_missing = ppid > 1 and not psutil.pid_exists(ppid)
            if ppid in {0, 1} or parent_missing:
                candidates.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    if not candidates:
        return killed

    logger.warning(
        f"Found {len(candidates)} orphan process(es) under {project_dir}; terminating before iteration start."
    )

    for proc in candidates:
        try:
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    gone, alive = psutil.wait_procs(candidates, timeout=3)
    for proc in gone:
        killed.append(proc.pid)

    for proc in alive:
        try:
            proc.kill()
            killed.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    if killed:
        logger.warning(f"Killed orphan process pid(s): {sorted(set(killed))}")
    return killed
