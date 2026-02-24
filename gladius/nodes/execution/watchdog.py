import os
import signal
import time
from pathlib import Path

import psutil

from gladius.state import GraphState

LOGS_DIR = Path("logs")
POLL_INTERVAL = 30
MAX_NO_OUTPUT_SECS = 1800  # 30 min
MAX_RAM_FRACTION = 0.90


def watchdog_node(state: GraphState) -> GraphState:
    pid = state.get("running_pid")
    run_id = state.get("run_id", "")

    if not pid:
        return {"experiment_status": "failed", "next_node": "knowledge_extractor"}

    log_path = LOGS_DIR / f"{run_id}.txt"
    last_size = 0
    last_output_time = time.time()

    while True:
        if not _is_running(pid):
            exit_code = _get_exit_code(pid)
            if exit_code == 0:
                return {"experiment_status": "done", "next_node": "validation_agent"}
            else:
                return {
                    "experiment_status": "failed",
                    "next_node": "knowledge_extractor",
                }

        ram_usage = _get_ram_usage()
        if ram_usage > MAX_RAM_FRACTION:
            _kill(pid)
            return {
                "experiment_status": "killed",
                "error_message": f"RAM usage {ram_usage:.0%} exceeded limit of {MAX_RAM_FRACTION:.0%}",
                "next_node": "knowledge_extractor",
            }

        if log_path.exists():
            current_size = log_path.stat().st_size
            if current_size != last_size:
                last_size = current_size
                last_output_time = time.time()
            elif time.time() - last_output_time > MAX_NO_OUTPUT_SECS:
                _kill(pid)
                return {
                    "experiment_status": "killed",
                    "error_message": "No log output for 30 minutes",
                    "next_node": "knowledge_extractor",
                }

        time.sleep(POLL_INTERVAL)


def _is_running(pid: int) -> bool:
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _get_exit_code(pid: int) -> int:
    try:
        proc = psutil.Process(pid)
        return proc.wait(timeout=5)
    except Exception:
        return -1


def _get_ram_usage() -> float:
    try:
        return psutil.virtual_memory().percent / 100.0
    except Exception:
        return 0.0


def _kill(pid: int):
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(3)
        if _is_running(pid):
            os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
