import subprocess
import sys
import threading
from pathlib import Path

from gladius.state import GraphState

LOGS_DIR = Path("logs")


def _wait_and_write_exitcode(proc: subprocess.Popen, exitcode_path: Path):
    """Wait for the child process and write its exit code to a sidecar file."""
    try:
        rc = proc.wait()
        exitcode_path.write_text(str(rc))
    except Exception:
        exitcode_path.write_text("-1")


def executor_node(state: GraphState) -> GraphState:
    script_path = state.get("generated_script_path")
    run_id = state.get("run_id", "run_001")

    if not script_path or not Path(script_path).exists():
        return {
            "experiment_status": "failed",
            "error_message": f"Script not found: {script_path}",
            "next_node": "knowledge_extractor",
        }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{run_id}.txt"
    exitcode_path = LOGS_DIR / f"{run_id}.exitcode"

    try:
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                [sys.executable, script_path],
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        # Background thread writes exit code when process finishes
        t = threading.Thread(
            target=_wait_and_write_exitcode,
            args=(proc, exitcode_path),
            daemon=True,
        )
        t.start()
        return {
            "running_pid": proc.pid,
            "experiment_status": "running",
            "next_node": "watchdog",
        }
    except Exception as e:
        return {
            "experiment_status": "failed",
            "error_message": str(e),
            "next_node": "knowledge_extractor",
        }
