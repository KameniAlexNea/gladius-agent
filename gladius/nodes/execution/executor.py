import subprocess
import sys
from pathlib import Path
from gladius.state import GraphState

LOGS_DIR = Path("logs")


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

    try:
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                [sys.executable, script_path],
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
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
