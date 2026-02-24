import subprocess
import time
from pathlib import Path
from gladius.state import GraphState

PREDICTIONS_DIR = Path("state/predictions")
MAX_RETRIES = 3
RETRY_BACKOFF = [30, 60, 120]


def submission_agent_node(state: GraphState) -> GraphState:
    run_id = state.get("run_id", "")
    competition = state.get("competition", {})
    competition_name = competition.get("name", "")

    pred_path = PREDICTIONS_DIR / f"{run_id}_submission.csv"
    if not pred_path.exists():
        return {
            "experiment_status": "failed",
            "error_message": f"Prediction file not found: {pred_path}",
            "next_node": "knowledge_extractor",
        }

    for attempt in range(MAX_RETRIES):
        try:
            result = subprocess.run(
                ["kaggle", "competitions", "submit",
                 "-c", competition_name,
                 "-f", str(pred_path),
                 "-m", f"gladius run {run_id}"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                return {
                    "experiment_status": "submitted",
                    "submissions_today": state.get("submissions_today", 0) + 1,
                    "last_submission_time": time.time(),
                    "next_node": "lb_tracker",
                }
            else:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF[attempt])
            else:
                return {
                    "experiment_status": "failed",
                    "error_message": str(e),
                    "next_node": "knowledge_extractor",
                }

    return {
        "experiment_status": "failed",
        "error_message": "Submission failed after max retries",
        "next_node": "knowledge_extractor",
    }
