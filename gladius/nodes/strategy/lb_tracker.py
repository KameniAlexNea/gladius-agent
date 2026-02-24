import time
import json
import subprocess
from pathlib import Path
from gladius.state import GraphState

LB_HISTORY_PATH = Path("state/leaderboard.json")
POLL_INTERVAL = 900   # 15 min
SCORE_TIMEOUT = 10800  # 3 hours


def lb_tracker_node(state: GraphState) -> GraphState:
    competition = state.get("competition", {})
    competition_name = competition.get("name", "")
    last_submission_time = state.get("last_submission_time", time.time())
    run_id = state.get("run_id", "")

    deadline = last_submission_time + SCORE_TIMEOUT

    while time.time() < deadline:
        lb_score = _poll_lb_score(competition_name, run_id)
        if lb_score is not None:
            oof_score = state.get("oof_score")
            gap = (oof_score - lb_score) if oof_score is not None else None
            gap_history = state.get("gap_history", [])
            if gap is not None:
                gap_history = (gap_history + [gap])[-10:]  # keep last 10

            _save_lb_history(run_id, lb_score, gap)

            return {
                "lb_score": lb_score,
                "gap_history": gap_history,
                "experiment_status": "complete",
                "next_node": "strategy",
            }
        time.sleep(POLL_INTERVAL)

    # Timeout
    return {
        "experiment_status": "score_timeout",
        "next_node": "notifier",
    }


def _poll_lb_score(competition_name: str, run_id: str) -> float | None:
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "submissions", "-c", competition_name, "-v"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if run_id in line:
                    parts = line.split(",")
                    if len(parts) >= 5:
                        try:
                            # Expected CSV columns from `kaggle competitions submissions -v`:
                            # fileName,date,description,status,publicScore,privateScore
                            return float(parts[4])
                        except ValueError:
                            pass
    except Exception:
        pass
    return None


def _save_lb_history(run_id: str, lb_score: float, gap: float | None):
    LB_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if LB_HISTORY_PATH.exists():
        try:
            history = json.loads(LB_HISTORY_PATH.read_text())
        except Exception:
            pass
    history.append({"run_id": run_id, "lb_score": lb_score, "gap": gap})
    LB_HISTORY_PATH.write_text(json.dumps(history, indent=2))
