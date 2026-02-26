import os

import requests

from gladius.state import GraphState


def notifier_node(state: GraphState) -> GraphState:
    status = state.get("experiment_status", "")
    run_id = state.get("run_id", "")
    error_msg = state.get("error_message", "")
    lb_score = state.get("lb_score")

    if status == "score_timeout":
        msg = f"⏰ Score timeout for run {run_id}. No LB score after 3 hours."
    elif status == "failed":
        msg = f"❌ Run {run_id} failed: {error_msg}"
    elif lb_score is not None:
        msg = f"✅ Run {run_id} LB score: {lb_score:.5f}"
    else:
        msg = f"ℹ️ Run {run_id} status: {status}"

    _send_telegram(msg)
    return {"next_node": "router"}


def _send_telegram(message: str):
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return  # Telegram not configured
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message},
            timeout=10,
        )
    except Exception:
        pass
