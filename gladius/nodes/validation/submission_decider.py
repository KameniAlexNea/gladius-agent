import time

from gladius.state import GraphState

GAP_WIDENING_THRESHOLD = 3
MIN_OOF_IMPROVEMENT = 1e-4
SUBMISSION_COOLDOWN_SECS = 7200  # 2 hours


def submission_decider_node(state: GraphState) -> GraphState:
    oof_score = state.get("oof_score")
    submissions_today = state.get("submissions_today", 0)
    competition = state.get("competition", {})
    sub_limit = competition.get("submission_limit", 5)
    gap_history = state.get("gap_history", [])
    best_oof = state.get("best_oof")

    # Budget check
    if submissions_today >= sub_limit:
        return {"experiment_status": "held", "next_node": "router"}

    # Must have a valid OOF score
    if oof_score is None:
        return {"experiment_status": "held", "next_node": "router"}

    # OOF improvement gate: new score must beat the best known OOF
    baseline = best_oof if best_oof is not None else 0.0
    if oof_score <= baseline + MIN_OOF_IMPROVEMENT:
        return {"experiment_status": "held", "next_node": "router"}

    # 2-hour cooldown since last submission
    last_sub = state.get("last_submission_time")
    if last_sub is not None and time.time() - last_sub < SUBMISSION_COOLDOWN_SECS:
        return {"experiment_status": "held", "next_node": "router"}

    # Gap-widening check
    if len(gap_history) >= GAP_WIDENING_THRESHOLD:
        recent_gaps = gap_history[-GAP_WIDENING_THRESHOLD:]
        if all(recent_gaps[i] > recent_gaps[i - 1] - 1e-6 for i in range(1, len(recent_gaps))):
            return {
                "experiment_status": "held",
                "next_node": "router",
                "error_message": "OOF-LB gap widening: holding submission",
            }

    return {
        "experiment_status": "submitted",
        "next_node": "submission_agent",
        "best_oof": oof_score,  # update best known OOF on approval
    }
