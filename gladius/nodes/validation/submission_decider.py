from gladius.state import GraphState

GAP_WIDENING_THRESHOLD = 3
MIN_OOF_IMPROVEMENT = 1e-4


def submission_decider_node(state: GraphState) -> GraphState:
    oof_score = state.get("oof_score")
    submissions_today = state.get("submissions_today", 0)
    competition = state.get("competition", {})
    sub_limit = competition.get("submission_limit", 5)
    gap_history = state.get("gap_history", [])

    if submissions_today >= sub_limit:
        return {"experiment_status": "held", "next_node": "router"}

    if oof_score is None:
        return {"experiment_status": "held", "next_node": "router"}

    if len(gap_history) >= GAP_WIDENING_THRESHOLD:
        recent_gaps = gap_history[-GAP_WIDENING_THRESHOLD:]
        if all(recent_gaps[i] > recent_gaps[i-1] - 1e-6 for i in range(1, len(recent_gaps))):
            return {
                "experiment_status": "held",
                "next_node": "router",
                "error_message": "OOF-LB gap widening: holding submission",
            }

    return {"experiment_status": "submitted", "next_node": "submission_agent"}
