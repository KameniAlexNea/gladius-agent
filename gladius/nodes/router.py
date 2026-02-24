from gladius.state import ExperimentStatus, GraphState


def router_node(state: GraphState) -> str:
    """Routes to the appropriate next node based on experiment status."""
    status = state.get("experiment_status", "")
    next_node = state.get("next_node", "")

    if status == ExperimentStatus.RUNNING:
        return "watchdog"
    if status == ExperimentStatus.DONE:
        return "validation_agent"
    if status == ExperimentStatus.VALIDATED:
        return "submission_decider"
    if status == ExperimentStatus.SUBMITTED:
        return "lb_tracker"
    if status == ExperimentStatus.SCORE_TIMEOUT:
        return "notifier"

    # Check budget before allowing submission routing
    submissions_today = state.get("submissions_today", 0)
    competition = state.get("competition", {})
    sub_limit = competition.get("submission_limit", 5)
    if next_node == "submission_decider" and submissions_today >= sub_limit:
        return "strategy"

    return next_node or "strategy"
