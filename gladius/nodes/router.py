from gladius.state import ExperimentStatus, GraphState


def router_node(state: GraphState) -> dict:
    """Router node: resolves next_node in state based on status hard-overrides and budget."""
    return {"next_node": _resolve_next_node(state)}


def _resolve_next_node(state: GraphState) -> str:
    """Returns the target node name. Used by both router_node and tests."""
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

    submissions_today = state.get("submissions_today", 0)
    competition = state.get("competition", {})
    sub_limit = competition.get("submission_limit", 5)
    if next_node == "submission_decider" and submissions_today >= sub_limit:
        return "strategy"

    return next_node or "strategy"
