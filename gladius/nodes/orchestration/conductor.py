"""Conductor agent - top-level orchestration."""
from gladius.state import GraphState


def conductor_node(state: GraphState) -> GraphState:
    """High-level conductor that monitors overall competition progress."""
    return {"next_node": state.get("next_node", "strategy")}
