from gladius.state import GraphState


def error_handler_node(state: GraphState) -> GraphState:
    node = state.get("next_node_before_error", "strategy")
    retries = state.get("node_retry_counts", {}).get(node, 0)
    if retries < 3:
        return {
            "node_retry_counts": {
                **state.get("node_retry_counts", {}),
                node: retries + 1,
            },
            "next_node": node,
        }
    else:
        _notify_telegram(state.get("error_message", ""), node)
        return {"next_node": "strategy", "error_message": None}


def _notify_telegram(msg: str, node: str):
    pass  # implemented by Notifier agent
