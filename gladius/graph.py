from __future__ import annotations

import json
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from gladius.nodes.code.code_generator import code_generator_node
from gladius.nodes.code.code_reviewer import code_reviewer_node
from gladius.nodes.code.versioning_agent import versioning_node
from gladius.nodes.error_handler import error_handler_node
from gladius.nodes.execution.executor import executor_node
from gladius.nodes.execution.resource_manager import resource_manager_node
from gladius.nodes.execution.watchdog import watchdog_node
from gladius.nodes.router import router_node
from gladius.nodes.strategy.ensemble_agent import ensemble_node
from gladius.nodes.strategy.hypothesis_generator import hypothesis_node
from gladius.nodes.strategy.knowledge_extractor import knowledge_extractor_node
from gladius.nodes.strategy.lb_tracker import lb_tracker_node
from gladius.nodes.strategy.strategy_agent import strategy_node
from gladius.nodes.validation.notifier import notifier_node
from gladius.nodes.validation.submission_agent import submission_agent_node
from gladius.nodes.validation.submission_decider import submission_decider_node
from gladius.nodes.validation.validation_agent import validation_node
from gladius.state import GraphState


def build_competition_graph() -> StateGraph:
    """Builds the LangGraph competition graph with all 18 agent nodes."""
    graph = StateGraph(GraphState)

    # Register all nodes
    graph.add_node("router", router_node)
    graph.add_node("strategy", strategy_node)
    graph.add_node("hypothesis", hypothesis_node)
    graph.add_node("code_generator", code_generator_node)
    graph.add_node("code_reviewer", code_reviewer_node)
    graph.add_node("versioning_agent", versioning_node)
    graph.add_node("executor", executor_node)
    graph.add_node("watchdog", watchdog_node)
    graph.add_node("validation_agent", validation_node)
    graph.add_node("submission_decider", submission_decider_node)
    graph.add_node("submission_agent", submission_agent_node)
    graph.add_node("lb_tracker", lb_tracker_node)
    graph.add_node("notifier", notifier_node)
    graph.add_node("knowledge_extractor", knowledge_extractor_node)
    graph.add_node("error_handler", error_handler_node)
    graph.add_node("ensemble_agent", ensemble_node)
    graph.add_node("resource_manager", resource_manager_node)

    # Set entry point
    graph.set_entry_point("router")

    # Add edges using conditional routing from the router
    graph.add_conditional_edges(
        "router",
        router_node,
        {
            "strategy": "strategy",
            "hypothesis": "hypothesis",
            "code_generator": "code_generator",
            "code_reviewer": "code_reviewer",
            "versioning_agent": "versioning_agent",
            "executor": "executor",
            "watchdog": "watchdog",
            "validation_agent": "validation_agent",
            "submission_decider": "submission_decider",
            "submission_agent": "submission_agent",
            "lb_tracker": "lb_tracker",
            "notifier": "notifier",
            "knowledge_extractor": "knowledge_extractor",
            "error_handler": "error_handler",
            "ensemble_agent": "ensemble_agent",
        },
    )

    # Direct edges (non-conditional flows)
    graph.add_edge("strategy", "hypothesis")
    graph.add_edge("hypothesis", "code_generator")
    graph.add_edge("code_generator", "code_reviewer")
    graph.add_conditional_edges(
        "code_reviewer",
        lambda s: s.get("next_node", "hypothesis"),
        {
            "versioning_agent": "versioning_agent",
            "hypothesis": "hypothesis",
            "strategy": "strategy",
        },
    )
    graph.add_edge("versioning_agent", "executor")
    graph.add_edge("executor", "watchdog")
    graph.add_conditional_edges(
        "watchdog",
        lambda s: s.get("next_node", "knowledge_extractor"),
        {
            "validation_agent": "validation_agent",
            "knowledge_extractor": "knowledge_extractor",
        },
    )
    graph.add_edge("validation_agent", "submission_decider")
    graph.add_conditional_edges(
        "submission_decider",
        lambda s: s.get("next_node", "router"),
        {
            "submission_agent": "submission_agent",
            "router": "router",
        },
    )
    graph.add_edge("submission_agent", "lb_tracker")
    graph.add_conditional_edges(
        "lb_tracker",
        lambda s: s.get("next_node", "router"),
        {
            "router": "router",
            "notifier": "notifier",
        },
    )
    graph.add_edge("notifier", "router")
    graph.add_edge("knowledge_extractor", "router")
    graph.add_conditional_edges(
        "error_handler",
        lambda s: s.get("next_node", "strategy"),
        {
            "strategy": "strategy",
            "hypothesis": "hypothesis",
            "code_generator": "code_generator",
            "code_reviewer": "code_reviewer",
            "executor": "executor",
            "watchdog": "watchdog",
            "validation_agent": "validation_agent",
            "submission_agent": "submission_agent",
            "lb_tracker": "lb_tracker",
        },
    )
    graph.add_edge("ensemble_agent", "hypothesis")

    return graph


def create_initial_state(competition_config: dict) -> GraphState:
    """Creates the initial graph state for a new competition run."""
    return GraphState(
        competition=competition_config,
        current_experiment=None,
        experiment_status="pending",
        running_pid=None,
        run_id=None,
        oof_score=None,
        lb_score=None,
        gap_history=[],
        submissions_today=0,
        last_submission_time=None,
        directive=None,
        exploration_flag=True,
        consecutive_same_directive=0,
        session_summary=None,
        generated_script_path=None,
        reviewer_feedback=None,
        code_retry_count=0,
        next_node="strategy",
        error_message=None,
        node_retry_counts={},
        next_node_before_error=None,
    )


def main(competition_config_path: str = "competition.json"):
    """Entry point: loads competition config and runs the graph."""
    Path("state").mkdir(exist_ok=True)

    config_path = Path(competition_config_path)
    if config_path.exists():
        competition_config = json.loads(config_path.read_text())
    else:
        competition_config = {
            "name": "example-competition",
            "metric": "auc",
            "target": "target",
            "deadline": "2026-03-31",
            "days_remaining": 35,
            "submission_limit": 5,
        }

    checkpointer = MemorySaver()
    graph = build_competition_graph()
    app = graph.compile(checkpointer=checkpointer)

    initial_state = create_initial_state(competition_config)
    run_config = {
        "configurable": {"thread_id": competition_config.get("name", "gladius_run")}
    }

    for event in app.stream(initial_state, config=run_config):
        node_name = list(event.keys())[0] if event else "unknown"
        print(f"[gladius] Completed node: {node_name}")


if __name__ == "__main__":
    import sys

    cfg = sys.argv[1] if len(sys.argv) > 1 else "competition.json"
    main(cfg)
