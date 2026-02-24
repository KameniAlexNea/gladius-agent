from gladius.state import GraphState
from gladius.utils.llm import call_llm
from gladius.utils.context_builder import ContextBuilder

DIRECTIVE_SCHEMA = {
    "directive_type": "tune_existing | new_features | new_model_type | ensemble | seed_average",
    "target_model": "catboost | lgbm | xgboost | nn | blend",
    "rationale": "one sentence",
    "exploration_flag": True,
    "priority": 3
}


def strategy_node(state: GraphState) -> GraphState:
    context = ContextBuilder.build_strategy_context(state)
    exploration_flag = _compute_exploration_flag(state)
    prompt = _build_prompt(context, exploration_flag)
    try:
        directive = call_llm(prompt, schema=DIRECTIVE_SCHEMA)
        directive["exploration_flag"] = exploration_flag
        return {
            "directive": directive,
            "exploration_flag": exploration_flag,
            "consecutive_same_directive": _count_same(state, directive),
            "next_node": "hypothesis",
        }
    except Exception as e:
        return {
            "error_message": str(e),
            "next_node_before_error": "strategy",
            "next_node": "error_handler",
        }


def _compute_exploration_flag(state: GraphState) -> bool:
    days_remaining = state.get("competition", {}).get("days_remaining", 10)
    if days_remaining < 2:
        return False
    consecutive = state.get("consecutive_same_directive", 0)
    if consecutive >= 5:
        return True
    return state.get("exploration_flag", True)


def _count_same(state: GraphState, new_directive: dict) -> int:
    old = state.get("directive", {})
    if old and old.get("directive_type") == new_directive.get("directive_type"):
        return state.get("consecutive_same_directive", 0) + 1
    return 0


def _build_prompt(context: dict, exploration_flag: bool) -> str:
    return f"""
You are the Strategy Agent for an ML competition system.
Competition context: {context}
exploration_flag={exploration_flag}

Based on the context, decide the next experiment. Respond with valid JSON matching this schema:
{{
  "directive_type": "tune_existing | new_features | new_model_type | ensemble | seed_average",
  "target_model": "catboost | lgbm | xgboost | nn | blend",
  "rationale": "one sentence",
  "exploration_flag": true | false,
  "priority": 1-5
}}
"""
