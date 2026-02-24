from gladius.state import GraphState
from gladius.utils.llm import call_llm
from gladius.utils.context_builder import ContextBuilder

SPEC_SCHEMA = {
    "parent_version": "v41",
    "changes": [],
    "estimated_runtime_multiplier": 1.0,
    "rationale": "..."
}


def hypothesis_node(state: GraphState) -> GraphState:
    directive = state.get("directive")
    if not directive:
        return {"next_node": "strategy", "error_message": "No directive available"}
    context = ContextBuilder.build_hypothesis_context(state)
    prompt = _build_prompt(directive, context)
    try:
        spec = call_llm(prompt, schema=SPEC_SCHEMA)
        return {"current_experiment": spec, "next_node": "code_generator"}
    except Exception as e:
        return {
            "error_message": str(e),
            "next_node_before_error": "hypothesis",
            "next_node": "error_handler",
        }


def _build_prompt(directive: dict, context: dict) -> str:
    return f"""
You are the Hypothesis Generator. Given a strategy directive, produce a concrete experiment spec.
Directive: {directive}
Context: {context}

Respond with valid JSON:
{{
  "parent_version": "<version>",
  "changes": [
    {{"type": "param_change", "param": "<name>", "old": "<val>", "new": "<val>"}},
    {{"type": "feature_add", "feature_name": "<name>", "code_snippet": "<code>"}},
    {{"type": "feature_remove", "feature_name": "<name>"}}
  ],
  "estimated_runtime_multiplier": 1.0,
  "rationale": "..."
}}
"""
