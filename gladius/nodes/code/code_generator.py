import re
from pathlib import Path

from gladius.state import GraphState

SCRIPTS_DIR = Path("state/scripts")
STAGING_PATH = SCRIPTS_DIR / "pending.py"


def code_generator_node(state: GraphState) -> GraphState:
    spec = state.get("current_experiment")
    if not spec:
        return {"next_node": "strategy", "error_message": "No experiment spec"}
    try:
        parent_version = spec.get("parent_version", "v0")
        parent_script = _load_parent_script(parent_version)
        modified = _apply_changes(parent_script, spec.get("changes", []), state)
        reviewer_feedback = state.get("reviewer_feedback")
        if reviewer_feedback:
            modified = _apply_llm_correction(modified, reviewer_feedback, spec)
        output_path = STAGING_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(modified)
        return {
            "generated_script_path": str(output_path),
            "next_node": "code_reviewer",
        }
    except Exception as e:
        return {
            "error_message": str(e),
            "next_node_before_error": "code_generator",
            "next_node": "error_handler",
        }


def _apply_llm_correction(script: str, feedback: str, spec: dict) -> str:
    """Call LLM to fix issues identified by the code reviewer."""
    try:
        from gladius.utils.llm import call_llm

        prompt = (
            f"Fix the following issues in this Python ML script.\n\n"
            f"Issues identified by code reviewer:\n{feedback}\n\n"
            f"Current script:\n```python\n{script}\n```\n\n"
            f"Return a JSON object with key \"fixed_script\" containing the corrected Python code. "
            f"Only fix the reported issues. Do not change anything else."
        )
        result = call_llm(prompt, schema={"fixed_script": "..."})
        return result.get("fixed_script", script)
    except Exception:
        return script  # graceful fallback — reviewer will re-check


def _load_parent_script(version: str) -> str:
    path = SCRIPTS_DIR / f"{version}.py"
    if path.exists():
        return path.read_text()
    return _default_template()


def _apply_changes(script: str, changes: list, state: GraphState) -> str:
    for change in changes:
        change_type = change.get("type")
        if change_type == "param_change":
            script = _apply_param_change(script, change)
        elif change_type == "feature_add":
            script = _apply_feature_add(script, change, state)
        elif change_type == "feature_remove":
            script = _apply_feature_remove(script, change)
    return script


def _apply_param_change(script: str, change: dict) -> str:
    param = change["param"]
    new_val = change["new"]
    pattern = rf"({re.escape(param)}\s*=\s*)[^\n,\)]*"
    replacement = rf"\g<1>{new_val}"
    return re.sub(pattern, replacement, script)


def _apply_feature_add(script: str, change: dict, state: GraphState) -> str:
    snippet = change.get("code_snippet", "")
    marker = "# --- FEATURES END ---"
    if marker in script:
        return script.replace(marker, f"{snippet}\n{marker}")
    return script + f"\n{snippet}"


def _apply_feature_remove(script: str, change: dict) -> str:
    feature_name = change.get("feature_name", "")
    lines = script.splitlines()
    filtered = [line for line in lines if feature_name not in line]
    return "\n".join(filtered)


def _default_template() -> str:
    return """import pandas as pd
import numpy as np

# --- FEATURES START ---
# --- FEATURES END ---

def train():
    pass

if __name__ == "__main__":
    train()
"""
