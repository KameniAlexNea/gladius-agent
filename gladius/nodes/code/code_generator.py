import re
from pathlib import Path

from gladius.state import GraphState

SCRIPTS_DIR = Path("state/scripts")
STAGING_PATH = SCRIPTS_DIR / "pending.py"

# Maximum number of context lines around an issue for targeted LLM correction
_CORRECTION_CONTEXT_LINES = 20


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
    """Call LLM to fix issues identified by the code reviewer.

    Uses targeted context around reported line numbers (±20 lines) instead of
    sending the entire script, reducing token cost and improving focus.
    Falls back to full-script mode when no line numbers are found in feedback.
    """
    try:
        from gladius.utils.llm import call_llm

        line_nums = _extract_line_numbers(feedback)
        lines = script.splitlines()

        if line_nums:
            # Build targeted context: ±_CORRECTION_CONTEXT_LINES around each issue
            snippets = []
            for ln in line_nums:
                start = max(1, ln - _CORRECTION_CONTEXT_LINES)
                end = min(len(lines), ln + _CORRECTION_CONTEXT_LINES)
                snippet_lines = lines[start - 1 : end]
                header = f"--- lines {start}-{end} ---"
                snippets.append(f"{header}\n" + "\n".join(
                    f"{i}: {l}" for i, l in enumerate(snippet_lines, start)
                ))
            context = "\n\n".join(snippets)
            prompt = (
                f"Fix the following issues in these Python ML script excerpts.\n\n"
                f"Issues identified by code reviewer:\n{feedback}\n\n"
                f"Relevant code sections (with line numbers):\n{context}\n\n"
                f"Return a JSON object with key \"replacements\": a list of objects, each with "
                f"\"start_line\" (int), \"end_line\" (int), and \"new_code\" (string) "
                f"representing the corrected lines. Only fix the reported issues."
            )
            result = call_llm(prompt, schema={"replacements": [{"start_line": 1, "end_line": 1, "new_code": ""}]})
            replacements = result.get("replacements", [])
            if replacements:
                return _apply_replacements(lines, replacements)
            return script
        else:
            # No line numbers in feedback: fall back to full-script approach
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


def _extract_line_numbers(feedback: str) -> list[int]:
    """Extract line numbers mentioned in reviewer feedback (e.g. 'line 42:', 'L42')."""
    import re as _re

    matches = _re.findall(r"(?:line\s+|L)(\d+)", feedback, _re.IGNORECASE)
    return sorted(set(int(m) for m in matches))


def _apply_replacements(lines: list[str], replacements: list[dict]) -> str:
    """Apply line-range replacements to the script, processing from bottom to top."""
    # Sort descending so earlier replacements don't shift later line numbers
    sorted_reps = sorted(replacements, key=lambda r: r.get("start_line", 0), reverse=True)
    for rep in sorted_reps:
        start = rep.get("start_line", 0)
        end = rep.get("end_line", start)
        new_code = rep.get("new_code", "")
        if start < 1 or end < start:
            continue
        new_lines = new_code.splitlines() if new_code else []
        lines[start - 1 : end] = new_lines
    return "\n".join(lines)


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
    if not snippet.strip():
        return script

    # Feature deduplication: check if any function defined in the snippet
    # already exists in the script
    duplicate = _check_feature_duplicate(script, snippet)
    if duplicate:
        return script  # skip — feature already present

    marker = "# --- FEATURES END ---"
    if marker in script:
        return script.replace(marker, f"{snippet}\n{marker}")
    return script + f"\n{snippet}"


def _check_feature_duplicate(script: str, snippet: str) -> bool:
    """Return True if any function/variable defined in snippet already exists in script."""
    import ast as _ast

    # Extract names defined in the snippet
    try:
        snippet_tree = _ast.parse(snippet)
    except SyntaxError:
        return False

    snippet_names = set()
    for node in _ast.walk(snippet_tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            snippet_names.add(node.name)
        elif isinstance(node, _ast.Assign):
            for target in node.targets:
                if isinstance(target, _ast.Name):
                    snippet_names.add(target.id)

    if not snippet_names:
        return False

    # Check if any of those names already exist in the script
    try:
        script_tree = _ast.parse(script)
    except SyntaxError:
        return False

    for node in _ast.walk(script_tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            if node.name in snippet_names:
                return True
        elif isinstance(node, _ast.Assign):
            for target in node.targets:
                if isinstance(target, _ast.Name) and target.id in snippet_names:
                    return True
    return False


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
