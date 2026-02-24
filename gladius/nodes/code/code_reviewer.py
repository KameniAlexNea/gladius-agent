import py_compile
import subprocess
import sys
from pathlib import Path

from gladius.state import GraphState


def code_reviewer_node(state: GraphState) -> GraphState:
    script_path = state.get("generated_script_path")
    if not script_path or not Path(script_path).exists():
        return {
            "reviewer_feedback": "Script not found",
            "next_node": "hypothesis",
        }
    issues = []
    # 1. Syntax check
    try:
        py_compile.compile(script_path, doraise=True)
    except py_compile.PyCompileError as e:
        issues.append(f"Syntax error: {e}")

    # 2. Pylint errors-only
    result = subprocess.run(
        [sys.executable, "-m", "pylint", "--errors-only", script_path],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        issues.append(f"Pylint: {(result.stdout + result.stderr).strip()}")

    # 3. No hardcoded paths (basic check)
    content = Path(script_path).read_text()
    for line_no, line in enumerate(content.splitlines(), 1):
        if "/home/" in line or "/root/" in line or "C:\\" in line:
            issues.append(f"line {line_no}: possible hardcoded path")

    if issues:
        feedback = "; ".join(issues)
        retry = state.get("code_retry_count", 0) + 1
        if retry >= 3:
            return {
                "reviewer_feedback": feedback,
                "code_retry_count": retry,
                "next_node": "strategy",
            }
        return {
            "reviewer_feedback": feedback,
            "code_retry_count": retry,
            "next_node": "hypothesis",
        }

    return {
        "reviewer_feedback": None,
        "next_node": "versioning_agent",
    }
