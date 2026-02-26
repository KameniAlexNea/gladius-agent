import ast
import importlib.util
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

from gladius.state import GraphState

SANDBOX_TIMEOUT_SECS = 60
SANDBOX_SAMPLE_FRACTION = 0.01  # 1% of data


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

    # 2. AST-based import availability check
    content = Path(script_path).read_text()
    issues.extend(_check_imports(content))

    # 3. No hardcoded paths (basic check)
    for line_no, line in enumerate(content.splitlines(), 1):
        if "/home/" in line or "/root/" in line or "C:\\" in line:
            issues.append(f"line {line_no}: possible hardcoded path")

    # 4. Sandbox test run (only if no static issues found)
    if not issues:
        sandbox_issues = _sandbox_run(script_path)
        issues.extend(sandbox_issues)

    if issues:
        feedback = "; ".join(issues)
        retry = state.get("code_retry_count", 0) + 1
        if retry >= 3:
            return {
                "reviewer_feedback": feedback,
                "code_retry_count": retry,
                "next_node": "strategy",
            }
        if retry == 2:
            return {
                "reviewer_feedback": feedback,
                "code_retry_count": retry,
                "next_node": "hypothesis",
            }
        # retry < 2: send feedback back to code_generator for correction
        return {
            "reviewer_feedback": feedback,
            "code_retry_count": retry,
            "next_node": "code_generator",
        }

    return {
        "reviewer_feedback": None,
        "next_node": "versioning_agent",
    }


def _sandbox_run(script_path: str) -> list[str]:
    """Execute the script in a controlled sandbox with a small data sample.

    Injects GLADIUS_SANDBOX=1 and GLADIUS_SAMPLE_FRACTION env vars so
    well-behaved training scripts can skip heavy computation and use
    only a fraction of the data.
    """
    issues = []
    import os

    env = os.environ.copy()
    env["GLADIUS_SANDBOX"] = "1"
    env["GLADIUS_SAMPLE_FRACTION"] = str(SANDBOX_SAMPLE_FRACTION)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            timeout=SANDBOX_TIMEOUT_SECS,
            env=env,
            text=True,
        )
        if result.returncode != 0:
            stderr_tail = (result.stderr or "")[-500:]
            issues.append(f"Sandbox run failed (exit {result.returncode}): {stderr_tail}")
    except subprocess.TimeoutExpired:
        issues.append(f"Sandbox run timed out after {SANDBOX_TIMEOUT_SECS}s")
    except Exception as e:
        issues.append(f"Sandbox run error: {e}")

    return issues


def _check_imports(content: str) -> list[str]:
    """Check that all imports in the script are available."""
    issues = []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []  # already caught by py_compile
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if importlib.util.find_spec(top) is None:
                    issues.append(f"import not found: {top}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if importlib.util.find_spec(top) is None:
                issues.append(f"import not found: {top}")
    return issues
