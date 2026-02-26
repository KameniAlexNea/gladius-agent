"""AST-based code reading and analysis utilities."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


def read_function(path: "str | Path", func_name: str) -> str:
    """Extract source lines for a named function from a Python file."""
    source = Path(path).read_text()
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise KeyError(f"{func_name!r} not found in {path}")


def read_lines(path: "str | Path", start: int, end: int) -> str:
    """Extract a line range (1-indexed, inclusive) from a file."""
    lines = Path(path).read_text().splitlines()
    return "\n".join(lines[start - 1 : end])


def list_functions(path: "str | Path") -> list[str]:
    """Return the names of all top-level functions defined in a file."""
    tree = ast.parse(Path(path).read_text())
    return [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]


def validate_syntax(code: str) -> list[str]:
    """Return a list of syntax error messages, or empty list if valid."""
    try:
        ast.parse(code)
        return []
    except SyntaxError as e:
        return [f"SyntaxError at line {e.lineno}: {e.msg}"]


def check_imports(code: str) -> list[str]:
    """Return a list of missing import names found in the code."""
    issues = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
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
