from __future__ import annotations

import json
import subprocess
from pathlib import Path

HOOK = Path(__file__).resolve().parents[1] / "gladius" / "hooks" / "after_edit.sh"


def _run_hook(file_path: str) -> subprocess.CompletedProcess[str]:
    payload = json.dumps({"tool_input": {"file_path": file_path}})
    return subprocess.run(
        ["bash", str(HOOK)],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )


def test_blocks_editing_hook_file():
    result = _run_hook("/tmp/after_edit.sh")
    assert result.returncode == 2
    assert "Editing hook infrastructure files is not allowed." in result.stderr


def test_allows_valid_python_file(tmp_path):
    py_file = tmp_path / "ok.py"
    py_file.write_text("x = 1\n", encoding="utf-8")

    result = _run_hook(str(py_file))
    assert result.returncode == 0


def test_blocks_python_syntax_error(tmp_path):
    py_file = tmp_path / "bad.py"
    py_file.write_text("def broken(:\n", encoding="utf-8")

    result = _run_hook(str(py_file))
    assert result.returncode == 2
    assert "Syntax error detected" in result.stderr


def test_blocks_experiment_state_with_string_agent_values(tmp_path):
    state_file = tmp_path / "EXPERIMENT_STATE.json"
    state_file.write_text('{"agent_a":"oops"}', encoding="utf-8")

    result = _run_hook(str(state_file))
    assert result.returncode == 2
    assert "non-dict agent values" in result.stderr


def test_allows_valid_experiment_state(tmp_path):
    state_file = tmp_path / "EXPERIMENT_STATE.json"
    state_file.write_text('{"agent_a": {"status": "ok"}}', encoding="utf-8")

    result = _run_hook(str(state_file))
    assert result.returncode == 0
