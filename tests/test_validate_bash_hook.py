from __future__ import annotations

import json
import subprocess
from pathlib import Path

HOOK = Path(__file__).resolve().parents[1] / "gladius" / "hooks" / "validate_bash.sh"


def _run_hook(command: str) -> subprocess.CompletedProcess[str]:
    payload = json.dumps({"tool_input": {"command": command}})
    return subprocess.run(
        ["bash", str(HOOK)],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )


def test_blocks_killall():
    result = _run_hook("killall python")
    assert result.returncode == 2
    assert "killall" in result.stderr


def test_blocks_pkill_f_broad_pattern():
    result = _run_hook('pkill -f "python"')
    assert result.returncode == 2
    assert "specific script filename" in result.stderr


def test_allows_pkill_f_specific_script_name():
    result = _run_hook('pkill -f "train.py" 2>/dev/null || true')
    assert result.returncode == 0


def test_allows_pkill_f_specific_script_path():
    result = _run_hook('pkill -f "python scripts/hpo.py" 2>/dev/null || true')
    assert result.returncode == 0


def test_allows_pkill_f_uv_run_train_script():
    # Regression: pkill -f targeting a train script must not be blocked by the
    # train-redirect rule (the pattern contains "train.py" but it's a kill, not a run).
    result = _run_hook('pkill -f "uv run python scripts/train.py" 2>/dev/null; sleep 2; echo "Killed existing processes"')
    assert result.returncode == 0
