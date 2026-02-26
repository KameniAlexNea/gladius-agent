import difflib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from gladius.state import GraphState

VERSIONS_DIR = Path("state/versions")
SCRIPTS_DIR = Path("state/scripts")


def versioning_node(state: GraphState) -> GraphState:
    script_path = state.get("generated_script_path", "")
    run_id = state.get("run_id", "run_001")
    directive = state.get("directive", {})
    spec = state.get("current_experiment", {})

    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(VERSIONS_DIR.glob("v*.json"))
    version_num = len(existing) + 1
    version_tag = f"v{version_num}"

    # Generate unified diff against parent version
    parent_version = spec.get("parent_version") if spec else None
    diff_text = _generate_diff(parent_version, script_path)

    metadata = {
        "version": version_tag,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parent_version": parent_version,
        "directive": directive,
        "what_changed": spec.get("changes", []) if spec else [],
        "reviewer_output": state.get("reviewer_feedback"),
        "script_path": script_path,
    }
    meta_path = VERSIONS_DIR / f"{version_tag}.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    # Write diff alongside metadata
    if diff_text:
        diff_path = VERSIONS_DIR / f"{version_tag}.diff"
        diff_path.write_text(diff_text)

    src = Path(script_path)
    if src.exists():
        versioned_script = SCRIPTS_DIR / f"{version_tag}.py"
        SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        versioned_script.write_text(src.read_text())

    try:
        subprocess.run(
            ["git", "add", str(script_path), str(meta_path)],
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                f"Add experiment {version_tag}: {directive.get('directive_type', 'unknown')}",
            ],
            capture_output=True,
            timeout=30,
        )
    except Exception:
        pass  # Git failures are non-fatal

    return {
        "run_id": version_tag,
        "next_node": "executor",
        "experiment_status": "queued",
    }


def _generate_diff(parent_version: "str | None", new_script_path: str) -> str:
    """Generate a unified diff between the parent version and the new script."""
    if not new_script_path or not Path(new_script_path).exists():
        return ""

    new_lines = Path(new_script_path).read_text().splitlines(keepends=True)

    if parent_version:
        parent_path = SCRIPTS_DIR / f"{parent_version}.py"
        if parent_path.exists():
            old_lines = parent_path.read_text().splitlines(keepends=True)
        else:
            old_lines = []
    else:
        old_lines = []

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"{parent_version or 'empty'}.py",
        tofile="pending.py",
        lineterm="",
    )
    return "\n".join(diff)
