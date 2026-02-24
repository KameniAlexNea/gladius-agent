import subprocess
import json
from pathlib import Path
from datetime import datetime, timezone
from gladius.state import GraphState

VERSIONS_DIR = Path("state/versions")


def versioning_node(state: GraphState) -> GraphState:
    script_path = state.get("generated_script_path", "")
    run_id = state.get("run_id", "run_001")
    directive = state.get("directive", {})
    spec = state.get("current_experiment", {})

    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(VERSIONS_DIR.glob("v*.json"))
    version_num = len(existing) + 1
    version_tag = f"v{version_num}"

    metadata = {
        "version": version_tag,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parent_version": spec.get("parent_version") if spec else None,
        "directive": directive,
        "what_changed": spec.get("changes", []) if spec else [],
        "reviewer_output": state.get("reviewer_feedback"),
        "script_path": script_path,
    }
    meta_path = VERSIONS_DIR / f"{version_tag}.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    src = Path(script_path)
    if src.exists():
        versioned_script = Path("state/scripts") / f"{version_tag}.py"
        versioned_script.write_text(src.read_text())

    try:
        subprocess.run(["git", "add", str(script_path), str(meta_path)],
                      capture_output=True, timeout=30)
        subprocess.run(
            ["git", "commit", "-m", f"Add experiment {version_tag}: {directive.get('directive_type', 'unknown')}"],
            capture_output=True, timeout=30
        )
    except Exception:
        pass  # Git failures are non-fatal

    return {"run_id": version_tag, "next_node": "executor", "experiment_status": "queued"}
