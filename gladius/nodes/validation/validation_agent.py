import numpy as np
from pathlib import Path
from gladius.state import GraphState

OOF_DIR = Path("state/oof")


def validation_node(state: GraphState) -> GraphState:
    run_id = state.get("run_id", "")
    oof_path = OOF_DIR / f"{run_id}_oof.npy"

    if not oof_path.exists():
        return {
            "experiment_status": "failed",
            "error_message": "OOF file not found",
            "next_node": "knowledge_extractor",
        }

    try:
        oof = np.load(str(oof_path))
    except Exception as e:
        return {
            "experiment_status": "failed",
            "error_message": f"Failed to load OOF: {e}",
            "next_node": "knowledge_extractor",
        }

    issues = []

    if oof.ndim == 0 or len(oof) == 0:
        issues.append("OOF array is empty or scalar")

    if np.any(np.isnan(oof)):
        issues.append("OOF contains NaN values")

    if np.any(oof < 0) or np.any(oof > 1):
        issues.append(f"OOF values out of [0,1] range: min={oof.min():.4f}, max={oof.max():.4f}")

    if issues:
        return {
            "experiment_status": "failed",
            "error_message": "; ".join(issues),
            "next_node": "knowledge_extractor",
        }

    return {
        "experiment_status": "validated",
        "next_node": "submission_decider",
    }
