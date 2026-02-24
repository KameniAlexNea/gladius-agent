import numpy as np
from pathlib import Path
from gladius.state import GraphState

OOF_DIR = Path("state/oof")
CORRELATION_THRESHOLD = 0.97
MIN_BASE_MODELS = 3


def ensemble_node(state: GraphState) -> GraphState:
    oof_files = sorted(OOF_DIR.glob("*_oof.npy"))
    if len(oof_files) < MIN_BASE_MODELS:
        return {"next_node": "strategy"}

    oofs = []
    paths = []
    for f in oof_files:
        try:
            arr = np.load(str(f))
            oofs.append(arr)
            paths.append(f.name)
        except Exception:
            continue

    if len(oofs) < MIN_BASE_MODELS:
        return {"next_node": "strategy"}

    uncorrelated = _find_uncorrelated(oofs, CORRELATION_THRESHOLD)
    if len(uncorrelated) < MIN_BASE_MODELS:
        return {"next_node": "strategy"}

    blend_directive = {
        "directive_type": "ensemble",
        "target_model": "blend",
        "rationale": f"Blending {len(uncorrelated)} uncorrelated base models",
        "exploration_flag": False,
        "priority": 4,
        "base_model_paths": [paths[i] for i in uncorrelated],
    }
    return {"directive": blend_directive, "next_node": "hypothesis"}


def _find_uncorrelated(oofs: list, threshold: float) -> list:
    selected = [0]
    for i in range(1, len(oofs)):
        correlated = False
        for j in selected:
            try:
                a, b = oofs[i], oofs[j]
                min_len = min(len(a), len(b))
                r = np.corrcoef(a[:min_len], b[:min_len])[0, 1]
                if abs(r) >= threshold:
                    correlated = True
                    break
            except Exception:
                pass
        if not correlated:
            selected.append(i)
    return selected
