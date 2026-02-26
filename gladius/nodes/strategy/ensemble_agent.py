from pathlib import Path

import numpy as np

from gladius.state import GraphState

OOF_DIR = Path("state/oof")
LABELS_PATH = Path("state/labels.npy")
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

    selected_oofs = [oofs[i] for i in uncorrelated]
    selected_paths = [paths[i] for i in uncorrelated]

    # Optimize blend weights using Nelder-Mead
    weights = _optimize_weights(selected_oofs, state)

    blend_directive = {
        "directive_type": "ensemble",
        "target_model": "blend",
        "rationale": f"Blending {len(uncorrelated)} uncorrelated base models",
        "exploration_flag": False,
        "priority": 4,
        "base_model_paths": selected_paths,
        "blend_weights": weights.tolist() if weights is not None else None,
    }
    return {"directive": blend_directive, "next_node": "hypothesis"}


def _optimize_weights(
    oofs: list[np.ndarray], state: GraphState
) -> "np.ndarray | None":
    """Optimise ensemble weights via Nelder-Mead on the competition metric.

    Falls back to uniform weights if scipy is unavailable or labels are missing.
    """
    n = len(oofs)
    if n == 0:
        return None

    # Load ground-truth labels
    if not LABELS_PATH.exists():
        return np.ones(n) / n  # uniform fallback

    try:
        labels = np.load(str(LABELS_PATH))
    except Exception:
        return np.ones(n) / n

    # Align lengths
    min_len = min(len(labels), *(len(o) for o in oofs))
    labels_trimmed = labels[:min_len]
    oofs_trimmed = [o[:min_len] for o in oofs]
    stacked = np.column_stack(oofs_trimmed)  # (min_len, n)

    metric = state.get("competition", {}).get("metric", "auc")

    def objective(raw_weights):
        # Softmax to enforce w_i >= 0, sum = 1
        w = np.exp(raw_weights) / np.exp(raw_weights).sum()
        blend = stacked @ w
        score = _evaluate_metric(labels_trimmed, blend, metric)
        # Nelder-Mead minimises, so negate for metrics where higher is better
        if metric in ("auc",):
            return -score
        return score  # rmse, logloss, mae: lower is better

    try:
        from scipy.optimize import minimize

        x0 = np.zeros(n)  # uniform in softmax space
        result = minimize(objective, x0, method="Nelder-Mead",
                          options={"maxiter": 500, "xatol": 1e-5, "fatol": 1e-7})
        w = np.exp(result.x) / np.exp(result.x).sum()
        return w
    except ImportError:
        return np.ones(n) / n
    except Exception:
        return np.ones(n) / n


def _evaluate_metric(labels: np.ndarray, preds: np.ndarray, metric: str) -> float:
    """Evaluate a competition metric (mirrors validation_agent)."""
    if metric == "auc":
        pos = preds[labels == 1]
        neg = preds[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        return float(
            np.mean(pos[:, None] > neg[None, :])
            + 0.5 * np.mean(pos[:, None] == neg[None, :])
        )
    if metric in ("logloss", "log_loss"):
        eps = 1e-15
        p = np.clip(preds, eps, 1 - eps)
        return -float(np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))
    if metric == "rmse":
        return float(np.sqrt(np.mean((labels - preds) ** 2)))
    if metric == "mae":
        return float(np.mean(np.abs(labels - preds)))
    return float(np.mean(preds))


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
