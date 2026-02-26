from pathlib import Path

import numpy as np

from gladius.state import GraphState

OOF_DIR = Path("state/oof")
LABELS_PATH = Path("state/labels.npy")


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
        issues.append(
            f"OOF values out of [0,1] range: min={oof.min():.4f}, max={oof.max():.4f}"
        )

    # Metric consistency: recompute OOF score and compare to script-reported value
    metric_issue = _check_metric_consistency(oof, state)
    if metric_issue:
        issues.append(metric_issue)

    if issues:
        return {
            "experiment_status": "failed",
            "error_message": "; ".join(issues),
            "next_node": "knowledge_extractor",
        }

    # Compute OOF score locally and update best_oof
    oof_score = _compute_oof_score(oof, state)
    best_oof = state.get("best_oof")
    updates: dict = {
        "experiment_status": "validated",
        "next_node": "submission_decider",
    }
    if oof_score is not None:
        updates["oof_score"] = oof_score
        if best_oof is None or oof_score > best_oof:
            updates["best_oof"] = oof_score
    return updates


def _compute_oof_score(oof: np.ndarray, state: GraphState) -> "float | None":
    """Recompute OOF metric from predictions + ground-truth labels."""
    if not LABELS_PATH.exists():
        return state.get("oof_score")  # fall back to script-reported score
    try:
        labels = np.load(str(LABELS_PATH))
        metric = state.get("competition", {}).get("metric", "auc")
        return _evaluate_metric(labels, oof, metric)
    except Exception:
        return state.get("oof_score")


def _evaluate_metric(labels: np.ndarray, preds: np.ndarray, metric: str) -> float:
    """Evaluate a competition metric. Supports common Kaggle metrics."""
    min_len = min(len(labels), len(preds))
    labels, preds = labels[:min_len], preds[:min_len]

    if metric == "auc":
        # Manual AUC-ROC (avoids sklearn dependency)
        return _auc_roc(labels, preds)
    if metric in ("logloss", "log_loss"):
        eps = 1e-15
        p = np.clip(preds, eps, 1 - eps)
        return -float(np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))
    if metric == "rmse":
        return float(np.sqrt(np.mean((labels - preds) ** 2)))
    if metric == "mae":
        return float(np.mean(np.abs(labels - preds)))
    # Default: return mean prediction as a placeholder
    return float(np.mean(preds))


def _auc_roc(labels: np.ndarray, preds: np.ndarray) -> float:
    """Compute AUC-ROC without sklearn using the Wilcoxon–Mann–Whitney statistic."""
    pos = preds[labels == 1]
    neg = preds[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Efficient vectorized U-statistic
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def _check_metric_consistency(oof: np.ndarray, state: GraphState) -> "str | None":
    """Compare script-reported OOF score to independently recomputed one."""
    reported = state.get("oof_score")
    if reported is None or not LABELS_PATH.exists():
        return None
    try:
        labels = np.load(str(LABELS_PATH))
        metric = state.get("competition", {}).get("metric", "auc")
        recomputed = _evaluate_metric(labels, oof, metric)
        tolerance = 0.005
        if abs(recomputed - reported) > tolerance:
            return (
                f"Metric mismatch: script reported {reported:.5f} "
                f"but recomputed {metric}={recomputed:.5f} (delta={abs(recomputed - reported):.5f})"
            )
    except Exception:
        pass
    return None
