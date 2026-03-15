"""
Ensembling utilities: hill-climbing (greedy forward selection) and weighted blend.

Usage:
    from scripts.blend import hill_climb, optimised_blend, rank_average

Example:
    oof_preds  = {"lgbm": lgbm_oof,  "xgb": xgb_oof}
    test_preds = {"lgbm": lgbm_test, "xgb": xgb_test}
    oof_blend, test_blend, weights = hill_climb(oof_preds, test_preds, y_train)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Diversity check ─────────────────────────────────────────────────────────

def diversity_check(oof_preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Return pairwise OOF correlation matrix.
    Warn if any pair has r > 0.97 (little value from adding both).
    """
    df = pd.DataFrame(oof_preds)
    corr = df.corr().round(3)
    for i, c1 in enumerate(corr.columns):
        for j, c2 in enumerate(corr.columns):
            if j > i and corr.loc[c1, c2] > 0.97:
                print(f"⚠️  High correlation ({corr.loc[c1, c2]:.3f}): {c1} vs {c2}")
    return corr


# ── Hill-climbing ────────────────────────────────────────────────────────────

def hill_climb(
    oof_preds:  dict[str, np.ndarray],
    test_preds: dict[str, np.ndarray],
    y_train:    np.ndarray,
    metric_fn=None,
    direction:  str = "maximize",
    n_steps:    int = 100,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Greedy forward selection — adds the model that most improves OOF each step.
    A model can be added multiple times (implicit weighting).
    Recommended for ensembles of ≥ 3 models.

    Returns:
        (ensemble_oof, ensemble_test, normalised_weights)
    """
    if metric_fn is None:
        from sklearn.metrics import roc_auc_score
        metric_fn = roc_auc_score

    names   = list(oof_preds.keys())
    weights = {n: 0 for n in names}
    n_test  = len(next(iter(test_preds.values())))
    ensemble_oof  = np.zeros(len(y_train))
    ensemble_test = np.zeros(n_test)

    for step in range(n_steps):
        cmp = -np.inf if direction == "maximize" else np.inf
        best_name = None
        for name in names:
            cand_oof = (ensemble_oof * step + oof_preds[name]) / (step + 1)
            score    = metric_fn(y_train, cand_oof)
            better   = (score > cmp) if direction == "maximize" else (score < cmp)
            if better:
                cmp, best_name = score, name
        weights[best_name] += 1
        ensemble_oof  = (ensemble_oof  * step + oof_preds[best_name])  / (step + 1)
        ensemble_test = (ensemble_test * step + test_preds[best_name]) / (step + 1)
        if (step + 1) % 10 == 0:
            print(f"Step {step+1:3d}: OOF={cmp:.6f}  (added {best_name})")

    total = sum(weights.values())
    norm_weights = {k: v / total for k, v in weights.items()}
    print(f"\nFinal weights: { {k: round(v, 4) for k, v in norm_weights.items()} }")
    return ensemble_oof, ensemble_test, norm_weights


# ── Weighted Nelder-Mead blend ────────────────────────────────────────────────

def optimised_blend(
    oof_preds:  dict[str, np.ndarray],
    test_preds: dict[str, np.ndarray],
    y_train:    np.ndarray,
    metric_fn=None,
    direction:  str = "maximize",
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Optimise blend weights via scipy.optimize.minimize (Nelder-Mead).
    Recommended for 2–4 models.

    Returns:
        (opt_oof, opt_test, weights_dict)
    """
    if metric_fn is None:
        from sklearn.metrics import roc_auc_score
        metric_fn = roc_auc_score

    from scipy.optimize import minimize

    names       = list(oof_preds.keys())
    oof_matrix  = np.column_stack([oof_preds[n]  for n in names])
    test_matrix = np.column_stack([test_preds[n] for n in names])

    def neg_score(w: np.ndarray) -> float:
        w = np.abs(w) / (np.abs(w).sum() + 1e-12)
        score = metric_fn(y_train, oof_matrix @ w)
        return -score if direction == "maximize" else score

    x0     = np.ones(len(names)) / len(names)
    result = minimize(neg_score, x0, method="Nelder-Mead",
                      options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
    w      = np.abs(result.x) / np.abs(result.x).sum()

    opt_oof  = oof_matrix  @ w
    opt_test = test_matrix @ w
    weights  = dict(zip(names, w.tolist()))
    print(f"Optimised weights: { {k: round(v, 4) for k, v in weights.items()} }")
    print(f"OOF score: {metric_fn(y_train, opt_oof):.6f}")
    return opt_oof, opt_test, weights


# ── Rank averaging ─────────────────────────────────────────────────────────────

def rank_average(preds_dict: dict[str, np.ndarray]) -> np.ndarray:
    """
    Scale-invariant blend — useful when model outputs are on different scales.
    Each array is rank-transformed to [0, 1] then averaged.
    """
    from scipy.stats import rankdata
    ranks = [rankdata(p) / len(p) for p in preds_dict.values()]
    return np.mean(ranks, axis=0)
