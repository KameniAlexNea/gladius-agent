"""
Ensembling: weighted blend, rank average, and diversity check.

Prerequisites:
    oof_preds  = {"model_name": np.array of shape (n_train,), ...}
    test_preds = {"model_name": np.array of shape (n_test,), ...}
    y_train    = np.array of true training labels
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score  # replace with competition metric


def diversity_check(oof_preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Return pairwise correlation matrix. Pairs > 0.97 add minimal ensemble value."""
    corr = pd.DataFrame(oof_preds).corr()
    print("Pairwise OOF correlation:")
    print(corr.to_string())
    high = [(a, b) for a in corr.columns for b in corr.columns
            if a < b and corr.loc[a, b] > 0.97]
    if high:
        print(f"\n⚠️  High-correlation pairs (> 0.97): {high}")
    return corr


def simple_average(oof_preds: dict, test_preds: dict, y_train, metric_fn=roc_auc_score):
    names = list(oof_preds.keys())
    oof  = np.column_stack([oof_preds[n] for n in names]).mean(axis=1)
    test = np.column_stack([test_preds[n] for n in names]).mean(axis=1)
    print(f"Simple average OOF: {metric_fn(y_train, oof):.6f}")
    return oof, test


def optimised_blend(
    oof_preds: dict,
    test_preds: dict,
    y_train,
    metric_fn=roc_auc_score,
    maximize: bool = True,
):
    """Nelder-Mead weight optimisation on OOF predictions."""
    names  = list(oof_preds.keys())
    y_oof  = np.column_stack([oof_preds[n] for n in names])
    y_tst  = np.column_stack([test_preds[n] for n in names])

    def neg_score(weights):
        w = np.abs(weights) / (np.abs(weights).sum() + 1e-12)
        pred = (y_oof * w).sum(axis=1)
        s = metric_fn(y_train, pred)
        return -s if maximize else s

    init_w = np.ones(len(names)) / len(names)
    result = minimize(neg_score, init_w, method="Nelder-Mead",
                      options={"maxiter": 5000, "xatol": 1e-6})
    opt_w  = np.abs(result.x) / np.abs(result.x).sum()
    opt_oof = (y_oof * opt_w).sum(axis=1)
    opt_tst = (y_tst * opt_w).sum(axis=1)

    score = metric_fn(y_train, opt_oof)
    print(f"Optimised blend OOF: {score:.6f}")
    for name, w in zip(names, opt_w):
        print(f"  {name}: {w:.4f}")

    return opt_oof, opt_tst, dict(zip(names, opt_w))


def rank_average(oof_preds: dict, test_preds: dict, y_train, metric_fn=roc_auc_score):
    """Scale-invariant blending via rank normalisation."""
    names = list(oof_preds.keys())
    n_train = len(next(iter(oof_preds.values())))
    n_test  = len(next(iter(test_preds.values())))

    rank_oof = np.column_stack(
        [rankdata(oof_preds[n]) / n_train for n in names]
    ).mean(axis=1)
    rank_tst = np.column_stack(
        [rankdata(test_preds[n]) / n_test for n in names]
    ).mean(axis=1)

    print(f"Rank-average OOF: {metric_fn(y_train, rank_oof):.6f}")
    return rank_oof, rank_tst


def save_submission(test_df: "pd.DataFrame", preds: np.ndarray, path: str = "submission_ensemble.csv"):
    sub = test_df[["id"]].copy() if "id" in test_df.columns else test_df.iloc[:, :1].copy()
    sub["target"] = preds  # replace with actual prediction column name
    assert not np.isnan(preds).any(), "NaN in ensemble predictions!"
    sub.to_csv(path, index=False)
    print(f"Ensemble submission saved to {path}")
