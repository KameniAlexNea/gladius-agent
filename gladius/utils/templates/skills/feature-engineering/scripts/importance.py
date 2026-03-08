"""
SHAP-based feature importance and incremental pruning.

Prerequisites: a trained LightGBM/XGBoost model and X_train, y_train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score  # replace with competition metric


def shap_importance(model, X_train: pd.DataFrame) -> pd.Series:
    """
    Compute mean |SHAP| per feature using TreeExplainer.
    Returns a Series sorted descending.
    """
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_train)
    # Binary classification: shap_values returns list [class0, class1]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    mean_abs = pd.Series(
        np.abs(shap_vals).mean(axis=0), index=X_train.columns
    ).sort_values(ascending=False)
    print("Top 30 features by mean |SHAP|:")
    print(mean_abs.head(30).to_string())
    return mean_abs


def permutation_importance_ranking(model, X_val: pd.DataFrame, y_val, n_repeats: int = 10) -> pd.Series:
    """Model-agnostic importance via permutation (slower, more general)."""
    result = permutation_importance(
        model, X_val, y_val, n_repeats=n_repeats, random_state=42, scoring="roc_auc"
    )
    perm = pd.Series(result.importances_mean, index=X_val.columns)
    print("Top 20 by permutation importance:")
    print(perm.sort_values(ascending=False).head(20).to_string())
    return perm


def incremental_pruning(
    X_base: pd.DataFrame,
    y: pd.Series,
    df_full: pd.DataFrame,
    candidate_features: list[str],
    evaluate_cv_fn,          # callable(X, y) -> float
    min_delta: float = 0.0005,
) -> tuple[list[str], float]:
    """
    Greedily add candidate features; keep only those improving OOF by >= min_delta.

    evaluate_cv_fn: a function that takes (X, y) and returns the OOF score.

    Returns:
        (features_to_keep, final_baseline_score)
    """
    baseline_score = evaluate_cv_fn(X_base, y)
    features_to_keep = list(X_base.columns)
    print(f"Baseline OOF: {baseline_score:.6f}")

    for feat in candidate_features:
        X_try = X_base.copy()
        X_try[feat] = df_full[feat]
        score = evaluate_cv_fn(X_try, y)
        delta = score - baseline_score
        print(f"  {feat}: delta={delta:+.5f}", end="")
        if delta >= min_delta:
            features_to_keep.append(feat)
            X_base = X_try
            baseline_score = score
            print(f"  → KEPT (baseline: {baseline_score:.6f})")
        else:
            print("  → DROPPED")

    return features_to_keep, baseline_score
