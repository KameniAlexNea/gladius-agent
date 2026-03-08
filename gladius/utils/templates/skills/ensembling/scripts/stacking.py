"""
Stacking: train a meta-learner on OOF predictions from base models.

Usage:
    meta_oof, meta_tst = stack(oof_preds, test_preds, y_train)
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


def stack(
    oof_preds: dict,
    test_preds: dict,
    y_train: np.ndarray,
    task: str = "classification",  # "classification" or "regression"
    metric_fn=roc_auc_score,
    n_folds: int = 5,
):
    names = list(oof_preds.keys())
    X_meta_train = np.column_stack([oof_preds[n] for n in names])
    X_meta_test = np.column_stack([test_preds[n] for n in names])

    if task == "classification":
        meta = LogisticRegression(C=1.0, max_iter=2000)
        cv_scores = cross_val_score(
            meta,
            X_meta_train,
            y_train,
            cv=StratifiedKFold(n_folds, shuffle=True, random_state=42),
            scoring="roc_auc",
        )
        print(f"Meta-learner CV: {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")
        meta.fit(X_meta_train, y_train)
        meta_tst = meta.predict_proba(X_meta_test)[:, 1]
    else:
        meta = Ridge(alpha=1.0)
        cv_scores = cross_val_score(
            meta,
            X_meta_train,
            y_train,
            cv=n_folds,
            scoring="neg_root_mean_squared_error",
        )
        print(f"Meta-learner CV RMSE: {-cv_scores.mean():.6f} ± {cv_scores.std():.6f}")
        meta.fit(X_meta_train, y_train)
        meta_tst = meta.predict(X_meta_test)

    return meta_tst
