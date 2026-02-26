"""
LightGBM 5-fold stratified CV baseline on 20 numeric features.

Hypothesis: A well-regularised LightGBM model with 5-fold stratified
cross-validation provides a strong baseline (expected OOF AUC ~0.88–0.92)
on this 800-sample binary classification dataset with 20 numeric features.
Early stopping (rounds=50) prevents overfitting on the small training folds.
"""

from __future__ import annotations

import json
import os
import pathlib
import warnings
from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GLADIUS_DIR = ROOT / ".gladius"
GLADIUS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
OOF_PATH = GLADIUS_DIR / "oof_v1.npy"
SUB_PATH = GLADIUS_DIR / "sub_v1.csv"

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Hyper-parameters ───────────────────────────────────────────────────────────
LGBM_PARAMS: dict = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

N_FOLDS = 5
EARLY_STOPPING_ROUNDS = 50


def load_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load train and test CSVs; return features, target, test features, test ids."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    feature_cols = [c for c in train.columns if c.startswith("feature_")]
    target_col = "target"

    X_train = train[feature_cols].astype(np.float32)
    y_train = train[target_col].astype(np.int32)
    X_test = test[feature_cols].astype(np.float32)
    test_ids = test["id"]

    return X_train, y_train, X_test, test_ids


def train_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run 5-fold stratified CV with LightGBM.

    Returns
    -------
    oof_preds  : shape (n_train,)  – OOF probability predictions
    test_preds : shape (n_test,)   – averaged test probability predictions
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_preds = np.zeros(len(X), dtype=np.float64)
    test_preds = np.zeros(len(X_test), dtype=np.float64)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred

        fold_auc = roc_auc_score(y_val, val_pred)
        best_iter = model.best_iteration_
        print(f"  Fold {fold_idx}/{N_FOLDS}  |  best_iter={best_iter:4d}  |  val_AUC={fold_auc:.6f}")

        test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS

    return oof_preds, test_preds


def save_outputs(
    oof_preds: np.ndarray,
    test_preds: np.ndarray,
    test_ids: pd.Series,
    oof_score: float,
) -> None:
    """Persist OOF array and submission CSV."""
    np.save(OOF_PATH, oof_preds)

    sub = pd.DataFrame({"id": test_ids, "target": test_preds})
    sub.to_csv(SUB_PATH, index=False)


def main() -> None:
    print("=" * 60)
    print("solution_v1  |  LightGBM 5-fold stratified CV baseline")
    print("=" * 60)

    X_train, y_train, X_test, test_ids = load_data()
    print(f"\nTrain shape : {X_train.shape}  |  pos_rate={y_train.mean():.3f}")
    print(f"Test  shape : {X_test.shape}\n")

    oof_preds, test_preds = train_evaluate(X_train, y_train, X_test)

    oof_score = roc_auc_score(y_train, oof_preds)
    save_outputs(oof_preds, test_preds, test_ids, oof_score)

    print(f"\nOOF predictions  → {OOF_PATH}")
    print(f"Submission       → {SUB_PATH}")
    print(f"\nOOF_SCORE: {oof_score:.6f}")


if __name__ == "__main__":
    main()
