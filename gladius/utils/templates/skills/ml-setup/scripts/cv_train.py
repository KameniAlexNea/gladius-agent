"""
Full cross-validation training script with LightGBM.

Copy this file, rename it descriptively (e.g. solution_lgbm_baseline.py),
then set TARGET, DATA_DIR, and the metric function.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score  # ← replace with competition metric
from sklearn.model_selection import StratifiedKFold  # use KFold for regression

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")  # ← adjust to actual data_dir from CLAUDE.md
TARGET = "target"  # ← replace with actual target column name
SEED = 42
N_FOLDS = 5
METRIC_NAME = "auc_roc"  # ← for logging only

LGB_PARAMS = {
    "objective": "binary",  # ← "regression" for RMSE, "multiclass" + num_class=N for multiclass
    "metric": "auc",
    "verbosity": -1,
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": SEED,
    "n_jobs": -1,
}

# ── Load data ─────────────────────────────────────────────────────────────────
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

# ── Basic preprocessing ───────────────────────────────────────────────────────
# Encode object/category columns for LightGBM
cat_cols = [c for c in train.columns if train[c].dtype == "object" and c != TARGET]
for col in cat_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")

feature_cols = [c for c in train.columns if c != TARGET]
X = train[feature_cols]
y = train[TARGET]
X_test = test[[c for c in feature_cols if c in test.columns]]

# ── CV loop ───────────────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
preds = np.zeros(len(test))

for fold, (tr, va) in enumerate(cv.split(X, y)):
    X_tr, y_tr = X.iloc[tr], y.iloc[tr]
    X_va, y_va = X.iloc[va], y.iloc[va]

    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof[va] = model.predict_proba(X_va)[:, 1]
    preds += model.predict_proba(X_test)[:, 1] / N_FOLDS

    fold_score = roc_auc_score(y_va, oof[va])
    print(f"  Fold {fold + 1}/{N_FOLDS}: {METRIC_NAME}={fold_score:.6f}")

oof_score = roc_auc_score(y, oof)
print(f"\nOOF {METRIC_NAME}: {oof_score:.6f}")

# ── Save OOF predictions ──────────────────────────────────────────────────────
np.save("artifacts/oof.npy", oof)
np.save("artifacts/test_preds.npy", preds)

# ── Build submission ──────────────────────────────────────────────────────────
sub = sample_sub.copy()
sub[sub.columns[-1]] = preds  # fill last column with predictions
assert len(sub) == len(sample_sub), (
    f"Row count mismatch: {len(sub)} vs {len(sample_sub)}"
)
assert not sub.isnull().any().any(), "NaN found in submission!"
sub.to_csv("submissions/submission_lgbm_baseline.csv", index=False)
print(f"Submission saved: {len(sub)} rows, columns: {list(sub.columns)}")
