"""
Diverse 4-model rank-averaged ensemble (LightGBM+XGBoost+CatBoost+LogReg) with 10-fold CV.

Hypothesis: Combining predictions from four structurally diverse models via rank
averaging reduces correlated errors and lifts AUC over the single-model v1 baseline
(0.9104). LightGBM OOF/test predictions are loaded directly from v1 to avoid
redundant compute. XGBoost, CatBoost, and LogisticRegression are trained fresh with
10-fold stratified CV. Rank averaging normalises differing probability scales so each
model contributes equally on the ordinal scale that AUC optimises.

Expected OOF AUC: ~0.932 (+0.022 over v1).
"""

from __future__ import annotations

import json
import pathlib
import warnings
from typing import Tuple

import catboost as cb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GLADIUS_DIR = ROOT / ".gladius"
GLADIUS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
OOF_V1_PATH = GLADIUS_DIR / "oof_v1.npy"
SUB_V1_PATH = GLADIUS_DIR / "sub_v1.csv"
OOF_PATH = GLADIUS_DIR / "oof_v2.npy"
SUB_PATH = GLADIUS_DIR / "sub_v2.csv"

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── CV config ─────────────────────────────────────────────────────────────────
N_FOLDS = 10
EARLY_STOPPING_ROUNDS = 50

# ── Model hyper-parameters ────────────────────────────────────────────────────
XGB_PARAMS: dict = dict(
    objective="binary:logistic",
    eval_metric="auc",
    max_depth=4,
    learning_rate=0.05,
    n_estimators=1000,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=SEED,
    n_jobs=-1,
    verbosity=0,
)

CB_PARAMS: dict = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    depth=5,
    learning_rate=0.05,
    iterations=1000,
    l2_leaf_reg=3.0,
    subsample=0.8,
    random_seed=SEED,
    verbose=False,
    allow_writing_files=False,
)

LR_PIPELINE = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=SEED, n_jobs=-1)),
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return (X_train, y_train, X_test, test_ids)."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    feature_cols = [c for c in train.columns if c.startswith("feature_")]
    X_train = train[feature_cols].astype(np.float32)
    y_train = train["target"].astype(np.int32)
    X_test = test[feature_cols].astype(np.float32)
    return X_train, y_train, X_test, test["id"]


def rank_norm(arr: np.ndarray) -> np.ndarray:
    """Map array to [0, 1] via rank normalisation (ties → average rank)."""
    return (rankdata(arr) - 1) / (len(arr) - 1)


def cv_xgb(
    X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, skf: StratifiedKFold
) -> Tuple[np.ndarray, np.ndarray]:
    """Train XGBoost with early stopping; return (oof_preds, test_preds)."""
    oof = np.zeros(len(X), dtype=np.float64)
    test_acc = np.zeros(len(X_test), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        val_pred = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = val_pred
        fold_auc = roc_auc_score(y_va, val_pred)
        print(f"    [XGB]  fold {fold:2d}/{N_FOLDS}  best_iter={model.best_iteration:4d}  val_AUC={fold_auc:.6f}")
        test_acc += model.predict_proba(X_test)[:, 1] / N_FOLDS

    return oof, test_acc


def cv_catboost(
    X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, skf: StratifiedKFold
) -> Tuple[np.ndarray, np.ndarray]:
    """Train CatBoost with early stopping; return (oof_preds, test_preds)."""
    oof = np.zeros(len(X), dtype=np.float64)
    test_acc = np.zeros(len(X_test), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = cb.CatBoostClassifier(**CB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False,
        )
        val_pred = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = val_pred
        fold_auc = roc_auc_score(y_va, val_pred)
        best_iter = model.get_best_iteration()
        print(f"    [CAT]  fold {fold:2d}/{N_FOLDS}  best_iter={best_iter:4d}  val_AUC={fold_auc:.6f}")
        test_acc += model.predict_proba(X_test)[:, 1] / N_FOLDS

    return oof, test_acc


def cv_logreg(
    X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, skf: StratifiedKFold
) -> Tuple[np.ndarray, np.ndarray]:
    """Train LogisticRegression pipeline; return (oof_preds, test_preds)."""
    oof = np.zeros(len(X), dtype=np.float64)
    test_acc = np.zeros(len(X_test), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=SEED, n_jobs=-1)),
        ])
        pipe.fit(X_tr, y_tr)
        val_pred = pipe.predict_proba(X_va)[:, 1]
        oof[va_idx] = val_pred
        fold_auc = roc_auc_score(y_va, val_pred)
        print(f"    [LR]   fold {fold:2d}/{N_FOLDS}                val_AUC={fold_auc:.6f}")
        test_acc += pipe.predict_proba(X_test)[:, 1] / N_FOLDS

    return oof, test_acc


def rank_ensemble(
    oof_arrays: list[np.ndarray],
    test_arrays: list[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rank-average OOF and test predictions across models.

    Each model's predictions are mapped to [0, 1] via rank normalisation,
    then the normalised ranks are averaged.
    """
    oof_ranks = np.column_stack([rank_norm(a) for a in oof_arrays])
    oof_ensemble = oof_ranks.mean(axis=1)

    # For test: rank each model's test preds jointly with its OOF preds so the
    # rank scale is consistent; then slice off the test portion.
    test_ensemble = np.zeros(len(test_arrays[0]), dtype=np.float64)
    for oof_a, test_a in zip(oof_arrays, test_arrays):
        combined = np.concatenate([oof_a, test_a])
        ranks = rank_norm(combined)
        test_ensemble += ranks[len(oof_a):]  # test portion
    test_ensemble /= len(test_arrays)

    return oof_ensemble, test_ensemble


def main() -> None:
    print("=" * 60)
    print("solution_v2  |  4-model rank-averaged ensemble (10-fold)")
    print("=" * 60)

    X_train, y_train, X_test, test_ids = load_data()
    print(f"\nTrain shape : {X_train.shape}  |  pos_rate={y_train.mean():.3f}")
    print(f"Test  shape : {X_test.shape}\n")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # ── 1. Load LightGBM predictions from v1 ─────────────────────────────────
    print("── LightGBM (reused from v1) ──")
    lgbm_oof = np.load(OOF_V1_PATH)
    lgbm_test = pd.read_csv(SUB_V1_PATH)["target"].values
    lgbm_oof_auc = roc_auc_score(y_train, lgbm_oof)
    print(f"    [LGBM] loaded v1 OOF  AUC={lgbm_oof_auc:.6f}  (5-fold, reused)\n")

    # ── 2. XGBoost ────────────────────────────────────────────────────────────
    print("── XGBoost (10-fold) ──")
    xgb_oof, xgb_test = cv_xgb(X_train, y_train, X_test, skf)
    xgb_auc = roc_auc_score(y_train, xgb_oof)
    print(f"    [XGB]  OOF AUC={xgb_auc:.6f}\n")

    # ── 3. CatBoost ───────────────────────────────────────────────────────────
    print("── CatBoost (10-fold) ──")
    cat_oof, cat_test = cv_catboost(X_train, y_train, X_test, skf)
    cat_auc = roc_auc_score(y_train, cat_oof)
    print(f"    [CAT]  OOF AUC={cat_auc:.6f}\n")

    # ── 4. Logistic Regression ────────────────────────────────────────────────
    print("── Logistic Regression (10-fold) ──")
    lr_oof, lr_test = cv_logreg(X_train, y_train, X_test, skf)
    lr_auc = roc_auc_score(y_train, lr_oof)
    print(f"    [LR]   OOF AUC={lr_auc:.6f}\n")

    # ── 5. Rank ensemble ──────────────────────────────────────────────────────
    print("── Rank-averaging ensemble ──")
    oof_ensemble, test_ensemble = rank_ensemble(
        oof_arrays=[lgbm_oof, xgb_oof, cat_oof, lr_oof],
        test_arrays=[lgbm_test, xgb_test, cat_test, lr_test],
    )
    ensemble_auc = roc_auc_score(y_train, oof_ensemble)

    # ── 6. Save outputs ───────────────────────────────────────────────────────
    np.save(OOF_PATH, oof_ensemble)
    sub = pd.DataFrame({"id": test_ids, "target": test_ensemble})
    sub.to_csv(SUB_PATH, index=False)

    print(f"\nModel OOF AUCs:")
    print(f"  LightGBM (v1, 5-fold) : {lgbm_oof_auc:.6f}")
    print(f"  XGBoost  (10-fold)    : {xgb_auc:.6f}")
    print(f"  CatBoost (10-fold)    : {cat_auc:.6f}")
    print(f"  LogReg   (10-fold)    : {lr_auc:.6f}")
    print(f"\nOOF predictions  → {OOF_PATH}")
    print(f"Submission       → {SUB_PATH}")
    print(f"\nOOF_SCORE: {ensemble_auc:.6f}")


if __name__ == "__main__":
    main()
