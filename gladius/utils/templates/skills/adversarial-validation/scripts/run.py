"""
Adversarial Validation — detects distribution shift between train and test.

Usage:
    python scripts/run.py

Adjust DATA_DIR and TARGET before running.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA_DIR = "data"    # <- adjust to your data_dir
TARGET   = "target"  # <- replace with actual target column name


def run_adversarial_validation(data_dir: str = DATA_DIR, target: str = TARGET):
    train = pd.read_csv(f"{data_dir}/train.csv")
    test  = pd.read_csv(f"{data_dir}/test.csv")

    train_feat = train.drop(columns=[target], errors="ignore")
    common_cols = [c for c in train_feat.columns if c in test.columns]

    adv = pd.concat(
        [train_feat[common_cols].assign(_is_test=0),
         test[common_cols].assign(_is_test=1)],
        ignore_index=True,
    )
    X = adv.drop(columns=["_is_test"])
    y = adv["_is_test"]

    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.codes
    X = X.fillna(-999)

    # --- OOF AUC ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    for tr, va in cv.split(X, y):
        clf = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
        clf.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = clf.predict_proba(X.iloc[va])[:, 1]

    auc = roc_auc_score(y, oof)
    print(f"\nAdversarial AUC: {auc:.4f}")

    if auc < 0.55:
        print("✅ No significant shift — proceed normally.")
    elif auc < 0.65:
        print("⚠️  Mild shift — check top features.")
    elif auc < 0.80:
        print("❌ Moderate shift — drop or transform top leaking features.")
    else:
        print("🚨 Severe shift — likely ID/time leak, investigate immediately.")

    # --- Feature importances ---
    clf_full = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
    clf_full.fit(X, y)
    importances = pd.Series(clf_full.feature_importances_, index=X.columns)
    top_feats = importances.sort_values(ascending=False).head(20)
    print("\nTop 20 leaking features:")
    print(top_feats.to_string())

    # --- Sample weights for train (rescale so mean=1) ---
    train_X = X.iloc[:len(train_feat)]
    train_y = y.iloc[:len(train_feat)]
    clf_adv = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
    clf_adv.fit(train_X, train_y)
    p = clf_adv.predict_proba(train_X)[:, 1]   # P(is_test | features)
    adv_weights = np.clip(p / (1 - p + 1e-6), 0.1, 10.0)
    adv_weights /= adv_weights.mean()
    print(f"\nSample weight range: {adv_weights.min():.3f}–{adv_weights.max():.3f}")
    print("Pass adv_weights as sample_weight= to your model's fit() call.")

    return auc, top_feats, adv_weights


if __name__ == "__main__":
    run_adversarial_validation()
