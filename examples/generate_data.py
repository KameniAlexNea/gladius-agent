"""
Generate synthetic binary-classification competition data.

Creates:
  data/train.csv          — 800 rows,  20 features + target
  data/test.csv           — 200 rows,  20 features (no target)
  data/sample_submission.csv — 200 rows, id + target=0.5 (trivial baseline)
  data/.answers.csv       — hidden test labels (used by the fake platform scorer)

Run:
    python generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

SEED = 42
TRAIN_SIZE = 800
TEST_SIZE = 200
N_FEATURES = 20
N_INFORMATIVE = 8

rng = np.random.default_rng(SEED)

# ── Generate base dataset ──────────────────────────────────────────────────────
X, y = make_classification(
    n_samples=TRAIN_SIZE + TEST_SIZE,
    n_features=N_FEATURES,
    n_informative=N_INFORMATIVE,
    n_redundant=4,
    n_repeated=2,
    n_classes=2,
    flip_y=0.05,
    random_state=SEED,
)

feature_cols = [f"feature_{i:02d}" for i in range(N_FEATURES)]

X_train, X_test = X[:TRAIN_SIZE], X[TRAIN_SIZE:]
y_train, y_test = y[:TRAIN_SIZE], y[TRAIN_SIZE:]

train_ids = np.arange(1, TRAIN_SIZE + 1)
test_ids  = np.arange(TRAIN_SIZE + 1, TRAIN_SIZE + TEST_SIZE + 1)

data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)

# ── train.csv ─────────────────────────────────────────────────────────────────
train_df = pd.DataFrame(X_train, columns=feature_cols)
train_df.insert(0, "id", train_ids)
train_df["target"] = y_train
train_df.to_csv(data_dir / "train.csv", index=False)
print(f"Wrote {len(train_df)} rows → data/train.csv")

# ── test.csv (no target) ──────────────────────────────────────────────────────
test_df = pd.DataFrame(X_test, columns=feature_cols)
test_df.insert(0, "id", test_ids)
test_df.to_csv(data_dir / "test.csv", index=False)
print(f"Wrote {len(test_df)} rows → data/test.csv")

# ── sample_submission.csv ─────────────────────────────────────────────────────
sub_df = pd.DataFrame({"id": test_ids, "target": 0.5})
sub_df.to_csv(data_dir / "sample_submission.csv", index=False)
print(f"Wrote {len(sub_df)} rows → data/sample_submission.csv")

# ── hidden answer key (used by fake platform scorer) ─────────────────────────
answers_df = pd.DataFrame({"id": test_ids, "target": y_test})
answers_path = data_dir / ".answers.csv"
answers_df.to_csv(answers_path, index=False)
print(f"Wrote {len(answers_df)} rows → data/.answers.csv  (keep this hidden!)")

# ── Quick sanity check ────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

clf = LogisticRegression(max_iter=200, random_state=SEED)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]
score = roc_auc_score(y_test, proba)
print(f"\nSanity check — LogReg AUC on test: {score:.4f}  (your agent should beat this)")
