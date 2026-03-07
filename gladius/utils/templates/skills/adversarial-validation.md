---
name: adversarial-validation
description: >
  Detect distribution shift between train and test sets. Run at the start of
  every competition and whenever the LB score diverges unexpectedly from OOF.
---

## When to run

- Start of competition — establish baseline understanding of train/test similarity.
- OOF score is high but leaderboard score is significantly lower (LB-OOF gap).
- After adding new features — verify each feature appears in both sets.

## Step 1 — run the classifier

```python
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DATA_DIR = "data"   # adjust to actual data_dir
TARGET    = "target"  # replace with actual target column name

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

train_feat = train.drop(columns=[TARGET], errors="ignore")
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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
for tr, va in cv.split(X, y):
    clf = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
    clf.fit(X.iloc[tr], y.iloc[tr])
    oof[va] = clf.predict_proba(X.iloc[va])[:, 1]

adv_auc = roc_auc_score(y, oof)
print(f"Adversarial AUC: {adv_auc:.4f}")
```

## Step 2 — interpret the AUC

| AUC | Verdict | Action |
| --- | --- | --- |
| 0.50–0.55 | ✅ No shift | Proceed normally |
| 0.55–0.65 | ⚠️ Mild shift | Monitor LB-OOF gap; check top features |
| 0.65–0.80 | ❌ Moderate shift | Drop or transform the top leaking features |
| 0.80–1.00 | 🚨 Severe shift | Likely ID/time leak — investigate immediately |

## Step 3 — find the leaking features

```python
clf_full = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
clf_full.fit(X, y)
importances = pd.Series(clf_full.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).head(20).to_string())
```

Common culprits:
- **Row ID / index columns** — always unique across splits; drop them.
- **Timestamp columns** — train is older, test is newer; engineer carefully or drop.
- **Count/aggregation features** computed on a different time window in train vs test.

## Step 4 — remediation

**Drop top leakers** and re-run adversarial validation until AUC < 0.60.

**Adversarial sample weighting** (when shift is structural and unavoidable):

```python
train_X = X.iloc[:len(train_feat)]
train_y = y.iloc[:len(train_feat)]
clf_adv = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
clf_adv.fit(train_X, train_y)

p = clf_adv.predict_proba(train_X)[:, 1]          # P(is_test | features)
adv_weights = np.clip(p / (1 - p + 1e-6), 0.1, 10.0)
adv_weights /= adv_weights.mean()                  # normalise mean to 1
print(f"Weight range: {adv_weights.min():.3f}–{adv_weights.max():.3f}")
# Pass adv_weights as sample_weight= to your model's fit() call.
```
