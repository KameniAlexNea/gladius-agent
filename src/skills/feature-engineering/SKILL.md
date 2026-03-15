---
name: feature-engineering
description: >
  Systematic feature generation recipes, SHAP-based importance measurement,
  and feature pruning for ML competitions. Use after a baseline is established.
  Covers numerical, categorical, temporal, and aggregation features — all
  with leakage prevention built in.
---

# Feature Engineering

## Overview

Feature engineering is the highest-ROI activity in tabular competitions after a
solid CV baseline exists. This skill provides ready-to-use recipes for every
feature type, a SHAP-based workflow for measuring importance, and a disciplined
pruning process.

## When to Use This Skill

- Baseline OOF score is established (LightGBM / XGBoost, raw features).
- You want to systematically improve features before HPO.
- OOF score has plateaued and you need orthogonal improvements.
- You need to identify which features are driving model decisions.

## Core Capabilities

1. **Leakage Prevention** — mandatory rules before writing any feature code. See `references/safety_rules.md`.
2. **Numerical Features** — log transforms, binning, Z-score outlier flags, pairwise interactions.
3. **Categorical Features** — frequency encoding, fold-safe target encoding, rare-group collapsing.
4. **Temporal / DateTime Features** — cyclic encoding, lag features, rolling statistics. Always sort by entity + time first.
5. **Aggregation Features** — per-group mean/std/min/max/count. Compute on train fold, map to val/test.
6. **SHAP Importance & Pruning** — TreeExplainer ranking + incremental pruning loop.

---

Read `references/safety_rules.md` before writing any feature code. It covers
fold-safe target encoding, scaler fitting rules, lag feature ordering, aggregation
leakage, and ID column removal — with correct and incorrect code examples.

**Quick rules:**
- Target encoding: computed inside each CV fold — never on full train.
- Scalers/encoders: fit on train fold only, transform val/test.
- Lag/rolling stats: sort by entity + time, use `.shift(1)` to exclude current row.
- Group aggregations: computed on train, mapped to val/test.
- Run adversarial validation after any new feature batch (`validation` skill).

### 2. Numerical Feature Recipes

```python
import numpy as np

def add_numerical_features(df):
    # Log transform (for skewed positives)
    df["amount_log1p"] = np.log1p(df["amount"].clip(lower=0))

    # Binning
    df["age_bin"] = pd.cut(df["age"], bins=10, labels=False)

    # Z-score outlier flag
    mu, sigma = df["value"].mean(), df["value"].std()
    df["value_outlier"] = (np.abs(df["value"] - mu) > 3 * sigma).astype(int)

    # Pairwise interaction
    df["ratio_a_b"] = df["feat_a"] / (df["feat_b"] + 1e-6)
    return df
```

### 3. Categorical Feature Recipes

```python
# Frequency encoding (safe — no target)
freq = df["cat_col"].value_counts() / len(df)
df["cat_col_freq"] = df["cat_col"].map(freq)

# Target encoding — MUST be fold-safe
def fold_safe_target_encode(train, val, col, target):
    mapping = train.groupby(col)[target].mean()
    return val[col].map(mapping).fillna(train[target].mean())

# Rare-group collapsing
counts = df["cat_col"].value_counts()
rare = counts[counts < 50].index
df["cat_col"] = df["cat_col"].where(~df["cat_col"].isin(rare), "RARE")
```

### 4. Temporal / DateTime Recipes

```python
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month

# Cyclic encoding
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# Lag features — MUST sort by entity+time first
df = df.sort_values(["entity_id", "timestamp"])
df["lag1"] = df.groupby("entity_id")["value"].shift(1)
df["rolling3_mean"] = df.groupby("entity_id")["value"].transform(lambda x: x.shift(1).rolling(3).mean())
```

### 5. Aggregation Features (Group Stats)

```python
# Compute on train, map to val/test — NEVER on full dataset
agg = train.groupby("group_col")["value"].agg(["mean", "std", "min", "max", "count"])
agg.columns = [f"grp_{c}" for c in agg.columns]
df = df.join(agg, on="group_col")
```

### 6. SHAP Importance & Feature Pruning

```python
import shap
import lightgbm as lgb

model = lgb.LGBMClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# For binary classification shap_values is a list — take class 1
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

importance = pd.Series(np.abs(sv).mean(axis=0), index=X_train.columns).sort_values(ascending=False)
print(importance.head(30))
```

**Pruning rule:** Drop a feature if mean |SHAP| < 0.001 AND adding it does not
improve OOF score by ≥ 0.0005.

## Quick Workflow

1. Run leakage checklist (Section 1).
2. Write candidate features using recipes above.
3. Compute SHAP values and rank features.
4. Run the pruning loop — keep only features that improve OOF by ≥ 0.0005.
5. Commit the feature set; each step must have a comment explaining the hypothesis.
