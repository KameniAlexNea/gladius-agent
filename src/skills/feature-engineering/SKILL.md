---
name: feature-engineering
description: Systematic feature generation for tabular ML competitions. Covers leakage-safe recipes for numerical, categorical, temporal, and aggregation features, plus SHAP-based importance measurement and disciplined pruning. Use after a baseline OOF score is established — engineering on top of a bad model is wasted effort.
---

# Feature Engineering

Feature engineering is the highest-ROI activity in tabular competitions once a solid CV baseline exists. Every new feature should have a hypothesis, be tested with a quick-fold sanity check, and pass the SHAP importance threshold before being committed.

## When to Use

- Baseline OOF score is established (LightGBM / XGBoost on raw features).
- OOF score has plateaued and you need orthogonal improvements.
- You want to understand which variables are driving model decisions (SHAP).
- Before HPO — tune on a strong feature set, not a weak one.

## Critical Rules

### ✅ DO

- **Read `references/safety_rules.md` first** — data leakage from feature engineering is the #1 competition mistake.
- **Test each feature batch quickly** — run n_splits=2 before full CV to catch leaks early.
- **Each feature needs a hypothesis comment** — silent features are technical debt.
- **Always sort by entity + time before any lag/rolling** — out-of-order lag computation silently leaks future data.
- **Compute group aggregations on train fold, map to val/test** — fitting on the full dataset is leakage.
- **Confirm no shift after each feature batch** — run adversarial validation (`validation` skill).

### ❌ DON'T

- **Don't encode target statistics on the full training set** — target encoding must be fold-safe.
- **Don't keep features with SHAP < 0.001 AND no OOF improvement** — noise features hurt generalisation.
- **Don't engineer before establishing a baseline** — you won't know if features are helping.
- **Don't fit scalers on val/test** — fit on train fold only, transform everything else.

## Anti-Patterns (NEVER)

```python
# ❌ BAD: Target encoding on full train — leaks target into val fold
train["cat_enc"] = train.groupby("cat_col")["target"].transform("mean")

# ✅ GOOD: Fold-safe target encoding — computed per fold
def fold_safe_target_encode(train_fold, val_fold, col, target):
    mapping = train_fold.groupby(col)[target].mean()
    global_mean = train_fold[target].mean()
    return val_fold[col].map(mapping).fillna(global_mean)
```

```python
# ❌ BAD: Rolling mean includes current row
df["rolling3"] = df.groupby("entity")["value"].transform(
    lambda x: x.rolling(3).mean()
)  # row i sees itself — future leak in time series

# ✅ GOOD: Shift before rolling to exclude current row
df = df.sort_values(["entity", "timestamp"])
df["rolling3"] = df.groupby("entity")["value"].transform(
    lambda x: x.shift(1).rolling(3).mean()
)
```

```python
# ❌ BAD: Group aggregation computed on full dataset (train+val+test)
full_agg = df.groupby("group")["value"].mean()
df["grp_mean"] = df["group"].map(full_agg)

# ✅ GOOD: Computed on train fold, mapped to val/test
train_agg = train_fold.groupby("group")["value"].mean()
val_fold["grp_mean"] = val_fold["group"].map(train_agg)
test["grp_mean"]     = test["group"].map(train_agg)
```

## Feature Recipes

### Numerical

```python
import numpy as np

# Log transform — for right-skewed positives
df["amount_log1p"] = np.log1p(df["amount"].clip(lower=0))

# Quantile binning — robust to outliers
df["age_bin"] = pd.qcut(df["age"], q=10, labels=False, duplicates="drop")

# Outlier flag
mu, sigma = df["value"].mean(), df["value"].std()
df["value_outlier"] = (np.abs(df["value"] - mu) > 3 * sigma).astype(int)

# Ratio interaction
df["ratio_a_b"] = df["feat_a"] / (df["feat_b"] + 1e-6)
```

### Categorical

```python
# Frequency encoding — no target, always safe
freq = train["cat"].value_counts() / len(train)
df["cat_freq"] = df["cat"].map(freq).fillna(0)

# Rare-group collapsing — stabilises low-count categories
counts = train["cat"].value_counts()
rare   = counts[counts < 50].index
df["cat"] = df["cat"].where(~df["cat"].isin(rare), "RARE")
```

### Temporal

```python
df["hour"]      = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["month"]     = df["timestamp"].dt.month

# Cyclic encoding — preserves circular nature of time
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# Lag + rolling — sort first, always
df = df.sort_values(["entity_id", "timestamp"])
df["lag1"]        = df.groupby("entity_id")["value"].shift(1)
df["roll3_mean"]  = df.groupby("entity_id")["value"].transform(
    lambda x: x.shift(1).rolling(3).mean()
)
```

### SHAP Importance & Pruning

```python
import shap, lightgbm as lgb

model = lgb.LGBMClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
shap_vals = shap.TreeExplainer(model).shap_values(X_val)
sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

importance = (
    pd.Series(np.abs(sv).mean(axis=0), index=X_train.columns)
    .sort_values(ascending=False)
)
print(importance.head(30))

# Pruning rule: drop if mean |SHAP| < 0.001 AND no OOF improvement ≥ 0.0005
```

## Common Pitfalls and Solutions

### The "Too Many Features" Problem

Adding 50 features at once makes it impossible to know which ones help. Overfitting risk rises sharply.

**Fix:** Add features in batches of 5–10. Test each batch with n_splits=2 before committing to full CV.

### The "SHAP Lies" Problem

SHAP says a feature is important, but removing it improves OOF. Collinear features share importance — both appear important, but one is redundant.

**Fix:** After SHAP ranking, run incremental removal: drop features one-by-one from the bottom; keep only if OOF does not increase.

### The "Adversarial AUC Jumps" Problem

Adding a new feature batch raises adversarial AUC from 0.52 to 0.71. The new features introduce distribution shift.

**Fix:** Check which new features have the highest adversarial importance. Lag features that use different lookback windows in train vs test are a common cause.

## Reference

- Leakage rules and examples: `references/safety_rules.md`

