---
name: feature-engineering
description: >
  Systematic feature generation recipes, SHAP-based importance measurement,
  and feature pruning. Use after a baseline is established to find high-impact
  engineered features.
---

## Workflow

1. **Establish baseline OOF score** with raw features only.
2. **Generate candidate features** using recipes below.
3. **Measure importance** — add candidates to the model and compute SHAP values.
4. **Prune** — drop features that do not improve CV score by ≥ 0.0005.
5. **Log** every feature with a hypothesis comment.

## Safety rules

- **NEVER fit target-based encodings on the full training set** — always compute
  within each CV fold to prevent leakage.
- Drop ID columns and raw timestamp columns before modeling.
- For lag / rolling features: use only past data — no future leakage.

---

## Recipe: numerical features

```python
import numpy as np

# Log / power transforms (handle zeros/negatives carefully)
df["feat_log1p"]  = np.log1p(df["feat"].clip(lower=0))
df["feat_sqrt"]   = np.sqrt(df["feat"].clip(lower=0))
df["feat_squared"] = df["feat"] ** 2

# Binning (quantile bins avoid outlier sensitivity)
df["feat_bin"] = pd.qcut(df["feat"], q=10, labels=False, duplicates="drop")

# Z-score outlier flag
z = (df["feat"] - df["feat"].mean()) / df["feat"].std()
df["feat_is_outlier"] = (z.abs() > 3).astype(int)

# Pairwise interactions (between top correlated numeric features)
df["feat_a_x_feat_b"] = df["feat_a"] * df["feat_b"]
df["feat_a_div_feat_b"] = df["feat_a"] / (df["feat_b"] + 1e-6)
df["feat_a_minus_feat_b"] = df["feat_a"] - df["feat_b"]
```

## Recipe: categorical features

```python
# Frequency encoding (safe — no target involved)
freq = df["cat_col"].value_counts()
df["cat_col_freq"] = df["cat_col"].map(freq).fillna(0)

# Target encoding — MUST be computed inside CV folds
# Pattern for inside a fold loop:
#   enc = train_fold.groupby("cat_col")["target"].mean()
#   val_fold["cat_col_te"] = val_fold["cat_col"].map(enc).fillna(enc.mean())
#   test["cat_col_te"]     = test["cat_col"].map(enc).fillna(enc.mean())

# Rare-category grouping
counts = df["cat_col"].value_counts()
rare   = counts[counts < 50].index
df["cat_col_grouped"] = df["cat_col"].replace(rare, "__RARE__")

# Ordinal encoding (for tree models — just label-encode)
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df[["cat_col_ord"]] = oe.fit_transform(df[["cat_col"]])

# Combination feature (concatenate two low-cardinality cats)
df["cat_a_x_cat_b"] = df["cat_a"].astype(str) + "_" + df["cat_b"].astype(str)
```

## Recipe: temporal / datetime features

```python
df["dt"] = pd.to_datetime(df["datetime_col"])
df["hour"]        = df["dt"].dt.hour
df["day_of_week"] = df["dt"].dt.dayofweek    # 0=Monday
df["day_of_month"] = df["dt"].dt.day
df["month"]       = df["dt"].dt.month
df["week_of_year"] = df["dt"].dt.isocalendar().week.astype(int)
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

# Cyclical encoding for periodic features
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# Time since reference date
ref = df["dt"].min()
df["days_since_start"] = (df["dt"] - ref).dt.days

# Lag features (sort by entity + time first!)
df = df.sort_values(["entity_id", "dt"])
df["lag_1"] = df.groupby("entity_id")["value"].shift(1)
df["lag_7"] = df.groupby("entity_id")["value"].shift(7)

# Rolling statistics (use min_periods=1 to avoid many NaNs)
df["roll_mean_7"] = (
    df.groupby("entity_id")["value"]
    .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
)
```

## Recipe: aggregation features (by group)

```python
# Aggregate statistics per group — fit only on train and map to val/test
agg = train.groupby("group_id")["value"].agg(["mean", "std", "min", "max", "count"])
agg.columns = [f"grp_{c}" for c in agg.columns]
df = df.merge(agg, on="group_id", how="left")
```

---

## SHAP importance workflow

```python
import shap
import lightgbm as lgb

# Train a quick model
model = lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1)
model.fit(X_train, y_train)

explainer  = shap.TreeExplainer(model)
shap_vals  = explainer.shap_values(X_train)
# For binary classification shap_values returns a list [class0, class1]
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

mean_abs_shap = pd.Series(
    np.abs(shap_vals).mean(axis=0), index=X_train.columns
).sort_values(ascending=False)
print(mean_abs_shap.head(30).to_string())
```

## Permutation importance (model-agnostic alternative)

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model, X_val, y_val, n_repeats=10, random_state=42, scoring="roc_auc"
)
perm_imp = pd.Series(result.importances_mean, index=X_val.columns)
print(perm_imp.sort_values(ascending=False).head(20).to_string())
```

## Feature pruning

Drop a candidate feature if:

1. SHAP mean absolute value < 0.001 (negligible contribution), *and*
2. Adding it does not improve OOF score by ≥ 0.0005.

```python
# Quick pruning loop — add one feature at a time and check CV delta
baseline_score = evaluate_cv(X_base, y)
features_to_keep = list(X_base.columns)
for feat in candidate_features:
    X_try = X_base.assign(**{feat: df[feat]})
    score  = evaluate_cv(X_try, y)
    delta  = score - baseline_score
    print(f"  {feat}: delta={delta:+.5f}")
    if delta >= 0.0005:
        features_to_keep.append(feat)
        baseline_score = score
        print(f"    → KEPT (new baseline: {baseline_score:.6f})")
    else:
        print(f"    → DROPPED")
```
