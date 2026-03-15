# Leakage Prevention Rules

**Read this before writing any feature code.**
Data leakage is the most common competition bug — it inflates OOF scores but kills leaderboard performance.

---

## Rule 1 — Target encoding is always fold-safe

Target encoding (mean target per category) leaks the target if computed on the full training set.

**Wrong:**
```python
# BUG: uses all train labels including the val fold
mapping = train.groupby("cat_col")["target"].mean()
df["cat_enc"] = df["cat_col"].map(mapping)
```

**Correct:**
```python
# In each CV fold:
def fold_safe_target_encode(train_fold, val_fold, col, target):
    mapping = train_fold.groupby(col)[target].mean()
    global_mean = train_fold[target].mean()
    return val_fold[col].map(mapping).fillna(global_mean)
```

---

## Rule 2 — Transformers fit on train fold only

StandardScaler, MinMaxScaler, PCA — any transformer that estimates statistics must be fit on the training fold, then applied to val/test.

**Wrong:**
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # uses all data including val
```

**Correct:**
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)     # no fit on val
X_test_scaled  = scaler.transform(X_test)
```

---

## Rule 3 — Lag and rolling features must use past data only

Sort by entity + timestamp before computing any lag or rolling statistic.
Use `.shift(1)` to prevent the current row from appearing in its own rolling window.

**Wrong:**
```python
df["rolling_mean"] = df.groupby("entity")["value"].transform(lambda x: x.rolling(3).mean())
# includes current row — leaks the present into itself
```

**Correct:**
```python
df = df.sort_values(["entity", "timestamp"])
df["rolling_mean"] = (
    df.groupby("entity")["value"]
    .transform(lambda x: x.shift(1).rolling(3).mean())
    # shift(1) excludes the current row
)
```

---

## Rule 4 — Aggregation features fit on train only

Group statistics (mean, std, count per group) must be computed on the training set and then mapped to val/test.

**Wrong:**
```python
df["grp_mean"] = df.groupby("group")["value"].transform("mean")
# test rows participate in computing train statistics
```

**Correct:**
```python
agg = train.groupby("group")["value"].mean().rename("grp_mean")
train = train.join(agg, on="group")
val   = val.join(agg,   on="group")    # same train-derived stats
test  = test.join(agg,  on="group")
```

---

## Rule 5 — No test rows in any training fold

When building CV splits, test rows must never appear in training or validation folds. This is automatically satisfied by `sklearn.model_selection.KFold` operating on the training set, but can be violated when combining datasets carelessly.

---

## Rule 6 — Drop ID columns

Row IDs, record IDs, or any column that is unique per row will cause adversarial AUC ≈ 1.0 and CV-LB correlation collapse. Drop them unless they encode meaningful entity identity.

---

## Quick Checklist

Before running any CV experiment with a new feature batch:

- [ ] Target encoding: fold-safe ✓
- [ ] Scalers/encoders: fit on train fold only ✓
- [ ] Lag/rolling features: shifted by ≥1 ✓
- [ ] Aggregations: computed on train, mapped to val/test ✓
- [ ] No ID or leaky columns included ✓
- [ ] Run adversarial validation to confirm AUC < 0.55 ✓
