# Feature Engineering — Safety Rules (Leakage Prevention)

## Cardinal Rule

**NEVER fit target-based encodings on the full training set.**
Always compute them inside each CV fold using only the fold's training rows.

---

## Leakage Sources to Check Before Any Feature

| Feature type | Leakage risk | Prevention |
| --- | --- | --- |
| Target encoding / mean encoding | High — uses target to create feature | Compute inside each CV fold only |
| StandardScaler / imputer | Medium — statistics from val/test bleed in | Fit on train fold, transform val/test |
| Lag/rolling features | Medium — future data in past | Sort by time; strict `shift(1)` before rolling |
| Count features on train+test combined | Medium — test info bleeds to train | Compute on train only, map to test |
| Row ID / index columns | High — perfectly separates train/test | Always drop before modeling |
| Aggregation features spanning train+test | High — data from test used in train stats | Compute on train, left-join to test |
| Date/time columns as-is | Low-medium | Extract only periodic features (hour, weekday, etc.) |

---

## Fold-Safe Target Encoding Pattern

```python
def target_encode_fold(train_fold, val_fold, test_df, col, target):
    """Compute target encoding inside a single CV fold."""
    enc = train_fold.groupby(col)[target].mean()
    global_mean = train_fold[target].mean()
    train_fold[f"{col}_te"] = train_fold[col].map(enc).fillna(global_mean)
    val_fold[f"{col}_te"]   = val_fold[col].map(enc).fillna(global_mean)
    test_df[f"{col}_te"]    = test_df[col].map(enc).fillna(global_mean)
    return train_fold, val_fold, test_df
```

---

## Checklist Before Submitting Any Script

- [ ] No `fit_transform(X_full_train)` where X_full_train includes validation rows.
- [ ] No features computed on the combined train+test dataframe (except frequency encoding of test IDs).
- [ ] Lag features: data is sorted by entity + time before `shift()`.
- [ ] Rolling features: use `shift(1).rolling()` not `rolling()` directly.
- [ ] ID columns (`id`, `row_id`, `SK_ID_CURR`, etc.) are dropped before training.
- [ ] Scaler/imputer fit only on train fold inside the CV loop.
