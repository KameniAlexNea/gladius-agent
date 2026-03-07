---
name: ensembling
description: >
  Combine predictions from multiple models to exceed any single model's score.
  Covers OOF blending, rank averaging, stacking, and hill-climbing selection.
---

## Prerequisite: collect OOF + test predictions

Every model that participates in the ensemble must produce:
1. **OOF predictions** — out-of-fold predictions on the training set (same CV splits).
2. **Test predictions** — average of fold predictions on the test set.

```python
import numpy as np

# Convention: oof_preds[i] is a 1-D array of length len(train)
#             test_preds[i] is a 1-D array of length len(test)
oof_preds  = {}   # {"lgbm_v1": array, "xgb_v2": array, ...}
test_preds = {}   # same keys
```

---

## Method 1 — Simple average / weighted blend

```python
from sklearn.metrics import roc_auc_score  # replace with competition metric

names = list(oof_preds.keys())
y_oof = np.column_stack([oof_preds[n] for n in names])   # (N_train, n_models)
y_tst = np.column_stack([test_preds[n] for n in names])  # (N_test,  n_models)

# Equal-weight average
blend_oof = y_oof.mean(axis=1)
blend_tst = y_tst.mean(axis=1)
print(f"Equal-weight blend OOF: {roc_auc_score(y_train, blend_oof):.6f}")

# Optimise weights with scipy
from scipy.optimize import minimize

def neg_score(weights):
    w = np.array(weights)
    w = np.abs(w) / np.abs(w).sum()          # softmax-like normalisation
    pred = (y_oof * w).sum(axis=1)
    return -roc_auc_score(y_train, pred)

init_w = np.ones(len(names)) / len(names)
result = minimize(neg_score, init_w, method="Nelder-Mead",
                  options={"maxiter": 5000, "xatol": 1e-6})
opt_w  = np.abs(result.x) / np.abs(result.x).sum()
opt_oof = (y_oof * opt_w).sum(axis=1)
opt_tst = (y_tst * opt_w).sum(axis=1)
print(f"Optimised blend OOF: {roc_auc_score(y_train, opt_oof):.6f}")
for name, w in zip(names, opt_w):
    print(f"  {name}: {w:.4f}")
```

## Method 2 — Rank averaging (robust across scales)

```python
from scipy.stats import rankdata

def rank_avg(*arrays):
    ranked = np.column_stack([rankdata(a) for a in arrays])
    return ranked.mean(axis=1) / len(arrays[0])    # normalise to [0,1]

rank_oof = rank_avg(*[oof_preds[n] for n in names])
rank_tst = rank_avg(*[test_preds[n] for n in names])
print(f"Rank-average OOF: {roc_auc_score(y_train, rank_oof):.6f}")
```

## Method 3 — Stacking (meta-learner on OOF)

```python
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score

# Level-1: OOF predictions become features for meta-learner
X_meta_train = y_oof                    # (N_train, n_models)
X_meta_test  = y_tst                    # (N_test,  n_models)

meta = LogisticRegression(C=1.0, max_iter=1000)  # Ridge for regression
meta_cv = cross_val_score(
    meta, X_meta_train, y_train,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="roc_auc",
)
print(f"Meta-learner CV: {meta_cv.mean():.6f} ± {meta_cv.std():.6f}")

meta.fit(X_meta_train, y_train)
stack_tst = meta.predict_proba(X_meta_test)[:, 1]
```

## Method 4 — Hill-climbing (greedy forward selection)

Greedily adds models to the ensemble when each addition improves OOF score.
Allows a model to be added multiple times (implicit weighting).

```python
def hill_climb(oof_dict, y_true, metric_fn, n_rounds=100, maximize=True):
    names = list(oof_dict.keys())
    oof_arr = np.column_stack([oof_dict[n] for n in names])

    best_score = -np.inf if maximize else np.inf
    chosen = []

    for _ in range(n_rounds):
        best_add = None
        for i, name in enumerate(names):
            candidate = np.column_stack(
                [oof_arr[:, j] for j in [names.index(c) for c in chosen]] +
                [oof_arr[:, i]]
            ).mean(axis=1)
            s = metric_fn(y_true, candidate)
            if (maximize and s > best_score) or (not maximize and s < best_score):
                best_score = s
                best_add   = name
        if best_add is None:
            break
        chosen.append(best_add)

    counts = {n: chosen.count(n) for n in set(chosen)}
    total  = sum(counts.values())
    weights = {n: c / total for n, c in counts.items()}
    print(f"Hill-climb OOF: {best_score:.6f}")
    for n, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {n}: {w:.4f}")
    return weights

weights = hill_climb(oof_preds, y_train, roc_auc_score, n_rounds=100)
final_tst = sum(test_preds[n] * w for n, w in weights.items())
```

## Diversity check (before ensembling)

Ensembling only helps when models make *different* errors.
Check pairwise OOF correlation — models with correlation > 0.97 add little value.

```python
import pandas as pd

corr_df = pd.DataFrame(oof_preds).corr()
print(corr_df.to_string())
# aim for no pair above 0.97; add a fundamentally different architecture if too correlated
```

## Saving the ensemble submission

```python
sample_sub = pd.read_csv("data/sample_submission.csv")
sub = sample_sub.copy()
sub[sub.columns[-1]] = final_tst
assert len(sub) == len(sample_sub), "Row count mismatch!"
assert not sub.isnull().any().any(), "NaN in ensemble submission!"
sub.to_csv("submission_ensemble.csv", index=False)
print(f"Ensemble submission saved: OOF {roc_auc_score(y_train, final_oof):.6f}")
```
