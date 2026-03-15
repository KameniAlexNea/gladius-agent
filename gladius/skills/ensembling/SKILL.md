---
name: ensembling
description: Combine predictions from multiple trained models to exceed any single model's OOF score. Covers pairwise diversity checking (prune r > 0.97), greedy hill-climbing (≥3 models), weighted Nelder-Mead blending (2–4 models), rank averaging, and Ridge stacking. Only effective when models are genuinely diverse — training on different architectures, feature sets, or seeds with independent CV evidence.
---

# Ensembling

Ensembling works because diverse models make *different* errors — averaging them cancels noise and reduces variance. Identical models (r > 0.97) add nothing. Run `diversity_check` first, always.

## When to Use

- You have ≥ 2 models with **solid independent OOF evidence** on the same CV splits.
- A single model's OOF score has plateaued (marginal feature improvements < 0.001).
- Models use different architectures, feature sets, or target transformations.
- **Not** before each model has been independently tuned — stale models dilute the blend.
- **Not** when all OOF correlations exceed 0.97 — no diversity means no gain.

## Critical Rules

### ✅ DO

- **Always run `diversity_check` first** — prune model pairs with OOF correlation > 0.97.
- **Use the same CV splits across all models** — OOF predictions must be comparable.
- **Report ensemble OOF vs each individual model** — if ensemble < best single model, stop.
- **Use hill-climbing for ≥ 3 models** — implicit weighting is more robust than Nelder-Mead in high dimensions.
- **Save OOF + test arrays consistently** — `artifacts/{model_name}_oof.npy` and `artifacts/{model_name}_test.npy`.

### ❌ DON'T

- **Don't ensemble before individual models are tuned** — weak models drag the blend down.
- **Don't blend more than ~6 models with Nelder-Mead** — optimisation becomes unreliable in high dimensions; use hill-climbing instead.
- **Don't skip the OOF check** — never trust a blend that appears to improve only on the leaderboard.
- **Don't use different CV seeds across models** — OOF predictions become incomparable.

## Anti-Patterns (NEVER)

```python
# ❌ BAD: Simple average ignores model quality differences
test_blend = np.mean(list(test_preds.values()), axis=0)

# ✅ GOOD: Diversity check first, then weighted blend based on OOF performance
from scripts.blend import diversity_check, hill_climb
print(diversity_check(oof_preds))  # prune r > 0.97 pairs
oof_blend, test_blend, weights = hill_climb(
    oof_preds, test_preds, y_train,
    metric_fn=roc_auc_score, direction="maximize",
)
```

```python
# ❌ BAD: Using test LB to tune blend weights — no OOF check
# Submitting 10 blend variants to find "best" weights is LB probing
blend = 0.7 * test_A + 0.3 * test_B   # chosen because LB was highest

# ✅ GOOD: Weights chosen entirely on OOF, never touching the LB
opt_oof, opt_test, weights = optimised_blend(
    oof_preds, test_preds, y_train,
    metric_fn=roc_auc_score, direction="maximize",
)
print(f"Best OOF: {metric_fn(y_train, opt_oof):.5f}, weights: {weights}")
# Only submit once — trust the OOF-selected weights
```

## Prerequisites: Collect OOF + Test Predictions

Every model must produce predictions on the **same folds**:

```python
# After each model training run:
np.save(f"artifacts/{model_name}_oof.npy",  oof_array)   # shape: (n_train,)
np.save(f"artifacts/{model_name}_test.npy", test_array)  # shape: (n_test,)

# Gather into dicts for blend functions:
oof_preds  = {n: np.load(f"artifacts/{n}_oof.npy")  for n in model_names}
test_preds = {n: np.load(f"artifacts/{n}_test.npy") for n in model_names}
```

## Method 1 — Hill-Climbing (recommended for ≥ 3 models)

Greedy forward selection. At each step adds the model (with replacement) that most improves OOF. Implicit weighting by repetition count — more robust than Nelder-Mead for ≥ 4 models.

```python
from scripts.blend import diversity_check, hill_climb
from sklearn.metrics import roc_auc_score

print(diversity_check(oof_preds))  # inspect correlation matrix first

oof_blend, test_blend, weights = hill_climb(
    oof_preds, test_preds, y_train,
    metric_fn=roc_auc_score,
    direction="maximize",
    n_steps=100,
)
print(f"Ensemble OOF: {roc_auc_score(y_train, oof_blend):.5f}")
print(f"Weights: {weights}")
```

## Method 2 — Weighted Blend (Nelder-Mead, 2–4 models)

Uses scipy optimisation to find the best convex combination of OOF predictions. Reliable for up to ~4 models.

```python
from scripts.blend import optimised_blend

opt_oof, opt_test, weights = optimised_blend(
    oof_preds, test_preds, y_train,
    metric_fn=roc_auc_score,
    direction="maximize",
)
print(f"Optimised OOF: {roc_auc_score(y_train, opt_oof):.5f}")
```

## Method 3 — Rank Averaging (scale-invariant)

Normalises each model's predictions to ranks before averaging. Use when model outputs are on very different scales (e.g., one produces log-odds, another probabilities).

```python
from scripts.blend import rank_average

oof_rank  = rank_average(oof_preds)
test_rank = rank_average(test_preds)
```

## Method 4 — Stacking (meta-learner)

Fits a simple Ridge or Logistic meta-learner on OOF predictions as features. Adds one more CV layer. Only worthwhile if the meta-learner OOF clearly beats simple blend.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

oof_matrix  = np.column_stack(list(oof_preds.values()))   # (n_train, n_models)
test_matrix = np.column_stack(list(test_preds.values()))  # (n_test, n_models)

meta      = Ridge(alpha=1.0)
meta_oof  = cross_val_predict(meta, oof_matrix, y_train, cv=5)
meta.fit(oof_matrix, y_train)
meta_test = meta.predict(test_matrix)
```

## Common Pitfalls and Solutions

### The "Correlated Models" Problem

Two "different" models (e.g., LightGBM with slightly different seeds) have OOF correlation of 0.98. Blending them barely moves OOF but adds noise to tuning.

**Fix:** Run `diversity_check` before any blending. Remove one of any pair with r > 0.97. Build diversity deliberately: different algorithms, feature sets, target transformations.

### The "LB Probing" Problem

Trying 10 different blend weights and selecting the one with the best leaderboard score. This overfits to public LB and destroys private LB rank.

**Fix:** Select blend weights entirely on OOF. Submit once. The OOF-selected blend is your true estimate of generalisation.

### The "Stale Model" Problem

Ensembling a first-baseline model from Day 1 with a well-tuned Day 5 model. The weak model drags the blend below the Day 5 model alone.

**Fix:** Run hill-climbing — if the weak model is never selected, drop it. Ensemble should exceed the best individual model OOF, or it's not worth the complexity.

## Reference

- `scripts/blend.py` — `diversity_check`, `hill_climb`, `optimised_blend`, `rank_average`

