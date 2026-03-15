---
name: ensembling
description: >
  Combine predictions from multiple models to exceed any single model's score.
  Covers weighted OOF blending, rank averaging, stacking, and greedy
  hill-climbing. Only use after at least two models with different architectures
  each have solid CV scores.
---

# Ensembling

## Overview

Ensembling works because diverse models make *different* errors — averaging their
mistakes cancels out noise and reduces variance. Before ensembling, verify model
diversity with pairwise OOF correlation. Models with correlation > 0.97 add
little value.

## When to Use This Skill

- You have ≥ 2 models with solid independent OOF evidence (different architectures, feature sets, or seeds).
- A single model's OOF score has plateaued.
- Do NOT ensemble premature or poorly-trained models.

## Core Capabilities

1. **Diversity Check** — pairwise OOF correlation; prune pairs with r > 0.97. See `scripts/blend.py::diversity_check`.
2. **Hill-Climbing** — greedy forward selection, recommended for ≥ 3 models; implicit weighting by repetition. See `scripts/blend.py::hill_climb`.
3. **Weighted Blend (Nelder-Mead)** — scipy optimisation for 2–4 models. See `scripts/blend.py::optimised_blend`.
4. **Rank Averaging** — scale-invariant blend; useful when model outputs differ in scale. See `scripts/blend.py::rank_average`.
5. **Stacking** — lightweight Ridge / Logistic meta-learner on OOF predictions as features.

---

## Prerequisites: Collect OOF + Test Predictions

Every model must produce:
1. **OOF predictions** — out-of-fold on the train set (same CV splits as all other models).
2. **Test predictions** — average of per-fold test predictions.

```python
# Save after each model training run:
np.save(f"artifacts/{model_name}_oof.npy",  oof_array)
np.save(f"artifacts/{model_name}_test.npy", test_array)

# Gather into dicts:
oof_preds  = {n: np.load(f"artifacts/{n}_oof.npy")  for n in model_names}
test_preds = {n: np.load(f"artifacts/{n}_test.npy") for n in model_names}
```

---

## Method 1 — Hill-Climbing (recommended for ≥ 3 models)

```python
from scripts.blend import diversity_check, hill_climb
from sklearn.metrics import roc_auc_score  # swap for your metric

print(diversity_check(oof_preds))  # prune pairs with r > 0.97 first

oof_blend, test_blend, weights = hill_climb(
    oof_preds, test_preds, y_train,
    metric_fn=roc_auc_score,
    direction="maximize",
    n_steps=100,
)
```

## Method 2 — Weighted Blend (Nelder-Mead, good for 2–4 models)

```python
from scripts.blend import optimised_blend

opt_oof, opt_test, weights = optimised_blend(
    oof_preds, test_preds, y_train,
    metric_fn=roc_auc_score,
    direction="maximize",
)
```

## Method 3 — Rank Averaging

```python
from scripts.blend import rank_average

oof_rank  = rank_average(oof_preds)
test_rank = rank_average(test_preds)
```

## Method 4 — Stacking (meta-learner)

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

oof_matrix  = np.column_stack(list(oof_preds.values()))
test_matrix = np.column_stack(list(test_preds.values()))

meta      = Ridge(alpha=1.0)
meta_oof  = cross_val_predict(meta, oof_matrix, y_train, cv=5)
meta.fit(oof_matrix, y_train)
meta_test = meta.predict(test_matrix)
```

## Quick Workflow

1. Collect `oof_preds` and `test_preds` from each trained model.
2. Run `diversity_check(oof_preds)` — prune pairs with r > 0.97.
3. ≥ 3 models → `hill_climb`; 2–4 models → `optimised_blend`.
4. Report ensemble OOF vs individual model scores.
5. Save blended predictions: `pd.DataFrame({"id": ids, TARGET: test_blend}).to_csv("artifacts/submission_ensemble.csv", index=False)`.
