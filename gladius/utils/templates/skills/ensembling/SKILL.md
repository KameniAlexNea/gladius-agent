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

Ensembling works because diverse models make *different* errors — averaging their mistakes cancels out noise and reduces variance.  Before ensembling, verify model diversity with pairwise OOF correlation (`scripts/blend.py` includes the check). Models with correlation > 0.97 add little value.

## When to Use This Skill

- You have ≥ 2 models with solid independent OOF evidence (different architectures, feature sets, or seeds).
- A single model's OOF score has plateaued — combining orthogonal predictions is the highest-ROI next step.
- Competition allows or encourages large ensembles. Do NOT ensemble premature/poorly-trained models.

## Core Capabilities

### 1. Collect OOF + Test Predictions

Every model must produce:
1. **OOF predictions** — out-of-fold predictions on the training set (same CV splits).
2. **Test predictions** — average of per-fold test predictions.

Store them per model name in two dicts: `oof_preds` and `test_preds`.

### 2. Weighted Blend (Nelder-Mead)

Optimise blend weights on OOF via `scipy.optimize.minimize`. See `scripts/blend.py`.

**Quick call:**
```python
from scripts.blend import optimised_blend
oof_preds = {"lgbm": ..., "xgb": ...}
test_preds = {"lgbm": ..., "xgb": ...}
opt_oof, opt_tst, weights = optimised_blend(oof_preds, test_preds, y_train)
```

### 3. Rank Averaging

Scale-invariant blending — useful when model outputs are on different scales. See `scripts/blend.py::rank_average`.

### 4. Stacking (Meta-Learner)

Train a lightweight meta-learner (Logistic Regression, Ridge) on OOF predictions as features. See `scripts/stacking.py`.

### 5. Hill-Climbing (Greedy Forward Selection)

Greedily adds models that improve OOF score. Allows a model to be added multiple times (implicit weighting). Most robust to overfitting on the validation set. See `scripts/hill_climb.py`.

**Recommended for large ensembles (> 5 models).** Start with hill-climbing before trying weighted blend.

## Quick Workflow

1. Collect `oof_preds` and `test_preds` dicts from each trained model.
2. Check pairwise OOF correlation — drop any pair with r > 0.97.
3. Run hill-climbing for large sets; weighted blend for small sets (2–4 models).
4. Validate with `scripts/blend.py`'s diversity check before submitting.

## Resources

### scripts/
- `blend.py` — diversity check, simple average, optimised weighted blend, rank averaging
- `stacking.py` — meta-learner (Logistic Regression / Ridge) stacking
- `hill_climb.py` — greedy forward selection (supports any metric function)
