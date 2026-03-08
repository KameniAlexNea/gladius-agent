---
name: hpo
description: >
  Bayesian hyperparameter optimisation with Optuna. Covers LightGBM, XGBoost,
  and CatBoost objective templates with fold-level pruning. Use after a solid
  baseline and feature set exist — HPO on weak features is wasted compute.
---

# Hyperparameter Optimisation (HPO)

## Overview

Optuna's TPE sampler finds good hyperparameters in 50–200 trials using Bayesian optimisation. The MedianPruner kills bad trials after 2 folds, saving 30–50% of compute. Results are persisted to SQLite so you can resume interrupted searches.

## When to Use This Skill

- A model architecture has proven competitive (OOF better than all alternatives tried).
- You have budget for 50–200 trials (≳ 30 min CPU / < 5 min GPU per trial).
- **Do NOT tune** before a solid feature set exists — HPO on bad features wastes compute.
- **Do NOT tune** more than one architecture simultaneously — pick the winner first.

## Core Capabilities

### 1. LightGBM + Optuna (primary template)

Full study with fold-level pruning. See `scripts/lgbm_optuna.py`.

**Quick usage:**
```bash
uv add optuna lightgbm
python scripts/lgbm_optuna.py
```
Results saved to `hpo.db` — resume if interrupted with `load_if_exists=True`.

### 2. XGBoost Template

See `scripts/lgbm_optuna.py` — the `XGB_PARAMS` section shows XGBoost parameter ranges.

### 3. CatBoost Template

See `scripts/lgbm_optuna.py` — the `CATBOOST_PARAMS` section shows CatBoost parameter ranges.

### 4. Best Practices

Read `references/best_practices.md` for tips on:
- Using `N_FOLDS=3` for HPO speed then retraining with 5 folds
- Early stopping inside the objective
- Resuming interrupted studies

## Quick Workflow

1. Set `DATA_DIR`, `TARGET`, `METRIC_DIRECTION` in `scripts/lgbm_optuna.py`
2. Run `python scripts/lgbm_optuna.py`
3. After `N_TRIALS` complete, the script prints best params
4. Copy best params into your main training script
5. Retrain with 5 folds using best params, report new OOF score

## Resources

### scripts/
- `lgbm_optuna.py` — LightGBM / XGBoost / CatBoost templates with Optuna

### references/
- `best_practices.md` — key rules for effective HPO
