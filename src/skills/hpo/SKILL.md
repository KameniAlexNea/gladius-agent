---
name: hpo
description: >
  Bayesian hyperparameter optimisation with Optuna. Covers LightGBM, XGBoost,
  and CatBoost objective templates with fold-level pruning. Use after a solid
  baseline and feature set exist — HPO on weak features is wasted compute.
---

# Hyperparameter Optimisation (HPO)

## Overview

Optuna's TPE sampler finds good hyperparameters in 50–200 trials using Bayesian
optimisation. The MedianPruner kills bad trials after 2 folds, saving 30–50% of
compute. Results are persisted to SQLite so you can resume interrupted searches.

## When to Use This Skill

- A model architecture has proven competitive (OOF better than all alternatives tried).
- You have budget for 50–200 trials (≳ 30 min CPU / < 5 min GPU per trial).
- **Do NOT tune** before a solid feature set exists — HPO on bad features wastes compute.
- **Do NOT tune** more than one architecture simultaneously — pick the winner first.

## Core Capabilities

1. **LightGBM + Optuna** — full study with fold-level MedianPruner. Results saved to `hpo.db` (SQLite) — resume any interrupted run automatically. See `scripts/lgbm_optuna.py`.
2. **XGBoost Template** — parameter ranges in `scripts/lgbm_optuna.py`, `xgb_objective` function.
3. **CatBoost Template** — parameter ranges in `scripts/lgbm_optuna.py`, `catboost_objective` function.

## Quick Start

```bash
uv add optuna lightgbm
# Edit scripts/lgbm_optuna.py: set DATA_DIR, TARGET, MODEL, METRIC_DIR
uv run python scripts/lgbm_optuna.py
```

After `N_TRIALS` completes: copy `best_params` into `src/models.py` and retrain with 5 folds.

---

## LightGBM Parameter Ranges

| Parameter | Range | Notes |
| --- | --- | --- |
| `n_estimators` | 200–2000 | Use early stopping with patience=50 |
| `learning_rate` | 1e-3–0.3 (log) | Lower rate → more trees needed |
| `num_leaves` | 20–300 | Primary complexity control |
| `max_depth` | 3–12 | |
| `min_child_samples` | 5–100 | Min samples per leaf |
| `subsample` | 0.5–1.0 | Row subsampling per tree |
| `colsample_bytree` | 0.5–1.0 | Column subsampling per tree |
| `reg_alpha` | 1e-8–10 (log) | L1 regularisation |
| `reg_lambda` | 1e-8–10 (log) | L2 regularisation |

## XGBoost Parameter Ranges

| Parameter | Range |
| --- | --- |
| `n_estimators` | 200–2000 |
| `learning_rate` | 1e-3–0.3 (log) |
| `max_depth` | 3–10 |
| `min_child_weight` | 1–20 |
| `subsample` | 0.5–1.0 |
| `colsample_bytree` | 0.5–1.0 |
| `gamma` | 0–5 |
| `reg_alpha` | 1e-8–10 (log) |
| `reg_lambda` | 1e-8–10 (log) |

## CatBoost Parameter Ranges

| Parameter | Range |
| --- | --- |
| `iterations` | 200–2000 |
| `learning_rate` | 1e-3–0.3 (log) |
| `depth` | 4–10 |
| `l2_leaf_reg` | 1e-8–10 (log) |
| `bagging_temperature` | 0–1 |
| `random_strength` | 0–10 |

---

## Best Practices

- Use `N_FOLDS=3` for HPO speed; retrain the winner with 5 folds.
- Always set `random_state=42` for reproducibility.
- Interrupted study? Re-run — `load_if_exists=True` resumes from `hpo.db`.
- HPO on a weak feature set wastes compute. Establish features first.

## Quick Workflow

1. Edit `scripts/lgbm_optuna.py` — set `DATA_DIR`, `TARGET`, `MODEL`, `METRIC_DIR`.
2. Run `uv add optuna lightgbm && uv run python scripts/lgbm_optuna.py`.
3. Study persists to `hpo.db` — re-run to resume if interrupted.
4. Copy printed `best_params` into `src/models.py`.
5. Retrain with 5 folds using best params and report new OOF score.

