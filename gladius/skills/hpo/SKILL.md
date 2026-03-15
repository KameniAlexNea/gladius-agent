---
name: hpo
description: Bayesian hyperparameter optimisation with Optuna for LightGBM, XGBoost, and CatBoost. Uses TPE sampler with fold-level MedianPruner to find good parameters in 50–200 trials. Results persist to SQLite so interrupted searches resume automatically. Only invoke after a competitive model architecture and stable feature set are established.
---

# Hyperparameter Optimisation (HPO)

Optuna's TPE sampler finds good hyperparameters in 50–200 trials. The MedianPruner kills bad trials after 2 folds, saving 30–50% of compute. HPO on a weak feature set is wasted — establish features first.

## When to Use

- One model architecture has proven best (best OOF across alternatives).
- Feature set is stable — at least one round of SHAP pruning done.
- You have budget for 50–200 trials (≳ 30 min CPU / GPU per trial).
- **Not** before selecting an architecture — pick the winner first.
- **Not** before feature engineering — HPO on bad features wastes compute.

## Critical Rules

### ✅ DO

- **Use `N_FOLDS=3` for HPO speed** — retrain the winner with 5 folds after search completes.
- **Use MedianPruner** — kills trials that underperform after 2 folds, cutting compute by 30–50%.
- **Persist study to SQLite** (`storage="sqlite:///hpo.db"`, `load_if_exists=True`) — interrupted searches resume automatically.
- **Fix `random_state=42`** inside the objective for reproducibility.
- **Run HPO with early stopping** — `n_estimators=2000` + `early_stopping_rounds=50` lets Optuna find the right tree count automatically.

### ❌ DON'T

- **Don't tune multiple architectures simultaneously** — pick the best first, then tune it.
- **Don't use `n_estimators` as a Optuna parameter if you use early stopping** — let early stopping determine it.
- **Don't run both XGBoost and LightGBM HPO in the same iteration** — too expensive and redundant.
- **Don't copy HPO params without retraining with 5 folds** — 3-fold OOF is noisier; recheck with 5.

## Anti-Patterns (NEVER)

```python
# ❌ BAD: Using cross_val_score inside Optuna — no fold-level pruning, no early stopping
def objective(trial):
    params = {"max_depth": trial.suggest_int("max_depth", 3, 10), ...}
    model = XGBClassifier(**params)
    return cross_val_score(model, X, y, cv=5).mean()  # slow, can't prune

# ✅ GOOD: Manual fold loop with fold-level reporting for pruning
def objective(trial):
    params = {...}
    oof = np.zeros(len(y))
    for fold, (tr, val) in enumerate(cv.split(X, y)):
        model = lgb.LGBMClassifier(**params)
        model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[val], y.iloc[val])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[val] = model.predict_proba(X.iloc[val])[:, 1]
        trial.report(metric_fn(y.iloc[val], oof[val]), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return metric_fn(y, oof)
```

```python
# ❌ BAD: Forgetting load_if_exists — restarts from scratch if interrupted
study = optuna.create_study(direction="maximize")

# ✅ GOOD: Study persists and resumes
study = optuna.create_study(
    direction="maximize",
    storage="sqlite:///hpo.db",
    study_name="lgbm_hpo",
    load_if_exists=True,    # resume if interrupted
)
```

## Quick Start

```bash
uv add optuna lightgbm
# Edit scripts/lgbm_optuna.py: set DATA_DIR, TARGET, MODEL, METRIC_DIR, N_TRIALS
uv run python scripts/lgbm_optuna.py
# Prints best_params — copy into src/models.py, retrain with N_FOLDS=5
```

See `scripts/lgbm_optuna.py` for the full template (LightGBM / XGBoost / CatBoost).

## Parameter Ranges

### LightGBM

| Parameter | Range | Notes |
| --- | --- | --- |
| `learning_rate` | 1e-3–0.3 (log) | Lower rate → more trees needed |
| `num_leaves` | 20–300 | Primary complexity control |
| `max_depth` | 3–12 | Caps `num_leaves` effective depth |
| `min_child_samples` | 5–100 | Min samples per leaf — regularises |
| `subsample` | 0.5–1.0 | Row subsampling per tree |
| `colsample_bytree` | 0.5–1.0 | Column subsampling per tree |
| `reg_alpha` | 1e-8–10 (log) | L1 regularisation |
| `reg_lambda` | 1e-8–10 (log) | L2 regularisation |

### XGBoost

| Parameter | Range | Notes |
| --- | --- | --- |
| `learning_rate` | 1e-3–0.3 (log) | |
| `max_depth` | 3–10 | |
| `min_child_weight` | 1–20 | |
| `subsample` | 0.5–1.0 | |
| `colsample_bytree` | 0.5–1.0 | |
| `gamma` | 0–5 | Minimum loss reduction to split |
| `reg_alpha` | 1e-8–10 (log) | L1 |
| `reg_lambda` | 1e-8–10 (log) | L2 |

### CatBoost

| Parameter | Range |
| --- | --- |
| `learning_rate` | 1e-3–0.3 (log) |
| `depth` | 4–10 |
| `l2_leaf_reg` | 1e-8–10 (log) |
| `bagging_temperature` | 0–1 |
| `random_strength` | 0–10 |

## Common Pitfalls and Solutions

### The "HPO Overfits Validation" Problem

After 200 trials, best params give great 3-fold OOF but 5-fold OOF drops. The Optuna objective implicitly overfit to the 3-fold split.

**Fix:** After HPO, always retrain with 5 folds and treat that as the true OOF. Report 5-fold, not HPO study best value.

### The "Low Learning Rate Trap"

`learning_rate=0.003` was chosen by Optuna, but OOF barely improves over default. With `n_estimators` capped, the model never converges.

**Fix:** Use `n_estimators=2000` (high cap) with `early_stopping_rounds=50`. Let early stopping find the right count for each learning rate.

### The "Resume Confusion"

Re-running the script after changing the objective function but forgetting to change `study_name` — Optuna loads old trials and corrupts the search.

**Fix:** Change `study_name` whenever the objective changes. Use descriptive names: `lgbm_hpo_v2_log1p_features`.


