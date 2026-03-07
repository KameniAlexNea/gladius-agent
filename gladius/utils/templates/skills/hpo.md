---
name: hpo
description: >
  Bayesian hyperparameter optimisation with Optuna. Use after a solid baseline
  exists to tune the best-performing model architecture.
---

## When to run HPO

- A model architecture has proven competitive (OOF better than all alternatives).
- You have budget for 50–200 trials (≳ 30 min on CPU / < 5 min on GPU).
- Do NOT tune hyperparameters before a solid feature set exists — HPO on bad
  features is wasted compute.

## Quick-start: LightGBM with Optuna

```python
import optuna
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score  # replace with competition metric

optuna.logging.set_verbosity(optuna.logging.WARNING)

METRIC_DIRECTION = "maximize"   # "minimize" for RMSE / log-loss
N_FOLDS    = 5
N_TRIALS   = 100
RANDOM_STATE = 42

def objective(trial):
    params = {
        "objective":        "binary",       # adjust per task
        "metric":           "auc",
        "verbosity":        -1,
        "boosting_type":    "gbdt",
        "num_leaves":       trial.suggest_int("num_leaves", 20, 300),
        "max_depth":        trial.suggest_int("max_depth", 3, 12),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators":     trial.suggest_int("n_estimators", 100, 2000),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state":     RANDOM_STATE,
    }

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X_train))
    for fold_i, (tr, va) in enumerate(cv.split(X_train, y_train)):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train.iloc[tr], y_train.iloc[tr],
            eval_set=[(X_train.iloc[va], y_train.iloc[va])],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        oof[va] = model.predict_proba(X_train.iloc[va])[:, 1]
        # Optuna pruning: report after each fold
        trial.report(roc_auc_score(y_train.iloc[va], oof[va]), fold_i)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return roc_auc_score(y_train, oof)


# Run the study
sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
pruner  = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
study   = optuna.create_study(
    direction=METRIC_DIRECTION,
    sampler=sampler,
    pruner=pruner,
    study_name="lgbm_hpo",
    storage="sqlite:///hpo.db",   # persist results
    load_if_exists=True,
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\nBest trial: {study.best_value:.6f}")
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
```

## XGBoost template (swap in place of LightGBM params)

```python
import xgboost as xgb

params = {
    "objective":      "binary:logistic",
    "eval_metric":    "auc",
    "tree_method":    "hist",
    "max_depth":      trial.suggest_int("max_depth", 3, 10),
    "learning_rate":  trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
    "n_estimators":   trial.suggest_int("n_estimators", 100, 2000),
    "subsample":      trial.suggest_float("subsample", 0.5, 1.0),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    "gamma":          trial.suggest_float("gamma", 1e-8, 5.0, log=True),
    "reg_alpha":      trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
    "reg_lambda":     trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    "random_state":   RANDOM_STATE,
}
```

## CatBoost template

```python
from catboost import CatBoostClassifier

params = {
    "iterations":      trial.suggest_int("iterations", 100, 2000),
    "depth":           trial.suggest_int("depth", 4, 10),
    "learning_rate":   trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
    "l2_leaf_reg":     trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    "random_seed":     RANDOM_STATE,
    "verbose":         0,
}
```

## Reading a persisted study

```python
study = optuna.load_study(study_name="lgbm_hpo", storage="sqlite:///hpo.db")
df_trials = study.trials_dataframe()
print(df_trials.sort_values("value", ascending=False).head(10).to_string())
```

## Best practices

- Always use **early stopping** inside the objective — set `n_estimators` high
  and let early stopping find the right count.
- Use the **MedianPruner** to kill bad trials fast (saves 30–50 % of compute).
- Run HPO with **`N_FOLDS=3`** first for speed; retrain the best config with 5.
- After finding best params, retrain on the full training set and regenerate
  the submission file.
- Report the HPO-tuned OOF score as the new `oof_score`.
