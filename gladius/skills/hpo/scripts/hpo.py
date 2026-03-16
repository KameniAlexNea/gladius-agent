"""
Bayesian hyperparameter optimisation with Optuna.

Supports LightGBM, XGBoost, and CatBoost.
Results persisted to SQLite — resume any interrupted study with load_if_exists=True.

Usage:
    uv add optuna lightgbm
    uv run python <skill-path>/scripts/lgbm_optuna.py

After N_TRIALS completes, copy best_params into src/models.py and retrain with 5 folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Configure these before running ───────────────────────────────────────────
DATA_DIR = "data"
TARGET = "target"  # target column name
N_FOLDS = 3  # 3 for HPO speed; retrain winner with 5
N_TRIALS = 100
METRIC_DIR = "maximize"  # or "minimize"
MODEL = "lgbm"  # "lgbm" | "xgb" | "catboost"
# ─────────────────────────────────────────────────────────────────────────────


def metric_fn(y_true, y_pred):
    """Swap for your competition metric."""
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(y_true, y_pred)


def lgbm_objective(trial, X, y):
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "verbose": -1,
    }
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for fold, (tr, val) in enumerate(cv.split(X, y)):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X.iloc[tr],
            y.iloc[tr],
            eval_set=[(X.iloc[val], y.iloc[val])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        oof[val] = model.predict_proba(X.iloc[val])[:, 1]
        trial.report(metric_fn(y.iloc[val], oof[val]), fold)
        if trial.should_prune():
            import optuna

            raise optuna.TrialPruned()
    return metric_fn(y, oof)


def xgb_objective(trial, X, y):
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "verbosity": 0,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    }
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for fold, (tr, val) in enumerate(cv.split(X, y)):
        model = xgb.XGBClassifier(**params)
        model.fit(
            X.iloc[tr],
            y.iloc[tr],
            eval_set=[(X.iloc[val], y.iloc[val])],
            verbose=False,
            early_stopping_rounds=50,
        )
        oof[val] = model.predict_proba(X.iloc[val])[:, 1]
        trial.report(metric_fn(y.iloc[val], oof[val]), fold)
        if trial.should_prune():
            import optuna

            raise optuna.TrialPruned()
    return metric_fn(y, oof)


def catboost_objective(trial, X, y):
    from catboost import CatBoostClassifier
    from sklearn.model_selection import StratifiedKFold

    params = {
        "iterations": trial.suggest_int("iterations", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "random_seed": 42,
        "verbose": 0,
    }
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for fold, (tr, val) in enumerate(cv.split(X, y)):
        model = CatBoostClassifier(**params)
        model.fit(
            X.iloc[tr],
            y.iloc[tr],
            eval_set=(X.iloc[val], y.iloc[val]),
            early_stopping_rounds=50,
        )
        oof[val] = model.predict_proba(X.iloc[val])[:, 1]
        trial.report(metric_fn(y.iloc[val], oof[val]), fold)
        if trial.should_prune():
            import optuna

            raise optuna.TrialPruned()
    return metric_fn(y, oof)


if __name__ == "__main__":
    import optuna

    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    X = train.drop(columns=[TARGET])
    y = train[TARGET]

    objectives = {
        "lgbm": lgbm_objective,
        "xgb": xgb_objective,
        "catboost": catboost_objective,
    }
    objective = lambda trial: objectives[MODEL](trial, X, y)  # noqa: E731

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(
        direction=METRIC_DIR,
        sampler=sampler,
        pruner=pruner,
        storage="sqlite:///hpo.db",
        study_name=f"{MODEL}_hpo",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\nBest OOF {METRIC_DIR}: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    print("\nNext: copy best_params into src/models.py and retrain with N_FOLDS=5")
