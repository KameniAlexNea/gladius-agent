"""
Hyperparameter optimisation with Optuna.
Supports LightGBM, XGBoost, and CatBoost via MODEL_TYPE flag.

Usage:
    python scripts/lgbm_optuna.py

Results are persisted to hpo.db — re-running resumes from where it stopped.
"""

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score  # ← replace with competition metric
from sklearn.model_selection import StratifiedKFold  # use KFold for regression

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = "data"  # ← adjust
TARGET = "target"  # ← replace
METRIC_DIRECTION = "maximize"  # "maximize" or "minimize"
N_FOLDS = 3  # use 3 for speed during HPO; retrain with 5 after
N_TRIALS = 100
RANDOM_STATE = 42
MODEL_TYPE = "lgbm"  # "lgbm" | "xgb" | "catboost"


def load_data():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    X = train.drop(columns=[TARGET])
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.codes
    X = X.fillna(-999)
    y = train[TARGET]
    return X, y


def objective_lgbm(trial, X, y):
    import lightgbm as lgb

    params = {
        "objective": "binary",  # ← adjust per task
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": RANDOM_STATE,
    }
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X))
    for fold_i, (tr, va) in enumerate(cv.split(X, y)):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X.iloc[tr],
            y.iloc[tr],
            eval_set=[(X.iloc[va], y.iloc[va])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        trial.report(roc_auc_score(y.iloc[va], oof[va]), fold_i)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return roc_auc_score(y, oof)


# XGB_PARAMS — swap into objective for XGBoost
XGB_PARAMS = lambda trial: {  # noqa E731
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "max_depth": trial.suggest_int("max_depth", 3, 10),
    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
    "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    "random_state": RANDOM_STATE,
}

# CATBOOST_PARAMS — swap into objective for CatBoost

CATBOOST_PARAMS = lambda trial: {  # noqa E731
    "iterations": trial.suggest_int("iterations", 100, 2000),
    "depth": trial.suggest_int("depth", 4, 10),
    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    "random_seed": RANDOM_STATE,
    "verbose": 0,
}


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, y = load_data()

    study = optuna.create_study(
        direction=METRIC_DIRECTION,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
        study_name=f"{MODEL_TYPE}_hpo",
        storage="sqlite:///hpo.db",
        load_if_exists=True,
    )
    study.optimize(
        lambda t: objective_lgbm(t, X, y), n_trials=N_TRIALS, show_progress_bar=True
    )

    print(f"\nBest trial: {study.best_value:.6f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Show top-10 trials
    df_trials = study.trials_dataframe()
    print("\nTop 10 trials:")
    print(
        df_trials.sort_values("value", ascending=(METRIC_DIRECTION == "minimize"))
        .head(10)
        .to_string()
    )
