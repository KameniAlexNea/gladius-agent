"""Temporal / datetime feature engineering recipes."""

import numpy as np
import pandas as pd


def add_datetime_features(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    """Decompose a datetime column into periodic features with cyclic encoding."""
    df = df.copy()
    dt = pd.to_datetime(df[dt_col])
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek  # 0=Monday
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["quarter"] = dt.dt.quarter
    # Cyclic encoding for periodic signals
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    # Days since start
    ref = dt.min()
    df["days_since_start"] = (dt - ref).dt.days
    return df


def add_lag_features(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    value_col: str,
    lags: list[int] = [1, 2, 3, 7, 14],
) -> pd.DataFrame:
    """
    Compute lag features for time-series data.
    MUST sort by entity + time before calling, or results will be silently wrong.
    """
    df = df.copy().sort_values([entity_col, time_col])
    g = df.groupby(entity_col)[value_col]
    for lag in lags:
        df[f"{value_col}_lag{lag}"] = g.shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    value_col: str,
    windows: list[int] = [3, 7, 14],
) -> pd.DataFrame:
    """
    Rolling mean and std — uses shift(1) before rolling to prevent leakage.
    MUST sort by entity + time before calling.
    """
    df = df.copy().sort_values([entity_col, time_col])
    g = df.groupby(entity_col)[value_col]
    for w in windows:
        shifted = g.shift(1)
        rolled = shifted.groupby(df[entity_col]).rolling(w, min_periods=1)
        df[f"{value_col}_roll_mean{w}"] = rolled.mean().reset_index(level=0, drop=True)
        df[f"{value_col}_roll_std{w}"] = (
            rolled.std().reset_index(level=0, drop=True).fillna(0)
        )
    return df


def make_group_aggs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    group_col: str,
    value_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-group statistics fit on train only, then mapped to test.
    Returns (train_with_aggs, test_with_aggs).
    """
    agg = train.groupby(group_col)[value_col].agg(
        ["mean", "std", "min", "max", "count"]
    )
    agg.columns = [f"{group_col}_{value_col}_{s}" for s in agg.columns]
    agg = agg.reset_index()
    train = train.merge(agg, on=group_col, how="left")
    test = test.merge(agg, on=group_col, how="left")
    return train, test
