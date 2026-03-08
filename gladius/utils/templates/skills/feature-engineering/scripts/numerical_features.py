"""Numerical feature engineering recipes."""
import numpy as np
import pandas as pd


def add_numerical_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Add log, sqrt, power, binning, outlier flag, and interaction features."""
    df = df.copy()
    for col in cols:
        x = df[col]
        df[f"{col}_log1p"]    = np.log1p(x.clip(lower=0))
        df[f"{col}_sqrt"]     = np.sqrt(x.clip(lower=0))
        df[f"{col}_squared"]  = x ** 2
        df[f"{col}_bin"]      = pd.qcut(x, q=10, labels=False, duplicates="drop")
        z = (x - x.mean()) / (x.std() + 1e-9)
        df[f"{col}_outlier"]  = (z.abs() > 3).astype(int)
    return df


def add_pairwise_interactions(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    """Multiply, divide, and subtract two numeric columns."""
    df = df.copy()
    df[f"{col_a}_x_{col_b}"]     = df[col_a] * df[col_b]
    df[f"{col_a}_div_{col_b}"]   = df[col_a] / (df[col_b] + 1e-9)
    df[f"{col_a}_minus_{col_b}"] = df[col_a] - df[col_b]
    return df
