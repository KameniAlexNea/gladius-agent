"""Categorical feature engineering recipes (all leakage-safe)."""

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def add_frequency_encoding(
    df: pd.DataFrame, col: str, train_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Frequency encode a column. Fit on train_df only (or df if not provided)."""
    df = df.copy()
    source = train_df if train_df is not None else df
    freq = source[col].value_counts()
    df[f"{col}_freq"] = df[col].map(freq).fillna(0)
    return df


def target_encode_fold(
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
    test_df: pd.DataFrame,
    col: str,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fold-safe target encoding.
    MUST be called inside the CV loop — never computed on the full training set.
    """
    enc = train_fold.groupby(col)[target].mean()
    global_mean = train_fold[target].mean()
    for part in (train_fold, val_fold, test_df):
        part = part.copy()
    train_fold = train_fold.copy()
    val_fold = val_fold.copy()
    test_df = test_df.copy()
    train_fold[f"{col}_te"] = train_fold[col].map(enc).fillna(global_mean)
    val_fold[f"{col}_te"] = val_fold[col].map(enc).fillna(global_mean)
    test_df[f"{col}_te"] = test_df[col].map(enc).fillna(global_mean)
    return train_fold, val_fold, test_df


def add_rare_grouping(
    df: pd.DataFrame, col: str, min_count: int = 50, label: str = "__RARE__"
) -> pd.DataFrame:
    """Group rare categories into a single bucket."""
    df = df.copy()
    counts = df[col].value_counts()
    rare = counts[counts < min_count].index
    df[f"{col}_grouped"] = df[col].replace(rare, label)
    return df


def add_ordinal_encoding(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Label-encode for tree models (handles unknown categories)."""
    df = df.copy()
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[cols] = oe.fit_transform(df[cols].astype(str))
    return df


def add_combination_feature(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    """Concatenate two low-cardinality categorical columns."""
    df = df.copy()
    df[f"{col_a}_x_{col_b}"] = df[col_a].astype(str) + "_" + df[col_b].astype(str)
    return df
