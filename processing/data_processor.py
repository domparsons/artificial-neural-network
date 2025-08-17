import numpy as np
import pandas as pd


def load_data(file_path: str, sheet_name: str, columns: list) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=columns)

    return df


def process_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = clean_data(df, target_col, impute_features=True)
    df = select_features(df, target_col, correlation_threshold=0.2)

    return df


def clean_data(
    df: pd.DataFrame, target_col: str, impute_features: bool = True, sentinel: int = -999
) -> pd.DataFrame:
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[df[target_col].notna() & (df[target_col] != sentinel)]

    if impute_features:
        feature_cols = [c for c in df.columns if c != target_col]
        df[feature_cols] = df[feature_cols].replace(sentinel, np.nan)
        if impute_features:
            df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    df = df.dropna()

    return df


def select_features(
    df: pd.DataFrame,
    target_col: str,
    correlation_threshold: float,
) -> pd.DataFrame:
    correlations = df.corr()[target_col]
    to_drop = [
        col
        for col, corr in correlations.items()
        if col != target_col and abs(corr) < correlation_threshold
    ]
    return df.drop(columns=to_drop)
