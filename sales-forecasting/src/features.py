# sales-forcasting/src/data_loader.py

import pandas as pd
import numpy as np


def add_lag_features(
    df: pd.DataFrame, 
    value_col: str = "total_global_sales", 
    lags: list[int] = [1, 2, 3, 4]
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["Platform", "Year"]).reset_index(drop=True)
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("Platform")[value_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, 
    value_col: str = "total_global_sales", 
    windows: list[int] = [2, 3]
) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"roll_mean_{w}"] = df.groupby("Platform")[value_col].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
    return df


def encode_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cat_cols:
        df[c] = df[c].astype(str)
        df[c + "_enc"] = df[c].factorize()[0]
    return df
