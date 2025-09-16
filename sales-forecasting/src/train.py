# sales-forcasting/src/train.py

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .data_loader import load_raw
from .preprocess import clean_basic, aggregate_by_year_platform
from .features import add_lag_features, add_rolling_features, encode_categoricals
from .models import build_ridge, build_rf, save_model
from .config import MODEL_DIR, TARGET


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    agg = aggregate_by_year_platform(df)

    # create features
    agg = add_lag_features(agg, value_col="total_global_sales", lags=[1, 2, 3])
    agg = add_rolling_features(agg, value_col="total_global_sales", windows=[2, 3])
    agg = encode_categoricals(agg, ["Platform"])

    # drop rows with NA due to lags
    agg = agg.dropna().reset_index(drop=True)
    return agg


def train_local(save_path: str = None):
    raw = load_raw()
    clean = clean_basic(raw)
    df_feat = prepare_features(clean)

    X = df_feat.drop(columns=["total_global_sales", "Year", "Platform"])
    y = df_feat["total_global_sales"]

    # time-series aware CV
    tscv = TimeSeriesSplit(n_splits=5)
    rf = build_rf()
    scores = -cross_val_score(
        rf, X, y, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    print(f"Cross-validated RMSE: {scores.mean():.4f} ± {scores.std():.4f}")

    # fit model on full dataset
    rf.fit(X, y)

    # save model if path provided
    if save_path:
        save_model(rf, save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sales forecasting model locally")
    parser.add_argument("--save", type=str, default=None, help="Path to save the trained model")
    args = parser.parse_args()

    train_local(args.save)
