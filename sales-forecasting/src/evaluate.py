# sales-forcasting/src/evaluate.py

import pandas as pd
import numpy as np
import fire
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .data_loader import load_raw
from .preprocess import clean_basic, aggregate_by_year_platform
from .features import add_lag_features, add_rolling_features, encode_categoricals
from .models import load_model



def evaluate(model_path: str):
    raw = load_raw()
    clean = clean_basic(raw)

    df_feat = aggregate_by_year_platform(clean)
    df_feat = add_lag_features(df_feat, value_col="total_global_sales", lags=[1, 2, 3])
    df_feat = add_rolling_features(df_feat, value_col="total_global_sales", windows=[2, 3])
    df_feat = encode_categoricals(df_feat, ["Platform"])
    df_feat = df_feat.dropna().reset_index(drop=True)

    X = df_feat.drop(columns=["total_global_sales", "Year", "Platform"])
    y = df_feat["total_global_sales"]

    model = load_model(model_path)
    preds = model.predict(X)

    rmse = mean_squared_error(y, preds, squared=False)
    mae = mean_absolute_error(y, preds)

    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")


if __name__ == "__main__":
    fire.Fire(evaluate)
