# sales-forcing/src/config.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import joblib
from typing import Any


def build_ridge() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42)),
    ])


def build_rf() -> Pipeline:
    return Pipeline([
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ])


def save_model(model: Any, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    return joblib.load(path)
