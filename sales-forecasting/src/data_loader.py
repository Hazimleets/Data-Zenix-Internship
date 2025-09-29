# sales-forcasting/src/data_loader.py

import os
import pandas as pd

CSV_NAME = "vgsales.csv"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", CSV_NAME)

def load_raw(path: str = None) -> pd.DataFrame:
    """
    Load raw CSV. If path is None, load from data/CSV_NAME.
    """
    if path is None:
        path = DATA_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Please download from Kaggle and place in the data/ folder."
        )

    df = pd.read_csv(path)
    return df
