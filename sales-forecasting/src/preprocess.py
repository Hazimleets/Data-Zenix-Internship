# sales-forcasting/src/preprocess.py

import pandas as pd
import numpy as np
from typing import Tuple


def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning and standardization on the raw dataset.
    """
    df = df.copy()
    
    # Standardize column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    
    # Year cleanup (convert to numeric, fill with median per Platform)
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["Year"] = df.groupby("Platform")["Year"].transform(
            lambda x: x.fillna(x.median())
        )

    # Clean string columns
    for c in ["Publisher", "Genre", "Platform", "Name"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace("nan", "Unknown")

    # Ensure all sales columns are numeric
    sales_cols = [c for c in df.columns if "Sales" in c or "sales" in c]
    for c in sales_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Derive Global_Sales if column variant exists
    if "Global_Sales" not in df.columns and "Global" in df.columns:
        df["Global_Sales"] = df["Global"]

    return df


def aggregate_by_year_platform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to a time-series style table by Year and Platform.
    Creates a consistent target column 'total_global_sales'.
    """
    agg = (
        df.groupby(["Year", "Platform"])["Global_Sales"]
        .sum()
        .reset_index()
        .rename(columns={"Global_Sales": "total_global_sales"})  # ✅ ensure correct name
        .sort_values(["Platform", "Year"])
    )
    return agg

