# sales-forcasting/src/streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_raw
from preprocess import clean_basic, aggregate_by_year_platform
from features import add_lag_features, add_rolling_features, encode_categoricals
from models import load_model

st.set_page_config(
    page_title="🎮 Video Game Sales Forecasting",
    page_icon="🎮",
    layout="wide"
)

# --- Load model ---
MODEL_PATH = "../models/rf_model.joblib"
model = load_model(MODEL_PATH)

# --- Load and preprocess data ---
raw = load_raw()
clean = clean_basic(raw)
df_feat = aggregate_by_year_platform(clean)
df_feat = add_lag_features(df_feat, value_col="total_global_sales", lags=[1,2,3])
df_feat = add_rolling_features(df_feat, value_col="total_global_sales", windows=[2,3])
df_feat = encode_categoricals(df_feat, ["Platform"])
df_feat = df_feat.dropna().reset_index(drop=True)

X = df_feat.drop(columns=["total_global_sales", "Year", "Platform"])
y = df_feat["total_global_sales"]
preds = model.predict(X)
df_feat["forecast"] = preds

# --- Sidebar Controls ---
st.sidebar.header("Controls")
platforms = sorted(df_feat["Platform"].unique())
platform = st.sidebar.selectbox("Select Platform", platforms)
year = st.sidebar.slider("Select Year", int(df_feat["Year"].min()), int(df_feat["Year"].max()), int(df_feat["Year"].max()))

# --- Filter Data ---
filtered = df_feat[df_feat["Platform"] == platform].sort_values("Year")

# --- Title ---
st.title("🎮 Video Game Sales Forecasting Dashboard")
st.markdown(f"📊 Forecasting global sales for **{platform}** platform")

# --- KPI Cards ---
latest_year = filtered[filtered["Year"] == year]
if not latest_year.empty:
    actual = latest_year["total_global_sales"].values[0]
    forecast = latest_year["forecast"].values[0]
else:
    actual = None
    forecast = None

col1, col2, col3 = st.columns(3)
col1.metric("Platform", platform)
col2.metric("Year", year)
col3.metric("Forecasted Sales", f"{forecast:.2f} M" if forecast else "N/A")

# --- Chart ---
st.subheader("📈 Historical vs Forecasted Sales")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(filtered["Year"], filtered["total_global_sales"], label="Actual Sales", marker="o")
ax.plot(filtered["Year"], filtered["forecast"], label="Forecasted Sales", marker="x", linestyle="--")
ax.set_xlabel("Year")
ax.set_ylabel("Global Sales (Millions)")
ax.set_title(f"{platform} Sales Forecast")
ax.legend()
st.pyplot(fig)

# --- Data Table ---
st.subheader("📋 Data Table")
st.dataframe(filtered[["Year", "total_global_sales", "forecast"]].round(2))

# --- Extra Insights ---
st.sidebar.markdown("### Insights")
st.sidebar.info(
    "💡 This dashboard uses lag features and rolling averages "
    "to forecast sales trends for each platform."
)
