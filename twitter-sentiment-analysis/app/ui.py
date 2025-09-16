# Streamlit UI for frontend

# app/ui.py
import os
import joblib
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from wordcloud import WordCloud

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "twitter_validation.csv")

# -------------------------
# Page / Sidebar
# -------------------------
st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="wide")
st.sidebar.title("🎨 Theme Mode")
theme = st.sidebar.radio("Select Theme", ["Light", "Dark"], index=1)

# -------------------------
# Chart Styling
# -------------------------
def set_chart_theme(mode: str):
    if mode == "Dark":
        plt.style.use("dark_background")
        mpl.rcParams.update({
            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "text.color": "white",
            "figure.facecolor": "#0E1117",
            "axes.facecolor": "#0E1117",
        })
        sns.set_palette("pastel")
    else:
        plt.style.use("default")
        sns.set_palette("deep")

set_chart_theme(theme)

# -------------------------
# Load Artifacts
# -------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts(model_path: str, vec_path: str):
    model_ = joblib.load(model_path)
    vectorizer_ = joblib.load(vec_path)
    return model_, vectorizer_

@st.cache_data(show_spinner=False)
def load_validation(path: str) -> pd.DataFrame:
    df_ = pd.read_csv(
        path,
        header=None,
        names=["id", "entity", "sentiment", "content"],
        encoding_errors="ignore",
    )
    df_["sentiment"] = df_["sentiment"].astype(str).str.capitalize()
    df_["content"] = df_["content"].astype(str)
    return df_

# Fail-fast checks
missing = []
if not os.path.exists(MODEL_PATH): missing.append(MODEL_PATH)
if not os.path.exists(VECTORIZER_PATH): missing.append(VECTORIZER_PATH)
if missing:
    st.error(
        "Model artifacts missing:\n\n" + "\n".join(f"- {p}" for p in missing) +
        "\n\nRun `python main.py` to train and save them into `/models`."
    )
    st.stop()

model, vectorizer = load_artifacts(MODEL_PATH, VECTORIZER_PATH)

if not os.path.exists(DATA_PATH):
    st.warning(
        f"Validation CSV not found at:\n{DATA_PATH}\n\n"
        "Using charts requires the validation file."
    )
    df = pd.DataFrame(columns=["id", "entity", "sentiment", "content"])
else:
    df = load_validation(DATA_PATH)

# -------------------------
# Header
# -------------------------
st.title("📊 Twitter Sentiment Analyzer")
st.write("Classify tweet sentiment and explore validation data with charts & word clouds.")

# -------------------------
# Input / Samples
# -------------------------
st.sidebar.header("⚙️ Options")
samples = [
    "I love the new iPhone design, it's amazing!",
    "This service is terrible, very disappointing.",
    "Nothing special, just another day.",
    "Great battery life but the camera is mediocre.",
    "Worst update ever. Everything is broken.",
]
prefill = st.sidebar.selectbox("Try a sample tweet:", [""] + samples, index=1)

st.subheader("✍️ Enter Tweet")
tweet = st.text_area("Paste a tweet to classify", prefill, height=120)

col_pred, col_hist = st.columns([1, 1])

with col_pred:
    if st.button("🔍 Analyze Sentiment", use_container_width=True):
        if tweet.strip():
            vec = vectorizer.transform([tweet])
            pred = model.predict(vec)[0]
            st.success(f"**Predicted Sentiment:** {pred}")
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({"text": tweet, "pred": pred})
        else:
            st.warning("Please enter a tweet first.")

with col_hist:
    st.markdown("#### 🗂️ Prediction History")
    if "history" in st.session_state and st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No predictions yet.")

st.markdown("---")

# -------------------------
# Charts (Validation Data)
# -------------------------
if not df.empty:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📊 Sentiment Distribution (Validation)")
        fig, ax = plt.subplots()
        sns.countplot(x="sentiment", data=df, ax=ax)
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        st.pyplot(fig, clear_figure=True)

    with c2:
        st.subheader("☁️ Word Cloud (Validation Tweets)")
        text = " ".join(df["content"].dropna().astype(str).tolist())
        if text.strip():
            wc_bg = "#0E1117" if theme == "Dark" else "white"
            wc = WordCloud(
                width=900, height=450,
                background_color=wc_bg,
                colormap="viridis",
                collocations=False,
            ).generate(text)
            fig2, ax2 = plt.subplots()
            ax2.imshow(wc, interpolation="bilinear")
            ax2.axis("off")
            st.pyplot(fig2, clear_figure=True)
        else:
            st.caption("No text available to build a word cloud.")

    st.subheader("📈 Entity-wise Sentiment (Top 20 Entities)")
    top_entities = df["entity"].value_counts().head(20).index.tolist()
    ent_df = df[df["entity"].isin(top_entities)].copy()
    if not ent_df.empty:
        fig3, ax3 = plt.subplots(figsize=(11, 5))
        sns.countplot(
            data=ent_df,
            x="entity",
            hue="sentiment",
            ax=ax3,
        )
        ax3.set_xlabel("Entity")
        ax3.set_ylabel("Count")
        ax3.set_title("Sentiment per Entity (Top 20)")
        ax3.tick_params(axis="x", rotation=45)
        st.pyplot(fig3, clear_figure=True)
    else:
        st.caption("Not enough entity data to plot.")
else:
    st.info("Load the validation CSV to see charts.")






