# frontend/app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px
import os
import zipfile
import io
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------ Load Model ------------------------
MODEL_PATH = "../saved_models/intel_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# ------------------------ Page Config ------------------------
st.set_page_config(
    page_title="Intel Image Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------ CSS Styling ------------------------
st.markdown("""
<style>
body {
    background-color: #f9f9f9;
}
h1 {
    color: #2c3e50;
    font-size: 36px;
    text-align: center;
    font-family: 'Helvetica', sans-serif;
}
.stButton>button {
    background-color: #1abc9c;
    color: white;
    font-size: 16px;
    height: 50px;
    width: 100%;
    border-radius: 8px;
}
.stFileUploader>div {
    border: 2px dashed #1abc9c;
    padding: 15px;
    border-radius: 15px;
}
.stDataFrame td, .stDataFrame th {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------------ Title ------------------------
st.title("🌄 Intel Image Classification Portfolio Dashboard")

# ------------------------ Tabs ------------------------
tabs = st.tabs(["Single Image", "Batch Prediction", "Metrics"])

# ------------------------ Single Image Prediction ------------------------
with tabs[0]:
    st.subheader("Upload a single image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Uploaded Image", width=300)

        with col2:
            st.markdown("**Prediction Results**")
            img = image.resize((150, 150))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            with st.spinner("Predicting..."):
                preds = model.predict(img_array)[0]

            pred_class = class_names[np.argmax(preds)]
            st.markdown(f"### Predicted Class: **{pred_class.upper()}**")

            fig = px.bar(
                x=class_names,
                y=preds,
                labels={'x': 'Class', 'y': 'Probability'},
                text=np.round(preds, 3),
                title="Prediction Probabilities"
            )
            fig.update_layout(yaxis=dict(range=[0, 1]), plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

# ------------------------ Batch Prediction ------------------------
with tabs[1]:
    st.subheader("Upload multiple images (ZIP)")
    uploaded_zip = st.file_uploader("Upload ZIP file", type=["zip"])

    if uploaded_zip is not None:
        with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as zip_ref:
            zip_ref.extractall("batch_images")
        image_files = [f for f in os.listdir("batch_images") if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        st.success(f"{len(image_files)} images loaded!")

        all_preds = []
        st.markdown("### Batch Preview & Predictions")
        cols = st.columns(3)
        for idx, img_name in enumerate(image_files):
            img_path = os.path.join("batch_images", img_name)
            img = Image.open(img_path).resize((150, 150))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            preds = model.predict(img_array)[0]
            all_preds.append((img_name, preds))

            with cols[idx % 3]:
                st.image(img, width=100)
                st.markdown(f"**{class_names[np.argmax(preds)]}**")

        # Save predictions to DataFrame
        df = pd.DataFrame({
            "Image": [x[0] for x in all_preds],
            "Predicted Class": [class_names[np.argmax(x[1])] for x in all_preds],
            "Probability": [round(np.max(x[1]), 3) for x in all_preds]
        })
        st.dataframe(df)
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Download CSV", csv, "batch_predictions.csv", "text/csv")

# ------------------------ Metrics / Extra Tab ------------------------
with tabs[2]:
    st.subheader("📊 Class Distribution & Sample Predictions")

    # Safe absolute path for test set
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TEST_DIR = os.path.join(BASE_DIR, "data", "seg_test")

    test_images = []
    test_labels = []

    for class_name in class_names:
        class_path = os.path.join(TEST_DIR, class_name)
        if os.path.exists(class_path):
            files = [f for f in os.listdir(class_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))][:20]  # sample 20 max
            for img_file in files:
                img_path = os.path.join(class_path, img_file)
                try:
                    img = Image.open(img_path).resize((150, 150))
                    img_array = np.array(img) / 255.0
                    test_images.append(img_array)
                    test_labels.append(class_name)
                except:
                    continue

    if test_images:
        test_images = np.array(test_images)
        test_labels_array = np.array(test_labels)

        # Predictions
        preds = model.predict(test_images)
        pred_classes = [class_names[np.argmax(p)] for p in preds]

        # Accuracy metric
        acc = np.mean(test_labels_array == pred_classes)
        st.metric("Sample Accuracy", f"{acc*100:.2f}%")

        # True distribution
        true_counts = pd.Series(test_labels_array).value_counts()
        fig_true = px.bar(true_counts, x=true_counts.index, y=true_counts.values,
                          labels={'x': 'Class', 'y': 'Count'}, title="True Class Distribution",
                          color=true_counts.index)
        st.plotly_chart(fig_true, use_container_width=True)

        # Confusion matrix
        cm = confusion_matrix(test_labels_array, pred_classes, labels=class_names)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names,
                    yticklabels=class_names, cmap='Blues')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)

        # Sample predictions
        st.markdown("### 🔍 Sample Predictions")
        cols = st.columns(5)
        for i, (img, t, p) in enumerate(zip(test_images[:10], test_labels_array[:10], pred_classes[:10])):
            with cols[i % 5]:
                st.image(img, width=100, caption=f"T: {t}\nP: {p}")
    else:
        st.warning("⚠️ No test images found. Check your TEST_DIR path.")

# ------------------------ Sidebar ------------------------
st.sidebar.header("About")
st.sidebar.info(
    """
    **Intel Image Classification Dashboard**  

    ✅ Upload single or batch images.  
    ✅ Predict buildings, forest, glacier, mountain, sea, or street.  
    ✅ Interactive charts and portfolio-ready design.  
    ✅ Metrics: Accuracy, Distribution, Confusion Matrix.  

    **Tech Stack:** TensorFlow, Streamlit, Plotly.  
    **Model:** CNN trained on Intel Image Classification Dataset.
    """
)
