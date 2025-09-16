# main.py

import pandas as pd
from src.preprocessing import preprocess_text
from src.features import vectorize_tfidf
from src.model import train_and_evaluate
from src.evaluation import evaluate_model
from src.utils import save_model, get_logger

logger = get_logger(__name__)

# 1. Load Data
logger.info("Loading training and validation datasets...")
train_df = pd.read_csv("data/raw/twitter_training.csv", header=None)
val_df = pd.read_csv("data/raw/twitter_validation.csv", header=None)

# Assign column names
train_df.columns = ["id", "entity", "sentiment", "content"]
val_df.columns = ["id", "entity", "sentiment", "content"]

# Keep only sentiment + content
train_df = train_df[["sentiment", "content"]].dropna()
val_df = val_df[["sentiment", "content"]].dropna()

logger.info(f"Training samples: {len(train_df)} | Validation samples: {len(val_df)}")

# 2. Preprocess text
logger.info("Preprocessing text...")
train_df["clean_text"] = train_df["content"].apply(preprocess_text)
val_df["clean_text"] = val_df["content"].apply(preprocess_text)

# 3. Feature extraction (TF-IDF)
logger.info("Extracting features with TF-IDF...")
X_train_vec, X_val_vec, vectorizer = vectorize_tfidf(
    train_df["clean_text"], val_df["clean_text"]
)
y_train, y_val = train_df["sentiment"], val_df["sentiment"]

# 4. Train & Evaluate
logger.info("Training model...")
model = train_and_evaluate(X_train_vec, y_train, X_val_vec, y_val)

logger.info("Evaluating model on validation set...")
evaluate_model(model, X_val_vec, y_val)

# 5. Save model + vectorizer
logger.info("Saving model and vectorizer...")
save_model(model, vectorizer)
logger.info("✅ Training pipeline complete. Model saved in 'models/'")
