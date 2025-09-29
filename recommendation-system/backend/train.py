# backend/train.py
import pandas as pd
import numpy as np
import pickle
import os

DATA_PATH = "data/books.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cf_matrix.pkl")

# Ensure models dir exists
os.makedirs(MODEL_DIR, exist_ok=True)

print("📖 Loading dataset...")
books_df = pd.read_csv(DATA_PATH)

# Dummy collaborative filtering (replace with ML later)
print("⚙️ Training dummy CF model...")
cf_matrix = np.random.rand(len(books_df), 100)

print("💾 Saving model...")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(cf_matrix, f)

print(f"✅ Model saved at {MODEL_PATH}")
