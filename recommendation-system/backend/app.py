# backend/app.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from scipy.sparse import csr_matrix
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

# --- Paths ---
ROOT = os.path.dirname(__file__)
MODEL_DIR = os.path.join(ROOT, "models")

# --- FastAPI app ---
app = FastAPI(title="Book Recommender API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Input schema ---
class PreferenceIn(BaseModel):
    user_id: str = None
    liked_book_ids: list[str] = []

# --- Lazy load models ---
_user_item_matrix = None
_cf_matrix = None
_content_matrix = None
_tfidf = None
_user_enc = None
_item_enc = None
_books_df = None


def load_models():
    global _user_item_matrix, _cf_matrix, _content_matrix, _tfidf, _user_enc, _item_enc, _books_df
    if _user_item_matrix is None:
        print("📦 Loading models from backend/models...")

        # Load saved artifacts
        _user_item_matrix = joblib.load(os.path.join(MODEL_DIR, "user_item_matrix.pkl"))
        _cf_matrix = joblib.load(os.path.join(MODEL_DIR, "cf_matrix.pkl"))
        _content_matrix = joblib.load(os.path.join(MODEL_DIR, "content_matrix.pkl"))
        _tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
        _user_enc = joblib.load(os.path.join(MODEL_DIR, "user_encoder.pkl"))
        _item_enc = joblib.load(os.path.join(MODEL_DIR, "item_encoder.pkl"))

        # Load books.csv for metadata
        import pandas as pd
        books_path = os.path.join(ROOT, "data", "Books.csv")
        _books_df = pd.read_csv(books_path, on_bad_lines="skip")
        _books_df.rename(columns={"ISBN": "book_id"}, inplace=True)

        print("✅ Models loaded successfully!")

    return _user_item_matrix, _cf_matrix, _content_matrix, _tfidf, _user_enc, _item_enc, _books_df


# --- Health endpoint ---
@app.get("/health")
async def health():
    return {"status": "ok"}


# --- Get books for search ---
@app.get("/books")
async def get_books(q: str = ""):
    _, _, _, _, _, _, books_df = load_models()

    if q:
        filtered = books_df[
            books_df["Book-Title"].str.contains(q, case=False, na=False)
            | books_df["Book-Author"].str.contains(q, case=False, na=False)
        ].head(20)
    else:
        filtered = books_df.head(20)

    return filtered[["book_id", "Book-Title", "Book-Author"]].to_dict(orient="records")


# --- Recommend endpoint ---
@app.post("/recommend")
async def recommend(pref: PreferenceIn, k: int = 10):
    user_item_matrix, cf_matrix, content_matrix, tfidf, user_enc, item_enc, books_df = load_models()

    results = []

    # Case 1: user_id known
    if pref.user_id:
        try:
            user_idx = user_enc.transform([pref.user_id])[0]
            user_ratings = user_item_matrix[user_idx].toarray().flatten()
            scores = cf_matrix[user_idx].dot(cf_matrix.T)
            top_indices = np.argsort(scores)[::-1][:k]
            for idx in top_indices:
                book_id = item_enc.inverse_transform([idx])[0]
                row = books_df[books_df["book_id"] == book_id].iloc[0]
                results.append({
                    "book_id": book_id,
                    "title": row["Book-Title"],
                    "author": row["Book-Author"]
                })
        except Exception:
            pass

    # Case 2: cold start with liked_book_ids
    elif pref.liked_book_ids:
        liked_idxs = []
        for bid in pref.liked_book_ids:
            try:
                liked_idxs.append(item_enc.transform([bid])[0])
            except Exception:
                continue

        if liked_idxs:
            sim_scores = content_matrix[liked_idxs].mean(axis=0).A1
            top_indices = np.argsort(sim_scores)[::-1][:k]
            for idx in top_indices:
                book_id = item_enc.inverse_transform([idx])[0]
                row = books_df[books_df["book_id"] == book_id].iloc[0]
                results.append({
                    "book_id": book_id,
                    "title": row["Book-Title"],
                    "author": row["Book-Author"]
                })

    return {"recommendations": results}
