# recommendation-system/backend/app.py

# backend/app.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import pickle
import os

app = FastAPI()

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
DATA_PATH = "./data/Books.csv"
PICKLE_PATH = "./models/cf_matrix.pkl"

# Load dataset
try:
    books_df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"✅ Loaded {len(books_df)} books")

    # Normalize columns: rename ISBN → book_id
    if "ISBN" in books_df.columns:
        books_df = books_df.rename(columns={"ISBN": "book_id"})
except FileNotFoundError:
    print(f"⚠️ Books dataset not found at {DATA_PATH}")
    books_df = pd.DataFrame()

# Load model
cf_matrix = {}
if os.path.exists(PICKLE_PATH):
    try:
        with open(PICKLE_PATH, "rb") as f:
            cf_matrix = pickle.load(f)
        print("✅ Model loaded")
    except Exception as e:
        print("⚠️ Error loading pickle:", e)
else:
    print("⚠️ Model not found, using dummy fallback.")


class RecommendationRequest(BaseModel):
    user_id: Optional[int] = None
    liked_book_ids: list[str] = []
    k: int = 10


@app.get("/books")
def list_books(q: str = Query("", min_length=0)):
    """Simple search in Books.csv"""
    if books_df.empty:
        return []

    results = books_df[
        books_df["Book-Title"].str.contains(q, case=False, na=False)
        | books_df["Book-Author"].str.contains(q, case=False, na=False)
    ].head(20)

    # Return consistent schema
    return [
        {
            "book_id": row.get("book_id", "unknown"),
            "title": row.get("Book-Title", "Unknown"),
            "author": row.get("Book-Author", "Unknown"),
        }
        for _, row in results.iterrows()
    ]


@app.post("/recommend")
def recommend(req: RecommendationRequest):
    """Dummy recommender – returns random top-k books."""
    print(f"📩 Request: {req.dict()}")

    if books_df.empty:
        return {"recommendations": [
            {"book_id": "dummy1", "title": "The Pragmatic Programmer", "author": "Andrew Hunt"},
            {"book_id": "dummy2", "title": "Clean Code", "author": "Robert C. Martin"},
            {"book_id": "dummy3", "title": "Deep Learning", "author": "Ian Goodfellow"},
        ]}

    # Just return top-k random books
    sample_books = books_df.sample(min(req.k, len(books_df)))
    recommendations = [
        {
            "book_id": row.get("book_id", row.get("ISBN", "unknown")),
            "title": row.get("Book-Title", "Unknown"),
            "author": row.get("Book-Author", "Unknown"),
        }
        for _, row in sample_books.iterrows()
    ]

    return {"recommendations": recommendations}

