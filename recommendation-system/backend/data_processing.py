# backend/data_processing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def load_raw_data(ratings_csv, books_csv, users_csv):
    """Load Ratings, Books, and Users data from Kaggle CSV files"""
    ratings = pd.read_csv(ratings_csv, low_memory=False)
    books = pd.read_csv(books_csv, low_memory=False)
    users = pd.read_csv(users_csv, low_memory=False)

    # Normalize column names to lowercase
    ratings.columns = [c.strip().lower() for c in ratings.columns]
    books.columns = [c.strip().lower() for c in books.columns]
    users.columns = [c.strip().lower() for c in users.columns]

    # Kaggle column mappings
    # Ratings.csv: user-id, isbn, book-rating
    # Books.csv: isbn, book-title, book-author, publisher
    # Users.csv: user-id, location, age
    if "isbn" in books.columns:
        books = books.rename(columns={"isbn": "book_id"})
    if "isbn" in ratings.columns:
        ratings = ratings.rename(columns={"isbn": "book_id"})
    if "user-id" in ratings.columns:
        ratings = ratings.rename(columns={"user-id": "user_id"})
    if "user-id" in users.columns:
        users = users.rename(columns={"user-id": "user_id"})

    return ratings, books, users


def build_user_item_matrix(ratings):
    """Build sparse user-item matrix for Collaborative Filtering"""
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    ratings["user_idx"] = user_encoder.fit_transform(ratings["user_id"])
    ratings["item_idx"] = item_encoder.fit_transform(ratings["book_id"])

    matrix = csr_matrix(
        (ratings["book-rating"], (ratings["user_idx"], ratings["item_idx"]))
    )

    meta = {
        "user_encoder": user_encoder,
        "item_encoder": item_encoder,
        "df": ratings,
        "n_items": len(item_encoder.classes_),
        "n_users": len(user_encoder.classes_),
    }
    return matrix, meta


def build_content_index(books, text_fields=None):
    """
    Build TF-IDF content matrix for book metadata.
    Kaggle dataset has: book-title, book-author, publisher
    """
    books = books.fillna("")
    books["content"] = (
        books["book-title"].astype(str)
        + " "
        + books["book-author"].astype(str)
        + " "
        + books["publisher"].astype(str)
    )

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    content_matrix = tfidf.fit_transform(books["content"])

    return content_matrix, tfidf
