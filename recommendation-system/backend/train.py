# backend/train.py

# backend/train.py
import os
import joblib
from data_processing import load_raw_data, build_user_item_matrix, build_content_index
from recommender import Recommender

# --- Paths ---
ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")  # dataset should be inside backend/data
RATINGS_CSV = os.path.join(DATA_DIR, "Ratings.csv")
BOOKS_CSV = os.path.join(DATA_DIR, "Books.csv")
USERS_CSV = os.path.join(DATA_DIR, "Users.csv")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    print("📥 Loading raw data...")
    ratings, books, users = load_raw_data(RATINGS_CSV, BOOKS_CSV, USERS_CSV)
    
    print("✅ Raw data loaded")
    print("Ratings columns:", ratings.columns)
    print("Books columns:", books.columns)
    print("Users columns:", users.columns)
    print(f"Ratings: {ratings.shape}, Books: {books.shape}, Users: {users.shape}")

    # Safety check
    if ratings.empty or books.empty or users.empty:
        print("❌ One of the datasets is empty! Check CSV files.")
        return

    print("🧩 Building user-item matrix...")
    user_item_matrix, meta = build_user_item_matrix(ratings)
    print(f"Matrix shape: {user_item_matrix.shape}")
    print(f"Number of users: {meta['n_users']}, Number of items: {meta['n_items']}")

    print("📚 Building content TF-IDF index...")
    item_encoder = meta["item_encoder"]
    item_ids_list = item_encoder.classes_.tolist()

    # Filter books to align with CF item encoder
    books_filtered = books[books["book_id"].isin(item_ids_list)]
    print("Books after filtering:", books_filtered.shape)
    if books_filtered.empty:
        print("❌ No books match the item IDs! Check CSV files.")
        return

    # Keep same order as CF item encoder
    books_filtered = books_filtered.set_index("book_id").loc[item_ids_list].reset_index()

    content_matrix, tfidf = build_content_index(books_filtered)
    print(f"Content TF-IDF matrix shape: {content_matrix.shape}")

    print("⚙️ Training Recommender (CF + Content Hybrid)...")
    rec = Recommender(cf_components=50, hybrid_alpha=0.6)
    rec.fit_cf(user_item_matrix)
    rec.fit_content(content_matrix, item_ids_list, tfidf)
    rec.build_popularity(
        meta["df"], meta["item_encoder"], lambda idx: item_encoder.inverse_transform([idx])[0]
    )

    # Save artifacts
    print("💾 Saving models and metadata...")
    joblib.dump(user_item_matrix, os.path.join(MODEL_DIR, "user_item_matrix.pkl"))
    rec.save(os.path.join(MODEL_DIR, "rec"))
    joblib.dump(meta, os.path.join(MODEL_DIR, "meta.pkl"))
    joblib.dump(users, os.path.join(MODEL_DIR, "users.pkl"))

    print("✅ Training complete. Models saved to backend/models/")


if __name__ == "__main__":
    main()
