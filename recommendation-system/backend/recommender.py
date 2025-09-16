# backend/recommender.py
import joblib
from sklearn.decomposition import TruncatedSVD
import numpy as np

class Recommender:
    def __init__(self, cf_components=50, hybrid_alpha=0.5):
        self.cf_components = cf_components
        self.hybrid_alpha = hybrid_alpha
        self.cf_model = None
        self.content_matrix = None
        self.item_ids = None
        self.tfidf = None
        self.popularity = None

    def fit_cf(self, user_item_matrix):
        svd = TruncatedSVD(n_components=self.cf_components, random_state=42)
        self.cf_matrix = svd.fit_transform(user_item_matrix)
        self.cf_model = svd

    def fit_content(self, content_matrix, item_ids, tfidf):
        self.content_matrix = content_matrix
        self.item_ids = item_ids
        self.tfidf = tfidf

    def build_popularity(self, ratings_df, item_encoder, id_lookup):
        counts = ratings_df.groupby("item_idx")["book-rating"].mean()
        self.popularity = [
            {"book_id": id_lookup(idx), "score": score}
            for idx, score in counts.sort_values(ascending=False).items()
        ]

    def recommend(self, liked_book_ids=None, k=10, use_popularity=False):
        if use_popularity or not liked_book_ids:
            return self.popularity[:k]

        # Hybrid scoring: CF + Content
        scores = {}
        for book_id in liked_book_ids:
            if book_id not in self.item_ids:
                continue
            idx = self.item_ids.index(book_id)
            sim_scores = self.content_matrix[idx].dot(self.content_matrix.T).toarray()[0]
            for i, score in enumerate(sim_scores):
                scores[self.item_ids[i]] = scores.get(self.item_ids[i], 0) + score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"book_id": b, "score": s} for b, s in ranked[:k]]

    def save(self, path):
        joblib.dump(self, f"{path}.pkl")

    @staticmethod
    def load(path):
        return joblib.load(f"{path}.pkl")
