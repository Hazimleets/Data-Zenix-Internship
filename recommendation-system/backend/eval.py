# backend/eval.py
import numpy as np
from sklearn.model_selection import train_test_split
from data_processing import build_user_item_matrix, load_raw_data
from recommender import Recommender
import joblib

def precision_recall_at_k(recommended_list, test_positive_set, k):
    rec_k = recommended_list[:k]
    hits = len([r for r in rec_k if r in test_positive_set])
    prec = hits / k
    rec = hits / len(test_positive_set) if len(test_positive_set)>0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return prec, rec, f1

def evaluate(ratings_path, books_path, k=10):
    ratings, books = load_raw_data(ratings_path, books_path)
    train, test = train_test_split(ratings, test_size=0.2, random_state=42, stratify=ratings['user_id'])
    # build training matrix
    user_item_matrix, meta = build_user_item_matrix(train)
    rec = Recommender()
    rec.fit_cf(user_item_matrix)
    # content
    item_ids_list = meta['item_encoder'].classes_.tolist()
    books_filtered = books[books['book_id'].isin(item_ids_list)].set_index('book_id').loc[item_ids_list].reset_index()
    from data_processing import build_content_index
    content_matrix, tfidf = build_content_index(books_filtered)
    rec.fit_content(content_matrix, item_ids_list, tfidf)

    # per-user evaluation
    users = test['user_id'].unique()
    precs, recs, f1s = [], [], []
    for u in users:
        # user test positives
        user_test = test[test['user_id']==u]
        pos_book_ids = set(user_test['book_id'].tolist())
        if len(pos_book_ids) == 0:
            continue
        # map user id to train index if exists
        ue = meta['user_encoder']
        try:
            uidx = int(ue.transform([u])[0])
            recs_list = [r[0] for r in rec.recommend_for_user(user_idx=uidx, user_item_matrix=user_item_matrix, top_k=k)]
        except Exception:
            # cold-start user: create profile from no items -> skip or use popularity
            recs_list = [r[0] for r in rec.popularity_ranking[:k]]

        p, r_, f = precision_recall_at_k(recs_list, pos_book_ids, k)
        precs.append(p); recs.append(r_); f1s.append(f)

    print(f"Precision@{k}: {np.mean(precs):.4f}")
    print(f"Recall@{k}: {np.mean(recs):.4f}")
    print(f"F1@{k}: {np.mean(f1s):.4f}")

if __name__ == "__main__":
    evaluate("data/ratings.csv", "data/books.csv", k=10)
