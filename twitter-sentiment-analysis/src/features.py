# Vectorization (TF-IDF, embeddings)

# src/features.py

from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_tfidf(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer
