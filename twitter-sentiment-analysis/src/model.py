# ML/DL training pipeline

# src/model.py

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train_and_evaluate(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=200, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return model
