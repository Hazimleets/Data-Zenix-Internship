# Flask/FastAPI for deployment

# app/api.py
import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

app = FastAPI(title="Twitter Sentiment API", version="1.0")

class TextIn(BaseModel):
    text: str

@app.post("/predict/")
def predict_sentiment(item: TextIn):
    text_vec = vectorizer.transform([item.text])
    prediction = model.predict(text_vec)[0]
    return {"text": item.text, "sentiment": prediction}

