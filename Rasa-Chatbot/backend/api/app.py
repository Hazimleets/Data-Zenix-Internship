# Rasa-Chatbot/backend/api/app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from api.hf_api import get_answer

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    sender: str
    message: str

RASA_URL = "http://127.0.0.1:5005/webhooks/rest/webhook"

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/send")
def send_message(msg: Message):
    payload = {"sender": msg.sender, "message": msg.message}

    try:
        # 🔹 Try Rasa first
        r = requests.post(RASA_URL, json=payload, timeout=10)
        r.raise_for_status()
        responses = r.json()

        if responses and "text" in responses[0]:
            return {"reply": responses[0]["text"]}

        # 🔹 Fallback to local GPT-like model
        answer = get_answer(msg.message)
        return {"reply": answer}

    except Exception:
        # 🔹 Final fallback
        answer = get_answer(msg.message)
        return {"reply": answer}

