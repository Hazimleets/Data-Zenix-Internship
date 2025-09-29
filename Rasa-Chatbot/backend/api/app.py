from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from hf_api import get_answer

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

@app.get("/")
def root():
    return {"message": "Welcome to the Local Chatbot API. Use /send to interact."}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/send")
def send_message(msg: Message):
    answer = get_answer(msg.message)
    return {"reply": answer}
