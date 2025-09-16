# backend/api/hf_api.py
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Load env vars
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME") or "google/flan-t5-large"
HF_TOKEN = os.getenv("HF_TOKEN") or None

print(f"🔧 Loading Hugging Face model: {MODEL_NAME}")

# Device
device = 0 if torch.cuda.is_available() else -1

# Load model (use_auth_token only if token is set)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, use_auth_token=HF_TOKEN if HF_TOKEN else None
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, use_auth_token=HF_TOKEN if HF_TOKEN else None
)

# Create pipeline
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

def get_answer(question: str) -> str:
    result = qa_pipeline(
        question,
        max_length=1024,  # increase output length
        min_length=200,   # optional, force some minimum
        do_sample=True,   # enable sampling for more natural outputs
        temperature=0.7,
        top_p=0.9
    )
    return result[0]["generated_text"]
