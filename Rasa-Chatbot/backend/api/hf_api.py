import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "facebook/blenderbot-400M-distill")

# Load model and tokenizer locally
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Create a pipeline for text generation
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if model.device.type == "cuda" else -1  # GPU if available
)

def format_prompt(question: str) -> str:
    """Make the bot answer professionally."""
    return f"You are a professional AI assistant. Answer clearly, politely, and concisely.\n\nUser: {question}\nAssistant:"

def get_answer(question: str) -> str:
    prompt = format_prompt(question)
    try:
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
        return outputs[0]["generated_text"].strip()
    except Exception as e:
        return f"❌ Inference failed: {str(e)}"
