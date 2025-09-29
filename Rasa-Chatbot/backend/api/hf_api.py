import os
from dotenv import load_dotenv
from transformers import pipeline

# Load env vars
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

print(f"🔧 Loading Hugging Face model: {MODEL_NAME}")

pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    use_auth_token=HF_TOKEN,
    device_map="auto"  # use GPU if available
)

def get_answer(question: str) -> str:
    """
    Returns the answer from the Hugging Face model.
    """
    try:
        outputs = pipe(
            question,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9
        )
        return outputs[0]["generated_text"]
    except Exception as e:
        return f"❌ Inference failed: {str(e)}"
