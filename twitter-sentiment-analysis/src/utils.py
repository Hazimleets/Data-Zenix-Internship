# Helper functions (logging, config mgmt)

# src/utils.py

import os
import joblib
import yaml
import logging
import time
from functools import wraps

# -----------------------------
# Logging Setup
# -----------------------------
def get_logger(name: str, log_file: str = "logs/project.log", level=logging.INFO):
    """Set up a logger with file + console output."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate logs
    if not logger.handlers:
        # Console Handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File Handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# -----------------------------
# Config Loader
# -----------------------------
def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config file as dictionary."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------
# Model Persistence
# -----------------------------
def save_model(model, vectorizer, model_path="models/sentiment_model.pkl", vec_path="models/vectorizer.pkl"):
    """Save trained model + vectorizer."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)


def load_model(model_path="models/sentiment_model.pkl", vec_path="models/vectorizer.pkl"):
    """Load trained model + vectorizer."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


# -----------------------------
# Timer Utility
# -----------------------------
def timeit(func):
    """Decorator to measure execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"⏱ {func.__name__} executed in {end - start:.2f}s")
        return result
    return wrapper
