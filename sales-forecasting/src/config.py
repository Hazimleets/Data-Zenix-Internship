# sales-forcing/src/config.py

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
SEED = 42
TARGET = "Global_Sales"
CSV_NAME = "vgsales.csv" # place dataset CSV here


# Create dirs at runtime if missing
MODEL_DIR.mkdir(parents=True, exist_ok=True)