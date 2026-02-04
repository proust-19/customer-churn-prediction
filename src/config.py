from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


DATA_PATH = BASE_DIR/"data"/"raw"/"Data.csv"
PROCESSED_DATA_PATH = BASE_DIR/"data"/"processed"/"processed.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_logreg_model.joblib"

TARGET = "Churn"
RANDOM_STATE = 42

