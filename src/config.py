from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DATA = BASE_DIR / "data/raw/train.csv"
TEST_DATA = BASE_DIR / "data/raw/test.csv"

PROCESSED_DATA = BASE_DIR / "data/processed/processed.csv"

BEST_MODEL = BASE_DIR / "models/best_model.pkl"
PIPELINE = BASE_DIR / "models/pipeline.pkl"

METRICS_FILE = BASE_DIR / "data/metrics/metrics.csv"