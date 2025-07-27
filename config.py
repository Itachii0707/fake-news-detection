from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = BASE_DIR / "models"

# Files
RAW_DATA_PATH = RAW_DIR / "news.csv"
MODEL_PATH = MODEL_DIR / "logreg_tfidf.joblib"

# Reproducibility
RANDOM_STATE = 42

# Columns expected in the dataset
TEXT_COLUMNS_CANDIDATES = ["text", "content", "article"]
TITLE_COLUMNS_CANDIDATES = ["title", "headline"]
LABEL_COLUMN_CANDIDATES = ["label", "target", "y"]

# Train / test split
TEST_SIZE = 0.2
