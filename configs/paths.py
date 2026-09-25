import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(
    os.environ.get(
        "AMAZON_ML_DATA_DIR",
        BASE_DIR / "dataset",
    )
)

ARTIFACTS_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR = BASE_DIR / "output"