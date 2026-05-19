"""
Preprocess raw LOC images for training.

Planned steps (see README.md):
    - Load images from data/raw/ using data/metadata.csv
    - Convert to grayscale
    - Resize to 224×224
    - Write outputs to data/processed/
"""

from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
META_FILE = Path("data/metadata.csv")


def main():
    raise NotImplementedError("Preprocessing not yet implemented")


if __name__ == "__main__":
    main()
