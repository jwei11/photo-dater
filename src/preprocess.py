"""
Preprocess raw LOC images for training.

Steps:
    - Load images from data/raw/ using data/metadata.csv
    - Convert to grayscale (prevents "color photo = recent" shortcut)
    - Resize to 300×300
    - Write outputs to data/processed/, mirroring the raw/ decade structure
    - Write data/processed_metadata.csv with updated file paths
    - Write data/preprocess_failures.csv for any row that could not be processed

Usage:
    python src/preprocess.py
"""

import csv
from pathlib import Path

from PIL import Image

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS  # Pillow < 10

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
META_FILE = Path("data/metadata.csv")
PROCESSED_META_FILE = Path("data/processed_metadata.csv")
FAILURES_FILE = Path("data/preprocess_failures.csv")

IMAGE_SIZE = (300, 300)

FIELDNAMES = [
    "item_id", "filename", "year", "decade", "date_raw",
    "source_url", "image_url", "title", "scraped_at",
]

FAILURE_FIELDNAMES = [
    "filename", "reason", "item_id", "year", "decade",
    "image_url", "source_url",
]


def make_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DIR.exists():
        for decade_dir in RAW_DIR.iterdir():
            if decade_dir.is_dir():
                (PROCESSED_DIR / decade_dir.name).mkdir(exist_ok=True)


def is_valid_processed(path: Path) -> bool:
    """True if path is a loadable 300×300 grayscale image."""
    try:
        with Image.open(path) as img:
            img.load()
            return img.mode == "L" and img.size == IMAGE_SIZE
    except Exception:
        return False


def process_image(src: Path, dst: Path) -> tuple[bool, str]:
    """Convert src to grayscale, resize to IMAGE_SIZE, save to dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    try:
        with Image.open(src) as img:
            img = img.convert("L")
            img = img.resize(IMAGE_SIZE, RESAMPLE)
            img.save(tmp, format="JPEG")
        tmp.replace(dst)
        return True, ""
    except Exception as e:
        tmp.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)
        print(f"  Failed {src.name}: {e}")
        return False, str(e)


def failure_row(row: dict, src: Path, reason: str) -> dict:
    return {
        "filename": str(src),
        "reason": reason,
        "item_id": row.get("item_id", ""),
        "year": row.get("year", ""),
        "decade": row.get("decade", ""),
        "image_url": row.get("image_url", ""),
        "source_url": row.get("source_url", ""),
    }


def main():
    if not META_FILE.exists():
        raise FileNotFoundError(f"{META_FILE} not found — run src/scraper.py first")

    with open(META_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No images in metadata.csv — run src/scraper.py first")
        return

    make_dirs()

    processed_rows = []
    failure_rows = []
    ok = skipped = failed = 0

    for i, row in enumerate(rows, 1):
        src = Path(row["filename"])
        if not src.is_file():
            print(f"  Missing raw file: {src}")
            failure_rows.append(failure_row(row, src, "missing_raw_file"))
            failed += 1
            continue

        try:
            rel = src.relative_to(RAW_DIR)
        except ValueError:
            print(f"  Path not under {RAW_DIR}: {src}")
            failure_rows.append(failure_row(row, src, "path_outside_raw_dir"))
            failed += 1
            continue

        dst = PROCESSED_DIR / rel

        if dst.exists() and is_valid_processed(dst):
            skipped += 1
        else:
            dst.unlink(missing_ok=True)
            success, err = process_image(src, dst)
            if not success:
                failure_rows.append(failure_row(row, src, err))
                failed += 1
                continue
            ok += 1

        processed_rows.append({**row, "filename": str(dst)})

        if i % 500 == 0:
            print(f"  {i}/{len(rows)} rows scanned...")

    with open(PROCESSED_META_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(processed_rows)

    if failure_rows:
        with open(FAILURES_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FAILURE_FIELDNAMES)
            writer.writeheader()
            writer.writerows(failure_rows)
    elif FAILURES_FILE.exists():
        FAILURES_FILE.unlink()

    total = ok + skipped + failed
    print(f"\nPreprocessing complete")
    print(f"  Newly processed : {ok}")
    print(f"  Already done    : {skipped}")
    print(f"  Failed          : {failed}")
    print(f"  Total ready     : {ok + skipped}")
    print(f"  Manifest        : {PROCESSED_META_FILE}")
    if failure_rows:
        print(f"  Failures log    : {FAILURES_FILE}")
    if total != len(rows):
        print(f"  Warning: counted {total} rows, expected {len(rows)}")


if __name__ == "__main__":
    main()
