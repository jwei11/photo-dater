"""
Library of Congress photo scraper
----------------------------------
Pulls dated photographs from the LOC's free API and saves them locally,
organized by decade. This is the first step in the photo-dating ML project.

API docs: https://libraryofcongress.github.io/data-exploration/

Usage:
    python src/scraper.py

Output structure:
    data/
        raw/
            1850s/
            1860s/
            ...
            1990s/
        metadata.csv   <- one row per image: filename, year, decade, source_url
"""

import csv
import re
import time
import requests
from pathlib import Path
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL          = "https://www.loc.gov/photos/"
OUTPUT_DIR        = Path("data/raw")
META_FILE         = Path("data/metadata.csv")

TARGET_PER_DECADE = 500
DECADES           = range(1850, 2000, 10)
RESULTS_PER_PAGE  = 25       # smaller JSON payloads — fewer IncompleteRead errors
RATE_LIMIT_DELAY  = 1.0      # be polite; LOC throttles aggressive scrapers
API_TIMEOUT       = 120      # search JSON can be ~1 MB per page
IMAGE_TIMEOUT     = 90
MAX_API_RETRIES   = 6
MAX_IMAGE_RETRIES = 4
PAGE_FAIL_PAUSE   = 30       # pause before retrying a failed API page
MIN_FILE_SIZE     = 500      # bytes — reject empty/partial downloads

FIELDNAMES = [
    "item_id", "filename", "year", "decade", "date_raw",
    "source_url", "image_url", "title", "scraped_at",
]

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "photo-dater/1.0 (educational; contact: local)",
})

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_dirs():
    """Create decade subfolders and metadata parent."""
    for decade in DECADES:
        (OUTPUT_DIR / f"{decade}s").mkdir(parents=True, exist_ok=True)
    META_FILE.parent.mkdir(parents=True, exist_ok=True)


def extract_year(date_str: str) -> int | None:
    if not date_str:
        return None
    match = re.search(r'\b(1[6-9]\d{2}|20[0-2]\d)\b', date_str)
    return int(match.group(1)) if match else None


def year_to_decade(year: int) -> int | None:
    decade = (year // 10) * 10
    return decade if decade in DECADES else None


def strip_url_fragment(url: str) -> str:
    return url.split("#")[0]


def is_valid_image_url(url: str) -> bool:
    """Skip LOC placeholder SVGs and non-tile URLs."""
    u = strip_url_fragment(url).lower()
    if ".svg" in u or "original-format" in u:
        return False
    return "tile.loc.gov" in u


def is_valid_image_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < MIN_FILE_SIZE:
        return False
    with open(path, "rb") as f:
        header = f.read(16)
    # SVG placeholders saved as .jpg
    if header.lstrip().startswith((b"<?xml", b"<svg", b"<SVG")):
        return False
    return True


def fetch_page(params: dict) -> dict | None:
    """Fetch one API page with retries and long timeout for large JSON."""
    for attempt in range(MAX_API_RETRIES):
        try:
            resp = SESSION.get(BASE_URL, params=params, timeout=API_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            wait = min(PAGE_FAIL_PAUSE, 2 ** attempt * 5)
            print(f"  Request failed ({e}), retrying in {wait}s...")
            time.sleep(wait)
    print(f"  Gave up on API page after {MAX_API_RETRIES} attempts")
    return None


def download_image(image_url: str, save_path: Path) -> bool:
    """Download a single image with retries. Returns True on success."""
    image_url = strip_url_fragment(image_url)

    if save_path.exists() and is_valid_image_file(save_path):
        return True
    if save_path.exists():
        save_path.unlink()

    tmp = save_path.with_suffix(".tmp")
    for attempt in range(MAX_IMAGE_RETRIES):
        try:
            resp = SESSION.get(image_url, timeout=IMAGE_TIMEOUT, stream=True)
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            if not is_valid_image_file(tmp):
                tmp.unlink(missing_ok=True)
                raise ValueError("download too small or not an image")
            tmp.replace(save_path)
            return True
        except (requests.RequestException, ValueError, OSError) as e:
            tmp.unlink(missing_ok=True)
            wait = 2 ** attempt
            print(f"  Image download failed ({e}), retry in {wait}s...")
            time.sleep(wait)
    return False


def normalize_title(title) -> str:
    if isinstance(title, list):
        return title[0] if title else ""
    return title or ""


def item_id_from_item(item: dict) -> str:
    return item.get("id", "").rstrip("/").split("/")[-1]


def item_key_from_path(path: Path) -> tuple[int, str] | None:
    decade_name = path.parent.name
    if not decade_name.endswith("s"):
        return None
    try:
        decade = int(decade_name[:-1])
    except ValueError:
        return None
    prefix = f"{decade}_"
    stem = path.stem
    if stem.startswith(prefix):
        return decade, stem[len(prefix):]
    return None


def get_best_image_url(item: dict) -> str | None:
    """Pick a medium tile.loc.gov image; skip SVG placeholders."""
    urls = [u for u in item.get("image_url", []) if is_valid_image_url(u)]
    if not urls:
        return None
    idx = min(1, len(urls) - 1)
    return strip_url_fragment(urls[idx])


def sanitize_existing_data() -> int:
    """
    Remove invalid rows/files from a previous run (SVG placeholders, orphans).
    Rewrites metadata.csv in place. Returns count of valid images kept.
    """
    if not META_FILE.exists():
        return 0

    kept_rows = []
    kept_paths = set()
    removed = 0

    with open(META_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = Path(row["filename"])
            url = row.get("image_url", "")
            if not is_valid_image_url(url) or not is_valid_image_file(path):
                path.unlink(missing_ok=True)
                removed += 1
                continue
            kept_rows.append(row)
            kept_paths.add(path.resolve())

    for path in OUTPUT_DIR.rglob("*.jpg"):
        if path.resolve() not in kept_paths:
            path.unlink(missing_ok=True)
            removed += 1

    with open(META_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(kept_rows)

    if removed:
        print(f"  Cleaned {removed} invalid/orphan files; {len(kept_rows)} valid images kept")
    return len(kept_rows)


def load_seen() -> set[tuple[int, str]]:
    """Load valid (decade, item_id) pairs from metadata only."""
    seen: set[tuple[int, str]] = set()
    if not META_FILE.exists():
        return seen

    with open(META_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = Path(row["filename"])
            if not is_valid_image_file(path):
                continue
            item_id = row.get("item_id") or path.stem.split("_", 1)[-1]
            decade = int(row["decade"])
            seen.add((decade, item_id))
    return seen


# ── Main scraping logic ───────────────────────────────────────────────────────

def scrape_decade(
    decade: int,
    writer: csv.DictWriter,
    csvfile,
    counts: dict,
    seen: set[tuple[int, str]],
):
    decade_dir = OUTPUT_DIR / f"{decade}s"
    start_year = decade
    end_year = decade + 9
    page = 1
    downloaded = sum(1 for d, _ in seen if d == decade)
    empty_pages = 0

    print(f"\nScraping {decade}s ({start_year}–{end_year})...")
    if downloaded:
        print(f"  Resuming with {downloaded}/{TARGET_PER_DECADE} images")

    while downloaded < TARGET_PER_DECADE:
        params = {
            "q":     "photograph",
            "dates": f"{start_year}/{end_year}",
            "fo":    "json",
            "c":     RESULTS_PER_PAGE,
            "sp":    page,
            "fa":    "online-format:image",
        }

        data = fetch_page(params)
        if not data:
            print(f"  Stopping {decade}s at page {page} (API unavailable — re-run to resume)")
            break

        results = data.get("results", [])
        if not results:
            empty_pages += 1
            if empty_pages >= 3:
                print(f"  No more results after page {page}")
                break
            page += 1
            continue
        empty_pages = 0

        for item in results:
            if downloaded >= TARGET_PER_DECADE:
                break

            date_str = item.get("date", "") or ""
            year = extract_year(date_str)
            if not year:
                continue

            if year_to_decade(year) != decade:
                continue

            image_url = get_best_image_url(item)
            if not image_url:
                continue

            item_id = item_id_from_item(item)
            if (decade, item_id) in seen:
                continue

            save_path = decade_dir / f"{decade}_{item_id}.jpg"
            if not download_image(image_url, save_path):
                continue

            seen.add((decade, item_id))
            writer.writerow({
                "item_id":    item_id,
                "filename":   str(save_path),
                "year":       year,
                "decade":     decade,
                "date_raw":   date_str,
                "source_url": item.get("url", ""),
                "image_url":  image_url,
                "title":      normalize_title(item.get("title", "")),
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            })
            csvfile.flush()

            downloaded += 1
            counts[decade] = downloaded

            if downloaded % 50 == 0:
                print(f"  {decade}s: {downloaded}/{TARGET_PER_DECADE} downloaded")

            time.sleep(RATE_LIMIT_DELAY)

        page += 1
        time.sleep(RATE_LIMIT_DELAY)

    print(f"  Done: {downloaded}/{TARGET_PER_DECADE} images for {decade}s")


def main():
    make_dirs()
    print("Checking existing data...")
    sanitize_existing_data()
    seen = load_seen()
    counts = {d: sum(1 for dec, _ in seen if dec == d) for d in DECADES}

    with open(META_FILE, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if csvfile.tell() == 0:
            writer.writeheader()

        for decade in DECADES:
            if counts.get(decade, 0) >= TARGET_PER_DECADE:
                print(f"\nSkipping {decade}s — already at {counts[decade]} images")
                continue
            scrape_decade(decade, writer, csvfile, counts, seen)

    print("\n── Scraping complete ──────────────────────────────────────")
    total = 0
    for decade in DECADES:
        n = counts.get(decade, 0)
        bar = "█" * (n // 20)
        status = "✓" if n >= TARGET_PER_DECADE else "…"
        print(f"  {decade}s: {n:>4}/{TARGET_PER_DECADE} {status}  {bar}")
        total += n
    print(f"\n  Total: {total} images")
    print(f"  Metadata: {META_FILE}")


if __name__ == "__main__":
    main()
