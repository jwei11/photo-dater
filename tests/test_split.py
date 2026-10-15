"""
Split integrity. These guard the property the reported metrics depend on: no
photograph may appear in more than one split, and the test split must be
identical across training seeds.

Synthetic rows throughout — data/ is gitignored, so a fresh clone has no images.
"""

import csv

import pytest

from dataset import (
    DECADES,
    DECADE_TO_IDX,
    IDX_TO_DECADE,
    SPLIT_SEED,
    load_metadata,
    train_val_test_split,
)


def make_rows(n_per_decade: int = 100) -> list[dict]:
    return [
        {"filename": f"img_{decade}_{i}.jpg", "decade": str(decade)}
        for decade in DECADES
        for i in range(n_per_decade)
    ]


def filenames(rows: list[dict]) -> set[str]:
    return {row["filename"] for row in rows}


def test_splits_are_disjoint():
    train, val, test = train_val_test_split(make_rows())
    assert not filenames(train) & filenames(val)
    assert not filenames(train) & filenames(test)
    assert not filenames(val) & filenames(test)


def test_split_loses_no_rows():
    rows = make_rows()
    train, val, test = train_val_test_split(rows)
    assert len(train) + len(val) + len(test) == len(rows)
    assert filenames(train) | filenames(val) | filenames(test) == filenames(rows)


def test_split_proportions():
    train, val, test = train_val_test_split(make_rows(100))
    assert (len(train), len(val), len(test)) == (1050, 225, 225)  # 70/15/15 of 1500


def test_split_is_stratified_by_decade():
    """Every decade must be represented in every split, or per-decade F1 is undefined."""
    _, val, test = train_val_test_split(make_rows(100))
    for split in (val, test):
        counts = {int(row["decade"]) for row in split}
        assert counts == set(DECADES)


def test_split_is_deterministic():
    first = train_val_test_split(make_rows())
    second = train_val_test_split(make_rows())
    for a, b in zip(first, second):
        assert filenames(a) == filenames(b)


def test_split_seed_is_independent_of_training_seed():
    """
    The test split must not move when training randomness changes. If it did, a
    multi-seed comparison would score each run on photos the others trained on.
    """
    rows = make_rows()
    baseline = filenames(train_val_test_split(rows)[2])
    assert filenames(train_val_test_split(rows, seed=SPLIT_SEED)[2]) == baseline
    # A genuinely different split seed should move it — proving the pin matters.
    assert filenames(train_val_test_split(rows, seed=SPLIT_SEED + 1)[2]) != baseline


def test_tiny_decade_is_rejected_not_silently_mangled():
    rows = [{"filename": "a.jpg", "decade": "1850"}, {"filename": "b.jpg", "decade": "1850"}]
    with pytest.raises(ValueError, match="cannot split three ways"):
        train_val_test_split(rows)


def test_load_metadata_drops_duplicate_rows(tmp_path):
    """
    The scraper can emit a row twice on retry. A duplicate landing either side of
    a split boundary would put the same photograph in train and test.
    """
    meta = tmp_path / "processed_metadata.csv"
    with open(meta, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "decade"])
        writer.writeheader()
        writer.writerow({"filename": "dup.jpg", "decade": "1850"})
        writer.writerow({"filename": "dup.jpg", "decade": "1850"})
        writer.writerow({"filename": "other.jpg", "decade": "1860"})

    rows = load_metadata(meta)
    assert len(rows) == 2
    assert filenames(rows) == {"dup.jpg", "other.jpg"}


def test_load_metadata_requires_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="preprocess"):
        load_metadata(tmp_path / "missing.csv")


def test_decade_index_round_trips():
    for decade in DECADES:
        assert IDX_TO_DECADE[DECADE_TO_IDX[decade]] == decade
    assert len(DECADES) == 15
