"""
Leakage-audit primitives.

The audit is only worth running if its building blocks are right: a perceptual
hash that collides for re-encodes of one photograph but not for different ones,
a correct minimum-distance search, and duplicate grouping that reports only
collisions actually spanning a split boundary.
"""

import io

import numpy as np
from PIL import Image

from leakage_check import (
    bit_matrix,
    cross_split_groups,
    dhash,
    min_cross_distance,
)


def photo_like(seed: int = 0, size: int = 300) -> Image.Image:
    """
    A stand-in for a photograph: smooth low-frequency structure plus one hard
    edge. Random noise would be the wrong fixture here — downsampling destroys
    high-frequency content entirely, so a noise image fails the rescaling test
    for reasons that have nothing to do with the hash.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size] / size
    field = np.sin(2 * np.pi * (x * 1.5 + rng.random())) + np.cos(2 * np.pi * (y * 2.0 + rng.random()))
    field = (field - field.min()) / (field.max() - field.min())
    arr = (field * 255).astype(np.uint8)
    arr[size // 4:size // 2, size // 3:size // 2] = 240  # a subject-like block
    return Image.fromarray(arr, mode="L")


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def test_dhash_is_deterministic():
    img = photo_like()
    assert dhash(img) == dhash(img)


def test_dhash_fits_in_64_bits():
    assert 0 <= dhash(photo_like(seed=1)) < 2**64


def test_dhash_survives_rescaling():
    """A re-scan at a different resolution must still look like the same image."""
    img = photo_like(seed=7)
    rescaled = img.resize((150, 150), Image.BILINEAR).resize((300, 300), Image.BILINEAR)
    assert hamming(dhash(img), dhash(rescaled)) <= 5


def test_dhash_survives_jpeg_reencoding():
    img = photo_like(seed=3)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    buf.seek(0)
    assert hamming(dhash(img), dhash(Image.open(buf).convert("L"))) <= 5


def test_dhash_separates_different_images():
    a, b = photo_like(seed=1), photo_like(seed=2)
    assert hamming(dhash(a), dhash(b)) > 10


def test_bit_matrix_shape_and_values():
    bits = bit_matrix([0, 2**64 - 1])
    assert bits.shape == (2, 64)
    assert bits[0].sum() == 0
    assert bits[1].sum() == 64


def test_min_cross_distance_finds_exact_match():
    reference = bit_matrix([0b1010, 0b1111, 0])
    query = bit_matrix([0b1111])
    assert min_cross_distance(query, reference).tolist() == [0]


def test_min_cross_distance_counts_differing_bits():
    reference = bit_matrix([0b0000])
    query = bit_matrix([0b0111])
    assert min_cross_distance(query, reference).tolist() == [3]


def test_min_cross_distance_takes_the_closest_reference():
    reference = bit_matrix([0b1111_1111, 0b0000_0001])
    query = bit_matrix([0b0000_0000])
    assert min_cross_distance(query, reference).tolist() == [1]


def test_min_cross_distance_handles_chunk_boundaries():
    """Chunking must not change the answer."""
    rng = np.random.default_rng(0)
    ref = bit_matrix(rng.integers(0, 2**62, 50).tolist())
    qry = bit_matrix(rng.integers(0, 2**62, 70).tolist())
    assert np.array_equal(
        min_cross_distance(qry, ref, chunk=7),
        min_cross_distance(qry, ref, chunk=1000),
    )


def test_cross_split_groups_reports_only_boundary_crossings():
    split_of = {"a.jpg": "train", "b.jpg": "test", "c.jpg": "train", "d.jpg": "train"}
    digests = {
        "a.jpg": "same",   # train + test -> leakage
        "b.jpg": "same",
        "c.jpg": "pair",   # both train -> harmless
        "d.jpg": "pair",
    }
    groups = cross_split_groups(digests, split_of)
    assert len(groups) == 1
    assert set(groups[0]) == {"a.jpg", "b.jpg"}


def test_cross_split_groups_ignores_unique_digests():
    split_of = {"a.jpg": "train", "b.jpg": "test"}
    digests = {"a.jpg": "x", "b.jpg": "y"}
    assert cross_split_groups(digests, split_of) == []
