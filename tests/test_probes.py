"""
Shortcut-control probe geometry.

The first version of the occlusion probe was mis-specified: `frac` is a linear
fraction, so frac=0.70 masks only 0.49 of the AREA, leaving a ring that was over
half the image and still full of subject matter. These tests pin the geometry so
that mistake can't return silently.
"""

import numpy as np
from PIL import Image

from shortcut_control import (
    IMG_SIZE,
    PROBES,
    BorderOnly,
    CenterCrop,
    CenterOnly,
    Downsample,
    Identity,
    PatchShuffle,
    _fill_value,
)


def gradient_image() -> Image.Image:
    """A smooth gradient: distinguishable from any constant fill."""
    x = np.linspace(0, 255, IMG_SIZE, dtype=np.uint8)
    return Image.fromarray(np.tile(x, (IMG_SIZE, 1)), mode="L")


def test_every_probe_preserves_shape_and_mode():
    img = gradient_image()
    for probe in PROBES:
        out = probe(img)
        assert out.size == (IMG_SIZE, IMG_SIZE), probe.name
        assert out.mode == "L", probe.name


def test_identity_leaves_the_image_alone():
    img = gradient_image()
    assert np.array_equal(np.asarray(Identity()(img)), np.asarray(img))


def test_downsample_destroys_detail():
    img = gradient_image()
    coarse = np.asarray(Downsample(8)(img), dtype=float)
    fine = np.asarray(Downsample(32)(img), dtype=float)
    original = np.asarray(img, dtype=float)
    # Coarser downsampling must deviate further from the original.
    assert np.abs(coarse - original).mean() > np.abs(fine - original).mean()


def test_patch_shuffle_permutes_rather_than_discards():
    """Tile contents should survive; only their arrangement changes."""
    img = gradient_image()
    out = PatchShuffle(5, seed=0)(img)
    assert not np.array_equal(np.asarray(out), np.asarray(img))
    # A permutation preserves the overall intensity distribution.
    assert abs(np.asarray(out, dtype=float).mean() - np.asarray(img, dtype=float).mean()) < 2.0


def test_patch_shuffle_is_reproducible():
    img = gradient_image()
    a, b = PatchShuffle(5, seed=0)(img), PatchShuffle(5, seed=0)(img)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_border_and_center_mask_complementary_regions():
    img = gradient_image()
    frac = 0.90
    fill = _fill_value(img, None)

    border = np.asarray(BorderOnly(frac, fill=fill)(img))
    center = np.asarray(CenterOnly(frac, fill=fill)(img))
    original = np.asarray(img)

    side = int(IMG_SIZE * frac)
    offset = (IMG_SIZE - side) // 2
    inner = (slice(offset, offset + side), slice(offset, offset + side))

    # BorderOnly blanks the middle and keeps the ring; CenterOnly does the inverse.
    assert (border[inner] == fill).all()
    assert np.array_equal(center[inner], original[inner])
    assert border[0, 0] == original[0, 0]
    assert center[0, 0] == fill


def test_ring_area_shrinks_as_frac_grows():
    """
    The sweep only isolates borders if the surviving ring genuinely narrows.
    frac=0.70 leaves ~51% of the area; frac=0.95 leaves ~10%.
    """
    img = gradient_image()
    fill = _fill_value(img, None)
    areas = []
    for frac in (0.70, 0.90, 0.95):
        out = np.asarray(BorderOnly(frac, fill=fill)(img))
        areas.append((out != fill).mean())

    assert areas[0] > areas[1] > areas[2]
    assert 0.45 < areas[0] < 0.55
    assert areas[2] < 0.15


def test_center_crop_rescales_while_center_only_does_not():
    """
    CenterCrop magnifies what it keeps, which is why CenterOnly exists: it tests
    border removal without confounding it with a change in scale.
    """
    img = gradient_image()
    cropped = np.asarray(CenterCrop(0.70)(img))
    masked = np.asarray(CenterOnly(0.70)(img))
    # Magnified content reaches the frame edge; the masked version does not.
    assert not np.array_equal(cropped, masked)
    assert (masked[0, 0] == _fill_value(img, None))


def test_fill_value_defaults_to_image_mean():
    img = gradient_image()
    assert _fill_value(img, None) == int(np.asarray(img).mean())
    assert _fill_value(img, 128) == 128


def test_probe_names_are_unique():
    names = [probe.name for probe in PROBES]
    assert len(names) == len(set(names))
