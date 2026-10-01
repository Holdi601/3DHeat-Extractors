"""
Tests for the frames handed to VGGT.

Preprocessing is the quiet half of a feed-forward reconstruction. The model
infers the camera's field of view from the picture, so an image squashed to a
square tells it the lens is wider than it was, and every recovered pose bends to
match — a lap that curves slightly wrong for a reason nothing downstream can see.
Both sides also have to be multiples of the patch size or the patch grid does not
divide the image.

No GPU and no weights here: this is the part that can be checked cheaply, which
is exactly why it lives in its own function rather than inside the inference call.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_capture.pose.vggt_model import LONG_SIDE, PATCH, prepare


def frame(height: int, width: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def test_both_sides_are_whole_patches():
    for height, width in [(1080, 1920), (720, 1280), (1200, 1600), (2160, 3840)]:
        out = prepare([frame(height, width)])
        assert out.shape[2] % PATCH == 0, (height, width)
        assert out.shape[3] % PATCH == 0, (height, width)


def test_long_side_is_the_trained_size():
    out = prepare([frame(1080, 1920)])
    assert max(out.shape[2], out.shape[3]) == LONG_SIDE


def test_aspect_ratio_is_preserved():
    """
    A 16:9 capture must not come back square.

    The model reads the field of view off the image. Squashing it misstates the
    lens, and the error goes into the poses rather than into anything visible.
    """
    height, width = 1080, 1920
    out = prepare([frame(height, width)])
    got = out.shape[3] / out.shape[2]

    # Rounding to whole patches moves the ratio a little; a squashed image would
    # move it to 1.0, which is nowhere near this tolerance.
    assert got == pytest.approx(width / height, rel=0.04)


def test_portrait_frames_keep_their_long_side():
    out = prepare([frame(1920, 1080)])
    assert out.shape[2] == LONG_SIDE
    assert out.shape[2] > out.shape[3]


def test_output_is_channels_first_and_normalised():
    out = prepare([frame(720, 1280), frame(720, 1280, seed=1)])

    assert out.shape[0] == 2
    assert out.shape[1] == 3
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_a_mixed_window_is_refused_rather_than_silently_resized():
    """
    Every frame in a window must be the same size.

    Resizing them to a common size regardless would mean two different fields of
    view in one solve, which the model has no way to represent and no way to
    report.
    """
    with pytest.raises(ValueError, match="uniform"):
        prepare([frame(720, 1280), frame(1080, 1920)])


def test_an_empty_window_is_refused():
    with pytest.raises(ValueError, match="no frames"):
        prepare([])
