"""
Tests for presenting a solved track as a pose source.

This is a seam, and seams are where the two sides quietly disagree. The
reconstructor asks for a pose at a timestamp; the solver produced poses at a
sample of frames roughly a second apart. What happens between those samples, and
what happens outside them, is the whole content of this module — and both have a
failure that looks like working geometry:

  - snapping to the nearest solved frame quantises the camera path into steps,
    and a car covers several metres in half a second;
  - extrapolating past the end places geometry confidently somewhere nothing
    ever looked at.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_capture.pose.feedforward import Track
from heat3d_capture.pose.feedforward_source import FeedForwardPoses


def straight_track(count: int = 10, step: float = 5.0) -> Track:
    centres = np.stack(
        [np.arange(count) * step, np.zeros(count), np.zeros(count)], axis=1
    ).astype(float)
    return Track(centres=centres, rotations=np.stack([np.eye(3)] * count))


def turning_track(count: int = 9) -> Track:
    """A quarter turn, so rotation interpolation has something to get wrong."""
    angles = np.linspace(0.0, np.pi / 2, count)
    rotations = np.stack(
        [
            np.array(
                [[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]]
            )
            for a in angles
        ]
    )
    centres = np.stack([np.cos(angles), np.zeros(count), np.sin(angles)], axis=1) * 30.0
    return Track(centres=centres, rotations=rotations)


def test_returns_the_solved_pose_at_a_solved_time():
    track = straight_track()
    source = FeedForwardPoses(track, [float(i) for i in range(10)])

    assert np.allclose(source.at(4.0).translation, track.centres[4])


def test_interpolates_between_solved_frames():
    """
    Halfway between two samples is halfway along, not one or the other.

    Nearest-neighbour would return one of the endpoints here, which is the
    failure that turns a smooth camera path into a staircase.
    """
    source = FeedForwardPoses(straight_track(), [float(i) for i in range(10)])

    midway = source.at(4.5)

    assert midway.translation[0] == pytest.approx(22.5)


def test_interpolated_rotations_stay_rotations():
    """
    Blending two rotation matrices elementwise does not give a rotation — it
    shrinks toward the mean — and an unorthonormal matrix skews every point the
    frame contributes, quietly.
    """
    source = FeedForwardPoses(turning_track(), [float(i) for i in range(9)])

    for t in (0.5, 2.25, 6.75):
        rotation = source.at(t).rotation
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9), t
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9), t


def test_outside_the_solved_span_there_is_no_pose():
    """
    None, not the nearest end.

    Extrapolating would place geometry confidently in a spot nothing looked at;
    the reconstructor reads None as "solve this frame visually", which is honest.
    """
    source = FeedForwardPoses(straight_track(), [float(i) for i in range(10)])

    assert source.at(-0.001) is None
    assert source.at(9.001) is None
    assert source.at(0.0) is not None
    assert source.at(9.0) is not None


def test_unsorted_timestamps_are_put_in_order():
    """Windows are solved in order, but nothing downstream should depend on it."""
    track = straight_track(4)
    shuffled = [3.0, 0.0, 2.0, 1.0]

    source = FeedForwardPoses(track, shuffled)

    assert source.times == [0.0, 1.0, 2.0, 3.0]
    assert np.allclose(source.at(0.0).translation, track.centres[1])


def test_a_mismatched_count_is_refused():
    with pytest.raises(ValueError, match="timestamps"):
        FeedForwardPoses(straight_track(5), [0.0, 1.0])


def test_a_single_camera_is_not_a_track():
    with pytest.raises(ValueError, match="at least two"):
        FeedForwardPoses(straight_track(1), [0.0])
