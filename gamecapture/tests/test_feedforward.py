"""
Tests for stitching feed-forward windows into one track.

Every window comes back in its own frame and its own scale, so the whole shape
of a lap rests on the similarity fit between overlapping windows. These tests
work on a known closed loop: if a synthetic circuit cut into windows, each
rotated, translated and rescaled at random, does not come back as the same
circuit, the stitch is wrong — and that is exactly the failure that shows up on
real footage as a lap that spirals instead of closing.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_capture.pose.feedforward import (
    MIN_STITCH,
    Window,
    plan_windows,
    similarity_from_points,
    stitch,
    to_trajectory,
)


def circuit(count: int = 60, radius: float = 40.0) -> np.ndarray:
    """A closed, non-planar loop — a track with elevation, not a flat circle."""
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.stack(
        [
            radius * np.cos(angle),
            radius * np.sin(angle) * 0.6,
            3.0 * np.sin(2.0 * angle),
        ],
        axis=1,
    )


def random_similarity(seed: int) -> tuple[float, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scale = float(rng.uniform(0.3, 3.0))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return scale, q, rng.normal(scale=20.0, size=3)


def test_similarity_recovers_a_known_transform():
    points = circuit(20)
    scale, rotation, translation = random_similarity(1)
    moved = (scale * (rotation @ points.T).T) + translation

    got_scale, got_rotation, got_translation = similarity_from_points(points, moved)

    assert got_scale == pytest.approx(scale, rel=1e-9)
    assert np.allclose(got_rotation, rotation, atol=1e-9)
    assert np.allclose(got_translation, translation, atol=1e-7)


def test_similarity_never_returns_a_reflection():
    """
    A mirrored fit can match the shared cameras and invert everything else.

    Three nearly collinear points — which is what an overlap looks like on a
    straight — are where the covariance is closest to singular and a reflection
    is most tempting, so that is the case worth pinning.
    """
    rng = np.random.default_rng(7)
    source = np.array([[0.0, 0, 0], [1, 0.001, 0], [2, -0.001, 0], [3, 0.002, 0]])
    target = source + rng.normal(scale=1e-3, size=source.shape)

    _, rotation, _ = similarity_from_points(source, target)

    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-6)


def test_similarity_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least 3"):
        similarity_from_points(np.zeros((2, 3)), np.zeros((2, 3)))


def split_into_windows(path: np.ndarray, window: int, overlap: int) -> list[Window]:
    """Cut a path into windows, each in its own arbitrary frame and scale."""
    windows = []
    for n, indices in enumerate(plan_windows(len(path), window, overlap)):
        scale, rotation, translation = random_similarity(100 + n)
        local = (scale * (rotation @ path[indices].T).T) + translation
        windows.append(
            Window(
                frames=indices,
                centres=local,
                rotations=np.stack([rotation] * len(indices)),
            )
        )
    return windows


def test_stitch_recovers_the_shape():
    """
    The stitched path is the original, up to one global similarity.

    `stitch` reports in the first window's frame, which is itself arbitrary, so
    the thing to check is the *shape*: fit one similarity onto the ground truth
    and require the residual to vanish. Comparing coordinates directly would
    only test that the first window happened to be the identity.
    """
    path = circuit(60)
    track = stitch(split_into_windows(path, window=12, overlap=5))

    assert track.frames == list(range(60))
    scale, rotation, translation = similarity_from_points(track.centres, path)
    aligned = (scale * (rotation @ track.centres.T).T) + translation
    spacing = np.linalg.norm(np.diff(path, axis=0), axis=1).mean()

    assert np.abs(aligned - path).max() < 1e-6 * spacing
    assert max(track.seam_error) < 1e-6


def test_stitch_closes_the_lap_it_started():
    """
    The measure the track shape is actually judged by.

    A lap driven once should end where it began. Scale error compounding across
    windows is what makes a reconstruction spiral, and it shows up here as a
    closure error larger than the one the path started with — the synthetic
    circuit has its own gap of a single segment, so the bar is that stitching
    adds nothing to it, not that the result is zero.
    """
    path = circuit(72)
    truth = float(
        np.linalg.norm(path[-1] - path[0])
        / np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    )
    track = stitch(split_into_windows(path, window=16, overlap=6))

    assert track.closure_error() == pytest.approx(truth, rel=1e-6)


def test_stitch_refuses_an_overlap_too_small_to_fix_scale():
    path = circuit(40)
    windows = split_into_windows(path, window=10, overlap=4)
    # Strip the shared frames down below what determines a similarity robustly.
    windows[1] = Window(
        frames=windows[1].frames[-3:],
        centres=windows[1].centres[-3:],
        rotations=windows[1].rotations[-3:],
    )

    with pytest.raises(ValueError, match=f"at least {MIN_STITCH}"):
        stitch(windows)


def test_plan_windows_covers_every_frame():
    for count in (1, 15, 16, 17, 40, 155):
        plan = plan_windows(count, window=16, overlap=6)
        covered = {i for indices in plan for i in indices}
        assert covered == set(range(count)), count
        assert plan[-1][-1] == count - 1, count


def test_plan_windows_keeps_neighbours_overlapping():
    plan = plan_windows(155, window=16, overlap=6)
    for before, after in zip(plan, plan[1:]):
        shared = set(before) & set(after)
        assert len(shared) >= MIN_STITCH, (before[0], after[0], len(shared))


def test_plan_windows_does_not_leave_a_short_tail():
    """
    The last window is pulled back rather than left short.

    A trailing window of two frames would be solved with almost no neighbours,
    and its cameras are the ones a lap has to close onto.
    """
    plan = plan_windows(100, window=16, overlap=6)
    assert all(len(indices) == 16 for indices in plan)


def test_plan_windows_rejects_an_overlap_that_never_advances():
    with pytest.raises(ValueError, match="must exceed"):
        plan_windows(50, window=8, overlap=8)


class TestHandingOverToTheRestOfThePipeline:
    """
    The fusion, coverage and export stages take a `Trajectory` and do not care
    where a pose came from. That seam is what lets the pose source be swapped
    without touching geometry, so it is worth a test of its own.
    """

    def track(self) -> "object":
        path = circuit(30)
        return stitch(split_into_windows(path, window=10, overlap=4))

    def test_every_camera_becomes_a_tracked_pose(self):
        trajectory = to_trajectory(self.track())

        assert len(trajectory.poses) == 30
        assert trajectory.breaks == 0
        assert trajectory.length > 0

    def test_poses_keep_the_camera_centres(self):
        """
        `Pose.translation` is where the camera is, not the other convention.

        Getting this backwards puts every camera at the reflection of its own
        position through the origin, which still produces a smooth path — so it
        would look like a working reconstruction of the wrong place.
        """
        track = self.track()
        trajectory = to_trajectory(track)

        assert np.allclose(trajectory.positions, track.centres)

    def test_feed_forward_poses_are_not_weighted_to_nothing(self):
        """
        A window solve counts no correspondences, and the usual confidence score
        is built from them — so without a case for it, every one of these poses
        would score zero and fuse at no weight, producing an empty mesh from a
        perfectly good track.
        """
        trajectory = to_trajectory(self.track())

        for result in trajectory.results:
            assert result.confidence > 0.5
            assert result.confidence <= 1.0
