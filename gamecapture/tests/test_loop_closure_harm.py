"""
Loop closure on a repetitive scene, which is where it destroys a scan.

The worst failure this project has had, and the hardest to see: tracking reported
every single frame placed, every pose was wrong, and nothing anywhere said so.

Measured on a real recording, 220 keyframes of a Forza rally lap:

    loop closure on    path 116.6 m    net displacement   5.3 m
    loop closure off   path 112.3 m    net displacement  83.7 m

Seventy-six "revisits" fired. None were revisits. On a rally stage every stretch
of road looks like every other stretch, feature matching cheerfully agrees, and
each false closure yanked the camera back onto a pose it had already left. The
car drove a hundred metres and the reconstruction stayed inside a five-metre box.

The general shape of it — a corridor, a row of trees, a tunnel, a car park — is
common enough that the feature is off by default. What is tested here is that the
default holds, that the gate is hard to pass if it is switched on, and above all
that a straight run through a repeating scene does not collapse.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_capture.pose.odometry import (
    LOOP_MIN_AGE,
    LOOP_MIN_INLIERS,
    MIN_INLIERS,
    Intrinsics,
    VisualOdometry,
)
from heat3d_capture.scan.reconstruct import Reconstructor


def corridor(n: int = 90, width: int = 480, height: int = 270) -> list[np.ndarray]:
    """
    A scene that repeats: the same wall pattern, sliding past, over and over.

    Deliberately self-similar. This is the property that breaks naive loop
    closure — not darkness, not speed, but the fact that where you are now looks
    like where you were.
    """
    import cv2

    rng = np.random.default_rng(5)
    tile = np.full((height, width, 3), 20, np.uint8)
    for _ in range(120):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        cv2.rectangle(
            tile,
            (x, y),
            (x + int(rng.integers(12, 40)), y + int(rng.integers(12, 40))),
            tuple(int(c) for c in rng.integers(70, 245, 3)),
            -1,
        )
    # Repeating the same tile three times makes the corridor genuinely periodic.
    strip = np.hstack([tile, tile, tile])
    return [
        np.ascontiguousarray(strip[:, i * 8 : i * 8 + width]) for i in range(n)
    ]


def flat(depth: float = 5.0, shape=(270, 480)) -> np.ndarray:
    return np.full(shape, 1.0 / depth, dtype=np.float32)


class TestTheDefault:
    def test_loop_closure_is_off(self):
        # The measurement in the module docstring is the argument. A weak place
        # recogniser is worse than none, and this one is weak.
        assert VisualOdometry(Intrinsics.from_fov(480, 270, 90.0))._loop_closure is False

    def test_the_reconstructor_agrees(self):
        r = Reconstructor(Intrinsics.from_fov(480, 270, 90.0), camera_height=None)
        assert r.odometry._loop_closure is False


class TestTheGate:
    def test_a_closure_needs_far_more_evidence_than_tracking(self):
        # Ordinary tracking is allowed to proceed on eighteen inliers. Claiming
        # to recognise a place already visited is a much stronger claim and must
        # cost much more.
        assert LOOP_MIN_INLIERS > MIN_INLIERS * 4

    def test_a_candidate_must_be_genuinely_old(self):
        # Twelve keyframes is a few seconds of driving — the road just behind
        # you, not a revisit. That was the original value.
        assert LOOP_MIN_AGE >= 60


class TestAgainstARepeatingScene:
    """The failure itself, reproduced without needing the recording."""

    @staticmethod
    def _run(loop_closure: bool):
        k = Intrinsics.from_fov(480, 270, 90.0)
        r = Reconstructor(k, voxel=0.4, mask_hud=False, camera_height=None,
                          loop_closure=loop_closure)
        depth = flat()
        for image in corridor():
            r.add(image, depth)
        return r

    def test_a_straight_run_does_not_collapse_onto_itself(self):
        r = self._run(loop_closure=False)
        positions = r.trajectory.positions
        assert len(positions) > 10
        net = float(np.linalg.norm(positions[-1] - positions[0]))
        # It went somewhere. The exact distance depends on the synthetic depth;
        # what matters is that the end is not the beginning.
        assert net > 1.0, f"net displacement {net:.2f} m — the track went nowhere"

    def test_enabling_it_on_a_repeating_scene_does_not_pin_the_camera(self):
        # With the gate tightened, even switched on it must not fire freely on a
        # scene like this. If this regresses, the gate has gone soft again.
        loose = self._run(loop_closure=True)
        tight = self._run(loop_closure=False)

        def net(r):
            p = r.trajectory.positions
            return float(np.linalg.norm(p[-1] - p[0]))

        assert net(loose) > net(tight) * 0.5, (
            f"closures collapsed the path: {net(loose):.2f} m against {net(tight):.2f} m"
        )

    def test_net_displacement_is_reported_not_just_path_length(self):
        # The number that hid the failure. 116 m of path with 5 m of net
        # displacement is a camera jittering in place, and path length alone
        # reads as a healthy scan.
        r = self._run(loop_closure=False)
        positions = r.trajectory.positions
        path = r.state.trajectory_length
        net = float(np.linalg.norm(positions[-1] - positions[0]))
        assert path > 0
        # Not an assertion about the ratio, but that both are available to be
        # compared at all — which is what the interface needs to warn on.
        assert np.isfinite(net)
