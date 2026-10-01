"""
A solved feed-forward track, presented as a pose source.

`Reconstructor` already accepts a `pose_source` — anything that can answer "where
was the camera at time t" — because Forza telemetry needed one. A stitched VGGT
track is the same shape of thing: a list of times and the poses at them. So the
offline solver plugs in exactly where the telemetry feed does, and the fusion,
coverage, texturing and export stages are reused unchanged rather than forked.

The difference from telemetry is only in what it knows. Telemetry is exact and
arrives live; this is an estimate and has to see the whole capture first. Neither
changes what happens downstream, which is the point of the seam.
"""

from __future__ import annotations

import bisect

import numpy as np

from .feedforward import Track
from .odometry import Pose
from .telemetry_pose import _slerp


class FeedForwardPoses:
    """
    Answers `at(t)` from a solved track, interpolating between solved frames.

    Interpolation rather than nearest-neighbour because the solve runs on a
    sample of frames — one every second or so — while fusion wants a pose for
    every keyframe it is given. Snapping to the nearest solved frame would
    quantise the camera path into steps half a second long, and a car covers
    several metres in that.
    """

    def __init__(self, track: Track, times: list[float]):
        if len(times) != len(track.centres):
            raise ValueError(
                f"{len(times)} timestamps for {len(track.centres)} solved cameras"
            )
        if len(times) < 2:
            raise ValueError("a pose source needs at least two solved cameras")
        order = np.argsort(np.asarray(times, dtype=np.float64))
        self.times = [float(times[i]) for i in order]
        self.poses = [
            Pose(
                rotation=np.asarray(track.rotations[i], dtype=np.float64),
                translation=np.asarray(track.centres[i], dtype=np.float64),
            )
            for i in order
        ]

    def __len__(self) -> int:
        return len(self.poses)

    @property
    def span(self) -> tuple[float, float]:
        return (self.times[0], self.times[-1])

    def at(self, t: float) -> Pose | None:
        """
        The pose at time `t`, or None outside the solved span.

        None rather than the nearest end, deliberately: a frame outside the solve
        has no pose, and extrapolating one would place geometry confidently in a
        spot nothing ever looked at. The reconstructor treats a missing pose as
        "solve this one visually", which is the honest fallback.
        """
        if t < self.times[0] or t > self.times[-1]:
            return None

        index = bisect.bisect_left(self.times, t)
        if index < len(self.times) and self.times[index] == t:
            return self.poses[index]

        after = index
        before = index - 1
        gap = self.times[after] - self.times[before]
        if gap < 1e-9:
            return self.poses[before]

        alpha = (t - self.times[before]) / gap
        translation = (
            self.poses[before].translation * (1 - alpha)
            + self.poses[after].translation * alpha
        )
        # Through the rotation between the two, not by blending matrices: a
        # blended matrix is not a rotation, and an unorthonormal one skews every
        # point the frame contributes.
        rotation = _slerp(self.poses[before].rotation, self.poses[after].rotation, alpha)
        return Pose(rotation=rotation, translation=translation)
