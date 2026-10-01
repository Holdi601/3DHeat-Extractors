"""
Odometry that solves a window of frames together instead of one at a time.

**Not wired in. It does not work on real footage yet — see the end of this note.**

The replacement for frame-to-frame tracking, and the reason is a number: on a
real Forza lap the old chain produced a camera that climbed 691 metres over a
level drive. Each pose was computed from its predecessor, so a small error in one
was inherited by every pose after it, and nothing ever went back to check.

Here each new frame is first *placed* against the map of 3D points already
triangulated — not against the previous frame — and then the last N poses and all
the points they see are adjusted together to minimise reprojection error. A point
tracked through fifteen frames constrains fifteen poses at once. There is no
chain for an error to travel along.

What is still missing, and worth naming: this is a *local* bundle adjustment over
a sliding window, not a global one over the whole scan. Drift within the window
is removed; drift accumulated across many windows is reduced, not eliminated. A
global solve would need the whole sequence in memory and cannot run live, and a
proper answer to the rest is loop closure with real place recognition, which is
also not here.

The depth map keeps two jobs and loses one. It no longer determines pose. It
still bootstraps the first few frames, before any point has been triangulated,
and it still supplies the dense geometry that gets fused — bundle adjustment
produces a few hundred points, which is a skeleton, not a level.

Where this stands
-----------------
The solver itself is verified against known truth: given poses pushed off by
tens of centimetres it recovers them to under a quarter of the error, holds its
anchor exactly, and tolerates half a pixel of observation noise. Those tests are
in `tests/test_bundle.py` and they pass.

The *integration* here does not work yet. On a real Forza lap, 400 keyframes:

    frame-to-frame   400 placed   38 m net   2 m vertical drift     6 s
    windowed + BA    399 placed    0 m net   0 m                  184 s

It collapses to nothing and is thirty times slower. Two causes are understood and
one is not:

- Gauge freedom. Fixing one camera removes six degrees of freedom and leaves the
  seventh, scale, free; the solver shrinks the whole configuration because that
  fits equally well. Fixing two was added and was not sufficient on its own.
- The bootstrap. The first poses come from the depth map, every later pose is
  placed against points triangulated from those poses, and the depth is then
  re-anchored against those same points. The scale has nothing outside itself to
  hold it.
- Something else, still unidentified, since the above does not account for a
  complete collapse.

So the working frame-to-frame path remains the default. This is kept because the
solver is sound and the front-end — long Lucas-Kanade tracks in `tracks.py` — is
what any real structure-from-motion needs; what is missing is a correct
initialisation, most likely a two-view essential-matrix bootstrap with the
baseline fixed from the ground-plane calibration rather than from the depth map.
"""

from __future__ import annotations

import numpy as np

from ..depth.estimator import AffineFit, fit_scale
from .bundle import (
    MIN_ANGLE_DEG,
    MIN_OBSERVATIONS,
    OUTLIER_PIXELS,
    Observation,
    Problem,
    bundle_adjust,
    triangulate,
)
from .odometry import Intrinsics, Pose, TrackResult
from .tracks import FeatureTracks

#: Keyframes solved together. Larger removes more drift and costs more than
#: linearly, because the solver's work grows with observations as well as poses.
WINDOW = 12

#: 2D-3D correspondences needed to place a frame against the map. Below this the
#: map has not been built yet and the depth map is used instead.
MIN_MAP_POINTS = 25

#: Run the solver every N keyframes rather than every one. The window overlaps
#: heavily between consecutive frames, so solving each is mostly repeated work.
SOLVE_EVERY = 3

#: Reprojection error, in pixels, above which a map point is thrown away after a
#: solve. Bundle adjustment has no defence against a point that is simply wrong:
#: it will happily move good poses to accommodate it, and the residual it reports
#: goes *down* while the geometry goes off. Measured on a fast-panning sequence,
#: leaving these in produced a 25-metre walk reconstructed as 190 metres with one
#: pose flung 66 metres clear of the truth.
MAX_POINT_ERROR = 6.0

#: A new pose further than this multiple of recent motion from the last one is
#: refused. Real movement between keyframes is bounded by how fast someone can
#: walk or drive; a jump far outside that is a solver failure, not a discovery.
MAX_JUMP_FACTOR = 6.0

#: Below this, motion estimates are too small to form a meaningful expectation.
MIN_JUMP_METRES = 0.5


class WindowedOdometry:
    """Tracks features, triangulates them, and bundle-adjusts a sliding window."""

    def __init__(
        self,
        intrinsics: Intrinsics,
        *,
        window: int = WINDOW,
        origin_fit: AffineFit | None = None,
    ):
        self.k = intrinsics
        self.window = window
        self._origin_fit = origin_fit
        self.features = FeatureTracks()

        self.poses: list[Pose] = []
        self.results: list[TrackResult] = []
        #: Triangulated world points, and which track each came from.
        self.points: list[np.ndarray] = []
        self._point_track: list[int] = []

        self.solves = 0
        self.dropped_points = 0
        self.last_error = 0.0
        self._since_solve = 0
        #: Kept only to bootstrap, before the map exists.
        self._previous_depth: np.ndarray | None = None
        self._previous_pose: Pose | None = None

    # -- the per-frame path ----------------------------------------------

    def track(
        self,
        image: np.ndarray,
        inverse_depth: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        known_pose: Pose | None = None,
    ) -> TrackResult:
        import cv2

        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        positions = self.features.add_frame(grey, mask)
        frame = self.features.frame

        if frame == 0:
            fit = self._origin_fit or AffineFit(1.0, 0.0, 0.0, 0)
            return self._accept(Pose.identity(), fit, 0, len(positions), "origin", inverse_depth)

        pose, inliers, how = self._place(positions, inverse_depth, known_pose)
        if pose is not None and known_pose is None and not self._plausible(pose):
            pose, how = None, "the solved pose jumped implausibly far"
        if pose is None:
            result = TrackResult(Pose.identity(), None, 0, len(positions), False, how)
            self.results.append(result)
            return result

        fit = self._anchor(pose, positions, inverse_depth)
        result = self._accept(pose, fit, inliers, len(positions), how, inverse_depth)

        self._grow_map()
        self._since_solve += 1
        if self._since_solve >= SOLVE_EVERY:
            self._since_solve = 0
            self._solve()
        return result

    def _plausible(self, pose: Pose) -> bool:
        """
        Is this pose a believable distance from the last few?

        A sanity bound, not a model of motion. Bundle adjustment and PnP both
        fail by producing a confident answer in the wrong place, and without a
        bound on how far a camera may move between two keyframes there is nothing
        to notice that with.
        """
        if len(self.poses) < 3:
            return True
        recent = np.array([p.translation for p in self.poses[-6:]])
        steps = np.linalg.norm(np.diff(recent, axis=0), axis=1)
        typical = max(float(np.median(steps)), MIN_JUMP_METRES)
        jump = float(np.linalg.norm(pose.translation - self.poses[-1].translation))
        return jump <= typical * MAX_JUMP_FACTOR

    def _drop_bad_points(self) -> int:
        """
        Remove map points that do not reproject where they are seen.

        Run after each solve. A point triangulated from a mistracked feature, or
        from views too close together, is not a weak constraint — it actively
        pulls good poses toward its own error, and it will keep doing so for as
        long as it is in the map.
        """
        keep_points: list[np.ndarray] = []
        keep_tracks: list[int] = []
        dropped = 0

        for point_index, track_id in enumerate(self._point_track):
            track = self.features.tracks.get(track_id)
            point = self.points[point_index]
            errors = []
            if track is not None:
                for frame, uv in track.seen.items():
                    if 0 <= frame < len(self.poses):
                        pose = self.poses[frame]
                        camera = (point - pose.translation) @ pose.rotation
                        if camera[2] <= 1e-6:
                            errors.append(np.inf)
                            continue
                        projected = self.k.project(camera[None])[0]
                        errors.append(float(np.linalg.norm(projected - uv)))
            if errors and float(np.median(errors)) > MAX_POINT_ERROR:
                if track is not None:
                    track.point = None
                dropped += 1
                continue
            if track is not None:
                track.point = len(keep_points)
            keep_points.append(point)
            keep_tracks.append(track_id)

        self.points = keep_points
        self._point_track = keep_tracks
        return dropped

    # -- placing this frame ----------------------------------------------

    def _place(self, positions, inverse_depth, known_pose):
        """Pose for this frame: told, solved against the map, or bootstrapped."""
        import cv2

        if known_pose is not None:
            return known_pose, 0, "telemetry"

        world, image_points = [], []
        for point_index, track_id in enumerate(self._point_track):
            uv = positions.get(track_id)
            if uv is not None:
                world.append(self.points[point_index])
                image_points.append(uv)

        if len(world) >= MIN_MAP_POINTS:
            ok, rvec, tvec, inlier_index = cv2.solvePnPRansac(
                np.array(world, dtype=np.float64),
                np.array(image_points, dtype=np.float64),
                self.k.matrix,
                None,
                reprojectionError=OUTLIER_PIXELS,
                confidence=0.999,
                iterationsCount=200,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            count = 0 if inlier_index is None else len(inlier_index)
            if ok and count >= MIN_MAP_POINTS // 2:
                rotation = cv2.Rodrigues(rvec)[0].T
                return (
                    Pose(rotation, -rotation @ tvec.reshape(3)),
                    count,
                    "",
                )

        # No map yet: fall back to the depth map, which is what the first few
        # frames have and all the old pipeline ever used.
        return self._bootstrap(positions, inverse_depth)

    def _bootstrap(self, positions, inverse_depth):
        import cv2

        if self._previous_depth is None or self._previous_pose is None:
            return None, 0, "nothing to bootstrap from"
        previous_frame = self.features.frame - 1

        world, image_points = [], []
        height, width = self._previous_depth.shape[:2]
        for track_id, uv in positions.items():
            earlier = self.features.tracks[track_id].seen.get(previous_frame)
            if earlier is None:
                continue
            x = int(np.clip(round(earlier[0]), 0, width - 1))
            y = int(np.clip(round(earlier[1]), 0, height - 1))
            depth = self._previous_depth[y, x]
            if not np.isfinite(depth) or depth <= 1e-3:
                continue
            world.append(self._previous_pose.to_world(self.k.unproject(earlier[None], [depth]))[0])
            image_points.append(uv)

        if len(world) < 12:
            return None, 0, f"only {len(world)} points with depth to bootstrap from"

        ok, rvec, tvec, inlier_index = cv2.solvePnPRansac(
            np.array(world, dtype=np.float64),
            np.array(image_points, dtype=np.float64),
            self.k.matrix, None,
            reprojectionError=OUTLIER_PIXELS, confidence=0.999, iterationsCount=200,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        count = 0 if inlier_index is None else len(inlier_index)
        if not ok or count < 10:
            return None, 0, f"bootstrap had {count} inliers"
        rotation = cv2.Rodrigues(rvec)[0].T
        return Pose(rotation, -rotation @ tvec.reshape(3)), count, "bootstrap"

    # -- the map ----------------------------------------------------------

    def _grow_map(self) -> None:
        """Triangulate tracks that have earned it and are not yet in the map."""
        first = max(0, self.features.frame - self.window + 1)
        already = set(self._point_track)

        for track in self.features.in_window(first):
            if track.id in already or track.point is not None:
                continue
            frames = sorted(f for f in track.seen if 0 <= f < len(self.poses))
            if len(frames) < MIN_OBSERVATIONS:
                continue
            views = [(self.poses[f], track.seen[f]) for f in frames]
            result = triangulate(views, self.k)
            if result is None:
                continue
            point, angle = result
            # An ill-conditioned point is not a weak constraint, it is a wrong
            # one: it will pull poses toward whatever its noise decided.
            if angle < MIN_ANGLE_DEG or not np.isfinite(point).all():
                continue
            # And it must be in front of the cameras that claim to see it.
            if any((point - self.poses[f].translation) @ self.poses[f].forward <= 0 for f in frames):
                continue
            track.point = len(self.points)
            self.points.append(point)
            self._point_track.append(track.id)

    def _solve(self) -> None:
        """Bundle-adjust the window."""
        first = max(0, len(self.poses) - self.window)
        window_poses = self.poses[first:]
        if len(window_poses) < 3 or not self.points:
            return

        local_index: dict[int, int] = {}
        local_points: list[np.ndarray] = []
        observations: list[Observation] = []

        for point_index, track_id in enumerate(self._point_track):
            track = self.features.tracks.get(track_id)
            if track is None:
                continue
            seen_here = [f for f in track.seen if f >= first and f < len(self.poses)]
            if len(seen_here) < 2:
                continue
            if point_index not in local_index:
                local_index[point_index] = len(local_points)
                local_points.append(self.points[point_index])
            for f in seen_here:
                observations.append(
                    Observation(
                        camera=f - first,
                        point=local_index[point_index],
                        uv=track.seen[f],
                    )
                )

        if len(observations) < 40 or len(local_points) < 10:
            return

        problem = Problem(
            poses=[Pose(p.rotation.copy(), p.translation.copy()) for p in window_poses],
            points=np.array(local_points, dtype=np.float64),
            observations=observations,
            # *Two* poses, not one. Fixing a single camera removes position and
            # rotation — six degrees of freedom — and leaves the seventh, scale,
            # entirely free. The solver then shrinks the whole configuration,
            # which fits the observations exactly as well and moves every camera
            # closer to every point. Measured on real footage before this was
            # understood: a 38-metre drive collapsed to 1 metre.
            #
            # Fixing the second camera as well pins the baseline between them and
            # with it the scale of everything the window contains.
            fixed={0, 1},
        )
        result = bundle_adjust(problem, self.k)
        self.solves += 1
        self.last_error = result.after
        if not result.improved:
            return

        for i, pose in enumerate(result.poses):
            self.poses[first + i] = pose
        for point_index, local in local_index.items():
            self.points[point_index] = result.points[local]

        # After the solve, not before: a point's error is only meaningful once
        # the poses that see it have settled.
        self.dropped_points += self._drop_bad_points()

    # -- scale and bookkeeping -------------------------------------------

    def _anchor(self, pose: Pose, positions, inverse_depth) -> AffineFit | None:
        """Fit this frame's depth to the map, so fused geometry is in step."""
        predicted, metric = [], []
        for point_index, track_id in enumerate(self._point_track):
            uv = positions.get(track_id)
            if uv is None:
                continue
            depth = float((self.points[point_index] - pose.translation) @ pose.forward)
            if depth <= 1e-3:
                continue
            x = int(np.clip(round(uv[0]), 0, inverse_depth.shape[1] - 1))
            y = int(np.clip(round(uv[1]), 0, inverse_depth.shape[0] - 1))
            predicted.append(inverse_depth[y, x])
            metric.append(depth)

        if len(predicted) >= 8:
            try:
                return fit_scale(predicted, metric)
            except ValueError:
                pass
        # Before the map exists, carry the previous frame's scale rather than
        # inventing one; the bootstrap poses are in those units already.
        for result in reversed(self.results):
            if result.fit is not None:
                return result.fit
        return self._origin_fit or AffineFit(1.0, 0.0, 0.0, 0)

    def _accept(self, pose, fit, inliers, matches, reason, inverse_depth) -> TrackResult:
        result = TrackResult(pose, fit, inliers, matches, True, reason)
        self.poses.append(pose)
        self.results.append(result)
        self._previous_pose = pose
        self._previous_depth = (
            fit.apply(inverse_depth) if fit is not None else None
        )
        # Sightings older than the window will never be solved again.
        self.features.trim(max(0, self.features.frame - self.window * 2))
        return result

    def reset(self) -> None:
        self.__init__(self.k, window=self.window, origin_fit=self._origin_fit)
