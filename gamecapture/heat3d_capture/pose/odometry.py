"""
Where the camera was, frame by frame.

Without this there is no reconstruction: a depth map alone says how far things
are from *a* viewpoint, and turning a pile of those into one level needs to know
where each viewpoint was.

The scale problem, and how it is chained
----------------------------------------
Each frame's predicted depth is correct only up to its own affine transform, and
crucially that transform is *different for every frame*. Fusing frames with
independent scales produces a level where one corridor is twice the size of the
next — geometry that looks locally plausible and is globally nonsense.

So scale is propagated rather than guessed, one frame at a time:

1. The first frame is declared to be the unit. Its affine is the identity, and
   everything downstream inherits whatever size that implies. One global factor
   for the whole capture is unknown; every *relative* size is not.
2. For each new frame, features are matched back to the previous one. The
   previous frame's depth is already anchored, so those matches lift to 3D points
   in world units.
3. `solvePnPRansac` puts the new camera among those known points. That yields a
   pose whose translation is in the same units — the scale has crossed the gap.
4. The new frame's own depth is then fitted against those same points, which
   anchors it, and the cycle repeats.

The failure this is built to survive is step 3 finding too few inliers — a blank
wall, a hard cut, someone spinning on the spot. That is reported as a broken
track rather than absorbed, because a pose guessed through a gap silently welds
two parts of a level together in the wrong place, and nothing downstream can
detect it.

Coming back after a break
-------------------------
Reporting the break is necessary and not sufficient. The first version simply
started again from the next frame, treating it as a fresh origin — and that is
the same fault wearing a different hat: everything after the gap lands in a
coordinate frame of its own, pinned at the origin, on top of the geometry from
before it. A ground-truth run made it obvious. Tracking was excellent, 35 frames
of 36 placed to within a couple of centimetres, and the path error still peaked
at 8.7 metres, entirely at the one break.

So a break is now followed by *relocalisation* against a small database of past
keyframes: match, solve PnP against their known 3D points, and continue in the
same coordinate frame. Until that succeeds, frames are dropped rather than fused.
Nothing is ever placed at a guessed origin.

That database also supports loop closure, which is **off by default**, because
measured on real footage it did far more harm than good.

The idea is sound: matching a new frame against keyframes from much earlier
detects a revisit and snaps the track back, bounding drift. The problem is
deciding what counts as a revisit. On a rally stage — or a corridor, or a row of
trees — every stretch looks like every other stretch, and feature matching
happily confirms it. Measured over 220 keyframes of a Forza lap:

    loop closure on   path 116.6 m   net displacement   5.3 m
    loop closure off  path 112.3 m   net displacement  83.7 m

Seventy-six false closures pinned the camera within five metres of where it
started while it drove more than a hundred. Tracking reported every frame placed
and every pose was wrong, which is the worst possible combination: a confident,
silent, total failure.

Doing this properly needs place recognition that can tell one stretch of road
from another — a bag-of-words vocabulary or a learned descriptor, plus geometric
verification against the map rather than against a single keyframe. That is not
built here, and a weak version of it is worse than none, so the honest default is
off. The switch remains for a scene with genuinely distinctive places.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..depth.estimator import AffineFit, fit_scale


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole camera parameters, in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fov(cls, width: int, height: int, horizontal_fov_deg: float = 90.0) -> "Intrinsics":
        """
        Build intrinsics from a field of view.

        A game does not publish its projection matrix, and this is the one number
        here that is genuinely assumed. Getting it wrong does not break tracking —
        it bends the reconstruction, because a too-narrow assumed field of view
        makes the scene read as deeper than it is. 90 degrees horizontal is the
        common default for first-person games; racing games sit nearer 65 and the
        setting is exposed for that reason.
        """
        half = np.radians(horizontal_fov_deg) / 2.0
        fx = (width / 2.0) / np.tan(half)
        # Square pixels: the vertical field of view follows from the aspect ratio
        # rather than being a second free parameter.
        return cls(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0, width=width, height=height)

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    def unproject(self, pixels: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Lift (N,2) pixel coordinates with (N,) depths to (N,3) camera-space points."""
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        depth = np.asarray(depth, dtype=np.float64).reshape(-1)
        x = (pixels[:, 0] - self.cx) / self.fx * depth
        y = (pixels[:, 1] - self.cy) / self.fy * depth
        return np.stack([x, y, depth], axis=1)

    def project(self, points: np.ndarray) -> np.ndarray:
        """Project (N,3) camera-space points to (N,2) pixels. Points behind become NaN."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        z = points[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(z > 1e-9, points[:, 0] / z * self.fx + self.cx, np.nan)
            v = np.where(z > 1e-9, points[:, 1] / z * self.fy + self.cy, np.nan)
        return np.stack([u, v], axis=1)


@dataclass
class Pose:
    """
    Camera pose as world-from-camera.

    Stored this way round because it is the useful one: `translation` is simply
    where the camera is, which is what the trajectory, the coverage map and the
    overlay all want. The other convention needs a negation at every use site and
    gets it wrong somewhere eventually.
    """

    rotation: np.ndarray  # (3,3), world from camera
    translation: np.ndarray  # (3,), camera centre in world

    @classmethod
    def identity(cls) -> "Pose":
        return cls(np.eye(3), np.zeros(3))

    def camera_from_world(self) -> tuple[np.ndarray, np.ndarray]:
        """The inverse, as OpenCV's solvePnP and projection functions want it."""
        r = self.rotation.T
        return r, -r @ self.translation

    def to_world(self, camera_points: np.ndarray) -> np.ndarray:
        points = np.asarray(camera_points, dtype=np.float64).reshape(-1, 3)
        return points @ self.rotation.T + self.translation

    @property
    def forward(self) -> np.ndarray:
        """Unit vector the camera looks along, in world space."""
        return self.rotation[:, 2]


#: Weight given to a pose that came from a feed-forward model. Not 1.0: it is an
#: estimate, unlike telemetry. Not derived from anything either, because there is
#: nothing per-frame to derive it from — the whole point of solving a window
#: jointly is that the frames constrain each other rather than one at a time.
FEEDFORWARD_CONFIDENCE = 0.85


@dataclass
class TrackResult:
    """One frame's outcome."""

    pose: Pose
    #: The affine that anchors this frame's depth into world units.
    fit: AffineFit | None
    #: Correspondences that survived RANSAC.
    inliers: int
    #: Correspondences offered to it.
    matches: int
    #: False when the camera could not be located and the track was broken.
    tracked: bool
    reason: str = ""

    @property
    def confidence(self) -> float:
        """0..1, how much this pose can be believed."""
        if not self.tracked:
            return 0.0
        if self.reason == "telemetry":
            # Published by the game, not inferred from pixels. Scoring it on
            # feature-match counts would have a perfectly exact pose fusing with
            # less weight than a lucky visual one.
            return 1.0
        if self.reason == "feedforward":
            # Solved by looking at a window of frames at once, so there are no
            # correspondences to count and the usual score would read as zero —
            # which would fuse every one of these at no weight at all. Below
            # telemetry, which is exact, and above a typical visual match, which
            # is what this exists to improve on.
            return FEEDFORWARD_CONFIDENCE
        if self.matches == 0:
            return 0.0
        ratio = self.inliers / self.matches
        # Both the ratio and the absolute count matter: 20 of 22 is a good ratio
        # on far too little evidence to trust a pose from.
        return float(min(1.0, ratio * min(1.0, self.inliers / 60.0)))


#: Below this many RANSAC inliers a pose is not believed. Chosen to be clearly
#: more than the six correspondences PnP needs at minimum — a solution from
#: barely enough points is exactly the one that is confidently wrong.
MIN_INLIERS = 18

#: A revisit is only informative if the drift has had time to accumulate, so
#: keyframes nearer than this in the database are not loop-closure candidates.
#: Raised hard after the measurement above: twelve keyframes is a few seconds of
#: driving, which is not a revisit, it is the road just behind you.
LOOP_MIN_AGE = 60

#: How near a past keyframe has to be, in world units, to count as the same
#: place. Without a radius, every textured wall is a candidate match for every
#: other one.
LOOP_RADIUS = 6.0

#: RANSAC inliers a closure must have, over and above what ordinary tracking
#: needs. A revisit to a place genuinely seen before matches *strongly*; anything
#: marginal is two pieces of road that happen to look alike.
LOOP_MIN_INLIERS = 90

#: A closure that moves the camera less than this is noise, not a correction.
LOOP_MIN_CORRECTION = 0.02

#: Two viewpoints closer together than this cannot triangulate usefully: the
#: rays are near-parallel and the depth that falls out is noise wearing a
#: number. Metres.
MIN_TRIANGULATION_BASELINE = 0.05


class VisualOdometry:
    """
    Incremental tracking from frames and their predicted depth.

    One previous keyframe is kept, not a map. That makes this odometry rather
    than SLAM: it has no loop closure, so walking a circuit and returning to the
    start will not quite line up. The drift is reported per frame instead of
    being hidden, and the coverage map shows where it is accumulating, which is
    the honest thing to do when the alternative is a much larger system.
    """

    def __init__(
        self,
        intrinsics: Intrinsics,
        *,
        max_features: int = 1500,
        keyframe_stride: int = 4,
        max_keyframes: int = 120,
        loop_closure: bool = False,
        origin_fit: AffineFit | None = None,
    ):
        import cv2

        self.k = intrinsics
        self._orb = cv2.ORB_create(nfeatures=max_features)
        # Hamming distance because ORB descriptors are binary; crossCheck keeps
        # only mutually-best pairs, which removes most of what a ratio test would.
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._previous: dict | None = None
        #: Past frames kept for relocalisation and loop closure. Bounded, because
        #: a half-hour scan would otherwise hold tens of thousands of descriptor
        #: sets; the oldest are dropped, which costs the ability to close a loop
        #: back to the very start of a very long walk and nothing else.
        self._keyframes: list[dict] = []
        self._keyframe_stride = max(1, keyframe_stride)
        self._max_keyframes = max_keyframes
        self._loop_closure = loop_closure
        #: What the very first frame's depth is worth, in metres. `None` declares
        #: it to be the unit — internally consistent, globally arbitrary, and on
        #: a real recording that meant a three-minute lap reconstructing as ten
        #: centimetres of road.
        self._origin_fit = origin_fit
        self._since_keyframe = 0
        self.poses: list[Pose] = []
        self.results: list[TrackResult] = []
        #: How many times the track was recovered, and how many loops were closed.
        self.relocalisations = 0
        self.loop_closures = 0

    def reset(self) -> None:
        self._previous = None
        self._keyframes.clear()
        self._since_keyframe = 0
        self.poses.clear()
        self.results.clear()
        self.relocalisations = 0
        self.loop_closures = 0

    def track(
        self,
        image: np.ndarray,
        inverse_depth: np.ndarray,
        *,
        known_pose: Pose | None = None,
        mask: np.ndarray | None = None,
    ) -> TrackResult:
        """
        Locate one frame.

        `known_pose` short-circuits the search when something already knows the
        answer — a telemetry feed, in practice. The pose is taken as given and
        only the depth anchoring is solved, which is both faster and strictly
        more accurate than re-deriving a pose that was published exactly.

        `mask` excludes pixels that are not the world — HUD, and anything bolted
        to the camera. Applying it *here*, at feature detection, is the whole
        point of having it: a crosshair is the most repeatable corner in any
        frame, it matches perfectly between every pair, and every match it
        contributes is evidence that the camera did not move. Masking it out of
        the fused geometry while still letting it vote on the pose would fix the
        visible symptom and keep the damaging one.
        """
        import cv2

        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        keypoints, descriptors = self._orb.detectAndCompute(grey, mask)

        if known_pose is not None:
            return self._with_known_pose(known_pose, keypoints, descriptors, inverse_depth)

        if self._previous is None and not self._keyframes:
            # The one true origin, and only ever once per run. Its depth is
            # declared to be the unit; everything else is measured against it.
            fit = self._origin_fit or AffineFit(scale=1.0, shift=0.0, residual=0.0, inliers=0)
            result = TrackResult(Pose.identity(), fit, 0, 0, tracked=True, reason="origin")
            self._accept(result, keypoints, descriptors, inverse_depth, keyframe=True)
            return result

        if self._previous is None:
            # Lost. Relocalise against what we have seen before rather than
            # declaring a second origin — everything after a gap would otherwise
            # land in its own coordinate frame, stacked on the geometry from
            # before it.
            recovered = self._relocalise(keypoints, descriptors, inverse_depth)
            if recovered is None:
                return self._still_lost("could not relocalise")
            self.relocalisations += 1
            return recovered

        # Checked here rather than inside the solver so the break says something
        # useful. "Lost the track" is true and tells nobody whether the frame was
        # blank, the match failed, or the geometry was degenerate — and that is
        # the first question asked when a scan comes out in pieces.
        if descriptors is None or len(keypoints) < MIN_INLIERS:
            return self._break_track(f"no features in the frame ({len(keypoints)} found)")

        result = self._track_against(self._previous, keypoints, descriptors, inverse_depth, reason="")
        if result is None:
            return self._break_track("too few matches to place the camera")

        self._accept(result, keypoints, descriptors, inverse_depth, keyframe=self._due())
        if self._loop_closure:
            self._try_loop_closure(keypoints, descriptors, inverse_depth, result)
        return result

    # -- the pieces ------------------------------------------------------

    def _track_against(
        self, reference: dict, keypoints, descriptors, inverse_depth, *, reason: str
    ) -> TrackResult | None:
        """
        Solve this frame's pose against one previous frame. None when it cannot.

        The single operation behind ordinary tracking, relocalisation and loop
        closure alike — they differ only in *which* previous frame is offered.
        """
        import cv2

        if descriptors is None or reference["descriptors"] is None:
            return None
        matches = self._matcher.match(reference["descriptors"], descriptors)
        if len(matches) < MIN_INLIERS:
            return None

        reference_pixels = np.array([reference["keypoints"][m.queryIdx].pt for m in matches])
        current_pixels = np.array([keypoints[m.trainIdx].pt for m in matches])

        reference_depth = _sample(reference["depth_metric"], reference_pixels)
        usable = np.isfinite(reference_depth) & (reference_depth > 1e-3)
        if int(usable.sum()) < MIN_INLIERS:
            return None

        world = reference["pose"].to_world(
            self.k.unproject(reference_pixels[usable], reference_depth[usable])
        )
        image_points = current_pixels[usable]

        ok, rvec, tvec, inlier_index = cv2.solvePnPRansac(
            world.astype(np.float64),
            image_points.astype(np.float64),
            self.k.matrix,
            None,
            reprojectionError=3.0,
            confidence=0.999,
            iterationsCount=200,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        inliers = 0 if inlier_index is None else len(inlier_index)
        if not ok or inliers < MIN_INLIERS:
            return None

        rotation_cw, _ = cv2.Rodrigues(rvec)
        rotation = rotation_cw.T
        pose = Pose(rotation=rotation, translation=(-rotation @ tvec.reshape(3)))

        chosen = inlier_index.reshape(-1)
        camera_points = (world[chosen] - pose.translation) @ pose.rotation
        fit = self._anchor(inverse_depth, image_points[chosen], camera_points[:, 2])
        if fit is None:
            return None
        return TrackResult(pose, fit, inliers, len(matches), tracked=True, reason=reason)

    def _relocalise(self, keypoints, descriptors, inverse_depth) -> TrackResult | None:
        """
        Find the camera again, in the coordinate frame it was already in.

        Most recent keyframes first: after a brief glitch the camera is usually
        still near where it was, so the nearest-in-time keyframe is both the
        likeliest match and the cheapest to find.
        """
        for keyframe in reversed(self._keyframes):
            result = self._track_against(
                keyframe, keypoints, descriptors, inverse_depth, reason="relocalised"
            )
            if result is not None:
                self._accept(result, keypoints, descriptors, inverse_depth, keyframe=True)
                return result
        return None

    def _try_loop_closure(self, keypoints, descriptors, inverse_depth, current: TrackResult) -> None:
        """
        Notice a place we have been before, and snap back onto it.

        Only old keyframes are considered — recent ones always match, because they
        are the frames just gone, and closing a loop with those is tracking under
        another name. A revisit is only informative if the drift has had time to
        accumulate away from it.

        What this does *not* do is redistribute the correction over the frames in
        between, which a pose graph would, nor move geometry already fused. It
        bounds the divergence rather than undoing it, and says so by counting.
        """
        if len(self._keyframes) < LOOP_MIN_AGE + 2:
            return
        candidates = self._keyframes[:-LOOP_MIN_AGE]
        near = [
            keyframe
            for keyframe in candidates
            if np.linalg.norm(keyframe["pose"].translation - current.pose.translation) < LOOP_RADIUS
        ]
        if not near:
            return
        for keyframe in reversed(near[-8:]):
            closed = self._track_against(
                keyframe, keypoints, descriptors, inverse_depth, reason="loop closed"
            )
            if closed is None:
                continue
            if closed.inliers < LOOP_MIN_INLIERS:
                continue
            moved = float(np.linalg.norm(closed.pose.translation - current.pose.translation))
            # A closure that changes nothing is not a closure, and one that
            # changes everything is a mismatch rather than a revisit.
            if LOOP_MIN_CORRECTION < moved < LOOP_RADIUS:
                self.loop_closures += 1
                self.poses[-1] = closed.pose
                self.results[-1] = closed
                if self._previous is not None and closed.fit is not None:
                    self._previous["pose"] = closed.pose
                    self._previous["depth_metric"] = closed.fit.apply(inverse_depth)
            return

    def _due(self) -> bool:
        self._since_keyframe += 1
        if self._since_keyframe >= self._keyframe_stride:
            self._since_keyframe = 0
            return True
        return False

    def _with_known_pose(self, pose: Pose, keypoints, descriptors, inverse_depth) -> TrackResult:
        """
        Take a published pose and solve the depth scale by triangulation.

        The game says where the camera is, not how far away anything it can see
        happens to be, so the scale still has to be found — and the first attempt
        at this got it exactly backwards. It anchored each frame against the
        *previous frame's* depth, the way the visual chain does. But that chain
        starts by declaring frame zero's depth to be the unit, an arbitrary
        choice, while the poses are in metres. The two scales then disagree by
        whatever that arbitrary factor was, every anchoring fit failed, and nine
        frames in ten came back positioned but unfusable.

        With two known poses there is no need to chain anything. The same feature
        seen from two known places triangulates directly, in metres, and the
        depth map is fitted to that. Every frame is anchored against the
        telemetry rather than against its predecessor, so there is no chain to
        drift and no arbitrary unit to inherit.
        """
        import cv2

        previous = self._previous
        fit = None
        inliers = matches = 0

        if previous is not None and descriptors is not None and previous["descriptors"] is not None:
            paired = self._matcher.match(previous["descriptors"], descriptors)
            matches = len(paired)
            if matches >= 8:
                previous_pixels = np.array(
                    [previous["keypoints"][m.queryIdx].pt for m in paired], dtype=np.float64
                )
                current_pixels = np.array(
                    [keypoints[m.trainIdx].pt for m in paired], dtype=np.float64
                )
                baseline = float(
                    np.linalg.norm(pose.translation - previous["pose"].translation)
                )
                # Triangulation from two poses that are nearly the same place is
                # numerically hopeless: the rays are parallel and the depth is
                # whatever the noise says. Better to leave the frame unanchored
                # than to anchor it to a number that means nothing.
                if baseline > MIN_TRIANGULATION_BASELINE:
                    r0, t0 = previous["pose"].camera_from_world()
                    r1, t1 = pose.camera_from_world()
                    p0 = self.k.matrix @ np.hstack([r0, t0.reshape(3, 1)])
                    p1 = self.k.matrix @ np.hstack([r1, t1.reshape(3, 1)])

                    homogeneous = cv2.triangulatePoints(p0, p1, previous_pixels.T, current_pixels.T)
                    w = homogeneous[3]
                    good = np.abs(w) > 1e-9
                    world = np.zeros((homogeneous.shape[1], 3))
                    world[good] = (homogeneous[:3, good] / w[good]).T

                    camera_points = (world - pose.translation) @ pose.rotation
                    ahead = good & (camera_points[:, 2] > 1e-3)
                    if int(ahead.sum()) >= 2:
                        inliers = int(ahead.sum())
                        fit = self._anchor(
                            inverse_depth, current_pixels[ahead], camera_points[ahead, 2]
                        )

        result = TrackResult(
            pose, fit, inliers, max(matches, inliers), tracked=True, reason="telemetry"
        )
        self._accept(result, keypoints, descriptors, inverse_depth, keyframe=self._due())
        return result

    def _accept(
        self, result: TrackResult, keypoints, descriptors, inverse_depth, *, keyframe: bool
    ) -> None:
        self.poses.append(result.pose)
        self.results.append(result)
        self._remember(keypoints, descriptors, inverse_depth, result)
        if keyframe and self._previous is not None:
            self._keyframes.append(dict(self._previous))
            if len(self._keyframes) > self._max_keyframes:
                del self._keyframes[0]

    def _anchor(
        self, inverse_depth: np.ndarray, pixels: np.ndarray, depths: np.ndarray
    ) -> AffineFit | None:
        predicted = _sample(inverse_depth, pixels)
        good = np.isfinite(predicted) & np.isfinite(depths) & (depths > 1e-3)
        if int(good.sum()) < 2:
            return None
        try:
            return fit_scale(predicted[good], depths[good])
        except ValueError:
            return None

    def _remember(self, keypoints, descriptors, inverse_depth, result: TrackResult) -> None:
        metric = (
            result.fit.apply(inverse_depth)
            if result.fit is not None
            else np.full_like(inverse_depth, np.nan)
        )
        self._previous = {
            "keypoints": keypoints,
            "descriptors": descriptors,
            "depth_metric": metric,
            "pose": result.pose,
        }

    def _break_track(self, reason: str) -> TrackResult:
        """
        Report a break and stop tracking until relocalisation succeeds.

        No pose is recorded. Carrying on from the last known one, or starting a
        fresh origin, both put the next stretch of level at an arbitrary offset —
        a seam that looks like real geometry and cannot be found afterwards.
        """
        result = TrackResult(Pose.identity(), None, 0, 0, tracked=False, reason=reason)
        self.results.append(result)
        self._previous = None
        return result

    def _still_lost(self, reason: str) -> TrackResult:
        result = TrackResult(Pose.identity(), None, 0, 0, tracked=False, reason=reason)
        self.results.append(result)
        return result


def _sample(field: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Nearest-neighbour sample of a (H,W) field at (N,2) pixel coordinates."""
    h, w = field.shape[:2]
    x = np.clip(np.round(pixels[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(pixels[:, 1]).astype(int), 0, h - 1)
    return field[y, x]


@dataclass
class Trajectory:
    """The path the camera took, and how well it was known."""

    poses: list[Pose] = field(default_factory=list)
    results: list[TrackResult] = field(default_factory=list)

    @property
    def positions(self) -> np.ndarray:
        if not self.poses:
            return np.zeros((0, 3))
        return np.array([p.translation for p in self.poses])

    @property
    def length(self) -> float:
        """Distance walked, in world units."""
        points = self.positions
        if len(points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

    @property
    def breaks(self) -> int:
        return sum(1 for r in self.results if not r.tracked)
