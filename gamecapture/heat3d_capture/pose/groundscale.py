"""
Fixing the one number monocular vision cannot recover: how big everything is.

A single camera sees shape, never size. A photograph of a street and a
photograph of a model of that street are the same photograph, and no amount of
processing separates them. The pipeline chains *relative* scale from frame to
frame correctly, so the level is internally consistent — but the whole thing
still floats on one unknown global factor, and the first frame was simply
declaring that factor to be 1.

That is not a harmless placeholder. Depth Anything's relative inverse depth comes
out in arbitrary units of order one to twenty, so `depth = 1 / predicted` puts
the entire world within a metre of the camera. Measured on a real recording — a
Forza lap, three minutes of driving — the recovered path came to **0.1 metres**.
The geometry was right; it was a doll's house of it, and useless next to
telemetry measured in metres.

So the factor is pinned to something known. The most reliable thing available in
both a driving game and a walking one is *how high the camera is above the
ground*: a driver's eyeline sits a bit over a metre, a standing player's a bit
under two. Neither is exact, and neither has to be — being within ten per cent of
true beats being out by a factor of a thousand.

Telemetry, where a game publishes it, is strictly better and makes this
unnecessary. This is what to do when it does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..depth.estimator import AffineFit
from .odometry import Intrinsics

#: Typical camera heights in metres, for the two cases this tool is built for.
DRIVING_HEIGHT = 1.15
WALKING_HEIGHT = 1.70

#: The band of the frame assumed to be ground, as fractions of height from the
#: top. The very bottom is bonnet, dashboard or weapon; above the middle is
#: horizon and sky. Between those, looking forward, is road or floor.
GROUND_BAND = (0.62, 0.92)

#: And horizontally, the middle, away from the verges and the frame edges where
#: depth is least reliable.
GROUND_WIDTH = (0.30, 0.70)

#: How un-flat the sampled band may be and still be believed. See `trustworthy`.
MAX_SCATTER = 0.60

#: Frames to calibrate from before settling on an answer.
#:
#: One, because the answer has to be available before the first frame's depth is
#: used as the unit, and waiting for more would mean either discarding the start
#: of the scan or re-basing it afterwards. Both were tried; both were worse than
#: the accuracy they bought. Single-frame estimates on real footage agreed to
#: within 6% across three minutes of driving.
CALIBRATION_FRAMES = 1


@dataclass
class GroundScale:
    """The result of a calibration, and how much to believe it."""

    fit: AffineFit
    #: Metres the camera was assumed to be above the ground.
    height: float
    #: How consistent the sampled ground points were; lower is better. This is
    #: the spread of implied heights divided by their median.
    scatter: float
    samples: int

    @property
    def trustworthy(self) -> bool:
        # Loosened from a third after measuring real footage. A rally stage at
        # night gave scatter of 0.40 to 0.45 on every frame sampled — and the
        # scale it implied was stable to within 6% across three minutes of
        # driving (18.4 to 19.6 metres per unit). The tight bound was rejecting a
        # calibration that was demonstrably working, on the grounds that a real
        # road is not as flat as a rendered one. It is not, and does not need to
        # be: the alternative was a three-minute lap reconstructing as ten
        # centimetres.
        return self.samples >= 200 and self.scatter < MAX_SCATTER


def calibrate_from_ground(
    inverse_depth: np.ndarray,
    k: Intrinsics,
    *,
    height: float = DRIVING_HEIGHT,
) -> GroundScale | None:
    """
    Turn one frame's relative depth into metres, using the ground as a ruler.

    Returns None when the frame has no usable ground band at all. The caller
    should then carry on unscaled rather than guess — an arbitrary factor that
    looks deliberate is worse than one that is visibly absent.
    """
    inverse_depth = np.asarray(inverse_depth, dtype=np.float64)
    h, w = inverse_depth.shape[:2]

    top, bottom = int(h * GROUND_BAND[0]), int(h * GROUND_BAND[1])
    left, right = int(w * GROUND_WIDTH[0]), int(w * GROUND_WIDTH[1])
    if bottom <= top or right <= left:
        return None

    band = inverse_depth[top:bottom, left:right]
    ys, xs = np.mgrid[top:bottom, left:right]
    predicted = band.ravel()
    usable = np.isfinite(predicted) & (predicted > 1e-6)
    if int(usable.sum()) < 200:
        return None

    # Depth in the network's own units, before any scaling.
    unit_depth = 1.0 / predicted[usable]
    # Height below the camera, in those same units. Camera space is y-down, so a
    # point on the ground has positive y.
    below = (ys.ravel()[usable] - k.cy) / k.fy * unit_depth
    # Only points actually below the optical axis are ground.
    ground = below > 1e-6
    if int(ground.sum()) < 200:
        return None
    below = below[ground]

    typical = float(np.median(below))
    if typical <= 1e-9:
        return None
    # Median absolute deviation over the median: a scale-free measure of how
    # plane-like the sample was, which is the only check available without
    # knowing the answer.
    scatter = float(np.median(np.abs(below - typical)) / typical)

    # The factor that turns the network's units into metres.
    metres_per_unit = height / typical
    # depth_metres = metres_per_unit / predicted, and the pipeline's affine is
    # `metric_disparity = scale * predicted + shift`, so scale is the reciprocal.
    fit = AffineFit(
        scale=1.0 / metres_per_unit, shift=0.0, residual=scatter, inliers=int(ground.sum())
    )
    return GroundScale(
        fit=fit, height=height, scatter=scatter, samples=int(ground.sum())
    )


class GroundCalibrator:
    """
    Accumulates ground estimates over several frames and settles on the median.

    One frame is a sample of whatever the car happened to be driving over at that
    instant — a crest, a dip, a kerb. Taking the median of a handful is both more
    robust and free, since the frames are arriving anyway.
    """

    def __init__(self, k: Intrinsics, *, height: float = DRIVING_HEIGHT):
        self.k = k
        self.height = height
        self._scales: list[float] = []
        self._scatters: list[float] = []
        self.note = "measuring the ground"

    @property
    def settled(self) -> bool:
        return len(self._scales) >= CALIBRATION_FRAMES

    def offer(self, inverse_depth: np.ndarray) -> AffineFit | None:
        """Feed one frame. Returns the fit once enough have been seen."""
        if self.settled:
            return self._result()
        estimate = calibrate_from_ground(inverse_depth, self.k, height=self.height)
        if estimate is not None and estimate.trustworthy:
            self._scales.append(estimate.fit.scale)
            self._scatters.append(estimate.scatter)
        return self._result() if self.settled else None

    def _result(self) -> AffineFit | None:
        if not self._scales:
            self.note = "no usable ground found — sizes are relative, not metres"
            return None
        scale = float(np.median(self._scales))
        spread = float(np.std(self._scales) / max(abs(scale), 1e-9))
        self.note = (
            f"scaled from a {self.height:.2f} m camera height, "
            f"{len(self._scales)} frames agreeing to {spread * 100:.0f}%"
        )
        return AffineFit(
            scale=scale, shift=0.0, residual=float(np.median(self._scatters)),
            inliers=len(self._scales),
        )
