"""
Preparing frames for a feed-forward solve.

A reconstruction model is handed pictures and told to work out where the camera
was. Two properties of a gameplay capture make that much harder than it needs to
be, and both are fixable before the model ever sees the frame.

**Half the picture is not the world.** In a cockpit view the dashboard, the
pillars, the wheel and the HUD are bolted to the camera, and they are exactly the
kind of thing a feature detector loves: high contrast, corner-rich, and
absolutely still. Measured on the reference lap, **60–79% of all detected
features sat on camera-attached pixels** — so the median apparent motion between
frames a second apart was 0–4 pixels, and the honest answer for the world behind
it was hidden underneath. A model told that most of the scene never moves will
conclude the camera barely moved.

**The world that is left is dark.** The reference lap is at night: outside the
headlight cone there is very little contrast to find anything in at all.

Masking the camera-attached region and lifting local contrast on what remains
changes the measurement completely. On the same footage, matched world features
between frames 1.17 s apart went from a median displacement of 0–4 px to 13–245
px, with 780–970 matches per pair — parallax that was always there, behind an
instrument panel that was drowning it out.

Local contrast, not global
--------------------------
CLAHE rather than a brightness curve, and on the lightness channel rather than on
all three. A night frame's histogram is dominated by black sky; anything global
either leaves the road flat or blows out the headlight cone. Equalising in tiles
lifts the road surface and the verge without touching what is already bright, and
keeping it off the colour channels means the result still looks like the scene
rather than like a false-colour image.
"""

from __future__ import annotations

import numpy as np

from ..capture.flowmask import FlowMask

#: Frames drawn from each sampled point of the recording to learn the mask from.
#: The mask wants *consecutive* frames — it measures how far a feature travels —
#: so it cannot reuse the solver's sparse sample.
LEARN_RUN = 90

#: How many places in the recording to learn from. Spread out, because a cockpit
#: is the same everywhere but the lighting and the scenery are not, and a mask
#: learned from one straight would carry that straight's accidents.
LEARN_POINTS = 4

#: CLAHE's ceiling on local contrast gain. Above about 4 the grain in a dark sky
#: starts being amplified into features of its own.
CLIP_LIMIT = 3.0
TILE_GRID = (8, 8)


def learn_camera_mask(
    video, *, points: int = LEARN_POINTS, run: int = LEARN_RUN
) -> tuple[np.ndarray | None, str]:
    """
    Watch a few runs of consecutive frames and return what is bolted to the camera.

    Returns the mask and the detector's own account of what it decided, so a
    caller can report it rather than silently applying something.
    """
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OSError(f"could not open {video}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < run:
            return None, "recording too short to learn a camera mask from"

        greys: list[np.ndarray] = []
        shape: tuple[int, int] | None = None
        for n in range(points):
            start = int((n + 0.5) * total / points) - run // 2
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, start))
            for _ in range(run):
                ok, frame = capture.read()
                if not ok:
                    break
                small = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
                shape = small.shape[:2]
                greys.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()

    if not greys or shape is None:
        return None, "no frames could be read to learn a camera mask from"

    detector = FlowMask()
    detector.observe_sequence(greys)
    return detector.mask(shape), detector.describe()


def enhance(frame: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """
    One frame, with the camera's own furniture removed and the rest lifted.

    The mask is resized to the frame rather than the other way round, so this
    works whatever resolution the caller is sampling at.
    """
    import cv2

    out = frame
    if frame.ndim == 3:
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID)
        out = clahe.apply(frame)

    if mask is None:
        return out

    if mask.shape[:2] != out.shape[:2]:
        mask = cv2.resize(
            mask, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    # Black, because it is the absence of a measurement rather than a dark
    # surface. A featureless region contributes nothing; a textured one that
    # never moves contributes a lie.
    out = out.copy()
    out[mask == 0] = 0
    return out


def enhance_all(frames: list[np.ndarray], mask: np.ndarray | None) -> list[np.ndarray]:
    return [enhance(f, mask) for f in frames]


#: Target displacement between frames handed to the solver, as a fraction of
#: frame width. About 5%: close enough that a feature seen in one frame is
#: comfortably inside the next, far enough that the pair actually carries
#: parallax rather than being two views of the same spot.
TARGET_FLOW = 0.05

#: Never take frames closer together than this, whatever the motion. A stationary
#: camera would otherwise emit nothing at all and a crawling one would emit the
#: whole recording.
MIN_GAP = 0.15

#: Nor further apart than this: a long stop should not swallow the rest of the lap
#: into one step.
MAX_GAP = 2.0


class _Prober:
    """
    Measures displacement between two moments of one recording.

    Holds the capture, the detector and the matcher open across measurements.
    Planning a lap takes a few hundred of them, and reopening a 4K file for each
    pair costs more than all the matching put together.
    """

    def __init__(self, video, mask: np.ndarray | None):
        import cv2

        self.capture = cv2.VideoCapture(str(video))
        if not self.capture.isOpened():
            raise OSError(f"could not open {video}")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.duration = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) / self.fps
        self.mask = mask
        self.clahe = cv2.createCLAHE(CLIP_LIMIT, TILE_GRID)
        self.orb = cv2.ORB_create(2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._cached: tuple[float, tuple] | None = None

    def close(self) -> None:
        self.capture.release()

    def __enter__(self) -> "_Prober":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _features(self, at: float):
        # Each step measures from where the last one ended, so the second frame
        # of one pair is the first of the next. Keeping it saves half the work.
        if self._cached is not None and abs(self._cached[0] - at) < 1e-9:
            return self._cached[1]
        import cv2

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(at * self.fps))
        ok, frame = self.capture.read()
        if not ok:
            return None
        small = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        grey = self.clahe.apply(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        found = self.orb.detectAndCompute(grey, self.mask)
        self._cached = (at, found)
        return found

    def flow(self, at: float, gap: float) -> float:
        """Median feature displacement between two frames `gap` seconds apart."""
        first = self._features(at)
        second = self._features(at + gap)
        if first is None or second is None or first[1] is None or second[1] is None:
            return float("nan")
        matches = self.matcher.match(first[1], second[1])
        if not matches:
            return float("nan")
        moved = [
            float(
                np.linalg.norm(
                    np.array(first[0][m.queryIdx].pt)
                    - np.array(second[0][m.trainIdx].pt)
                )
            )
            for m in matches
        ]
        return float(np.median(moved))


def measure_flow(video, mask: np.ndarray | None, *, at: float, gap: float) -> float:
    """Median feature displacement between two frames `gap` seconds apart."""
    with _Prober(video, mask) as prober:
        return prober.flow(at, gap)


def plan_by_motion(
    video,
    mask: np.ndarray | None,
    *,
    target: float = TARGET_FLOW,
    budget: int = 160,
) -> list[float]:
    """
    Timestamps spaced by how far the picture moved, not by the clock.

    A fixed interval is the wrong criterion for a lap, and the measurement says
    so plainly. At 1.17 s between frames, the median feature displacement over
    one recording ran from 13 px where the car was crawling to 149 px where it
    was quick — and the windows that came out wrong were the *fast* ones, where
    consecutive frames no longer overlapped enough for the solver to relate them.
    A clock gives the slow sections frames they do not need and starves the fast
    ones of the frames they do.

    So the gap is chosen per step: measure the displacement, and lengthen or
    shorten the next step to hold it near the target. Bounded at both ends, since
    a stationary camera would otherwise emit nothing and a long pause would
    swallow the rest of the recording into a single step.

    `budget` caps the count, because solving is minutes per window and an
    unbounded plan is not a plan. Hitting it is worth reporting: it means the
    recording is longer or faster than can be solved at this spacing.
    """
    want = target * 960.0
    with _Prober(video, mask) as prober:
        total = prober.duration
        times = [0.0]
        gap = MIN_GAP * 2
        while times[-1] < total - MIN_GAP and len(times) < budget:
            moved = prober.flow(times[-1], gap)
            if not np.isfinite(moved) or moved <= 1e-3:
                # Nothing matched, so the gap tells us nothing. Shrink rather
                # than grow: too far apart is the failure this exists to avoid.
                gap = max(MIN_GAP, gap * 0.6)
                times.append(min(total, times[-1] + gap))
                continue
            times.append(min(total, times[-1] + gap))
            # Aim at the target, but move there gradually: one noisy measurement
            # should not swing the whole rest of the plan.
            gap = float(np.clip(gap * (0.5 + 0.5 * want / moved), MIN_GAP, MAX_GAP))
    return times
