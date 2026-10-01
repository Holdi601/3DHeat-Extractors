"""
Is this frame worth reconstructing from, and if not, what should the user do?

The coverage map already answers "where have I been". This answers the other
question someone has while driving, which is "is what I am doing right now any
good" — and it has to answer it in time to change the answer.

Three things ruin a frame for reconstruction, and each has a different remedy:

    blurred    the camera moved too far while the shutter was open   → slow down
    dark       there is nothing in the picture to measure            → find light
    empty      flat sky, a wall, a loading screen: no structure      → look around

Telling them apart matters, because the advice is opposite. "Slow down" is wrong
for a dark tunnel and useless on a featureless wall.

Measured against the capture, not against a constant
----------------------------------------------------
Sharpness has no absolute scale: the Laplacian variance of a night rally stage
and of a bright city street differ by an order of magnitude with both perfectly
in focus. So every reading is judged against the running median of *this*
capture. "Blurrier than this capture usually is" is actionable; "below 120" is
not, and would have this shouting at someone the entire time on dark footage —
which is exactly the footage where it would be most annoying and least useful.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

#: Frames kept for the running median. A few seconds at capture rate: long
#: enough to be stable, short enough to follow a change of scene.
HISTORY = 90

#: Readings needed before any judgement is offered. Until then the honest answer
#: is that there is nothing to compare against.
WARMUP = 20

#: Sharpness below this fraction of the capture's own median is blurred. Not
#: tighter, because ordinary frame-to-frame variation is substantial and advice
#: that fires on half the frames is noise.
BLUR_RATIO = 0.55

#: Mean luminance, 0..255, below which there is too little signal to measure.
DARK_LEVEL = 26.0

#: Absolute sharpness below which the frame has no structure at all, whatever
#: the capture's median says. A loading screen, a solid wall or a clear sky sits
#: here — and so does a capture pointed somewhere useless, in which case the
#: median is meaningless and the ratio test above cannot help.
#:
#: Near zero on purpose, because "flat" and "blurred" are different problems with
#: opposite advice and the gap between them is enormous: measured on a synthetic
#: frame, crisp reads 4231, heavily blurred reads 1.9, and genuinely featureless
#: reads 0.0. Set anywhere in the middle this calls blurred footage featureless
#: and tells the user to point the camera somewhere else, which is no help at all.
FLAT_SHARPNESS = 1.0


@dataclass(frozen=True)
class FrameQuality:
    """One frame, measured."""

    sharpness: float
    #: Sharpness as a fraction of the capture's own running median.
    relative: float
    luminance: float

    @property
    def blurred(self) -> bool:
        return self.relative < BLUR_RATIO and self.sharpness > FLAT_SHARPNESS

    @property
    def dark(self) -> bool:
        return self.luminance < DARK_LEVEL

    @property
    def flat(self) -> bool:
        return self.sharpness <= FLAT_SHARPNESS


@dataclass(frozen=True)
class Advice:
    """What to tell the user, and how loudly."""

    text: str
    #: "ok", "warn" or "stop" — the interface picks a colour from it.
    level: str

    @property
    def urgent(self) -> bool:
        return self.level == "stop"


def measure(image: np.ndarray, *, mask: np.ndarray | None = None) -> tuple[float, float]:
    """
    Sharpness and mean luminance of one frame.

    Variance of the Laplacian for sharpness: it is the standard measure, it is
    one pass, and it responds to exactly what motion blur destroys — the
    high-frequency edges a feature detector needs.

    The mask is honoured because a cockpit is always perfectly sharp. Judging a
    frame that is three-quarters dashboard on the whole picture reports the
    dashboard, and would call a hopelessly blurred road crisp.
    """
    import cv2

    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    if mask is not None:
        if mask.shape[:2] != grey.shape[:2]:
            mask = cv2.resize(mask, (grey.shape[1], grey.shape[0]), interpolation=cv2.INTER_NEAREST)
        keep = mask != 0
        if keep.sum() < grey.size // 20:
            keep = np.ones_like(grey, dtype=bool)
    else:
        keep = np.ones_like(grey, dtype=bool)

    laplacian = cv2.Laplacian(grey, cv2.CV_64F)
    return float(laplacian[keep].var()), float(grey[keep].mean())


class QualityWatcher:
    """
    Follows frame quality over a capture and says what to do about it.

    Holds only a bounded history, so it can run for an hour without growing, and
    every judgement is relative to that history rather than to a constant.
    """

    def __init__(self, *, history: int = HISTORY):
        self._sharpness: deque[float] = deque(maxlen=history)
        self.last: FrameQuality | None = None

    @property
    def ready(self) -> bool:
        return len(self._sharpness) >= WARMUP

    @property
    def median(self) -> float:
        return float(np.median(self._sharpness)) if self._sharpness else 0.0

    def observe(self, image: np.ndarray, *, mask: np.ndarray | None = None) -> FrameQuality:
        sharpness, luminance = measure(image, mask=mask)
        # Compared against the history *before* this frame joins it, or a long
        # blurred stretch drags the median down to meet itself and the warning
        # quietly stops.
        reference = self.median if self.ready else sharpness
        relative = sharpness / reference if reference > 1e-9 else 1.0
        self._sharpness.append(sharpness)
        self.last = FrameQuality(sharpness=sharpness, relative=relative, luminance=luminance)
        return self.last

    def advise(self, state=None, *, moving: bool = True) -> Advice:
        """
        One instruction, chosen by what is most wrong.

        Ordered by what blocks the scan hardest rather than by what is easiest
        to detect: a frame nobody can place is worse than a frame that is merely
        blurred, and both are worse than thin coverage, which is only a matter of
        driving round again.
        """
        quality = self.last
        if quality is None or not self.ready:
            return Advice("Getting a feel for the picture…", "ok")

        if quality.flat:
            return Advice(
                "Nothing to measure — point the camera at the level, not at the sky.",
                "stop",
            )
        if quality.dark:
            return Advice(
                "Too dark to reconstruct. Headlights on, or drive this stretch in daylight.",
                "stop",
            )
        if quality.blurred and moving:
            return Advice("Blurred — slow down.", "warn")
        if quality.blurred:
            return Advice("Blurred, and the camera is not moving. Check the capture.", "warn")

        if state is not None:
            if state.frames and state.tracking_health < 0.5:
                return Advice(
                    "Losing track — slow down and keep more of the level in view.", "warn"
                )
            if state.tracked and state.fusion_health < 0.4:
                return Advice(
                    "Positions are good but depth is not sticking. Drive nearer the scenery.",
                    "warn",
                )
            if state.weak_spots:
                return Advice(
                    f"{len(state.weak_spots)} thin spots — the map marks them.", "ok"
                )

        return Advice("Good — keep going.", "ok")
