"""
Finding the parts of the frame that are not the world.

Two things in a game frame are not level geometry and both wreck a
reconstruction if they are treated as though they were:

- **The HUD.** A health bar, a minimap, a crosshair, a kill feed. Sharp, high
  contrast, full of corners — which is precisely what makes it poison. ORB loves
  a crosshair: it is the most repeatable feature in the frame, it matches
  perfectly between every pair of frames, and it says the camera never moved.
  A handful of HUD features among the real ones is enough to drag a pose solution
  toward "stationary" and quietly compress the whole reconstruction.
- **Things bolted to the camera.** The weapon in a shooter, the bonnet in a
  racing game, a cockpit frame. These are real geometry, and they are in the
  wrong coordinate system: they travel with the camera, so fusing them smears a
  gun barrel along the entire path walked.

One signal covers both
----------------------
Both are *static in screen space while the world moves*. That is the whole
detector. Accumulate how much each pixel changes from frame to frame, and
compare each pixel against the frame as a whole: where the world is sliding past
and one region is not, that region is attached to the camera rather than standing
in the level — whether it is drawn by the HUD or modelled in the scene.

The comparison has to be relative, not an absolute threshold, because "how much
does the picture change" depends entirely on how fast someone is walking. A
fixed threshold masks half the screen when they stand still and nothing at all
when they sprint.

Accumulated only while the camera is known to be moving. A stationary camera
makes *everything* static, and a detector that learned from those frames would
decide the whole level was a HUD.
"""

from __future__ import annotations

import numpy as np

#: Frames of confirmed motion before the mask is believed at all. Below this the
#: statistics are dominated by whatever happened to be on screen, and masking on
#: that basis throws away real geometry.
MIN_FRAMES = 12

#: A pixel changing less than this fraction of the frame's median change is
#: considered static. Generous, because the cost of masking a little real
#: geometry is a small hole, and the cost of *not* masking a crosshair is a
#: compressed reconstruction.
STATIC_RATIO = 0.22

#: Refuse to mask more than this much of the frame. A mask covering most of the
#: screen means the premise failed — a menu, a cutscene, a loading screen — and
#: masking almost everything would silently stop the scan rather than degrade it.
MAX_COVERAGE = 0.45


class ScreenMask:
    """
    Learns which pixels belong to the camera rather than to the level.

    Cheap enough to update on every keyframe: one blur, one absolute difference
    and one running average over a half-resolution frame.
    """

    def __init__(self, *, downscale: int = 2, decay: float = 0.9):
        self.downscale = max(1, downscale)
        self.decay = decay
        self._change: np.ndarray | None = None
        self._previous: np.ndarray | None = None
        self._shape: tuple[int, int] | None = None
        self.frames = 0

    @property
    def ready(self) -> bool:
        return self.frames >= MIN_FRAMES and self._change is not None

    def _small(self, image: np.ndarray) -> np.ndarray:
        import cv2

        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        h, w = grey.shape[:2]
        small = cv2.resize(
            grey, (max(1, w // self.downscale), max(1, h // self.downscale)),
            interpolation=cv2.INTER_AREA,
        )
        # Blurred so that a one-pixel jitter in an otherwise static HUD element
        # does not read as motion.
        #
        # Halved rather than quartered, which costs four times the arithmetic on
        # an operation that was already negligible and is the difference between
        # catching a crosshair and missing it: two pixels of line at quarter
        # resolution is half a pixel, and it averages into the moving world
        # behind it before the statistics ever see it.
        return cv2.GaussianBlur(small, (3, 3), 0).astype(np.float32)

    def update(self, image: np.ndarray, *, moving: bool = True) -> None:
        """
        Fold one frame in. `moving` says whether the camera actually moved.

        Frames where it did not are used to refresh the reference picture but not
        to learn from — otherwise standing still would teach the detector that
        the whole level is attached to the camera.
        """
        small = self._small(image)
        if self._previous is None or self._previous.shape != small.shape:
            self._previous = small
            self._change = np.zeros_like(small)
            self._shape = image.shape[:2]
            self.frames = 0
            return

        if moving:
            difference = np.abs(small - self._previous)
            # Exponential average, so the mask follows a HUD that appears or
            # disappears part-way through a scan instead of being fixed by the
            # first dozen frames.
            self._change = self.decay * self._change + (1.0 - self.decay) * difference
            self.frames += 1
        self._previous = small
        self._shape = image.shape[:2]

    def static_fraction(self) -> float:
        """How much of the frame currently reads as attached to the camera."""
        small = self._small_mask()
        return 0.0 if small is None else float((small == 0).mean())

    def _small_mask(self) -> np.ndarray | None:
        if not self.ready or self._change is None:
            return None
        # Compared against the frame's own median, so the threshold follows how
        # fast the camera is being moved instead of assuming a speed.
        typical = float(np.median(self._change))
        if typical < 1e-4:
            # Nothing moved anywhere, in frames we believed were moving. No
            # conclusion to draw.
            return None
        static = self._change < (typical * STATIC_RATIO)
        if static.mean() > MAX_COVERAGE:
            # The premise has failed — a menu, a cutscene, a loading screen.
            # Masking most of the screen would stop the scan rather than help it.
            return None
        return np.where(static, 0, 255).astype(np.uint8)

    def mask(self, shape: tuple[int, int] | None = None) -> np.ndarray | None:
        """
        A full-resolution mask: 255 where the pixel is world, 0 where it is not.

        None while there is not yet enough evidence, which callers must treat as
        "use everything" rather than "use nothing".
        """
        import cv2

        small = self._small_mask()
        if small is None:
            return None
        target = shape or self._shape
        if target is None:
            return None
        # Grown by a little before upscaling: the boundary of a HUD element is
        # antialiased against the world behind it, and those blended pixels
        # belong to neither.
        grown = cv2.erode(small, np.ones((3, 3), np.uint8), iterations=1)
        full = cv2.resize(grown, (target[1], target[0]), interpolation=cv2.INTER_NEAREST)
        return full

    def describe(self) -> str:
        """One line for the interface."""
        if not self.ready:
            return f"learning what is HUD ({self.frames}/{MIN_FRAMES} frames)"
        fraction = self.static_fraction()
        if fraction <= 0.001:
            return "no HUD or attached geometry detected"
        return f"ignoring {fraction * 100:.0f}% of the frame — HUD or camera-mounted"
