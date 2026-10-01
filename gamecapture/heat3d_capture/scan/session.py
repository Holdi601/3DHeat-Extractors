"""
A scan, as a thing with settings and progress, independent of any interface.

Kept apart from the window deliberately. Everything here is testable without a
display, and the same session drives a command line, so the interface stays a
view of a scan rather than the place a scan lives.

Keyframes, and why most frames are thrown away
----------------------------------------------
A capture arrives at sixty frames a second and almost all of it is redundant:
standing still produces hundreds of views of one thing, and fusing them adds
nothing but a hundredfold cost and a bias toward whatever the user happened to
stare at. What reconstruction wants is *parallax* — views from positions far
enough apart to triangulate.

So frames are kept on how much the picture changed since the last kept one, not
on a timer. That also makes the tool behave the way someone expects when they
slow down at something detailed: turning slowly through a doorway keeps frames
steadily, and standing still keeps almost none.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np


class Quality(str, Enum):
    """
    Presets, because the individual knobs only make sense together.

    Naming them by the *walk* rather than by resolution: what the user is really
    choosing is how carefully they are willing to move.
    """

    DRAFT = "draft"
    BALANCED = "balanced"
    DETAILED = "detailed"

    @property
    def capture_side(self) -> int:
        return {"draft": 640, "balanced": 960, "detailed": 1280}[self.value]

    @property
    def keep_quantile(self) -> float:
        """
        How selective to be, as a quantile of recent observed change.

        Relative, not absolute, and that is the whole point. The first version
        used fixed fractions — 0.12, 0.07, 0.035 — chosen against a brightly
        textured test scene. Measured on a real recording, a Forza lap at night,
        the change between consecutive frames had a median of **0.0025**: every
        threshold was between fourteen and eighty times too high, and a
        three-minute clip yielded twenty-two keyframes out of eleven thousand.

        A quantile cannot be wrong that way. "Keep the busiest third of frames"
        means the same thing in a bright room and on a dark rally stage, and the
        keep *rate* is what the user is really choosing.
        """
        return {"draft": 0.85, "balanced": 0.65, "detailed": 0.40}[self.value]

    @property
    def keyframe_change(self) -> float:
        """A starting threshold, before enough frames have been seen to adapt."""
        return {"draft": 0.020, "balanced": 0.010, "detailed": 0.004}[self.value]

    @property
    def summary(self) -> str:
        return {
            "draft": "Fast and rough. Good for checking a route before a real scan.",
            "balanced": "The usual choice. Walk at a normal pace.",
            "detailed": "Slow and thorough. Walk deliberately and turn gently.",
        }[self.value]


@dataclass
class ScanSettings:
    """Everything a scan can be told to do."""

    source: str = "monitor:1"
    quality: Quality = Quality.BALANCED
    output: Path = Path("scans/untitled.glb")
    #: Stop after this many seconds. 0 means keep going until stopped.
    max_seconds: float = 0.0
    #: Stop after this many keyframes, as a guard on memory. 0 means no limit.
    max_keyframes: int = 3000
    #: Listen for Forza telemetry, which makes the camera path exact.
    use_telemetry: bool = False
    telemetry_port: int = 5300
    #: Override backend selection; None picks the best available.
    prefer_backend: str | None = None
    #: The game's horizontal field of view, degrees. The one genuinely assumed
    #: number in the whole pipeline: too narrow makes the level reconstruct
    #: deeper than it is, too wide flattens it. 90 suits most first-person
    #: games; racing games sit nearer 65.
    field_of_view: float = 90.0
    #: Surface voxel size in metres. Smaller is finer and quadratically heavier.
    voxel: float = 0.25
    #: Build geometry while scanning. Off makes this a pure recorder.
    reconstruct: bool = True
    #: Detect and ignore HUD and camera-mounted geometry.
    mask_hud: bool = True
    #: Metres the camera sits above the ground, used once to fix the global
    #: scale. Roughly 1.15 in a car, 1.7 on foot. None leaves the result
    #: self-consistent but unit-less.
    camera_height: float | None = 1.15

    def __post_init__(self) -> None:
        # Coerced rather than trusted. `Quality` is a str enum, and anything that
        # round-trips a value through a system that special-cases strings hands
        # back a plain one: Qt stores `addItem(..., Quality.DRAFT)` as a QVariant,
        # sees a str subclass, and `currentData()` returns "draft" with the enum
        # gone. Every attribute access downstream then dies on a bare string,
        # inside a worker thread, mid-scan.
        #
        # The window is fixed to pass the value explicitly, but the guard stays:
        # this is the boundary where an outside value arrives, and it is cheaper
        # to be right here than to trust every caller.
        if not isinstance(self.quality, Quality):
            self.quality = Quality(str(self.quality).lower())

    def capture_side(self) -> int:
        return self.quality.capture_side


@dataclass
class ScanProgress:
    """A snapshot of a running scan, for whatever is displaying it."""

    running: bool = False
    frames_seen: int = 0
    keyframes: int = 0
    frames_dropped: int = 0
    elapsed: float = 0.0
    fps: float = 0.0
    #: Fraction of the last frame that changed; how close to keeping the next one.
    change: float = 0.0
    telemetry_frames: int = 0
    message: str = ""
    #: Live reconstruction figures, when reconstruction is on.
    reconstruction: object | None = None

    @property
    def keyframe_rate(self) -> float:
        return self.keyframes / self.elapsed if self.elapsed > 0 else 0.0


def picture_change(a: np.ndarray, b: np.ndarray, *, grid: int = 16, mask=None) -> float:
    """
    How much two frames differ, as a fraction from 0 to 1.

    Compared on a coarse grid of block means rather than per pixel. Per-pixel
    difference is dominated by things that are not camera motion — film grain,
    foliage, a flickering light, the HUD — and would keep frames while the player
    stands still watching a fire. Block means respond to the picture *moving*,
    which is the thing that produces parallax.

    `mask` excludes what is not the world, and it is not optional in practice.
    Measured on a real recording — Forza, cockpit view, at night — the dashboard,
    A-pillars and windscreen frame hold about 40% of the picture perfectly still,
    and the rest is dark. Averaged over the whole frame the change never reached
    the threshold: **22 keyframes out of 11,147**. The scan was not slow, it was
    barely happening, and the reason was that most of what it was measuring was
    bolted to the camera.
    """
    import cv2

    def signature(image: np.ndarray) -> np.ndarray:
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        return cv2.resize(grey, (grid, grid), interpolation=cv2.INTER_AREA).astype(np.float32)

    difference = np.abs(signature(a) - signature(b))
    if mask is not None:
        # Weighted by how much of each block is world rather than cockpit, so a
        # block that is half dashboard counts half.
        import cv2 as _cv2

        weight = _cv2.resize(
            (mask > 0).astype(np.float32), (grid, grid), interpolation=_cv2.INTER_AREA
        )
        total = float(weight.sum())
        if total > grid * grid * 0.05:
            return float((difference * weight).sum() / total / 255.0)
    return float(difference.mean() / 255.0)


class ScanSession:
    """
    Runs a scan on a worker thread and reports progress.

    Stoppable at any point, because a scan is something a person is physically
    doing and they have to be able to change their mind. Stopping keeps whatever
    was gathered rather than discarding it — a partial level is still a level.
    """

    def __init__(
        self,
        settings: ScanSettings,
        *,
        estimator=None,
        on_progress: Callable[[ScanProgress], None] | None = None,
        on_preview: Callable[[np.ndarray, np.ndarray], None] | None = None,
        keep_frames: bool = False,
    ):
        self.settings = settings
        self.estimator = estimator
        self.on_progress = on_progress
        self.on_preview = on_preview
        # Frames are *not* kept by default. A half-hour scan at three keyframes a
        # second is thousands of 960x540 images plus their depth maps, which is
        # tens of gigabytes — and once a frame has been fused, nothing needs it
        # again. Kept only when something means to re-run reconstruction on them.
        self.keep_frames = keep_frames
        self.reconstructor = None
        #: Started with the scan when telemetry is asked for. Owned here rather
        #: than by the reconstructor, because it is a socket with a lifetime and
        #: the reconstructor is pure geometry.
        self.telemetry = None
        self.progress = ScanProgress()
        self.keyframes: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    # -- control ---------------------------------------------------------

    def start(self, source=None) -> None:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("this scan is already running")
        self._stop.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, args=(source,), daemon=True)
        self._thread.start()

    def wait(self, *, timeout: float | None = None) -> bool:
        """
        Block until the scan ends on its own. Returns True if it did.

        Distinct from `stop()`, which *ends* it. A scan finishes by itself when a
        video file runs out or a limit is reached, and a caller that wants the
        result — a test, or a batch run over a recording — has to be able to wait
        for that without cutting it short.
        """
        if not self._thread:
            return True
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def stop(self, *, timeout: float = 5.0) -> None:
        """Ask the scan to end now, keeping whatever it has gathered."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    @property
    def error(self) -> BaseException | None:
        """Whatever ended the scan, if it was not asked to stop."""
        return self._error

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # -- the scan --------------------------------------------------------

    def _emit(self) -> None:
        if self.on_progress:
            self.on_progress(self.progress)

    def _run(self, source=None) -> None:
        from ..capture.sources import open_source

        self.progress = ScanProgress(running=True, message="starting")
        self._emit()
        started = time.perf_counter()
        owned = source is None
        last_kept: np.ndarray | None = None
        recent: list[float] = []
        #: Recent change measurements, for the adaptive threshold below.
        changes: list[float] = []

        try:
            # Opening the source sits *inside* the guard. It used to be above it,
            # which meant a bad monitor number, a missing file or a mistyped
            # window title escaped as an unhandled exception on a worker thread —
            # printed to a console nobody reads, while the window sat there
            # saying "scanning" forever. The one place most likely to fail was
            # the one place not covered.
            if source is None:
                source = open_source(
                    self.settings.source, longest_side=self.settings.capture_side()
                )
            threshold = self.settings.quality.keyframe_change

            for frame in source.frames():
                if self._stop.is_set():
                    self.progress.message = "stopped"
                    break

                self.progress.frames_seen += 1
                self.progress.elapsed = time.perf_counter() - started
                # Surfaced, not just counted. Dropped frames are the single most
                # diagnostic number in a live scan: they mean reconstruction is
                # slower than capture, which widens the gap between keyframes
                # until feature matching stops working. Tracking then fails for
                # a reason that has nothing to do with tracking.
                dropped = getattr(source, "dropped", 0)
                if dropped:
                    self.progress.frames_dropped = dropped
                if self.progress.message == "starting":
                    # Left at "starting" for the whole scan in the first
                    # version, which is a small thing that reads as a large
                    # one: it is the only word on screen saying whether
                    # anything is happening at all.
                    self.progress.message = "scanning"

                # The mask the reconstructor has learned, so the change
                # measure ignores the cockpit rather than being flattened by it.
                screen = getattr(self.reconstructor, "screen", None)
                world_mask = screen.mask(frame.image.shape[:2]) if screen is not None else None
                change = (
                    1.0
                    if last_kept is None
                    else picture_change(last_kept, frame.image, mask=world_mask)
                )
                self.progress.change = change

                changes.append(change)
                # A rolling window, so the threshold follows the scene: standing
                # in a dark corridor and sprinting across a bright field want
                # very different absolute numbers and the same keep rate.
                del changes[:-240]
                if len(changes) >= 30:
                    adapted = float(
                        np.quantile(changes, self.settings.quality.keep_quantile)
                    )
                    # Never zero: a perfectly still scene would otherwise keep
                    # every frame of nothing.
                    threshold = max(adapted, 1e-4)

                if change >= threshold or last_kept is None:
                    self._keep(frame, recent)
                    last_kept = frame.image

                if self.progress.frames_seen % 5 == 0:
                    self._emit()

                if self._finished():
                    break
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            self._error = exc
            self.progress.message = f"failed: {exc}"
        finally:
            if owned and source is not None:
                source.close()
            if self.telemetry is not None:
                self.telemetry.stop()
            self.progress.running = False
            if not self.progress.message or self.progress.message in (
                "starting",
                "scanning",
            ):
                self.progress.message = "finished"
            self.progress.elapsed = time.perf_counter() - started
            self._emit()

    def _ensure_reconstructor(self, image: np.ndarray):
        if self.reconstructor is not None or not self.settings.reconstruct:
            return self.reconstructor
        from ..pose.odometry import Intrinsics
        from .reconstruct import Reconstructor

        height, width = image.shape[:2]
        source = None
        if self.settings.use_telemetry:
            from ..pose.telemetry_pose import ForzaPoseSource

            self.telemetry = ForzaPoseSource(port=self.settings.telemetry_port)
            # Failure to bind is reported, not raised: telemetry is an upgrade,
            # and a scan that would have worked on vision alone must not be
            # stopped because a port was busy.
            if self.telemetry.start():
                source = self.telemetry.track

        self.reconstructor = Reconstructor(
            Intrinsics.from_fov(width, height, self.settings.field_of_view),
            voxel=self.settings.voxel,
            mask_hud=self.settings.mask_hud,
            pose_source=source,
            camera_height=self.settings.camera_height,
        )
        return self.reconstructor

    def _keep(self, frame, recent: list[float]) -> None:
        depth = None
        if self.estimator is not None:
            began = time.perf_counter()
            (depth,) = self.estimator.predict(frame.image)
            recent.append(time.perf_counter() - began)
            # A short window, so the figure tracks what is happening now rather
            # than averaging away a machine that has started thermal throttling.
            del recent[:-20]
            mean = sum(recent) / len(recent)
            self.progress.fps = 1.0 / mean if mean > 0 else 0.0

        self.progress.keyframes += 1
        if self.keep_frames:
            self.keyframes.append(
                {"t": frame.t, "index": frame.index, "image": frame.image, "depth": depth}
            )

        if depth is not None:
            reconstructor = self._ensure_reconstructor(frame.image)
            if reconstructor is not None:
                # The frame's own capture time, on the same monotonic clock the
                # telemetry track stamps packets with — which is the only thing
                # that lets the two be lined up at all.
                self.progress.reconstruction = reconstructor.add(
                    frame.image, depth.values, timestamp=frame.monotonic
                )
                if self.telemetry is not None:
                    self.progress.telemetry_frames = self.telemetry.track.received

        if self.on_preview is not None and depth is not None:
            self.on_preview(frame.image, depth.values)

    def _finished(self) -> bool:
        limit = self.settings.max_seconds
        if limit and self.progress.elapsed >= limit:
            self.progress.message = f"reached the {limit:.0f}s limit"
            return True
        cap = self.settings.max_keyframes
        if cap and self.progress.keyframes >= cap:
            self.progress.message = f"reached {cap} keyframes"
            return True
        return False

    # -- results ---------------------------------------------------------

    def export(self, path=None, *, name: str = "scan"):
        """Write the reconstructed level. None when there is nothing to write."""
        if self.reconstructor is None:
            return None
        return self.reconstructor.export(path or self.settings.output, name=name)

    def weak_spots(self):
        """Places worth revisiting. Empty when not reconstructing."""
        if self.reconstructor is None:
            return []
        return self.reconstructor.refresh_weak_spots()
