"""
Where frames come from.

Two sources, one shape. Live capture is the one that matters — the point of the
tool is to walk through a level and watch it build — but a video file is what
makes any of it testable, reproducible, and re-runnable with different settings
without going back into the game. A reconstruction bug found live is nearly
impossible to fix, because the input is gone; the same bug on a recorded file can
be worked until it is understood.

Capture is external: Windows Graphics Capture, the same public API screen
recorders use. Nothing is injected into the game and nothing is hooked, so this
does not put an account at risk with any anti-cheat. That constraint is also why
the pipeline downstream has to estimate geometry rather than read it — the depth
buffer is right there in the game's memory and entirely off limits.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np


@dataclass
class CapturedFrame:
    """One frame, in the terms the rest of the pipeline works in."""

    #: (H, W, 3) uint8, **RGB**. Not BGR: every model downstream expects RGB and
    #: converting once here beats converting in four places.
    image: np.ndarray
    #: Seconds from the start of the capture, monotonic.
    t: float
    #: Position in the source, counting every frame that arrived.
    index: int
    #: Raw `time.monotonic()` at capture, or None for a file.
    #:
    #: Kept separate from `t` because it is the only clock a telemetry feed and a
    #: capture have in common: the game's own timestamp counts from something
    #: unrelated to either, and `t` counts from the start of this scan. Lining
    #: poses up to frames needs a shared origin, and this is it.
    monotonic: float | None = None

    @property
    def size(self) -> tuple[int, int]:
        return (self.image.shape[1], self.image.shape[0])


class FrameSource(Protocol):
    def frames(self) -> Iterator[CapturedFrame]: ...
    def close(self) -> None: ...


def _resize_longest(image: np.ndarray, longest: int) -> np.ndarray:
    """
    Scale so the longer side is `longest`, preserving aspect.

    Reconstruction runs at a few hundred pixels, not at 4K. A 3840x2160 frame is
    thirty times the pixels the depth model will use and costs that much again in
    the copy out of the capture buffer; resizing at the source keeps everything
    after it cheap. Area interpolation on the way down because it averages rather
    than samples, which matters when the next stage is looking for gradients.
    """
    import cv2

    h, w = image.shape[:2]
    longest_side = max(h, w)
    if longest_side <= longest:
        return image
    scale = longest / longest_side
    return cv2.resize(
        image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA
    )


class VideoSource:
    """Frames from a file, as fast as they can be decoded."""

    def __init__(self, path: str | Path, *, longest_side: int = 960, stride: int = 1):
        import cv2

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open {self.path}")
        self.longest_side = longest_side
        self.stride = max(1, stride)
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        # A container with no frame rate is common for screen recordings; the
        # timestamps only have to be monotonic and evenly spaced for the pose
        # estimator, so a sane default beats refusing the file.
        self.fps = fps if fps and fps > 1e-3 else 30.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def frames(self) -> Iterator[CapturedFrame]:
        import cv2

        index = 0
        emitted = 0
        while True:
            ok, bgr = self._cap.read()
            if not ok:
                return
            if index % self.stride == 0:
                rgb = cv2.cvtColor(_resize_longest(bgr, self.longest_side), cv2.COLOR_BGR2RGB)
                # A file has no wall-clock capture time; its own timeline is the
                # only one it has, and telemetry cannot be aligned to it anyway.
                yield CapturedFrame(
                    image=rgb, t=index / self.fps, index=emitted, monotonic=index / self.fps
                )
                emitted += 1
            index += 1

    def close(self) -> None:
        self._cap.release()


class WindowSource:
    """
    Live frames from a window or a monitor.

    The capture library hands frames to a callback on its own thread and expects
    that callback to return promptly — it is the thread the compositor is
    delivering on. Reconstruction takes far longer than a frame interval, so the
    callback does nothing but drop the frame into a queue.

    The queue is deliberately shallow and **drops the oldest** when it fills. A
    growing backlog is the wrong behaviour for this: it would mean the overlay
    showing progress through a level the user walked past a minute ago, and
    memory climbing until the capture dies. Falling behind should cost frames,
    not correctness — the frames are highly redundant anyway, which is the whole
    premise of keyframe selection downstream.
    """

    def __init__(
        self,
        *,
        window_name: str | None = None,
        monitor_index: int | None = None,
        longest_side: int = 960,
        queue_size: int = 4,
    ):
        if (window_name is None) == (monitor_index is None):
            raise ValueError("give exactly one of window_name or monitor_index")
        self.longest_side = longest_side
        self._queue: queue.Queue[CapturedFrame | None] = queue.Queue(maxsize=queue_size)
        self._dropped = 0
        self._index = 0
        self._t0: float | None = None
        self._control = None
        self._stop = threading.Event()
        self._window_name = window_name
        self._monitor_index = monitor_index

    @property
    def dropped(self) -> int:
        """Frames thrown away because reconstruction could not keep up."""
        return self._dropped

    def _build(self):
        import cv2
        from windows_capture import Frame, InternalCaptureControl, WindowsCapture

        capture = WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            window_name=self._window_name,
            monitor_index=self._monitor_index,
        )

        @capture.event
        def on_frame_arrived(frame: "Frame", control: "InternalCaptureControl") -> None:
            if self._stop.is_set():
                control.stop()
                return
            now = time.monotonic()
            if self._t0 is None:
                self._t0 = now
            # The buffer is BGRA and is reused by the capture library between
            # callbacks, so this must copy before the callback returns.
            bgra = np.asarray(frame.frame_buffer)
            rgb = cv2.cvtColor(_resize_longest(bgra, self.longest_side), cv2.COLOR_BGRA2RGB)
            captured = CapturedFrame(
                image=rgb, t=now - self._t0, index=self._index, monotonic=now
            )
            self._index += 1
            try:
                self._queue.put_nowait(captured)
            except queue.Full:
                # Drop the oldest, keep the newest: the live view should show
                # where the user is now, not where they were.
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(captured)
                except (queue.Empty, queue.Full):
                    pass
                self._dropped += 1

        @capture.event
        def on_closed() -> None:
            self._queue.put(None)

        return capture

    def frames(self) -> Iterator[CapturedFrame]:
        capture = self._build()
        self._control = capture.start_free_threaded()
        try:
            while not self._stop.is_set():
                try:
                    item = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if item is None:
                    return
                yield item
        finally:
            self.close()

    def close(self) -> None:
        self._stop.set()
        control = self._control
        if control is not None:
            self._control = None
            try:
                control.stop()
            except Exception:
                # Already closed, or the window went away. Nothing to recover.
                pass


def open_source(target: str, *, longest_side: int = 960, stride: int = 1) -> FrameSource:
    """
    Resolve a command-line target to a source.

    `monitor:1`, `window:Forza Horizon 5`, or a path to a video file.
    """
    if target.startswith("monitor:"):
        return WindowSource(monitor_index=int(target.split(":", 1)[1]), longest_side=longest_side)
    if target.startswith("window:"):
        return WindowSource(window_name=target.split(":", 1)[1], longest_side=longest_side)
    return VideoSource(target, longest_side=longest_side, stride=stride)
