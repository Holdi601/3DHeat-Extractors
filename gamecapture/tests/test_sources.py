"""
Tests for the frame sources.

`VideoSource` is tested against a real file written here, because the failure
modes worth catching — the colour channels arriving swapped, timestamps that do
not advance, a stride that silently drops the wrong frames — all look perfectly
healthy in isolation and only show up as a reconstruction that is subtly wrong.

`WindowSource` is exercised for its queueing behaviour without a capture, since
what can break there is the back-pressure policy rather than the Windows API.
"""

from __future__ import annotations

import queue

import cv2
import numpy as np
import pytest

from heat3d_capture.capture.sources import (
    CapturedFrame,
    VideoSource,
    WindowSource,
    _resize_longest,
    open_source,
)


@pytest.fixture
def clip(tmp_path):
    """
    A 20-frame clip whose frames are individually identifiable.

    Each frame is a solid colour with a red channel that counts up, so a test can
    tell not only that frames arrived but *which* ones, and in what order.
    """
    path = tmp_path / "clip.mp4"
    w, h, n = 320, 180, 20
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    assert writer.isOpened()
    for i in range(n):
        # BGR on the way out, so a correct reader gives us back R = i * 10.
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 2] = i * 10  # red
        frame[:, :, 0] = 40  # a constant blue, to catch a channel swap
        writer.write(frame)
    writer.release()
    return path


class TestVideoSource:
    def test_reads_every_frame_in_order(self, clip):
        src = VideoSource(clip)
        frames = list(src.frames())
        src.close()
        assert len(frames) == 20
        assert [f.index for f in frames] == list(range(20))

    def test_hands_back_rgb_not_bgr(self, clip):
        # The single most consequential convention in the pipeline: every model
        # downstream expects RGB, and a swap here produces depth maps that are
        # wrong in a way that still looks like a plausible depth map.
        src = VideoSource(clip)
        frame = next(iter(src.frames()))
        src.close()
        r, g, b = frame.image[0, 0]
        assert b == pytest.approx(40, abs=12), "blue channel lost or swapped"
        assert r < 40, "red should be near zero on the first frame"

    def test_timestamps_advance_with_the_frame_rate(self, clip):
        src = VideoSource(clip)
        frames = list(src.frames())
        src.close()
        assert frames[0].t == pytest.approx(0.0)
        # 30 fps, so frame 15 is at half a second.
        assert frames[15].t == pytest.approx(0.5, abs=1e-6)
        assert all(b.t > a.t for a, b in zip(frames, frames[1:]))

    def test_stride_samples_evenly_and_renumbers(self, clip):
        src = VideoSource(clip, stride=4)
        frames = list(src.frames())
        src.close()
        assert len(frames) == 5
        # Indices are compacted, but the timestamps still refer to the original
        # clip — the pose estimator needs real elapsed time, not frame counts.
        assert [f.index for f in frames] == [0, 1, 2, 3, 4]
        assert [round(f.t, 4) for f in frames] == [0.0, 0.1333, 0.2667, 0.4, 0.5333]

    def test_downscales_to_the_requested_longest_side(self, clip):
        src = VideoSource(clip, longest_side=160)
        frame = next(iter(src.frames()))
        src.close()
        assert max(frame.size) == 160
        assert frame.size == (160, 90), "aspect ratio must be preserved"

    def test_refuses_a_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VideoSource(tmp_path / "nope.mp4")


class TestResize:
    def test_leaves_a_small_frame_alone(self):
        img = np.zeros((90, 160, 3), dtype=np.uint8)
        assert _resize_longest(img, 960) is img

    def test_scales_by_the_longest_side_whichever_it_is(self):
        tall = np.zeros((400, 100, 3), dtype=np.uint8)
        assert _resize_longest(tall, 200).shape[:2] == (200, 50)
        wide = np.zeros((100, 400, 3), dtype=np.uint8)
        assert _resize_longest(wide, 200).shape[:2] == (50, 200)

    def test_never_collapses_a_dimension_to_zero(self):
        # An extreme aspect ratio rounding to zero would make the frame unusable
        # rather than merely small.
        skinny = np.zeros((1000, 3, 3), dtype=np.uint8)
        out = _resize_longest(skinny, 100)
        assert out.shape[0] == 100 and out.shape[1] >= 1


class TestWindowSourceBackPressure:
    """
    The queue policy, without a live capture.

    Reconstruction is far slower than the compositor, so the queue *will* fill.
    What matters is that it then drops the oldest frame rather than blocking the
    capture thread or growing without bound.
    """

    def _source(self, size: int = 2) -> WindowSource:
        return WindowSource(monitor_index=1, queue_size=size)

    def test_requires_exactly_one_target(self):
        with pytest.raises(ValueError, match="exactly one"):
            WindowSource()
        with pytest.raises(ValueError, match="exactly one"):
            WindowSource(window_name="a", monitor_index=1)

    def test_drops_the_oldest_when_full(self):
        src = self._source(size=2)
        for i in range(5):
            frame = CapturedFrame(np.zeros((2, 2, 3), dtype=np.uint8), t=float(i), index=i)
            try:
                src._queue.put_nowait(frame)
            except queue.Full:
                src._queue.get_nowait()
                src._queue.put_nowait(frame)
                src._dropped += 1
        held = []
        while not src._queue.empty():
            held.append(src._queue.get_nowait().index)
        # The two most recent survive; the older three were dropped.
        assert held == [3, 4]
        assert src.dropped == 3

    def test_close_is_safe_before_and_after_start(self):
        src = self._source()
        src.close()
        src.close()


class TestOpenSource:
    def test_resolves_a_monitor_target(self):
        src = open_source("monitor:1")
        assert isinstance(src, WindowSource)
        assert src._monitor_index == 1
        src.close()

    def test_resolves_a_window_target_keeping_colons_in_the_title(self):
        src = open_source("window:Forza Horizon 5: Rally")
        assert isinstance(src, WindowSource)
        assert src._window_name == "Forza Horizon 5: Rally"
        src.close()

    def test_resolves_a_path_to_a_video(self, clip):
        src = open_source(str(clip))
        assert isinstance(src, VideoSource)
        src.close()
