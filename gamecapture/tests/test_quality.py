"""
Tests for the live judgement of frame quality, and the advice that follows.

Two things have to hold for this to be worth putting on screen.

It has to tell the *causes* apart, because their remedies are opposite: "slow
down" is wrong for a dark tunnel and useless pointed at a wall. And it has to
judge against the capture rather than against a constant, because a night rally
stage and a bright city street differ by an order of magnitude with both in
focus — a fixed threshold would shout continuously at exactly the footage where
the advice would be least welcome and least true.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_capture.capture.quality import (
    BLUR_RATIO,
    DARK_LEVEL,
    FLAT_SHARPNESS,
    QualityWatcher,
    measure,
)


def textured(height=120, width=200, seed=0, level=160) -> np.ndarray:
    """A frame with plenty of edges in it, at a chosen brightness."""
    import cv2

    rng = np.random.default_rng(seed)
    frame = np.full((height, width, 3), level // 3, np.uint8)
    for _ in range(220):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        shade = int(rng.integers(level // 2, min(255, level + 60)))
        cv2.rectangle(frame, (x, y), (x + 9, y + 9), (shade, shade, shade), -1)
    return frame


def blur(frame, amount=2):
    import cv2

    return cv2.GaussianBlur(frame, (0, 0), amount)


def flat(height=120, width=200, level=140) -> np.ndarray:
    return np.full((height, width, 3), level, np.uint8)


class TestMeasuring:
    def test_a_blurred_frame_reads_as_less_sharp(self):
        crisp, _ = measure(textured())
        soft, _ = measure(blur(textured()))

        assert soft < crisp * BLUR_RATIO

    def test_luminance_follows_brightness(self):
        _, dark = measure(textured(level=30))
        _, bright = measure(textured(level=220))

        assert dark < bright
        assert bright > DARK_LEVEL

    def test_a_flat_frame_has_almost_no_sharpness(self):
        sharpness, _ = measure(flat())

        assert sharpness <= FLAT_SHARPNESS

    def test_the_mask_keeps_the_cockpit_out_of_the_verdict(self):
        """
        A dashboard is always perfectly sharp, and it fills a lot of the frame.

        Judged on the whole picture, a hopelessly blurred road behind a crisp
        instrument panel reads as crisp — which is the one case where the advice
        matters most and would be exactly wrong.
        """
        frame = blur(textured(seed=1))
        # A crisp panel across the bottom third.
        frame[80:] = textured(height=40, width=200, seed=2)[:]

        whole, _ = measure(frame)
        mask = np.full(frame.shape[:2], 255, np.uint8)
        mask[80:] = 0
        masked, _ = measure(frame, mask=mask)

        assert masked < whole


class TestJudgingAgainstTheCapture:
    def test_it_says_nothing_until_it_has_something_to_compare_with(self):
        watcher = QualityWatcher()
        watcher.observe(textured())

        assert not watcher.ready
        assert watcher.advise().level == "ok"
        assert "feel" in watcher.advise().text.lower()

    def test_a_consistently_soft_capture_is_not_nagged(self):
        """
        Night footage is soft throughout and there is nothing to be done about
        it. A fixed threshold would warn on every frame of it.
        """
        watcher = QualityWatcher()
        for i in range(40):
            watcher.observe(blur(textured(seed=i), amount=2))

        assert watcher.ready
        assert watcher.advise().level == "ok"

    def test_a_frame_blurrier_than_the_capture_is_flagged(self):
        watcher = QualityWatcher()
        for i in range(40):
            watcher.observe(textured(seed=i))

        watcher.observe(blur(textured(seed=99), amount=2))

        advice = watcher.advise(moving=True)
        assert advice.level == "warn"
        assert "slow" in advice.text.lower()

    def test_a_long_blurred_stretch_keeps_being_flagged(self):
        """
        The comparison is against the history *before* each frame joins it.

        Otherwise a sustained blurred stretch drags the median down to meet
        itself, the ratio returns to one, and the warning quietly stops in the
        middle of the footage it is most needed for.
        """
        watcher = QualityWatcher()
        for i in range(40):
            watcher.observe(textured(seed=i))
        for i in range(25):
            watcher.observe(blur(textured(seed=100 + i), amount=2))

        assert watcher.advise(moving=True).level == "warn"


class TestTellingTheCausesApart:
    def settled(self, seed_shift=0):
        watcher = QualityWatcher()
        for i in range(40):
            watcher.observe(textured(seed=i + seed_shift))
        return watcher

    def test_darkness_is_not_reported_as_speed(self):
        watcher = self.settled()
        watcher.observe(textured(seed=7, level=12))

        advice = watcher.advise(moving=True)
        assert advice.level == "stop"
        assert "dark" in advice.text.lower()
        assert "slow" not in advice.text.lower()

    def test_nothing_to_look_at_is_not_reported_as_speed(self):
        watcher = self.settled()
        watcher.observe(flat())

        advice = watcher.advise(moving=True)
        assert advice.level == "stop"
        assert "slow" not in advice.text.lower()

    def test_a_blurred_still_camera_is_not_told_to_slow_down(self):
        watcher = self.settled()
        watcher.observe(blur(textured(seed=99), amount=2))

        advice = watcher.advise(moving=False)
        assert advice.level == "warn"
        assert "slow down" not in advice.text.lower()


class TestAdviceFromTheScanState:
    class State:
        def __init__(self, **kw):
            self.frames = kw.get("frames", 100)
            self.tracked = kw.get("tracked", 100)
            self.fused = kw.get("fused", 100)
            self.weak_spots = kw.get("weak_spots", [])

        @property
        def tracking_health(self):
            return self.tracked / self.frames if self.frames else 0.0

        @property
        def fusion_health(self):
            return self.fused / self.tracked if self.tracked else 0.0

    def settled(self):
        watcher = QualityWatcher()
        for i in range(40):
            watcher.observe(textured(seed=i))
        return watcher

    def test_losing_track_outranks_thin_coverage(self):
        """Ordered by what blocks the scan hardest, not by what is easiest to see."""
        watcher = self.settled()
        state = self.State(tracked=20, fused=20, weak_spots=[object(), object()])

        advice = watcher.advise(state)
        assert "track" in advice.text.lower()

    def test_depth_not_sticking_is_its_own_message(self):
        watcher = self.settled()
        state = self.State(tracked=100, fused=10)

        advice = watcher.advise(state)
        assert advice.level == "warn"
        assert "depth" in advice.text.lower()

    def test_thin_spots_are_mentioned_with_their_number(self):
        watcher = self.settled()
        state = self.State(weak_spots=[object()] * 3)

        advice = watcher.advise(state)
        assert "3" in advice.text
        assert advice.level == "ok"

    def test_a_healthy_scan_says_so_briefly(self):
        watcher = self.settled()

        advice = watcher.advise(self.State())
        assert advice.level == "ok"
        assert len(advice.text) < 40
