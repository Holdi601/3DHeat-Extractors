"""
Tests for what a frame looks like by the time the solver sees it.

Two things happen to it, and both were added because of a measurement rather
than a hunch.

The camera's own furniture is blacked out. On the reference lap, 60–79% of every
feature a detector found sat on the dashboard, the pillars or the HUD — so the
median apparent motion between frames a second apart was 0–4 pixels, and the
model was being told that the scene barely moves.

What remains gets its local contrast lifted, because the reference lap is at
night and there is very little to find outside the headlights.

The tests below check the properties that make those two things safe: that the
geometry is untouched, that a frame with nothing to mask is passed through, and
that the enhancement is local rather than a brightness curve.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_capture.pose.solveprep import enhance, enhance_all, learn_camera_mask


def night_frame(height: int = 120, width: int = 200, seed: int = 0) -> np.ndarray:
    """Mostly dark, with one bright patch — a headlight cone at night."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 40, size=(height, width, 3), dtype=np.uint8)
    top, left, right = height // 2, width // 3, 2 * width // 3
    frame[top:, left:right] = rng.integers(
        120, 200, size=(height - top, right - left, 3), dtype=np.uint8
    )
    return frame


def test_masked_pixels_are_blacked_out():
    frame = night_frame()
    mask = np.full(frame.shape[:2], 255, np.uint8)
    mask[-30:, :] = 0  # a dashboard across the bottom

    out = enhance(frame, mask)

    assert (out[-30:, :] == 0).all()
    assert out[:-30, :].any(), "the world above the dashboard was blacked out too"


def test_the_frame_keeps_its_shape_so_the_geometry_is_unchanged():
    """
    Masked rather than cropped, deliberately.

    Cropping would move the principal point off centre, and the model assumes it
    is centred — so the camera path would bend by a constant amount for a reason
    nothing downstream could see. Blacking out costs the pixels and keeps the
    lens honest.
    """
    frame = night_frame()
    mask = np.full(frame.shape[:2], 255, np.uint8)
    mask[-40:, :60] = 0

    out = enhance(frame, mask)

    assert out.shape == frame.shape
    assert out.dtype == frame.dtype


def test_a_frame_with_nothing_to_mask_is_still_enhanced():
    frame = night_frame()

    out = enhance(frame, None)

    assert out.shape == frame.shape
    # The dark part should have gained contrast; a no-op would leave it alone.
    assert out[:40].std() > frame[:40].std()


def test_enhancement_is_local_rather_than_a_brightness_curve():
    """
    A night histogram is dominated by black sky. Anything global either leaves
    the road flat or blows out the headlights, which is why this equalises in
    tiles instead.
    """
    frame = night_frame()

    out = enhance(frame, None)

    dark_before, dark_after = frame[:40].mean(), out[:40].mean()
    bright_before, bright_after = frame[-40:].mean(), out[-40:].mean()

    # The dark region gains much more than the already-bright one.
    assert dark_after - dark_before > bright_after - bright_before


def test_a_mask_of_a_different_size_is_fitted_to_the_frame():
    """The mask is learned at one resolution and used at whatever is sampled."""
    frame = night_frame(240, 400)
    mask = np.full((120, 200), 255, np.uint8)
    mask[-30:, :] = 0

    out = enhance(frame, mask)

    assert out.shape == frame.shape
    assert (out[-55:, :] == 0).all()


def test_enhance_all_leaves_the_input_alone():
    """The caller keeps the originals for fusion, which wants the true pixels."""
    frames = [night_frame(seed=i) for i in range(3)]
    before = [f.copy() for f in frames]
    mask = np.full(frames[0].shape[:2], 255, np.uint8)
    mask[-20:, :] = 0

    enhance_all(frames, mask)

    for original, kept in zip(frames, before):
        assert np.array_equal(original, kept)


def test_a_missing_recording_says_so(tmp_path):
    with pytest.raises(OSError, match="could not open"):
        learn_camera_mask(tmp_path / "nothing.mp4")


class TestSpacingFramesByMotion:
    """
    A clock is the wrong criterion for spacing frames, and the measurement says
    so plainly.

    On the reference lap, at a fixed 1.17 s the median feature displacement
    between consecutive frames ran from 13 px where the car crawled to 149 px
    where it was quick — and the windows that came out wrong were the *fast*
    ones, where consecutive frames no longer overlapped enough to be related to
    each other. A clock gives the slow sections frames they do not need and
    starves the fast ones of the frames they do.

    Spacing by measured displacement instead brought the same footage to a median
    of 56 px with a 11–123 px spread, on the same frame budget.
    """

    @staticmethod
    def moving_clip(path, *, frames=150, fps=30.0, speed_of):
        """A textured world sliding past at a speed that varies over the clip."""
        import cv2

        rng = np.random.default_rng(5)
        wide = np.full((180, 4000), 30, np.uint8)
        for _ in range(3000):
            x, y = int(rng.integers(0, 4000)), int(rng.integers(0, 180))
            cv2.rectangle(wide, (x, y), (x + 9, y + 9), int(rng.integers(90, 250)), -1)

        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 180)
        )
        if not writer.isOpened():
            pytest.skip("no video encoder available")
        at = 0.0
        for i in range(frames):
            at += speed_of(i / frames)
            start = int(min(at, wide.shape[1] - 321))
            tile = np.ascontiguousarray(wide[:, start : start + 320])
            writer.write(cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR))
        writer.release()
        return path

    def test_it_takes_more_frames_where_the_picture_moves_faster(self, tmp_path):
        from heat3d_capture.pose.solveprep import plan_by_motion

        # Slow for the first half, fast for the second.
        clip = self.moving_clip(
            tmp_path / "vary.mp4", speed_of=lambda f: 1.0 if f < 0.5 else 12.0
        )

        times = plan_by_motion(clip, None, budget=60)

        assert len(times) >= 4
        half = times[-1] / 2
        slow = sum(1 for t in times if t < half)
        fast = sum(1 for t in times if t >= half)
        assert fast > slow, f"{slow} frames in the slow half, {fast} in the fast half"

    def test_gaps_stay_within_their_bounds(self, tmp_path):
        """
        A stationary camera must not emit nothing, and a long pause must not
        swallow the rest of the recording into one step.
        """
        from heat3d_capture.pose.solveprep import MAX_GAP, MIN_GAP, plan_by_motion

        clip = self.moving_clip(tmp_path / "still.mp4", speed_of=lambda f: 0.0)

        times = plan_by_motion(clip, None, budget=40)

        gaps = np.diff(times)
        assert len(times) > 1
        assert gaps.min() >= MIN_GAP - 1e-6
        assert gaps.max() <= MAX_GAP + 1e-6

    def test_the_budget_is_respected(self, tmp_path):
        """Solving is minutes per window, so an unbounded plan is not a plan."""
        from heat3d_capture.pose.solveprep import plan_by_motion

        clip = self.moving_clip(tmp_path / "fast.mp4", speed_of=lambda f: 20.0)

        assert len(plan_by_motion(clip, None, budget=12)) <= 12

    def test_it_starts_at_the_beginning_and_moves_forward(self, tmp_path):
        from heat3d_capture.pose.solveprep import plan_by_motion

        clip = self.moving_clip(tmp_path / "fwd.mp4", speed_of=lambda f: 6.0)

        times = plan_by_motion(clip, None, budget=40)

        assert times[0] == 0.0
        assert all(b > a for a, b in zip(times, times[1:]))
