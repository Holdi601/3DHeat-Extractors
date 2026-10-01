"""
The two numbers that real footage corrected.

Both were set against a brightly-lit synthetic room and both were wrong by more
than an order of magnitude on a real recording — a Forza rally lap, at night,
from the cockpit. They are kept together here because they have the same moral:
a threshold expressed as an absolute quantity is a guess about the world, and the
world was not consulted.

- **Keyframe selection.** Fixed change thresholds of 0.035 to 0.12 against a
  measured median inter-frame change of **0.0025**. Twenty-two keyframes from
  eleven thousand frames.
- **Global scale.** The first frame declared its depth to be the unit, so a
  three-minute lap reconstructed as **0.1 metres** of road.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_capture.pose.groundscale import (
    MAX_SCATTER,
    GroundCalibrator,
    calibrate_from_ground,
)
from heat3d_capture.pose.odometry import Intrinsics
from heat3d_capture.scan.session import Quality


def ground_plane_depth(k: Intrinsics, height: float) -> np.ndarray:
    """
    Inverse depth of a flat floor `height` below a level camera.

    For a pixel row v, a ray hits the ground at depth = height * fy / (v - cy).
    Rows at or above the horizon never do.
    """
    ys, _ = np.mgrid[0 : k.height, 0 : k.width]
    below = ys - k.cy
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(below > 0, height * k.fy / np.maximum(below, 1e-9), np.inf)
    return np.where(np.isfinite(depth), 1.0 / depth, 0.0).astype(np.float32)


class TestGroundScale:
    @pytest.mark.parametrize("height", [1.15, 1.70, 2.40])
    def test_recovers_the_scale_of_a_known_floor(self, height):
        k = Intrinsics.from_fov(960, 540, 65.0)
        # Depth already in metres, so a correct calibration returns scale 1.
        estimate = calibrate_from_ground(ground_plane_depth(k, height), k, height=height)
        assert estimate is not None and estimate.trustworthy
        assert estimate.fit.scale == pytest.approx(1.0, rel=0.1)

    def test_a_wrongly_scaled_floor_is_corrected(self):
        k = Intrinsics.from_fov(960, 540, 65.0)
        # What the network actually hands over: relative inverse depth in units
        # of its own, here twenty times too large.
        metric = ground_plane_depth(k, 1.15)
        estimate = calibrate_from_ground(metric * 20.0, k, height=1.15)
        assert estimate is not None
        # Inverse depth twenty times too large means depths twenty times too
        # small, so a metre is twenty of the network's units. The factor that
        # undoes it is 20, not 1/20 — the direction this assertion originally
        # had backwards.
        assert 1.0 / estimate.fit.scale == pytest.approx(20.0, rel=0.15)

    def test_a_perfectly_flat_floor_has_almost_no_scatter(self):
        k = Intrinsics.from_fov(960, 540, 65.0)
        estimate = calibrate_from_ground(ground_plane_depth(k, 1.15), k, height=1.15)
        assert estimate.scatter < 0.2

    def test_a_scene_with_no_ground_is_refused(self):
        # Pointing at the sky, or standing at a wall. Better to say so than to
        # invent a factor that looks deliberate.
        k = Intrinsics.from_fov(960, 540, 65.0)
        assert calibrate_from_ground(np.zeros((540, 960), np.float32), k) is None

    def test_the_tolerance_admits_a_real_road(self):
        # Measured on the reference recording: every frame sampled across three
        # minutes of driving gave scatter between 0.40 and 0.45, and the scale
        # they implied agreed to within 6%. The original bound of 0.35 rejected
        # all of it — a calibration that was demonstrably working, refused for
        # being less flat than a rendered floor.
        assert MAX_SCATTER > 0.45


class TestCalibrator:
    def test_settles_and_reports_how_it_did(self):
        k = Intrinsics.from_fov(960, 540, 65.0)
        calibrator = GroundCalibrator(k, height=1.15)
        fit = calibrator.offer(ground_plane_depth(k, 1.15))
        assert fit is not None
        assert calibrator.settled
        assert "1.15 m camera height" in calibrator.note

    def test_says_plainly_when_it_could_not(self):
        k = Intrinsics.from_fov(960, 540, 65.0)
        calibrator = GroundCalibrator(k)
        for _ in range(3):
            calibrator.offer(np.zeros((540, 960), np.float32))
        assert calibrator._result() is None
        assert "relative, not metres" in calibrator.note


class TestKeyframeSelectivity:
    def test_the_presets_are_quantiles_not_fixed_amounts(self):
        # The fix for the measurement above. A quantile means the same thing in a
        # bright room and on a dark rally stage; a fixed fraction of the picture
        # does not, and was between fourteen and eighty times too large.
        for q in Quality:
            assert 0.0 < q.keep_quantile < 1.0

    def test_more_detail_keeps_more(self):
        assert Quality.DETAILED.keep_quantile < Quality.BALANCED.keep_quantile
        assert Quality.BALANCED.keep_quantile < Quality.DRAFT.keep_quantile

    def test_the_starting_thresholds_suit_real_footage(self):
        # Used only for the first thirty frames, before there is enough history
        # to adapt. They must at least be the right order of magnitude for a dark
        # scene, whose measured median change was 0.0025.
        for q in Quality:
            assert q.keyframe_change < 0.03, (
                f"{q.value} would keep almost nothing on real footage"
            )

    def test_a_scan_adapts_its_threshold_to_the_scene(self):
        # End to end: a low-contrast sequence whose changes are far below every
        # fixed threshold must still yield keyframes.
        from heat3d_capture.capture.sources import CapturedFrame
        from heat3d_capture.scan.session import ScanSession, ScanSettings

        rng = np.random.default_rng(0)
        base = (rng.random((120, 200, 3)) * 30 + 10).astype(np.uint8)

        class Faint:
            """Frames that drift by a hair — a dark scene, slowly panning."""

            def frames(self):
                for i in range(200):
                    shifted = np.roll(base, i, axis=1)
                    yield CapturedFrame(image=shifted, t=i / 30, index=i, monotonic=i / 30)

            def close(self):
                pass

        session = ScanSession(ScanSettings(quality=Quality.BALANCED, reconstruct=False))
        session.start(source=Faint())
        assert session.wait(timeout=30)
        # The old fixed threshold kept one frame in this situation.
        assert session.progress.keyframes > 20, (
            f"only {session.progress.keyframes} keyframes — the threshold did not adapt"
        )
