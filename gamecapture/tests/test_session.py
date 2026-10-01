"""
Tests for the scan session.

Driven by a scripted frame source rather than a capture, so the keyframe policy
— the part that decides what the reconstruction is even built from — can be
tested against known motion instead of against whatever was on screen.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from heat3d_capture.capture.sources import CapturedFrame
from heat3d_capture.scan.session import (
    Quality,
    ScanSession,
    ScanSettings,
    picture_change,
)


class ScriptedSource:
    """A frame source that plays a list of images."""

    def __init__(self, images, *, fps: float = 30.0, delay: float = 0.0):
        self.images = images
        self.fps = fps
        self.delay = delay
        self.closed = False

    def frames(self):
        for i, image in enumerate(self.images):
            if self.delay:
                time.sleep(self.delay)
            yield CapturedFrame(image=image, t=i / self.fps, index=i)

    def close(self):
        self.closed = True


def solid(value: int, size=(90, 160)) -> np.ndarray:
    return np.full((*size, 3), value, dtype=np.uint8)


def noise(seed: int, size=(90, 160)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((*size, 3)) * 255).astype(np.uint8)


class TestPictureChange:
    def test_identical_frames_have_not_changed(self):
        assert picture_change(solid(120), solid(120)) == pytest.approx(0.0)

    def test_a_completely_different_frame_is_a_large_change(self):
        assert picture_change(solid(0), solid(255)) == pytest.approx(1.0, abs=0.02)

    def test_is_proportional_to_the_difference(self):
        small = picture_change(solid(100), solid(110))
        large = picture_change(solid(100), solid(200))
        assert 0 < small < large

    def test_fine_detail_does_not_register_as_camera_motion(self):
        # Two different noise fields have enormous per-pixel difference and
        # almost no block-mean difference. That is the point: grain, foliage and
        # a flickering HUD must not keep frames while the player stands still.
        assert picture_change(noise(1), noise(2)) < 0.05


class TestQuality:
    def test_more_detail_means_a_lower_threshold_and_more_pixels(self):
        assert Quality.DETAILED.keyframe_change < Quality.BALANCED.keyframe_change
        assert Quality.BALANCED.keyframe_change < Quality.DRAFT.keyframe_change
        assert Quality.DETAILED.capture_side > Quality.DRAFT.capture_side

    def test_every_preset_explains_itself(self):
        for q in Quality:
            assert q.summary and q.capture_side > 0


class TestKeyframes:
    def _run(self, images, **over):
        settings = ScanSettings(**over)
        session = ScanSession(settings)
        session.start(source=ScriptedSource(images))
        assert session.wait(timeout=10), "scan did not finish"
        return session

    def test_the_first_frame_is_always_kept(self):
        session = self._run([solid(100)])
        assert session.progress.keyframes == 1

    def test_a_static_scene_keeps_almost_nothing(self):
        # Standing still produces hundreds of views of one thing. Fusing them
        # adds cost and a bias toward whatever the player stared at.
        session = self._run([solid(100) for _ in range(50)])
        assert session.progress.frames_seen == 50
        assert session.progress.keyframes == 1

    def test_a_changing_scene_keeps_frames(self):
        images = [solid(v) for v in range(0, 250, 25)]
        session = self._run(images)
        assert session.progress.keyframes == len(images)

    def test_quality_changes_how_much_is_kept(self):
        # The same walk, scanned two ways. Detailed must keep at least as much.
        images = [solid(100 + (i % 2) * 12) for i in range(40)]
        draft = self._run(images, quality=Quality.DRAFT)
        detailed = self._run(images, quality=Quality.DETAILED)
        assert detailed.progress.keyframes >= draft.progress.keyframes

    def test_reports_the_frames_it_looked_at(self):
        session = self._run([solid(100) for _ in range(12)])
        assert session.progress.frames_seen == 12


class TestLimits:
    def test_stops_at_the_keyframe_cap(self):
        images = [solid(v % 256) for v in range(0, 2000, 40)]
        session = ScanSession(ScanSettings(max_keyframes=5))
        session.start(source=ScriptedSource(images))
        assert session.wait(timeout=10)
        assert session.progress.keyframes == 5
        assert "5 keyframes" in session.progress.message

    def test_stops_at_the_time_limit(self):
        images = [solid(v % 256) for v in range(0, 4000, 7)]
        session = ScanSession(ScanSettings(max_seconds=0.25, max_keyframes=0))
        session.start(source=ScriptedSource(images, delay=0.002))
        assert session.wait(timeout=10)
        assert "limit" in session.progress.message
        assert session.progress.elapsed >= 0.25

    def test_zero_means_no_limit(self):
        images = [solid(v) for v in range(0, 250, 25)]
        session = ScanSession(ScanSettings(max_seconds=0, max_keyframes=0))
        session.start(source=ScriptedSource(images))
        assert session.wait(timeout=10)
        assert session.progress.keyframes == len(images)


class TestControl:
    def test_can_be_stopped_part_way_and_keeps_what_it_had(self):
        # A scan is something a person is physically doing; they must be able to
        # change their mind, and a partial level is still a level.
        images = [solid(v % 256) for v in range(0, 100_000, 30)]
        session = ScanSession(ScanSettings(max_keyframes=0))
        session.start(source=ScriptedSource(images, delay=0.001))
        time.sleep(0.2)
        assert session.is_running()
        session.stop(timeout=10)
        assert not session.is_running()
        assert session.progress.keyframes > 0
        assert session.progress.message == "stopped"

    def test_refuses_to_start_twice(self):
        session = ScanSession(ScanSettings())
        session.start(source=ScriptedSource([solid(1) for _ in range(500)], delay=0.002))
        with pytest.raises(RuntimeError, match="already running"):
            session.start(source=ScriptedSource([solid(2)]))
        session.stop(timeout=10)

    def test_reports_progress_while_running(self):
        seen = []
        session = ScanSession(ScanSettings(), on_progress=lambda p: seen.append(p.frames_seen))
        session.start(source=ScriptedSource([solid(v % 256) for v in range(0, 500, 10)]))
        assert session.wait(timeout=10)
        assert seen and seen[-1] > 0

    def test_a_failing_source_is_reported_not_swallowed(self):
        class Broken:
            def frames(self):
                raise RuntimeError("capture device went away")
                yield  # pragma: no cover

            def close(self):
                pass

        session = ScanSession(ScanSettings())
        session.start(source=Broken())
        assert session.wait(timeout=10)
        assert isinstance(session.error, RuntimeError)
        assert "went away" in session.progress.message

    def test_closes_a_source_it_opened_itself(self):
        source = ScriptedSource([solid(1)])
        session = ScanSession(ScanSettings())
        # Passed in, so the session must *not* close it — the caller owns it.
        session.start(source=source)
        assert session.wait(timeout=10)
        assert source.closed is False


class TestWithAnEstimator:
    class FakeEstimator:
        def __init__(self):
            self.calls = 0

        def predict(self, image):
            self.calls += 1
            from heat3d_capture.depth.estimator import DepthMap

            return [DepthMap(values=np.ones(image.shape[:2], dtype=np.float32), metric=False)]

    def test_depth_is_only_run_on_keyframes(self):
        # The whole reason keyframes exist: depth is the expensive step, and
        # running it on every frame of a static scene is pure waste.
        estimator = self.FakeEstimator()
        session = ScanSession(ScanSettings(), estimator=estimator)
        session.start(source=ScriptedSource([solid(100) for _ in range(30)]))
        assert session.wait(timeout=10)
        assert estimator.calls == 1
        assert session.progress.keyframes == 1

    def test_previews_are_offered_for_display(self):
        previews = []
        session = ScanSession(
            ScanSettings(),
            estimator=self.FakeEstimator(),
            on_preview=lambda rgb, depth: previews.append((rgb.shape, depth.shape)),
        )
        session.start(source=ScriptedSource([solid(v) for v in range(0, 250, 25)]))
        assert session.wait(timeout=10)
        assert previews
        assert previews[0][0][:2] == previews[0][1]

    def test_measures_a_throughput(self):
        session = ScanSession(ScanSettings(), estimator=self.FakeEstimator())
        session.start(source=ScriptedSource([solid(v) for v in range(0, 250, 25)]))
        assert session.wait(timeout=10)
        assert session.progress.fps > 0


class TestMemory:
    """Frames must not be hoarded: a long scan would otherwise eat the machine."""

    def _session(self, **over):
        return ScanSession(ScanSettings(reconstruct=False, **over), estimator=None)

    def test_frames_are_not_kept_by_default(self):
        # A half-hour scan at three keyframes a second is thousands of images
        # plus their depth maps — tens of gigabytes, for data nothing reads again
        # once it has been fused.
        session = self._session()
        session.start(source=ScriptedSource([solid(v) for v in range(0, 250, 10)]))
        assert session.wait(timeout=10)
        assert session.progress.keyframes > 1
        assert session.keyframes == []

    def test_frames_can_be_kept_when_something_will_reuse_them(self):
        session = ScanSession(ScanSettings(reconstruct=False), keep_frames=True)
        session.start(source=ScriptedSource([solid(v) for v in range(0, 250, 25)]))
        assert session.wait(timeout=10)
        assert len(session.keyframes) == session.progress.keyframes

    def test_the_count_is_right_either_way(self):
        kept = ScanSession(ScanSettings(reconstruct=False), keep_frames=True)
        dropped = self._session()
        frames = [solid(v) for v in range(0, 250, 25)]
        for session in (kept, dropped):
            session.start(source=ScriptedSource(list(frames)))
            assert session.wait(timeout=10)
        assert kept.progress.keyframes == dropped.progress.keyframes


class TestReconstructionWiring:
    def test_no_reconstructor_without_depth(self):
        # Reconstruction needs depth; without an estimator there is nothing to
        # reconstruct from, and it must not half-start.
        session = ScanSession(ScanSettings(reconstruct=True), estimator=None)
        session.start(source=ScriptedSource([solid(v) for v in range(0, 250, 25)]))
        assert session.wait(timeout=10)
        assert session.reconstructor is None
        assert session.export() is None
        assert session.weak_spots() == []

    def test_disabling_reconstruction_makes_it_a_recorder(self):
        session = ScanSession(
            ScanSettings(reconstruct=False), estimator=TestWithAnEstimator.FakeEstimator()
        )
        session.start(source=ScriptedSource([solid(v) for v in range(0, 250, 25)]))
        assert session.wait(timeout=10)
        assert session.reconstructor is None
        assert session.progress.keyframes > 0

    def test_reconstruction_runs_when_there_is_depth(self):
        import numpy as np

        from heat3d_capture.depth.estimator import DepthMap

        class PlaneEstimator:
            """Depth of a plane four metres away, which is fusible geometry."""

            def predict(self, image):
                return [
                    DepthMap(
                        values=np.full(image.shape[:2], 0.25, dtype=np.float32), metric=False
                    )
                ]

        import cv2

        def textured(shift):
            rng = np.random.default_rng(11)
            canvas = np.full((240, 960, 3), 25, np.uint8)
            for _ in range(200):
                x, y = int(rng.integers(0, 960)), int(rng.integers(0, 240))
                cv2.rectangle(canvas, (x, y), (x + 20, y + 20),
                              tuple(int(c) for c in rng.integers(60, 255, 3)), -1)
            return np.ascontiguousarray(np.roll(canvas, -shift, axis=1)[:, :320])

        session = ScanSession(
            ScanSettings(reconstruct=True, quality=Quality.DETAILED), estimator=PlaneEstimator()
        )
        session.start(source=ScriptedSource([textured(s) for s in range(0, 300, 30)]))
        assert session.wait(timeout=60)
        assert session.reconstructor is not None
        assert session.progress.reconstruction is not None
        assert session.progress.reconstruction.frames > 0
