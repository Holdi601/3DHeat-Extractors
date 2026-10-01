"""
Tests for the offline reconstruction run.

The expensive part of this run — VGGT — is injected, so everything around it can
be tested on a laptop with no GPU: how the video is sampled, how windows are
handed to the solver and stitched back, how the solved path becomes a pose
source, and whether the run reports honestly when the result is poor.

That last one matters as much as the geometry. A reconstruction that comes out
wrong and says nothing is worse than one that fails, because the failure is then
discovered by someone looking at a map and wondering why the corner is in the
wrong place.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from heat3d_capture.pose.feedforward import Window
from heat3d_capture.scan.offline import (
    MAX_SOLVE_FRAMES,
    OfflineProgress,
    sample_video,
    solve_track,
)


def circuit(count: int, radius: float = 50.0) -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.stack(
        [radius * np.cos(angle), np.zeros(count), radius * np.sin(angle)], axis=1
    )


class FakeRunner:
    """
    Stands in for VGGT: returns the truth for a window, in its own frame.

    Each window gets a different rotation, translation and scale, because that is
    what the real model does — it has no idea how big anything is, and no idea
    which way is north. A stand-in that returned everything in one frame would
    let a broken stitch pass.
    """

    def __init__(self, truth: np.ndarray):
        self.truth = truth
        self.calls: list[list[int]] = []

    def solve(self, frames, indices, *, keep_points: bool = False) -> Window:
        self.calls.append(list(indices))
        rng = np.random.default_rng(len(self.calls))
        scale = float(rng.uniform(0.5, 2.0))
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        shift = rng.normal(scale=10.0, size=3)
        local = (scale * (q @ self.truth[indices].T).T) + shift
        return Window(
            frames=list(indices),
            centres=local,
            rotations=np.stack([q] * len(indices)),
        )


def test_solve_track_recovers_the_shape_from_windows_in_their_own_frames():
    truth = circuit(48)
    runner = FakeRunner(truth)

    track = solve_track([None] * 48, runner=runner, window=12, overlap=5)

    from heat3d_capture.pose.feedforward import similarity_from_points

    scale, rotation, translation = similarity_from_points(track.centres, truth)
    aligned = (scale * (rotation @ track.centres.T).T) + translation
    spacing = np.linalg.norm(np.diff(truth, axis=0), axis=1).mean()

    assert np.abs(aligned - truth).max() < 1e-6 * spacing
    assert max(track.seam_error) < 1e-6


def test_every_frame_reaches_the_solver_exactly_through_some_window():
    truth = circuit(40)
    runner = FakeRunner(truth)

    track = solve_track([None] * 40, runner=runner, window=12, overlap=5)

    seen = {i for chunk in runner.calls for i in chunk}
    assert seen == set(range(40))
    assert track.frames == list(range(40))


def test_progress_is_reported_once_per_window():
    truth = circuit(40)
    seen: list[OfflineProgress] = []

    solve_track(
        [None] * 40,
        runner=FakeRunner(truth),
        window=12,
        overlap=5,
        on_progress=seen.append,
    )

    assert seen, "no progress reported"
    assert all(p.stage == "solving" for p in seen)
    assert seen[-1].total == len(seen)


def test_a_recording_too_long_for_one_run_is_refused_with_a_way_out():
    """
    Better than running for hours and then failing on memory.

    The message has to say what to do about it, because the caller's options —
    sample less often, or cut the recording — are not obvious from the failure.
    """
    truth = circuit(MAX_SOLVE_FRAMES + 1)

    with pytest.raises(ValueError, match="sample less often"):
        solve_track([None] * (MAX_SOLVE_FRAMES + 1), runner=FakeRunner(truth))


class TestSamplingAVideo:
    @staticmethod
    def write(path: Path, frames: int = 90, fps: float = 30.0) -> Path:
        import cv2

        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 180)
        )
        if not writer.isOpened():
            pytest.skip("no video encoder available")
        for i in range(frames):
            frame = np.full((180, 320, 3), i * 2 % 256, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return path

    def test_samples_at_about_the_asked_interval(self, tmp_path: Path):
        video = self.write(tmp_path / "clip.mp4", frames=90, fps=30.0)

        frames, times = sample_video(video, 0.5, width=160)

        assert len(frames) == len(times)
        assert len(frames) >= 5
        gaps = np.diff(times)
        assert np.allclose(gaps, 0.5, atol=0.05), gaps

    def test_frames_come_back_as_rgb_at_the_asked_width(self, tmp_path: Path):
        video = self.write(tmp_path / "clip.mp4", frames=60)

        frames, _ = sample_video(video, 0.5, width=160)

        assert frames[0].shape[1] == 160
        assert frames[0].shape[2] == 3
        # 320x180 scaled to 160 wide keeps its aspect ratio.
        assert frames[0].shape[0] == 90

    def test_a_missing_video_says_so(self, tmp_path: Path):
        with pytest.raises(OSError, match="could not open"):
            sample_video(tmp_path / "nothing.mp4", 1.0)


class TestTheWholeRun:
    """
    Sample, mask, solve, fuse, export — with the expensive parts injected.

    The point is the wiring, not the geometry: that the solved track reaches the
    reconstructor as a pose source, that frames outside the solved span are
    skipped rather than placed, and that a run which produces no surface says so
    instead of writing an empty file. Each of those is a seam between two pieces
    that are tested separately and could still be joined wrongly.
    """

    @staticmethod
    def clip(path, *, frames=120, fps=30.0):
        import cv2

        rng = np.random.default_rng(11)
        wide = np.full((180, 3000), 40, np.uint8)
        for _ in range(2500):
            x, y = int(rng.integers(0, 3000)), int(rng.integers(0, 180))
            cv2.rectangle(wide, (x, y), (x + 8, y + 8), int(rng.integers(90, 250)), -1)
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 180)
        )
        if not writer.isOpened():
            pytest.skip("no video encoder available")
        for i in range(frames):
            start = min(i * 8, wide.shape[1] - 321)
            tile = np.ascontiguousarray(wide[:, start : start + 320])
            writer.write(cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR))
        writer.release()
        return path

    class Depth:
        """A stand-in depth model: a plane sloping away from the camera."""

        def predict(self, image):
            from heat3d_capture.depth.estimator import DepthMap

            height, width = image.shape[:2]
            rows = np.linspace(1.0, 0.2, height, dtype=np.float32)
            return [DepthMap(values=np.tile(rows[:, None], (1, width)), metric=False)]

    def test_it_runs_end_to_end_and_reports_what_it_did(self, tmp_path):
        from heat3d_capture.scan.offline import reconstruct_video

        video = self.clip(tmp_path / "run.mp4")
        truth = circuit(40)

        result = reconstruct_video(
            video,
            tmp_path / "out.glb",
            runner=FakeRunner(truth),
            depth=self.Depth(),
            fuse_interval=0.5,
        )

        assert len(result.track.centres) > 4
        assert len(result.times) == len(result.track.centres)
        assert result.state.frames > 0
        # A run that wrote nothing has to say so rather than leave the caller to
        # discover an empty file.
        assert result.export is not None or any("no surface" in n for n in result.notes)
        assert "cameras solved" in result.summary()

    def test_a_drifting_lap_is_called_out_rather_than_shipped_quietly(self, tmp_path):
        """
        A lap that does not close is the failure that looks like success: the
        picture is smooth and the shape is wrong. It has to be in the notes.
        """
        from heat3d_capture.scan.offline import reconstruct_video

        video = self.clip(tmp_path / "drift.mp4")
        # A straight line: its ends are as far apart as the path is long.
        straight = np.stack(
            [np.arange(40) * 5.0, np.zeros(40), np.zeros(40)], axis=1
        )

        result = reconstruct_video(
            video,
            tmp_path / "out.glb",
            runner=FakeRunner(straight),
            depth=self.Depth(),
            fuse_interval=0.5,
        )

        assert result.closure > 0.5
        assert any("does not close" in n for n in result.notes), result.notes
