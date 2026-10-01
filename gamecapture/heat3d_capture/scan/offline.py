"""
Reconstructing a recording, with the camera path solved all at once.

The live path in `session.py` tracks frame to frame, which is what makes the
overlay useful while walking. This one is for the case that path loses: a
racetrack, where the road is a smooth ribbon with little texture, the horizon
barely moves, and a pose that drifts is never pulled back. It needs the whole
recording before it can answer, so it is not live — and in exchange every frame
constrains every other one instead of only its neighbour.

Everything after the poses is the same code. The solved track is handed to the
reconstructor as a `pose_source`, exactly where Forza telemetry goes, so fusion,
coverage, colour and the GLB export are reused rather than forked. The only thing
this module decides is where the camera was.

Two passes over the video
-------------------------
The solve runs on a sample — one frame every second or so — because cost grows
faster than linearly with window size and a lap of dense frames would not fit.
Fusion then runs on a denser set, taking its poses by interpolation between the
solved ones. Sampling both from one pass would either make the solve enormous or
the geometry sparse; they want different rates, so they get them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..pose.feedforward import WINDOW, OVERLAP, Track, plan_windows, stitch
from ..pose.feedforward_source import FeedForwardPoses
from ..pose.odometry import Intrinsics
from ..pose.solveprep import learn_camera_mask, plan_by_motion

#: Frames handed to the solver, at most. Not an interval: they are spaced by
#: how far the picture moved rather than by the clock, so a fast section gets
#: the frames it needs and a slow one does not waste them.
#:
#: A clock was the first attempt and it was measurably wrong. At a fixed 1.2 s
#: the median feature displacement over one lap ran from 13 px where the car
#: crawled to 149 px where it was quick, and the windows that came out wrong
#: were the fast ones — consecutive frames no longer overlapped enough to be
#: related to each other at all.
SOLVE_BUDGET = 260

#: Seconds between frames fused into geometry. Denser than the solve, because
#: surface coverage wants many views and the poses in between are interpolated.
FUSE_INTERVAL = 0.4

#: A lap this long or shorter is solved in one pass if it fits the window.
MAX_SOLVE_FRAMES = 400


@dataclass
class OfflineProgress:
    """What the run is doing, for a caller that wants to show it."""

    stage: str = "starting"
    done: int = 0
    total: int = 0
    note: str = ""


@dataclass
class OfflineResult:
    """What a run produced, and how much of it can be believed."""

    track: Track
    times: list[float]
    export: Path | None = None
    #: Distance between the first and last camera, over the path length. On a
    #: single lap this is the honest measure of drift.
    closure: float = 0.0
    #: Worst disagreement at a window seam, in units of frame spacing.
    worst_seam: float = 0.0
    state: object | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"{len(self.track.centres)} cameras solved over {self.times[-1]:.0f}s",
            f"closure error {self.closure * 100:.1f}% of path length",
            f"worst seam {self.worst_seam:.2f} frame spacings",
        ]
        lines += self.notes
        if self.export:
            lines.append(f"wrote {self.export}")
        return "\n".join(lines)


def sample_video(
    path: str | Path, interval: float, *, width: int = 960
) -> tuple[list[np.ndarray], list[float]]:
    """
    Frames every `interval` seconds, as RGB, with their timestamps.

    Downscaled on the way out: the solver works at 518 pixels and the depth model
    below 1000, so decoding 4K straight into those is the slowest step in the
    whole run for no gain in either.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"could not open {path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(1, int(round(fps * interval)))

        frames: list[np.ndarray] = []
        times: list[float] = []
        for index in range(0, total, stride):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                break
            height = int(round(frame.shape[0] * width / frame.shape[1]))
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            times.append(index / fps)
        return frames, times
    finally:
        capture.release()


def frames_at(path, times: list[float], *, width: int = 960) -> list[np.ndarray]:
    """The frames at given timestamps, as RGB, downscaled on the way out."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"could not open {path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        out = []
        for when in times:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(when * fps))
            ok, frame = capture.read()
            if not ok:
                break
            height = int(round(frame.shape[0] * width / frame.shape[1]))
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return out
    finally:
        capture.release()


def solve_track(
    frames: list[np.ndarray],
    indices: list[int] | None = None,
    *,
    runner=None,
    window: int = WINDOW,
    overlap: int = OVERLAP,
    on_progress: Callable[[OfflineProgress], None] | None = None,
) -> Track:
    """
    Solve every window and stitch them into one path.

    `runner` is injected so this can be exercised with a stand-in: loading VGGT
    costs several gigabytes and a GPU, and the stitching logic — which is where
    the mistakes are — needs neither.
    """
    if runner is None:
        from ..pose.vggt_model import VggtRunner

        runner = VggtRunner()

    if len(frames) > MAX_SOLVE_FRAMES:
        raise ValueError(
            f"{len(frames)} frames is more than this solves in one run "
            f"({MAX_SOLVE_FRAMES}); sample less often or cut the recording"
        )

    plan = plan_windows(len(frames), window, overlap)
    windows = []
    for n, chunk in enumerate(plan):
        if on_progress:
            on_progress(
                OfflineProgress(
                    stage="solving", done=n, total=len(plan), note=f"frames {chunk[0]}–{chunk[-1]}"
                )
            )
        windows.append(runner.solve([frames[i] for i in chunk], chunk))
    return stitch(windows)


def reconstruct_video(
    video: str | Path,
    out: str | Path,
    *,
    intrinsics: Intrinsics | None = None,
    solve_budget: int = SOLVE_BUDGET,
    fuse_interval: float = FUSE_INTERVAL,
    runner=None,
    depth=None,
    on_progress: Callable[[OfflineProgress], None] | None = None,
) -> OfflineResult:
    """
    The whole run: sample, solve, fuse, export.

    Returns a result even when nothing was worth exporting, because "the track
    came out but the geometry did not" and "the track did not come out" are
    different problems and the caller should be able to tell them apart.
    """
    from .reconstruct import Reconstructor

    def report(stage: str, done: int = 0, total: int = 0, note: str = "") -> None:
        if on_progress:
            on_progress(OfflineProgress(stage, done, total, note))

    # The camera mask is learned but *not* applied to the solver's frames, and
    # that is a correction rather than an oversight.
    #
    # Blacking out the cockpit was justified by a feature count: 60-79% of every
    # feature an ORB detector found sat on camera-attached pixels. That is a true
    # statement about a feature detector and it says nothing about a trained
    # network, which was the thing being fed. Measured on what actually matters —
    # how far the recovered path turns between consecutive steps, where a car is
    # a few degrees and a random walk is ninety:
    #
    #     raw frames          31.5 degrees
    #     contrast lifted     34.6
    #     cockpit masked      45.0
    #     both                85.4
    #
    # So the preprocessing that looked best by feature statistics was the worst
    # by the only measure that counts. The mask is still learned, because the
    # incremental tracker in `odometry.py` does match features and does benefit,
    # and because the interface should say what it found.
    report("masking")
    camera_mask, mask_note = learn_camera_mask(video)
    report("masking", note=mask_note)

    report("planning")
    solve_times = plan_by_motion(video, camera_mask, budget=solve_budget)
    if len(solve_times) < 4:
        raise ValueError(
            f"{len(solve_times)} frames planned from {video}; too few to solve a path"
        )
    report("planning", note=f"{len(solve_times)} frames spaced by motion")
    solve_frames = frames_at(video, solve_times)

    track = solve_track(solve_frames, runner=runner, on_progress=on_progress)
    source = FeedForwardPoses(track, [solve_times[i] for i in track.frames])

    notes = [mask_note]
    closure = track.closure_error()
    worst = max(track.seam_error) if track.seam_error else 0.0
    if closure > 0.1:
        notes.append(
            f"The lap does not close: its end is {closure * 100:.0f}% of the path "
            "length away from its start. Treat the shape as unreliable."
        )
    if worst > 1.0:
        notes.append(
            f"Windows disagreed by up to {worst:.1f} frame spacings at a seam, "
            "so part of the path is stitched on weak evidence."
        )

    report("fusing")
    fuse_frames, fuse_times = sample_video(video, fuse_interval)
    height, width = fuse_frames[0].shape[:2]
    k = intrinsics or Intrinsics.from_fov(width, height, 65.0)

    if depth is None:
        from ..depth.estimator import DepthEstimator

        depth = DepthEstimator()

    reconstructor = Reconstructor(k, pose_source=source)
    kept = 0
    for n, (frame, t) in enumerate(zip(fuse_frames, fuse_times)):
        if source.at(t) is None:
            continue  # outside the solved span; nothing to place it with
        maps = depth.predict(frame)
        if not maps:
            continue
        reconstructor.add(frame, maps[0].values, timestamp=t)
        kept += 1
        if n % 10 == 0:
            report("fusing", n, len(fuse_frames), f"{kept} placed")

    report("exporting")
    written = reconstructor.export(out, name=Path(video).stem)
    if written is None:
        notes.append("Fusion produced no surface, so nothing was written.")

    return OfflineResult(
        track=track,
        times=[solve_times[i] for i in track.frames],
        export=written,
        closure=closure,
        worst_seam=worst,
        state=reconstructor.state,
        notes=notes,
    )
