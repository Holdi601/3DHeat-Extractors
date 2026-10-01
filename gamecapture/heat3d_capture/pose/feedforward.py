"""
Camera track and geometry from a feed-forward reconstruction model.

The incremental pipeline in `odometry.py` builds a pose one frame at a time from
feature matches, and on a racetrack it fails in a specific way: the road is a
smooth ribbon with little texture, the horizon barely moves, and once a few
frames drift there is nothing to pull them back. A lap comes out as a curve that
wanders instead of closing.

VGGT takes a different route. It looks at a *set* of frames at once and predicts
their cameras, depths and point maps in a single pass, so every frame constrains
every other one directly rather than through a chain of pairwise estimates. That
is the property that matters here: it is what the user meant by asking the model
to take earlier frames into account, and it is why a run of this can recover a
shape that frame-to-frame tracking loses.

Chunking, and why the overlap is not optional
---------------------------------------------
The model attends across all frames at once, so cost grows faster than linearly
and a 3-minute lap does not fit in one pass. The lap is therefore cut into
overlapping windows, each solved independently, and the windows are stitched.

Each window comes back in its own arbitrary frame *and its own arbitrary scale* —
the model cannot know how big anything is, exactly as the monocular depth model
cannot. So consecutive windows share several frames, and those shared cameras
determine the similarity transform (rotation, translation and one scale factor)
that carries one window into the previous one's frame. With too few shared frames
that transform is poorly conditioned and the scale estimate in particular is
noisy; the error then compounds window over window and the lap spirals. The
overlap is what keeps the stitch rigid, which is why it defaults to a third of
the window rather than to one or two frames.

Licensing
---------
`facebook/VGGT-1B` is **CC-BY-NC-4.0**: non-commercial only, and a reconstruction
made with it inherits that. A separately licensed `VGGT-1B-Commercial` exists but
is a gated repository that has to be requested from Meta. Nothing here chooses
for the user — `MODELS` names both, the non-commercial default is stated at the
point of use, and `describe_licence()` exists so the interface can say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: The weights, and what each obliges. See the module docstring.
MODELS = {
    "vggt-1b": "facebook/VGGT-1B",
    "vggt-1b-commercial": "facebook/VGGT-1B-Commercial",
}

#: Non-commercial. Deliberately the default only because the commercial variant
#: is gated and cannot be fetched without the user's own approved access.
DEFAULT_MODEL = "vggt-1b"

COMMERCIAL_MODELS = {"vggt-1b-commercial"}

#: Frames per window. Cost is dominated by fixed per-window overhead rather than
#: by frame count, so bigger is cheaper per frame until memory runs out.
WINDOW = 16

#: Shared frames between neighbouring windows. A third of the window: enough for
#: the similarity fit to be well conditioned in scale as well as in rotation.
OVERLAP = 6

#: Below this many shared cameras a stitch is not attempted, because three
#: points determine a similarity transform exactly and leave nothing over to
#: detect a bad one with.
MIN_STITCH = 4


def describe_licence(model: str = DEFAULT_MODEL) -> str:
    """One line the interface can show before a run starts."""
    if model in COMMERCIAL_MODELS:
        return f"{MODELS[model]} — separately licensed; check the terms you were granted."
    return (
        f"{MODELS[model]} — CC-BY-NC-4.0, non-commercial use only. "
        "Reconstructions made with it inherit that restriction."
    )


@dataclass
class Window:
    """One solved window: cameras, and the points they saw."""

    #: Indices into the caller's frame list, in order.
    frames: list[int]
    #: (N, 3) camera centres in this window's own frame and scale.
    centres: np.ndarray
    #: (N, 3, 3) camera rotations, world-from-camera.
    rotations: np.ndarray
    #: (N, H, W, 3) predicted world points, or None when not kept.
    points: np.ndarray | None = None
    #: (N, H, W) confidence for those points.
    confidence: np.ndarray | None = None


@dataclass
class Track:
    """A stitched camera path over the whole capture."""

    centres: np.ndarray
    rotations: np.ndarray
    frames: list[int] = field(default_factory=list)
    #: Residual of each stitch, in units of the local inter-frame spacing. A
    #: value near zero means the windows agreed; a large one marks the seam
    #: where the shape most likely went wrong.
    seam_error: list[float] = field(default_factory=list)

    def length(self) -> float:
        """Path length in the track's own units."""
        if len(self.centres) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.centres, axis=0), axis=1).sum())

    def closure_error(self) -> float:
        """
        How far the end lands from the start, relative to the path length.

        For a single lap this is the honest measure of whether the shape is
        right: a lap that closes to within a few per cent has not drifted, and
        one that does not has, whatever the picture looks like.
        """
        if len(self.centres) < 2:
            return float("inf")
        total = self.length()
        if total <= 0:
            return float("inf")
        return float(np.linalg.norm(self.centres[-1] - self.centres[0]) / total)


def similarity_from_points(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    The similarity transform taking `source` onto `target`, least squares.

    Umeyama's solution: centre both clouds, take the SVD of their covariance for
    the rotation, then one scale factor from the ratio of spreads. The reflection
    guard matters — without it a noisy overlap can produce a mirrored rotation
    that fits the shared points and turns the rest of the window inside out.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"need matching (N, 3) arrays, got {source.shape} and {target.shape}")
    if len(source) < 3:
        raise ValueError(f"a similarity transform needs at least 3 points, got {len(source)}")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    a = source - source_mean
    b = target - target_mean

    u, singular, vt = np.linalg.svd(b.T @ a / len(source))
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1.0
    rotation = u @ correction @ vt

    variance = (a**2).sum() / len(source)
    scale = float((singular * np.diag(correction)).sum() / variance) if variance > 0 else 1.0
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def similarity_from_poses(
    source: np.ndarray,
    source_rotations: np.ndarray,
    target: np.ndarray,
    target_rotations: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    The similarity between two windows, taking the rotation from the *cameras*.

    Fitting a rotation to the shared camera *positions* alone is ill-posed here,
    and in exactly the case that matters. The shared frames are consecutive
    positions along a path, so on any straight they are nearly collinear — and
    collinear points leave rotation about their own axis completely
    undetermined, with the rest of it barely determined. The fit then returns
    whatever the noise suggests.

    The symptom is unmistakable once seen: every window individually recovers a
    curving path, and the stitched result is a smooth straight line. The turning
    is not lost by any one window; it is discarded at every seam.

    Each window also reports where each camera *looks*, and that is fully
    determined whatever the positions do. So the rotation comes from the
    orientations — the average of `target_R · source_Rᵀ`, projected back onto a
    true rotation — and the positions are left to do only what they can do,
    which is fix the scale and the offset.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"need matching (N, 3) arrays, got {source.shape} and {target.shape}")
    if len(source) < 2:
        raise ValueError(f"need at least 2 shared cameras, got {len(source)}")

    # Average the relative rotation, then project back onto SO(3): a mean of
    # rotation matrices is not itself a rotation.
    stack = np.einsum("nij,nkj->ik", np.asarray(target_rotations), np.asarray(source_rotations))
    u, _, vt = np.linalg.svd(stack / len(source))
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1.0
    rotation = u @ correction @ vt

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    a = source - source_mean
    b = target - target_mean
    # Scale from the spread along the rotated source, which is what the
    # positions genuinely determine.
    turned = (rotation @ a.T).T
    denominator = float((turned * turned).sum())
    scale = float((turned * b).sum() / denominator) if denominator > 1e-18 else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def stitch(windows: list[Window]) -> Track:
    """
    Carry every window into the first one's frame and concatenate.

    Each window is placed using only the frames it shares with the running
    result, so a window that solved badly disturbs its own seam rather than
    silently rescaling everything after it.
    """
    if not windows:
        raise ValueError("nothing to stitch")

    placed: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for frame, centre, rotation in zip(
        windows[0].frames, windows[0].centres, windows[0].rotations
    ):
        placed[frame] = (centre, rotation)

    seams: list[float] = []
    for window in windows[1:]:
        shared = [i for i, f in enumerate(window.frames) if f in placed]
        if len(shared) < MIN_STITCH:
            raise ValueError(
                f"window starting at frame {window.frames[0]} shares only "
                f"{len(shared)} cameras with what is already placed; at least "
                f"{MIN_STITCH} are needed to fix a similarity transform"
            )
        source = window.centres[shared]
        target = np.stack([placed[window.frames[i]][0] for i in shared])
        # From the orientations, not the positions: shared cameras sit along a
        # path and are nearly collinear on any straight, which leaves a
        # position-only fit unable to see the rotation at all.
        scale, rotation, translation = similarity_from_poses(
            source,
            window.rotations[shared],
            target,
            np.stack([placed[window.frames[i]][1] for i in shared]),
        )

        moved = (scale * (rotation @ source.T).T) + translation
        spacing = np.linalg.norm(np.diff(target, axis=0), axis=1).mean() or 1.0
        seams.append(float(np.abs(moved - target).max() / spacing))

        for i, frame in enumerate(window.frames):
            if frame in placed:
                continue
            centre = scale * (rotation @ window.centres[i]) + translation
            placed[frame] = (centre, rotation @ window.rotations[i])

    order = sorted(placed)
    return Track(
        centres=np.stack([placed[f][0] for f in order]),
        rotations=np.stack([placed[f][1] for f in order]),
        frames=order,
        seam_error=seams,
    )


def to_trajectory(track: Track):
    """
    Present a stitched track as the `Trajectory` the rest of the pipeline takes.

    The fusion, coverage and export stages are written against the incremental
    tracker's output and have nothing to do with how a pose was obtained, so
    swapping the pose source should not touch them. This is that seam.

    `Pose` wants world-from-camera and the camera centre, which is what a window
    already holds — the conversion from the model's own convention happens once,
    in `vggt_model.solve`, rather than being repeated here.

    Imported lazily because `odometry` pulls in OpenCV, and the stitching maths
    above is deliberately usable without it.
    """
    from .odometry import Pose, TrackResult, Trajectory

    poses = [
        Pose(rotation=r.astype(np.float64), translation=c.astype(np.float64))
        for r, c in zip(track.rotations, track.centres)
    ]
    # A feed-forward solve has no inlier count to report: it did not match
    # features, it looked at the frames. Saying so in `reason` keeps the coverage
    # map from scoring these as if they were weak triangulations.
    results = [
        TrackResult(
            pose=pose,
            fit=None,
            inliers=0,
            matches=0,
            tracked=True,
            reason="feedforward",
        )
        for pose in poses
    ]
    return Trajectory(poses=poses, results=results)


def plan_windows(count: int, window: int = WINDOW, overlap: int = OVERLAP) -> list[list[int]]:
    """
    Split `count` frames into overlapping windows.

    The last window is pulled back to end on the final frame rather than left
    short, so the end of a capture is solved with as many neighbours as the
    middle instead of by a window of two.
    """
    if window <= overlap:
        raise ValueError(f"window {window} must exceed overlap {overlap}")
    if count <= window:
        return [list(range(count))]

    stride = window - overlap
    starts = list(range(0, count - window + 1, stride))
    if starts[-1] + window < count:
        starts.append(count - window)
    return [list(range(s, s + window)) for s in starts]
