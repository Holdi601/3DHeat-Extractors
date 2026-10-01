"""
Bundle adjustment: solving every pose against every observation at once.

The difference between this and what came before is the whole difference between
a reconstruction and a drift. Frame-to-frame odometry computes each pose from the
one before it, so every error is inherited and nothing ever corrects it — measured
on a real Forza lap, the recovered camera climbed **691 metres** over a drive that
was essentially level, because a small pitch error repeated a thousand times is a
large pitch error.

Bundle adjustment asks a different question. Given a set of 3D points and a set of
camera poses, it adjusts *all of them together* to minimise how far each point
lands from where it was actually observed, across every frame that saw it. A point
seen in twenty frames constrains twenty poses simultaneously. Errors cannot
accumulate along a chain because there is no chain.

This is the step Microsoft's Hyperlapse used on exactly this kind of footage — a
forward-facing camera on a bicycle — and it is why that worked in 2014 on material
that looks much like a rally stage at night.

Scope
-----
A *local* bundle adjustment over a sliding window of recent keyframes, not a
global one over the whole scan. Global is better and costs more than a live scan
can spend: the window bounds the work per keyframe, and the oldest pose in the
window is held fixed so the solution cannot drift as a whole.

Implemented on `scipy.optimize.least_squares` with an explicit sparsity pattern.
That pattern is not an optimisation detail — without it the solver builds a dense
Jacobian of (2 x observations) by (6 x cameras + 3 x points), which for a modest
window is millions of entries that are almost all zero, and it will not finish.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .odometry import Intrinsics, Pose

#: Observations a point needs before it is worth solving for. Two views
#: triangulate, but the result sits on the line between them and contributes
#: nothing to the poses; three is where a point starts constraining geometry.
MIN_OBSERVATIONS = 3

#: Reprojection error, in pixels, beyond which an observation is discarded before
#: solving. A least-squares solver has no defence against a gross outlier — one
#: mistracked feature at two hundred pixels drags every pose in the window.
OUTLIER_PIXELS = 4.0

#: Degrees of triangulation angle below which a point is too ill-conditioned to
#: keep. A point straight ahead of a forward-moving camera has almost none, and
#: its depth is whatever the noise says.
MIN_ANGLE_DEG = 0.8


@dataclass
class Observation:
    """One feature, seen in one frame."""

    camera: int
    point: int
    uv: np.ndarray  # (2,)


@dataclass
class Problem:
    """Everything the solver needs, in its own index space."""

    poses: list[Pose]
    points: np.ndarray  # (P, 3) world
    observations: list[Observation]
    #: Cameras held fixed, by index. The window's oldest, normally.
    fixed: set[int] = field(default_factory=set)

    @property
    def camera_count(self) -> int:
        return len(self.poses)

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])


def _rotvec_from_matrix(r: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.Rodrigues(np.ascontiguousarray(r))[0].reshape(3)


def _matrix_from_rotvec(v: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.Rodrigues(np.ascontiguousarray(v.reshape(3, 1)))[0]


def pack(problem: Problem) -> np.ndarray:
    """
    Flatten poses and points into the solver's parameter vector.

    Cameras are stored camera-from-world, which is the form the projection uses;
    converting inside the residual would repeat the work on every evaluation.
    """
    values = []
    for pose in problem.poses:
        r, t = pose.camera_from_world()
        values.append(np.concatenate([_rotvec_from_matrix(r), t]))
    return np.concatenate([np.concatenate(values), problem.points.ravel()])


def unpack(x: np.ndarray, cameras: int, points: int) -> tuple[list[Pose], np.ndarray]:
    """The inverse of `pack`, back to world-from-camera poses."""
    camera_block = x[: cameras * 6].reshape(cameras, 6)
    world_points = x[cameras * 6 :].reshape(points, 3)
    poses = []
    for row in camera_block:
        r = _matrix_from_rotvec(row[:3])
        rotation = r.T
        poses.append(Pose(rotation=rotation, translation=-rotation @ row[3:]))
    return poses, world_points


def _residuals(x: np.ndarray, problem: Problem, k: Intrinsics) -> np.ndarray:
    cameras, points = problem.camera_count, problem.point_count
    camera_block = x[: cameras * 6].reshape(cameras, 6)
    world = x[cameras * 6 :].reshape(points, 3)

    out = np.empty((len(problem.observations), 2))
    # Grouped by camera so each rotation is built once rather than per
    # observation; a window of 15 cameras and 4000 observations would otherwise
    # do four thousand Rodrigues conversions per evaluation.
    by_camera: dict[int, list[int]] = {}
    for i, ob in enumerate(problem.observations):
        by_camera.setdefault(ob.camera, []).append(i)

    for camera, rows in by_camera.items():
        r = _matrix_from_rotvec(camera_block[camera, :3])
        t = camera_block[camera, 3:]
        indices = np.array([problem.observations[i].point for i in rows])
        seen = world[indices] @ r.T + t
        z = np.where(np.abs(seen[:, 2]) < 1e-9, 1e-9, seen[:, 2])
        u = seen[:, 0] / z * k.fx + k.cx
        v = seen[:, 1] / z * k.fy + k.cy
        observed = np.array([problem.observations[i].uv for i in rows])
        out[rows, 0] = u - observed[:, 0]
        out[rows, 1] = v - observed[:, 1]
    return out.ravel()


def _sparsity(problem: Problem):
    """
    Which parameters each residual actually depends on.

    An observation touches exactly one camera and one point, so each of its two
    residual rows has nine non-zero columns out of thousands. Handing that
    structure to the solver is the difference between seconds and never.
    """
    from scipy.sparse import lil_matrix

    cameras, points = problem.camera_count, problem.point_count
    rows = 2 * len(problem.observations)
    columns = cameras * 6 + points * 3
    pattern = lil_matrix((rows, columns), dtype=int)

    for i, ob in enumerate(problem.observations):
        if ob.camera not in problem.fixed:
            for j in range(6):
                pattern[2 * i, ob.camera * 6 + j] = 1
                pattern[2 * i + 1, ob.camera * 6 + j] = 1
        for j in range(3):
            pattern[2 * i, cameras * 6 + ob.point * 3 + j] = 1
            pattern[2 * i + 1, cameras * 6 + ob.point * 3 + j] = 1
    return pattern


@dataclass
class BundleResult:
    poses: list[Pose]
    points: np.ndarray
    #: Root-mean-square reprojection error, pixels, before and after.
    before: float
    after: float
    observations: int
    converged: bool

    @property
    def improved(self) -> bool:
        return self.after <= self.before


def bundle_adjust(
    problem: Problem,
    k: Intrinsics,
    *,
    iterations: int = 30,
) -> BundleResult:
    """
    Solve the window. Returns refined poses and points, and what it achieved.

    Fixed cameras stay fixed by zeroing their Jacobian columns rather than by
    removing them from the parameter vector, which keeps the indexing simple at
    the cost of a few unused parameters.
    """
    from scipy.optimize import least_squares

    if not problem.observations or problem.point_count == 0:
        return BundleResult(problem.poses, problem.points, 0.0, 0.0, 0, True)

    x0 = pack(problem)
    start = _residuals(x0, problem, k)
    before = float(np.sqrt(np.mean(start**2)))

    result = least_squares(
        _residuals,
        x0,
        jac_sparsity=_sparsity(problem),
        # Soft L1 rather than plain squares: even after outlier rejection a few
        # observations are worse than the rest, and squaring gives them the loudest
        # voice in the solution precisely because they are wrong.
        loss="soft_l1",
        f_scale=2.0,
        method="trf",
        max_nfev=iterations * 10,
        args=(problem, k),
        verbose=0,
    )
    after = float(np.sqrt(np.mean(result.fun**2)))
    poses, points = unpack(result.x, problem.camera_count, problem.point_count)

    # Fixed cameras are restored exactly. The solver leaves them alone through the
    # sparsity pattern, but round-tripping through rotvec and back is not the
    # identity to floating-point precision, and a window anchor that shifts by a
    # micron every keyframe is a drift of its own.
    for index in problem.fixed:
        poses[index] = problem.poses[index]

    return BundleResult(
        poses=poses,
        points=points,
        before=before,
        after=after,
        observations=len(problem.observations),
        converged=bool(result.success),
    )


def triangulate(
    observations: list[tuple[Pose, np.ndarray]], k: Intrinsics
) -> tuple[np.ndarray, float] | None:
    """
    Multi-view triangulation by the linear (DLT) method.

    Returns the world point and the widest angle any pair of views subtends at
    it, which is the only honest measure of how much the point is worth: a point
    that every camera saw from the same bearing is not measured, it is assumed.
    """
    if len(observations) < 2:
        return None

    rows = []
    centres = []
    for pose, uv in observations:
        r, t = pose.camera_from_world()
        projection = k.matrix @ np.hstack([r, t.reshape(3, 1)])
        rows.append(uv[0] * projection[2] - projection[0])
        rows.append(uv[1] * projection[2] - projection[1])
        centres.append(pose.translation)

    _, _, vt = np.linalg.svd(np.array(rows))
    homogeneous = vt[-1]
    if abs(homogeneous[3]) < 1e-12:
        return None
    point = homogeneous[:3] / homogeneous[3]

    rays = np.array(centres) - point
    lengths = np.linalg.norm(rays, axis=1, keepdims=True)
    if (lengths < 1e-9).any():
        return None
    rays = rays / lengths
    # The widest pair, via the smallest dot product.
    cos = rays @ rays.T
    angle = float(np.degrees(np.arccos(np.clip(cos.min(), -1.0, 1.0))))
    return point, angle
