"""
Tests for bundle adjustment.

Built from known poses and known points, so "did it work" is a number rather than
an impression: perturb the truth, solve, and measure how much of the truth came
back. That is the only way to test an optimiser — one that merely reduces its own
residual can be converging confidently on the wrong answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_capture.pose.bundle import (
    MIN_ANGLE_DEG,
    Observation,
    Problem,
    bundle_adjust,
    pack,
    triangulate,
    unpack,
)
from heat3d_capture.pose.odometry import Intrinsics, Pose


def k() -> Intrinsics:
    return Intrinsics.from_fov(640, 480, 70.0)


def truth(cameras: int = 8, points: int = 60, seed: int = 0):
    """
    A camera moving sideways past a cloud, which triangulates well.

    Sideways on purpose: it is the motion that gives parallax, so any failure
    here is the solver's and not the geometry's.
    """
    rng = np.random.default_rng(seed)
    cloud = np.stack(
        [
            rng.uniform(-4, 4, points),
            rng.uniform(-3, 3, points),
            rng.uniform(8, 16, points),
        ],
        axis=1,
    )
    poses = [
        Pose(np.eye(3), np.array([i * 0.6, 0.0, 0.0], dtype=float)) for i in range(cameras)
    ]
    return poses, cloud


def observe(poses, cloud, k_, *, noise: float = 0.0, seed: int = 1):
    """Project every point into every camera that can see it."""
    rng = np.random.default_rng(seed)
    out = []
    for ci, pose in enumerate(poses):
        r, t = pose.camera_from_world()
        camera_points = cloud @ r.T + t
        pixels = k_.project(camera_points)
        for pi, (uv, z) in enumerate(zip(pixels, camera_points[:, 2])):
            if z <= 0.1 or not np.isfinite(uv).all():
                continue
            if not (0 <= uv[0] < k_.width and 0 <= uv[1] < k_.height):
                continue
            jitter = rng.normal(0, noise, 2) if noise else 0.0
            out.append(Observation(camera=ci, point=pi, uv=np.asarray(uv) + jitter))
    return out


class TestPacking:
    def test_round_trips_exactly(self):
        poses, cloud = truth(4, 10)
        problem = Problem(poses=poses, points=cloud, observations=[])
        back_poses, back_points = unpack(pack(problem), 4, 10)
        assert back_points == pytest.approx(cloud)
        for a, b in zip(poses, back_poses):
            assert a.translation == pytest.approx(b.translation, abs=1e-9)
            assert a.rotation == pytest.approx(b.rotation, abs=1e-9)

    def test_survives_a_real_rotation(self):
        angle = 0.7
        rotation = np.array(
            [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        )
        pose = Pose(rotation, np.array([2.0, -1.0, 3.0]))
        problem = Problem(poses=[pose], points=np.zeros((1, 3)), observations=[])
        back, _ = unpack(pack(problem), 1, 1)
        assert back[0].rotation == pytest.approx(rotation, abs=1e-9)
        assert back[0].translation == pytest.approx(pose.translation, abs=1e-9)


class TestTriangulation:
    def test_recovers_a_known_point(self):
        k_ = k()
        point = np.array([1.0, -0.5, 12.0])
        poses = [Pose(np.eye(3), np.array([x, 0.0, 0.0])) for x in (0.0, 2.0, 4.0)]
        views = []
        for pose in poses:
            r, t = pose.camera_from_world()
            views.append((pose, k_.project((point @ r.T + t)[None])[0]))
        got, angle = triangulate(views, k_)
        assert got == pytest.approx(point, abs=1e-6)
        assert angle > MIN_ANGLE_DEG

    def test_reports_a_tiny_angle_for_views_from_one_place(self):
        # The case that ruins a forward-facing drive: every view from the same
        # bearing, so the depth is whatever the noise says.
        k_ = k()
        point = np.array([0.0, 0.0, 40.0])
        poses = [Pose(np.eye(3), np.array([0.0, 0.0, z])) for z in (0.0, 0.05, 0.1)]
        views = []
        for pose in poses:
            r, t = pose.camera_from_world()
            views.append((pose, k_.project((point @ r.T + t)[None])[0]))
        # Exactly degenerate: three cameras on the optical axis, a point straight
        # ahead. The DLT has no solution and refusing is better than returning
        # one — which is the answer here, so either outcome is acceptable as
        # long as it is not a confident wrong point.
        result = triangulate(views, k_)
        if result is not None:
            _, angle = result
            assert angle < MIN_ANGLE_DEG

    def test_one_view_triangulates_nothing(self):
        assert triangulate([(Pose.identity(), np.array([10.0, 10.0]))], k()) is None


class TestSolving:
    def test_perfect_data_stays_perfect(self):
        k_ = k()
        poses, cloud = truth()
        problem = Problem(poses=poses, points=cloud, observations=observe(poses, cloud, k_),
                          fixed={0})
        result = bundle_adjust(problem, k_)
        assert result.before < 1e-6
        assert result.after < 1e-5

    def test_recovers_poses_that_were_pushed_off(self):
        # The thing that matters. Start from poses that are wrong by tens of
        # centimetres — which is what a drifting chain produces — and see how much
        # of the truth comes back.
        k_ = k()
        poses, cloud = truth()
        observations = observe(poses, cloud, k_)

        rng = np.random.default_rng(7)
        drifted = [poses[0]] + [
            Pose(p.rotation.copy(), p.translation + rng.normal(0, 0.25, 3))
            for p in poses[1:]
        ]
        problem = Problem(poses=drifted, points=cloud.copy(),
                          observations=observations, fixed={0})
        result = bundle_adjust(problem, k_)

        before = np.mean([
            np.linalg.norm(a.translation - b.translation) for a, b in zip(drifted, poses)
        ])
        after = np.mean([
            np.linalg.norm(a.translation - b.translation)
            for a, b in zip(result.poses, poses)
        ])
        assert after < before * 0.25, f"pose error {before:.3f} m -> {after:.3f} m"
        assert result.after < result.before

    def test_the_anchor_does_not_move(self):
        # The window's first camera holds the solution in place. If it drifts,
        # the whole reconstruction slides and nothing downstream can tell.
        k_ = k()
        poses, cloud = truth()
        rng = np.random.default_rng(3)
        drifted = [poses[0]] + [
            Pose(p.rotation.copy(), p.translation + rng.normal(0, 0.2, 3)) for p in poses[1:]
        ]
        problem = Problem(poses=drifted, points=cloud.copy(),
                          observations=observe(poses, cloud, k_), fixed={0})
        result = bundle_adjust(problem, k_)
        assert result.poses[0].translation == pytest.approx(poses[0].translation, abs=1e-12)
        assert result.poses[0].rotation == pytest.approx(poses[0].rotation, abs=1e-12)

    def test_it_tolerates_noisy_observations(self):
        k_ = k()
        poses, cloud = truth()
        noisy = observe(poses, cloud, k_, noise=0.5)
        rng = np.random.default_rng(11)
        drifted = [poses[0]] + [
            Pose(p.rotation.copy(), p.translation + rng.normal(0, 0.15, 3)) for p in poses[1:]
        ]
        problem = Problem(poses=drifted, points=cloud.copy(), observations=noisy, fixed={0})
        result = bundle_adjust(problem, k_)
        after = np.mean([
            np.linalg.norm(a.translation - b.translation)
            for a, b in zip(result.poses, poses)
        ])
        assert after < 0.12, f"pose error {after:.3f} m with half-pixel noise"

    def test_nothing_to_solve_is_not_an_error(self):
        problem = Problem(poses=[Pose.identity()], points=np.zeros((0, 3)), observations=[])
        result = bundle_adjust(problem, k())
        assert result.converged and result.observations == 0

    def test_the_sparsity_pattern_is_actually_sparse(self):
        # Without it the solver builds a dense Jacobian and does not finish. This
        # is a correctness property of the implementation, not a nicety.
        from heat3d_capture.pose.bundle import _sparsity

        k_ = k()
        poses, cloud = truth(10, 100)
        problem = Problem(poses=poses, points=cloud,
                          observations=observe(poses, cloud, k_), fixed={0})
        pattern = _sparsity(problem)
        density = pattern.nnz / (pattern.shape[0] * pattern.shape[1])
        assert density < 0.05, f"Jacobian is {density * 100:.1f}% dense"
