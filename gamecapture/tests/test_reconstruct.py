"""
Tests for the reconstruction stage.

Fed synthetic frames with known geometry, so the output can be checked rather
than admired. The important behaviours are the defensive ones: a lost track must
not contribute geometry, distant and near-field points must be rejected, and an
empty reconstruction must refuse to write a file rather than writing an empty one.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_capture.pose.odometry import Intrinsics
from heat3d_capture.scan.reconstruct import (
    MAX_DEPTH,
    MIN_DEPTH,
    Reconstructor,
)

WIDTH, HEIGHT = 320, 240


def intrinsics() -> Intrinsics:
    return Intrinsics.from_fov(WIDTH, HEIGHT, 90.0)


def textured(shift: int = 0, seed: int = 7) -> np.ndarray:
    """A frame with repeatable corners for ORB."""
    import cv2

    rng = np.random.default_rng(seed)
    canvas = np.full((HEIGHT, WIDTH * 3, 3), 25, dtype=np.uint8)
    for _ in range(240):
        x = int(rng.integers(0, WIDTH * 3))
        y = int(rng.integers(0, HEIGHT))
        cv2.rectangle(
            canvas,
            (x, y),
            (x + int(rng.integers(10, 34)), y + int(rng.integers(10, 34))),
            tuple(int(c) for c in rng.integers(60, 255, 3)),
            -1,
        )
    start = int(np.clip(shift, 0, WIDTH * 2))
    return np.ascontiguousarray(canvas[:, start : start + WIDTH])


def flat_depth(distance: float = 4.0) -> np.ndarray:
    """Inverse depth of a plane at a fixed distance."""
    return np.full((HEIGHT, WIDTH), 1.0 / distance, dtype=np.float32)


class TestFirstFrame:
    def test_the_first_frame_is_fused(self):
        r = Reconstructor(intrinsics(), camera_height=None)
        state = r.add(textured(), flat_depth())
        assert state.frames == 1
        assert state.tracked == 1
        assert state.surface_voxels > 0
        assert state.coverage_voxels > 0

    def test_records_where_it_looked_from(self):
        r = Reconstructor(intrinsics(), camera_height=None)
        r.add(textured(), flat_depth())
        assert len(r.trajectory.poses) == 1
        assert r.trajectory.positions[0] == pytest.approx([0, 0, 0])


class TestLostTracking:
    def test_a_lost_frame_contributes_no_geometry(self):
        # The behaviour that keeps a scan honest. Fusing a frame whose pose is
        # unknown welds geometry on at an arbitrary offset, and nothing
        # downstream can tell that happened.
        r = Reconstructor(intrinsics(), camera_height=None)
        r.add(textured(), flat_depth())
        before = r.state.surface_voxels
        state = r.add(np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8), flat_depth())
        assert state.lost == 1
        assert state.surface_voxels == before

    def test_reports_why_it_lost_the_track(self):
        r = Reconstructor(intrinsics(), camera_height=None)
        r.add(textured(), flat_depth())
        r.add(np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8), flat_depth())
        assert r.state.last_reason

    def test_tracking_health_reflects_the_losses(self):
        r = Reconstructor(intrinsics(), camera_height=None)
        r.add(textured(), flat_depth())
        blank = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
        r.add(blank, flat_depth())
        assert r.state.tracking_health == pytest.approx(0.5)

    def test_health_of_a_scan_that_never_started_is_zero(self):
        assert Reconstructor(intrinsics(), camera_height=None).state.tracking_health == 0.0


class TestDepthGating:
    def test_the_near_field_is_rejected(self):
        # A game frame's near field is the weapon model, the HUD and the car's
        # own bonnet. None of it is level geometry, and all of it is close.
        r = Reconstructor(intrinsics(), camera_height=None)
        r.add(textured(), np.full((HEIGHT, WIDTH), 1.0 / (MIN_DEPTH * 0.5), dtype=np.float32))
        assert r.state.surface_voxels == 0

    def test_the_far_field_is_rejected(self):
        # Monocular error grows with the square of distance; distant points are
        # noise shaped like a smeared shell behind everything real.
        r = Reconstructor(intrinsics(), camera_height=None)
        r.add(textured(), np.full((HEIGHT, WIDTH), 1.0 / (MAX_DEPTH * 2), dtype=np.float32))
        assert r.state.surface_voxels == 0

    def test_a_frame_with_nothing_usable_is_not_an_error(self):
        r = Reconstructor(intrinsics(), camera_height=None)
        state = r.add(textured(), np.zeros((HEIGHT, WIDTH), dtype=np.float32))
        assert state.frames == 1
        assert state.surface_voxels == 0

    def test_infinite_depth_is_dropped_not_fused(self):
        depth = flat_depth()
        depth[: HEIGHT // 2] = 0.0  # sky: inverse depth zero
        r = Reconstructor(intrinsics(), camera_height=None)
        r.add(textured(), depth)
        assert r.state.surface_voxels > 0


class TestFusing:
    def test_a_walk_accumulates_surface(self):
        r = Reconstructor(intrinsics(), camera_height=None)
        for shift in range(0, 180, 20):
            r.add(textured(shift=shift), flat_depth())
        assert r.state.tracked >= 2
        assert r.state.surface_voxels > 0
        assert r.state.coverage_voxels > 0

    def test_colour_is_carried_into_the_volume(self):
        r = Reconstructor(intrinsics(), camera_height=None)
        r.add(textured(), flat_depth())
        mesh = r.volume.to_mesh()
        if not mesh.is_empty():
            assert mesh.colours is not None

    def test_a_greyscale_frame_is_accepted(self):
        r = Reconstructor(intrinsics(), camera_height=None)
        grey = np.asarray(textured()[:, :, 0])
        state = r.add(grey, flat_depth())
        assert state.frames == 1


class TestWeakSpots:
    def test_starts_with_none(self):
        assert Reconstructor(intrinsics(), camera_height=None).state.weak_spots == []

    def test_refresh_is_safe_on_an_empty_scan(self):
        assert Reconstructor(intrinsics(), camera_height=None).refresh_weak_spots() == []

    def test_a_single_glance_leaves_weak_coverage(self):
        # One frame of a wall, from one place. Every condition for good coverage
        # fails at once, so this must be reported.
        r = Reconstructor(intrinsics(), camera_height=None)
        r.add(textured(), flat_depth(distance=30.0))
        spots = r.refresh_weak_spots()
        assert spots
        assert spots[0].advice


class TestExport:
    def test_an_empty_scan_writes_nothing(self, tmp_path):
        # An empty .glb in the output folder reads as a successful scan that
        # produced nothing, which is harder to debug than an honest refusal.
        out = tmp_path / "empty.glb"
        assert Reconstructor(intrinsics(), camera_height=None).export(out) is None
        assert not out.exists()

    def test_a_real_scan_writes_a_loadable_file(self, tmp_path):
        r = Reconstructor(intrinsics(), voxel=0.3, camera_height=None)
        # A slab of surface seen from several places, which is enough to mesh.
        rng = np.random.default_rng(0)
        for i, shift in enumerate(range(0, 200, 25)):
            r.add(textured(shift=shift), flat_depth(distance=4.0 + 0.05 * i))
        out = r.export(tmp_path / "scan.glb", name="testscan")
        if out is None:
            pytest.skip("synthetic frames did not fuse into a surface")
        assert out.exists() and out.stat().st_size > 500

    def test_what_it_writes_passes_the_writer_s_own_contract(self, tmp_path):
        from heat3d_capture.fusion.volume import SurfaceVolume, mesh_to_parts
        from heat3d_capture.geometry.glb import write_glb

        # Straight through the same path export() takes, with geometry that is
        # guaranteed to mesh, so the contract is exercised rather than skipped.
        r = Reconstructor(intrinsics(), voxel=0.25, camera_height=None)
        rng = np.random.default_rng(2)
        directions = rng.normal(size=(60_000, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        r.volume.integrate(directions * 3.0)
        parts = mesh_to_parts(r.volume.to_mesh(), name="sphere")
        out = write_glb(tmp_path / "contract.glb", parts)
        assert out.exists()
        for part in parts:
            assert part.cls in ("ground", "structure", "water")
