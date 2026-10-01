"""
Tests for fusion and meshing.

Driven by analytic shapes — a sphere, a plane, a box — because those have known
answers. A reconstruction of a real capture can only be eyeballed; a sphere of
radius 3 either comes back with radius 3 or it does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_capture.fusion.volume import (
    SURFACE_LEVEL,
    Mesh,
    SurfaceVolume,
    classify,
    mesh_to_parts,
)


def sphere(radius: float = 3.0, n: int = 40_000, centre=(0, 0, 0), seed: int = 0) -> np.ndarray:
    """Points on a sphere, which has a known radius to measure back."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions * radius + np.asarray(centre, dtype=float)


def plane(size: float = 8.0, y: float = 0.0, n: int = 40_000, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.uniform(-size, size, n), np.full(n, y), rng.uniform(-size, size, n)], axis=1
    )


class TestIntegrate:
    def test_records_voxels(self):
        volume = SurfaceVolume(voxel=0.25)
        assert volume.integrate(sphere()) > 0
        assert len(volume) > 0

    def test_the_same_points_twice_adds_weight_not_voxels(self):
        volume = SurfaceVolume(voxel=0.25)
        points = sphere()
        first = volume.integrate(points)
        volume.integrate(points)
        assert len(volume) == first

    def test_drops_non_finite_points(self):
        volume = SurfaceVolume(voxel=0.25)
        points = sphere(n=500)
        points[0] = [np.inf, 0, 0]
        points[1] = [0, np.nan, 0]
        assert volume.integrate(points) > 0

    def test_empty_input_is_not_an_error(self):
        volume = SurfaceVolume(voxel=0.25)
        assert volume.integrate(np.zeros((0, 3))) == 0
        assert volume.to_mesh().is_empty()

    def test_rejects_a_nonsense_voxel_size(self):
        with pytest.raises(ValueError, match="positive"):
            SurfaceVolume(voxel=-1)

    def test_bounds_follow_the_data(self):
        volume = SurfaceVolume(voxel=0.5)
        volume.integrate(sphere(radius=4.0))
        low, high = volume.bounds()
        # Radius 4 at half-metre voxels spans roughly sixteen cells.
        assert (high - low).min() >= 14


class TestMeshing:
    def test_a_sphere_comes_back_as_a_sphere(self):
        volume = SurfaceVolume(voxel=0.2)
        volume.integrate(sphere(radius=3.0, n=120_000))
        mesh = volume.to_mesh()
        assert not mesh.is_empty()
        radii = np.linalg.norm(mesh.vertices, axis=1)
        # Within a voxel or so of the truth, which is all this method promises.
        assert radii.mean() == pytest.approx(3.0, abs=0.35)
        assert radii.std() < 0.35

    def test_a_sphere_lands_where_it_was_put(self):
        volume = SurfaceVolume(voxel=0.2)
        volume.integrate(sphere(radius=2.0, centre=(10.0, -4.0, 6.0), n=120_000))
        mesh = volume.to_mesh()
        assert mesh.vertices.mean(axis=0) == pytest.approx([10.0, -4.0, 6.0], abs=0.4)

    def test_produces_a_closed_surface_not_a_point_cloud(self):
        volume = SurfaceVolume(voxel=0.25)
        volume.integrate(sphere(n=80_000))
        mesh = volume.to_mesh()
        assert mesh.triangles > 100
        assert mesh.faces.max() < len(mesh.vertices)

    def test_stray_points_are_dropped_rather_than_becoming_spikes(self):
        volume = SurfaceVolume(voxel=0.25)
        volume.integrate(sphere(radius=3.0, n=80_000), weight=1.0)
        # A handful of outliers far away, as a depth network produces at
        # silhouette edges.
        volume.integrate(np.array([[40.0, 40.0, 40.0], [-38.0, 12.0, 5.0]]), weight=1.0)
        mesh = volume.to_mesh()
        assert np.abs(mesh.vertices).max() < 12.0

    def test_smoothing_reduces_surface_roughness(self):
        # Measured as excess surface area against the analytic sphere. A noisy
        # field wrinkles the surface, and a wrinkled surface has more area for
        # the same shape — so the ratio is a direct measure of how rough it came
        # out, with a known correct answer to compare against.
        #
        # Neither figure approaches 1.0, and should not: the shell is one voxel
        # thick, so marching cubes finds an inner and an outer surface and the
        # floor for this ratio is about 2. What matters is that smoothing moves
        # it down rather than up.
        volume = SurfaceVolume(voxel=0.3)
        volume.integrate(sphere(radius=3.0, n=120_000))

        def excess(mesh) -> float:
            corners = mesh.vertices[mesh.faces]
            area = 0.5 * np.linalg.norm(
                np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
            ).sum()
            return float(area / (4 * np.pi * 3.0**2))

        rough = volume.to_mesh(smooth=False)
        smooth = volume.to_mesh(smooth=True)
        assert excess(smooth) < excess(rough)
        assert smooth.triangles < rough.triangles

    def test_too_little_evidence_produces_nothing_rather_than_noise(self):
        volume = SurfaceVolume(voxel=0.25)
        volume.integrate(sphere(n=200), weight=0.01)
        assert volume.to_mesh().is_empty()

    def test_carries_colour_when_given_it(self):
        volume = SurfaceVolume(voxel=0.25)
        points = sphere(n=60_000)
        red = np.tile(np.array([[1.0, 0.1, 0.1]], dtype=np.float32), (len(points), 1))
        volume.integrate(points, colours=red)
        mesh = volume.to_mesh()
        assert mesh.colours is not None
        assert mesh.colours.shape == mesh.vertices.shape
        assert mesh.colours[:, 0].mean() > mesh.colours[:, 1].mean()


class TestClassify:
    def test_a_floor_is_ground(self):
        volume = SurfaceVolume(voxel=0.25)
        # A slab, so the surface has a genuine top and bottom to face.
        for y in np.linspace(-0.4, 0.0, 5):
            volume.integrate(plane(size=6.0, y=float(y), n=40_000))
        mesh = volume.to_mesh()
        labels = classify(mesh)
        assert (labels == "ground").mean() > 0.35

    def test_a_wall_is_structure(self):
        volume = SurfaceVolume(voxel=0.25)
        rng = np.random.default_rng(4)
        for x in np.linspace(-0.4, 0.0, 5):
            volume.integrate(
                np.stack(
                    [
                        np.full(40_000, x),
                        rng.uniform(0, 6, 40_000),
                        rng.uniform(-6, 6, 40_000),
                    ],
                    axis=1,
                )
            )
        labels = classify(volume.to_mesh())
        assert (labels == "structure").mean() > 0.6

    def test_never_claims_water(self):
        # Flat ground and water are indistinguishable by shape, and a wrong
        # `water:` label renders a courtyard as a lake.
        volume = SurfaceVolume(voxel=0.25)
        for y in np.linspace(-0.4, 0.0, 5):
            volume.integrate(plane(y=float(y)))
        assert "water" not in set(classify(volume.to_mesh()))

    def test_an_empty_mesh_classifies_to_nothing(self):
        empty = Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros((0, 3), np.float32))
        assert classify(empty).size == 0


class TestParts:
    def test_splits_into_labelled_parts_the_writer_accepts(self):
        volume = SurfaceVolume(voxel=0.25)
        for y in np.linspace(-0.4, 0.0, 5):
            volume.integrate(plane(size=6.0, y=float(y), n=40_000))
        parts = mesh_to_parts(volume.to_mesh(), name="testlevel")
        assert parts
        assert {p.cls for p in parts} <= {"ground", "structure"}
        for part in parts:
            assert part.label.split(":")[0] in ("ground", "structure")

    def test_each_part_carries_only_the_vertices_it_uses(self):
        volume = SurfaceVolume(voxel=0.3)
        volume.integrate(sphere(radius=3.0, n=100_000))
        mesh = volume.to_mesh()
        parts = mesh_to_parts(mesh)
        for part in parts:
            assert int(part.indices.max()) < len(part.positions)
            assert len(part.positions) <= len(mesh.vertices)

    def test_the_parts_round_trip_through_the_glb_writer(self, tmp_path):
        # The end of the whole pipeline: fused geometry must be writable as the
        # file the viewer reads, with no special handling.
        from heat3d_capture.geometry.glb import write_glb

        volume = SurfaceVolume(voxel=0.25)
        volume.integrate(sphere(radius=4.0, n=120_000))
        parts = mesh_to_parts(volume.to_mesh(), name="sphere")
        out = write_glb(tmp_path / "scan.glb", parts)
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_an_empty_mesh_produces_no_parts(self):
        empty = Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros((0, 3), np.float32))
        assert mesh_to_parts(empty) == []


class TestSurfaceLevel:
    def test_the_level_sits_between_empty_and_solid(self):
        assert 0.0 < SURFACE_LEVEL < 1.0
