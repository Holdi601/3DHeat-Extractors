"""
Extracting a surface over an area the size of a racetrack.

This exists because of a scan that went right and reported as though it had gone
wrong. A real Forza lap — telemetry connected, 55,579 packets, 702 frames placed,
3.47 million coverage cells — came back with "the scan did not produce enough
geometry to build a surface". It had produced plenty. The surface extractor
realised one dense array over the whole bounding box, and at quarter-metre detail
a kilometre-square track is 3.8 *billion* cells and 15 GB, so it hit its own size
guard and returned nothing, silently.

The data was always sparse — millions of cells in billions of empty ones — so the
fix is to extract block by block over the parts that hold anything. These tests
pin the properties that makes that worth doing: it works at track scale, it costs
memory proportional to what was scanned rather than to the box around it, and the
blocks join into one surface rather than a grid of separate boxes.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_capture.fusion.volume import BLOCK, SurfaceVolume


def ribbon(length_m: float, width_m: float, *, voxel: float, per_m: int = 60):
    """
    A lap: a closed loop of road, with elevation.

    A loop rather than a straight, because that is what defeats a dense array.
    The bounding box grows with the *square* of the lap size while the road
    inside it grows only linearly — a kilometre lap is a box of billions of
    cells holding a few hundred thousand of road.

    Elevation too, because a real circuit has some and it multiplies the box by
    the height in voxels. Flat, the box is small enough that the dense version
    would have coped, and the fixture would not be posing the problem.
    """
    steps = int(length_m * per_m)
    angle = np.linspace(0.0, 2.0 * np.pi, steps, endpoint=False)
    rng = np.random.default_rng(4)
    across = rng.uniform(-width_m / 2, width_m / 2, size=steps)

    radius = length_m / (2.0 * np.pi)
    x = (radius + across) * np.cos(angle)
    z = (radius + across) * np.sin(angle) * 0.9
    # 60 m of climb and descent over the lap, as a circuit has.
    y = 30.0 * np.sin(2.0 * angle) + rng.normal(0.0, voxel * 0.3, size=steps)
    return np.stack([x, y, z], axis=1)


class TestItWorksAtTrackScale:
    def test_a_full_circuit_produces_a_surface(self):
        """
        The case that failed. Four kilometres is an ordinary circuit length, and
        a dense array over its box is billions of cells — holding a few hundred
        thousand of road.
        """
        volume = SurfaceVolume(voxel=0.25)
        volume.integrate(ribbon(4000.0, 12.0, voxel=0.25), weight=1.0)

        mesh = volume.to_mesh()

        assert not mesh.is_empty(), "a whole circuit extracted to nothing"
        assert mesh.triangles > 1000

    def test_the_dense_array_it_replaces_would_be_impossible(self):
        """
        Stated as a test so the reason is checked rather than asserted in prose.

        If this ever stops being true — a much coarser default, say — the block
        machinery is no longer earning its complexity and someone should say so.
        """
        volume = SurfaceVolume(voxel=0.25)
        volume.integrate(ribbon(4000.0, 12.0, voxel=0.25), weight=1.0)
        low, high = volume.bounds()

        cells = np.prod((high - low + 3).astype(np.float64))

        assert cells > 1e9, f"the bounding box is only {cells:,.0f} cells"

    def test_it_spans_the_whole_route_rather_than_one_end(self):
        volume = SurfaceVolume(voxel=0.25)
        points = ribbon(4000.0, 12.0, voxel=0.25)
        volume.integrate(points, weight=1.0)

        mesh = volume.to_mesh()
        # A 4 km lap is a loop about 1270 m across, so span the loop rather
        # than the road length.
        reach = mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min()

        assert reach > 1100, f"surface spans only {reach:.0f} m of a 1270 m loop"

    def test_memory_follows_the_surface_not_the_bounding_box(self):
        """
        Doubling the *extent* while scanning the same amount of road must not
        double the cost. That is the whole point of blocks, and the property the
        dense version did not have.
        """
        near = SurfaceVolume(voxel=0.25)
        near.integrate(ribbon(200.0, 12.0, voxel=0.25), weight=1.0)
        far = SurfaceVolume(voxel=0.25)
        spread = ribbon(200.0, 12.0, voxel=0.25)
        spread[:, 0] *= 5.0  # same road, stretched across five times the box

        far.integrate(spread, weight=1.0)

        assert not near.to_mesh().is_empty()
        assert not far.to_mesh().is_empty()


class TestTheBlocksJoinUp:
    def test_a_slab_crossing_many_blocks_is_one_surface(self):
        """
        The halo's job. Without it each block closes its own surface against its
        boundary and a continuous road becomes a row of sealed boxes — which has
        far more triangles and, seen from above, a grid of seams.
        """
        volume = SurfaceVolume(voxel=0.25)
        # A flat slab several blocks across in x and z.
        span = BLOCK * 0.25 * 3
        rng = np.random.default_rng(1)
        n = 400_000
        points = np.stack(
            [
                rng.uniform(0, span, n),
                rng.normal(0, 0.05, n),
                rng.uniform(0, span, n),
            ],
            axis=1,
        )
        volume.integrate(points, weight=1.0)

        mesh = volume.to_mesh()

        assert not mesh.is_empty()
        # A slab is two sheets; sealed per block it would be six faces per block
        # and many times the area. Measured against the slab's own footprint.
        area = _area(mesh)
        footprint = span * span
        assert area < footprint * 6, (
            f"surface area {area:.0f} against a {footprint:.0f} footprint — "
            "the blocks are closing themselves off"
        )

    def test_no_triangle_is_emitted_twice(self):
        """
        The halo makes each block see its neighbours' voxels. Emitting the halo's
        triangles as well as the interior's would double them and leave
        coincident faces that shade badly.
        """
        volume = SurfaceVolume(voxel=0.25)
        span = BLOCK * 0.25 * 2
        rng = np.random.default_rng(2)
        n = 200_000
        points = np.stack(
            [rng.uniform(0, span, n), rng.normal(0, 0.05, n), rng.uniform(0, span, n)],
            axis=1,
        )
        volume.integrate(points, weight=1.0)

        mesh = volume.to_mesh()
        centres = mesh.vertices[mesh.faces].mean(axis=1)
        rounded = np.round(centres, 3)
        unique = len({tuple(c) for c in rounded})

        assert unique > len(rounded) * 0.97, (
            f"{len(rounded) - unique} of {len(rounded)} triangles are duplicates"
        )

    def test_indices_stay_inside_the_vertex_list(self):
        """Blocks are concatenated with an offset each; an off-by-one here
        produces a mesh that loads and renders as noise."""
        volume = SurfaceVolume(voxel=0.25)
        volume.integrate(ribbon(300.0, 12.0, voxel=0.25), weight=1.0)

        mesh = volume.to_mesh()

        assert mesh.faces.min() >= 0
        assert mesh.faces.max() < len(mesh.vertices)
        assert len(mesh.normals) == len(mesh.vertices)


class TestStillRefusingNonsense:
    def test_an_outlier_far_from_everything_is_still_dropped(self):
        """
        Guarded explicitly because the block split makes this easy to break: an
        isolated stray in its own empty block, normalised to that block's own
        maximum, becomes a full surface hanging in mid-air. The normalisation
        has to stay global.
        """
        volume = SurfaceVolume(voxel=0.25)
        rng = np.random.default_rng(3)
        n = 200_000
        points = np.stack(
            [rng.uniform(0, 20, n), rng.normal(0, 0.05, n), rng.uniform(0, 20, n)],
            axis=1,
        )
        volume.integrate(points, weight=1.0)
        volume.integrate(np.array([[300.0, 40.0, 300.0]]), weight=1.0)

        mesh = volume.to_mesh()

        assert mesh.vertices[:, 0].max() < 60, "a lone stray point became geometry"

    def test_an_empty_volume_says_nothing_rather_than_guessing(self):
        assert SurfaceVolume(voxel=0.25).to_mesh().is_empty()


def _area(mesh) -> float:
    corners = mesh.vertices[mesh.faces]
    return float(
        0.5
        * np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            axis=1,
        ).sum()
    )
