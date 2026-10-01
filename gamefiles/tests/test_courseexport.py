"""
Turning an extraction into textured parts: tile UVs and the road/verge split.

The orientation is the thing to protect. A tile's image has row 0 at its north
edge (largest z) and column 0 at its west edge (smallest x) - found by laying
recorded laps over it every way a square can be flipped or turned - and a UV
that gets either backwards puts the asphalt in the grass without any error
anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_gamefiles.courseexport import (
    Baked,
    planar_uvs,
    plan_parts,
    tile_origin,
    vertex_tiles,
)
from heat3d_gamefiles.forzaterrain import NAME_STEP, TILE_METRES, Extraction

#: The tile whose name says x=1023, z=2046: its corner is at 512 m, 1024 m.
KEY = (NAME_STEP, 2 * NAME_STEP)


def one_tile(surfaces: list[str], *, points=None) -> Extraction:
    """Triangles in one tile, laid out along its west-to-east middle."""
    x0, z0 = tile_origin(KEY)
    rows = []
    faces = []
    for i, _ in enumerate(surfaces):
        x = x0 + 20.0 + i * 40.0
        z = z0 + 256.0
        base = len(rows)
        rows += [[x, 0, z], [x + 10, 0, z], [x, 0, z + 10]]
        faces.append([base, base + 1, base + 2])
    return Extraction(
        positions=np.asarray(points if points is not None else rows, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        surfaces=np.asarray(surfaces, dtype=object),
        tile_of=np.zeros(len(surfaces), dtype=np.int32),
        tile_keys=[KEY],
    )


class TestTileUvs:
    def test_a_tiles_corner_follows_its_name(self):
        assert tile_origin(KEY) == (512.0, 1024.0)

    def test_u_runs_east_and_v_runs_south(self):
        x0, z0 = tile_origin(KEY)
        extraction = Extraction(
            positions=np.array(
                [[x0, 0, z0], [x0 + TILE_METRES, 0, z0], [x0, 0, z0 + TILE_METRES]], dtype=np.float32
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
            tile_of=np.zeros(1, dtype=np.int32),
            tile_keys=[KEY],
        )
        uv = planar_uvs(extraction)
        # South-west corner: left, bottom row. East: u = 1. North: top row, v = 0.
        assert np.allclose(uv, [[0, 1], [1, 1], [0, 0]])

    def test_a_vertex_a_hair_outside_its_tile_stays_in_its_image(self):
        x0, z0 = tile_origin(KEY)
        extraction = Extraction(
            positions=np.array([[x0 - 0.01, 0, z0], [x0 + 1, 0, z0], [x0, 0, z0 + 1]], dtype=np.float32),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
            tile_of=np.zeros(1, dtype=np.int32),
            tile_keys=[KEY],
        )
        assert planar_uvs(extraction).min() >= 0.0

    def test_a_vertex_shared_between_tiles_is_refused(self):
        """Its UV would point into one image or the other, silently."""
        extraction = Extraction(
            positions=np.zeros((4, 3), dtype=np.float32),
            faces=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
            tile_of=np.array([0, 1], dtype=np.int32),
            tile_keys=[KEY, (0, 0)],
        )
        with pytest.raises(ValueError, match="shared between tiles"):
            vertex_tiles(extraction)


class TestThePlan:
    def baked(self, asphalt_west: float, asphalt_east: float, size: int = 64) -> dict:
        """A tile whose asphalt mask differs between its west and east halves."""
        mask = np.empty((size, size), dtype=np.float32)
        mask[:, : size // 2] = asphalt_west
        mask[:, size // 2 :] = asphalt_east
        image = np.zeros((size, size, 3), dtype=np.uint8)
        return {KEY: Baked(image=image, asphalt=mask)}

    def test_road_strips_split_into_asphalt_and_verge_by_the_mask(self):
        """
        The strips are wider than the carriageway; the mask says where the
        asphalt ends. West of the middle is asphalt here, east is not.
        """
        extraction = one_tile(["road", "road", "road", "road", "road", "road", "road", "road"])
        plan = plan_parts(extraction, self.baked(1.0, 0.0))
        names = {p.name: len(p.faces) for p in plan.parts}
        x, z = KEY
        assert names[f"ground:road x{x} z{z}"] == 6
        assert names[f"ground:verge x{x} z{z}"] == 2
        assert plan.textured

    def test_every_triangle_lands_in_exactly_one_part(self):
        extraction = one_tile(["road", "terrain", "markings", "road", "water"])
        plan = plan_parts(extraction, self.baked(1.0, 1.0))
        faces = np.concatenate([p.faces for p in plan.parts])
        assert sorted(map(tuple, faces.tolist())) == sorted(map(tuple, extraction.faces.tolist()))

    def test_the_ground_parts_carry_the_tile_image_and_decals_do_not(self):
        extraction = one_tile(["road", "terrain", "markings"])
        baked = self.baked(1.0, 1.0)
        plan = plan_parts(extraction, baked)
        by_name = {p.name.split(" ")[0]: p for p in plan.parts}
        assert by_name["ground:road"].texture is baked[KEY].image
        assert by_name["ground:terrain"].texture is baked[KEY].image
        assert by_name["ground:markings"].texture is None

    def test_without_maps_the_strips_stay_road_and_nothing_is_textured(self):
        extraction = one_tile(["road", "road", "terrain"])
        plan = plan_parts(extraction, None)
        assert not plan.textured
        assert plan.counts == {"road": 2, "terrain": 1}


class TestPaint:
    def test_markings_get_their_own_vertices_lifted_off_the_road(self):
        from heat3d_gamefiles.courseexport import DECAL_LIFT, lift_decals
        from heat3d_gamefiles.glb import Part

        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]], dtype=np.float32)
        normals = np.tile([0.0, 1.0, 0.0], (4, 1)).astype(np.float32)
        road = Part(name="ground:road x0 z0", faces=np.array([[0, 1, 2]]))
        paint = Part(name="ground:markings", faces=np.array([[1, 3, 2]]))
        parts, extra, extra_normals = lift_decals([road, paint], positions, normals)
        assert parts[0] is road
        assert len(extra) == 3 and np.allclose(extra[:, 1], DECAL_LIFT)
        assert parts[1].faces.min() >= len(positions)
        # The road's own corners did not move.
        assert np.allclose(positions[:, 1], 0.0)
        assert np.allclose(extra_normals, [0, 1, 0])

    def test_nothing_to_lift_is_nothing_added(self):
        from heat3d_gamefiles.courseexport import lift_decals
        from heat3d_gamefiles.glb import Part

        road = Part(name="ground:road x0 z0", faces=np.array([[0, 1, 2]]))
        parts, extra, _ = lift_decals([road], np.zeros((3, 3), np.float32), np.zeros((3, 3), np.float32))
        assert parts == [road] and len(extra) == 0
