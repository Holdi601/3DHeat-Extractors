"""
Cutting a piece of terrain out by world coordinates.

The workflow this serves is: drive a lap, keep the telemetry, ask for the map
where the lap was. So what these test is the translation from *place* to
*entries* — which is where the map-sized mistakes live. A decoder that reads a
mesh wrongly produces a visibly broken shape; a lookup that reads the grid
wrongly produces perfectly good terrain from somewhere else entirely, and
nothing about it looks wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_gamefiles.forzaterrain import (
    NAME_STEP,
    TILE_METRES,
    Tile,
    choose_variant,
    clip,
    extract,
    index_tiles,
)
from heat3d_gamefiles.forzatech import Geometry
from heat3d_gamefiles.minizip import LZ4, Contents, MiniZip

from .forza_fixtures import build_contents, build_minizip, build_modelbin
from .test_forza_geometry import grid


def tile_name(x: int, z: int, variant: str = "", cluster: int = 0) -> str:
    suffix = f"_{variant}" if variant else ""
    return (
        f"scene\\tbheightfield\\autoterrain_x{x}_z{z}{suffix}"
        f"_cluster{cluster:03}.i.modelbin"
    )


def make_track(tmp_path, tiles):
    """
    An archive holding one small mesh per named tile, each sitting where its
    name says it should.
    """
    names, blobs = [], []
    for x, z, variant in tiles:
        origin = (x / NAME_STEP * TILE_METRES, 100.0, z / NAME_STEP * TILE_METRES)
        points, faces = grid(4, size=TILE_METRES, origin=origin)
        names.append(tile_name(x, z, variant))
        blobs.append(
            build_modelbin(
                positions=points,
                faces=faces,
                scale=(512.0, 512.0, 512.0),
                bias=(origin[0] + 256.0, origin[1], origin[2] + 256.0),
            )
        )
    archive = build_minizip(
        tmp_path / "GeoChunk0.minizip", blobs, codecs=[LZ4] * len(blobs)
    )
    contents = build_contents(tmp_path / "ChunkContentsMiniZip0.txt", names)
    return MiniZip(archive), Contents(contents)


class TestTheGrid:
    def test_a_tile_is_512_metres_and_its_name_steps_by_1023(self):
        """
        The trap in the whole file. The two numbers in a tile's name step by
        1023, the tile is 512 metres across, and taking the name as metres
        misses by a factor of two - which does not fail, it returns terrain
        from the wrong half of the map.
        """
        assert Tile(x=0, z=0, variant="").origin == (0.0, 0.0)
        assert Tile(x=-8184, z=13299, variant="").origin == (-4096.0, 6656.0)
        assert Tile(x=NAME_STEP, z=NAME_STEP, variant="").origin == (512.0, 512.0)

    def test_a_tile_covers_the_square_after_its_origin(self):
        tile = Tile(x=NAME_STEP * 2, z=0, variant="")

        assert tile.touches((1024.0, 0.0), (1100.0, 100.0))
        assert tile.touches((900.0, -50.0), (1030.0, 10.0))
        assert not tile.touches((0.0, 0.0), (1023.0, 100.0))
        assert not tile.touches((1536.1, 0.0), (2000.0, 100.0))

    def test_the_plain_variant_wins(self, tmp_path):
        """
        A tile ships up to three times over - plain, `_cb` and `_ul` - and the
        three are the same ground. Taking all of them triples the triangle
        count and leaves three surfaces in the same place.
        """
        tiles = index_tiles(
            Contents(
                build_contents(
                    tmp_path / "c.txt",
                    [tile_name(0, 0), tile_name(0, 0, "cb"), tile_name(0, 0, "ul")],
                )
            )
        )

        assert choose_variant(tiles, 0, 0).variant == ""

    def test_a_tile_that_only_ships_as_a_variant_is_still_found(self, tmp_path):
        tiles = index_tiles(
            Contents(
                build_contents(tmp_path / "c.txt", [tile_name(0, 0, "ul")])
            )
        )

        assert choose_variant(tiles, 0, 0).variant == "ul"

    def test_clusters_of_one_tile_are_gathered(self, tmp_path):
        tiles = index_tiles(
            Contents(
                build_contents(
                    tmp_path / "c.txt",
                    [tile_name(0, 0, cluster=k) for k in range(5)],
                )
            )
        )

        assert len(tiles[(0, 0, "")].entries) == 5


class TestClipping:
    def test_a_triangle_reaching_into_the_box_is_kept(self):
        """
        Any corner inside, not all three. A triangle with one corner in is part
        of the surface at the boundary, and requiring all three leaves a fringe
        of holes along every edge of the cut - most visible exactly where
        someone is looking, since the box was drawn around what they wanted.
        """
        points = np.array(
            [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 0.0, 100.0]], dtype=np.float32
        )
        geometry = Geometry(positions=points, faces=np.array([[0, 1, 2]], dtype=np.int32))

        kept, faces, _mask, _used = clip(geometry, (-10.0, -10.0), (10.0, 10.0))

        assert len(faces) == 1
        assert len(kept) == 3

    def test_a_triangle_that_swallows_the_box_is_kept(self):
        """
        No corner of this triangle is inside the box; the box is inside the
        triangle. Terrain ships LOD meshes whose triangles are tens of metres
        across, so asking for a short stretch of road can land entirely within
        one - and a corner test returns an empty region for a place that is
        plainly covered.
        """
        points = np.array(
            [[-500.0, 0.0, -500.0], [500.0, 0.0, -500.0], [0.0, 0.0, 500.0]],
            dtype=np.float32,
        )
        geometry = Geometry(positions=points, faces=np.array([[0, 1, 2]], dtype=np.int32))

        _kept, faces, _mask, _used = clip(geometry, (-5.0, -5.0), (5.0, 5.0))

        assert len(faces) == 1

    def test_a_triangle_entirely_outside_goes(self):
        points = np.array(
            [[500.0, 0.0, 500.0], [600.0, 0.0, 500.0], [500.0, 0.0, 600.0]],
            dtype=np.float32,
        )
        geometry = Geometry(positions=points, faces=np.array([[0, 1, 2]], dtype=np.int32))

        _kept, faces, _mask, _used = clip(geometry, (-10.0, -10.0), (10.0, 10.0))

        assert len(faces) == 0

    def test_height_is_never_clipped(self):
        """
        The box is drawn in the plane on purpose. Bounding a lap's altitude and
        cutting to it removes the ground under a hill the road climbs, which is
        the part of the map a driver most wants to see.
        """
        points = np.array(
            [[0.0, -800.0, 0.0], [1.0, 900.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        geometry = Geometry(positions=points, faces=np.array([[0, 1, 2]], dtype=np.int32))

        _kept, faces, _mask, _used = clip(geometry, (-10.0, -10.0), (10.0, 10.0))

        assert len(faces) == 1

    def test_the_indices_are_renumbered_to_what_survived(self):
        points, faces = grid(6, size=100.0, origin=(0.0, 0.0, 0.0))
        geometry = Geometry(
            positions=points.astype(np.float32), faces=faces.astype(np.int32)
        )

        kept, cut, _mask, _used = clip(geometry, (-1.0, -1.0), (50.0, 50.0))

        assert len(cut)
        assert cut.max() < len(kept)


class TestExtracting:
    def test_it_takes_the_tiles_the_box_touches_and_no_others(self, tmp_path):
        archive, contents = make_track(
            tmp_path, [(0, 0, ""), (NAME_STEP, 0, ""), (NAME_STEP * 8, 0, "")]
        )

        with archive:
            out = extract(archive, contents, (10.0, 10.0), (600.0, 100.0))

        assert out.tiles == 2, "the distant tile was read, or a near one was not"
        assert len(out.faces)

    def test_the_merged_indices_stay_inside_the_merged_vertices(self, tmp_path):
        """
        Concatenating meshes means adding an offset per block, and an off-by-one
        produces a file that loads and renders as noise rather than failing.
        """
        archive, contents = make_track(
            tmp_path, [(0, 0, ""), (NAME_STEP, 0, ""), (0, NAME_STEP, "")]
        )

        with archive:
            out = extract(archive, contents, (-100.0, -100.0), (1100.0, 1100.0))

        assert out.faces.min() >= 0
        assert out.faces.max() < len(out.positions)

    def test_the_result_is_where_the_box_was(self, tmp_path):
        archive, contents = make_track(tmp_path, [(0, 0, ""), (NAME_STEP * 4, 0, "")])

        with archive:
            out = extract(archive, contents, (2048.0, 0.0), (2560.0, 512.0))

        low, high = out.bounds
        assert low[0] >= 2000.0
        assert high[0] <= 2600.0

    def test_a_box_over_empty_map_returns_nothing_rather_than_raising(self, tmp_path):
        archive, contents = make_track(tmp_path, [(0, 0, "")])

        with archive:
            out = extract(archive, contents, (50_000.0, 50_000.0), (51_000.0, 51_000.0))

        assert len(out.faces) == 0
        assert out.tiles == 0

    def test_an_inverted_box_is_a_mistake_worth_reporting(self, tmp_path):
        archive, contents = make_track(tmp_path, [(0, 0, "")])

        with archive:
            with pytest.raises(ValueError, match="is empty"):
                extract(archive, contents, (100.0, 100.0), (0.0, 0.0))

    def test_one_unreadable_mesh_does_not_cost_the_region(self, tmp_path):
        """
        Tens of thousands of terrain meshes ship in one map archive. Any of them may use a
        vertex format this does not read, and stopping on the first would mean
        a region is all or nothing. The count is reported so that losing most
        of one says so.
        """
        archive, contents = make_track(tmp_path, [(0, 0, ""), (NAME_STEP, 0, "")])
        broken = build_contents(
            tmp_path / "ChunkContentsMiniZip0.txt",
            [tile_name(0, 0), tile_name(NAME_STEP, 0)],
        )

        with archive:
            # Corrupt the first entry's payload in place, leaving the index alone.
            original = archive.read
            archive.read = lambda i: b"burG" + bytes(16) if i == 0 else original(i)
            out = extract(archive, Contents(broken), (-100.0, -100.0), (1100.0, 100.0))

        assert out.skipped == 1
        assert out.meshes == 1
        assert len(out.faces), "the readable tile was lost with the unreadable one"
