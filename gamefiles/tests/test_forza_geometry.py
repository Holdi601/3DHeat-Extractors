"""
Decoding a ForzaTech mesh.

The point of these is that the numbers that come out are *world* coordinates.
A decoder that returns a plausible shape in some arbitrary frame is no use for
what this is for — finding the stretch of map a lap of telemetry ran over — so
every test here checks where the geometry lands, not just that it parsed.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_gamefiles.forzatech import (
    UnsupportedForza,
    read_geometry,
    read_layout,
    read_model,
)

from .forza_fixtures import build_modelbin


def grid(n: int = 5, *, size: float = 400.0, origin=(-4096.0, 150.0, 6656.0)):
    """A little patch of terrain: a square grid of quads with some relief."""
    u = np.linspace(0.0, size, n)
    x, z = np.meshgrid(u, u, indexing="ij")
    y = 20.0 * np.sin(x / size * np.pi) * np.cos(z / size * np.pi)
    points = np.stack([x + origin[0], y + origin[1], z + origin[2]], axis=-1).reshape(-1, 3)

    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces += [[a, a + 1, a + n], [a + n, a + 1, a + n + 1]]
    return points, np.array(faces, dtype=np.uint16)


class TestItLandsWhereTheGameHasIt:
    def test_the_positions_come_back_in_world_metres(self, tmp_path):
        points, faces = grid()
        raw = build_modelbin(
            positions=points, faces=faces, scale=(256.0, 256.0, 256.0),
            bias=(-3896.0, 150.0, 6856.0),
        )

        geometry = read_geometry(read_model(tmp_path / "t.modelbin", raw))

        # Quantised to int16 across a 256 m half-width, so a centimetre.
        assert np.allclose(geometry.positions, points, atol=0.02)

    def test_the_triangles_survive(self, tmp_path):
        points, faces = grid()
        raw = build_modelbin(positions=points, faces=faces)

        geometry = read_geometry(read_model(tmp_path / "t.modelbin", raw))

        assert len(geometry.faces) == len(faces)
        assert np.array_equal(geometry.faces, faces.astype(np.int32))

    def test_the_bias_is_what_places_the_tile(self, tmp_path):
        """
        Two tiles with the same local shape and different biases have to come
        back a tile apart. This is the whole mechanism by which a terrain mesh
        knows where it is - there is no instance transform anywhere.
        """
        points, faces = grid(origin=(0.0, 0.0, 0.0))
        here = read_geometry(
            read_model(
                tmp_path / "a", build_modelbin(positions=points, faces=faces, bias=(0.0, 0.0, 0.0))
            )
        )
        shifted = points + np.array([512.0, 0.0, 0.0])
        there = read_geometry(
            read_model(
                tmp_path / "b",
                build_modelbin(positions=shifted, faces=faces, bias=(512.0, 0.0, 0.0)),
            )
        )

        assert there.bounds[0][0] - here.bounds[0][0] == pytest.approx(512.0, abs=0.05)


class TestTheVertexLayout:
    def test_a_second_channel_of_the_same_attribute_is_not_a_second_attribute(
        self, tmp_path
    ):
        """
        The element record splits its first four bytes into two u16s, and the
        second is the channel - a second UV set names the same string as the
        first. Read as one u32 it becomes attribute 65536, and the whole layout
        after it reads as plausible nonsense.
        """
        points, faces = grid()
        raw = build_modelbin(positions=points, faces=faces, extra_attribute=True)
        model = read_model(tmp_path / "t.modelbin", raw)
        layout = read_layout([model.raw(c) for c in model.chunks if c.tag == "VLay"][0])

        names = [(e.semantic, e.channel) for e in layout]

        assert names == [("POSITION", 0), ("TEXCOORD", 0), ("TEXCOORD", 1)]

    def test_extra_attributes_do_not_disturb_the_positions(self, tmp_path):
        points, faces = grid()
        plain = read_geometry(
            read_model(tmp_path / "a", build_modelbin(positions=points, faces=faces))
        )
        rich = read_geometry(
            read_model(
                tmp_path / "b",
                build_modelbin(positions=points, faces=faces, extra_attribute=True),
            )
        )

        assert np.allclose(plain.positions, rich.positions)


class TestItRefusesFormatsItDoesNotKnow:
    def test_an_unknown_position_format(self, tmp_path):
        """
        Guessing here is the expensive mistake: an unknown vertex format
        decoded as the known one returns a mesh, and a wrong mesh looks right
        until it is next to the track.
        """
        points, faces = grid()
        raw = build_modelbin(positions=points, faces=faces, position_format=2)

        with pytest.raises(UnsupportedForza, match="stores positions as format 2"):
            read_geometry(read_model(tmp_path / "t.modelbin", raw))

    def test_an_unknown_index_format(self, tmp_path):
        points, faces = grid()
        raw = build_modelbin(positions=points, faces=faces, index_format=99)

        with pytest.raises(UnsupportedForza, match="stores indices as format 99"):
            read_geometry(read_model(tmp_path / "t.modelbin", raw))

    def test_a_model_with_no_geometry_chunks(self, tmp_path):
        import struct

        head = struct.pack("<4sIIII", b"burG", 1, 0, 20, 0)

        with pytest.raises(UnsupportedForza, match="has no Mesh chunk"):
            read_geometry(read_model(tmp_path / "t.modelbin", head))


# ---------------------------------------------------------------------------
# Submeshes, levels of detail, normals and UVs: what a placed prop carries.


def two_level_prop(variants=None) -> bytes:
    """
    Two unit squares: the first drawn at level 0, the second at level 1, with
    materials 0 and 1. Normals lean differently at every vertex so a swapped
    component shows.
    """
    from .forza_fixtures import build_textured_model

    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    positions = np.concatenate([square, square + [2, 0, 0]])
    normals = np.array(
        [[0.6, 0.0, 0.8], [0.0, 0.6, 0.8], [0.0, 0.0, 1.0], [-0.6, 0.0, 0.8]] * 2, dtype=np.float64
    )
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]] * 2, dtype=np.float64) * [1.0, 0.5]
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    return build_textured_model(
        positions=positions,
        faces=faces,
        normals=normals,
        uvs=uvs,
        submeshes=[(variants or 0, 0x2, 0, 2), (1, 0x4, 2, 2)],
    ), normals, uvs


class TestSubmeshes:
    def test_each_triangle_knows_its_level_and_material(self):
        data, _n, _uv = two_level_prop()
        geometry = read_geometry(read_model("prop.modelbin", data), lod=None)
        assert geometry.lods.tolist() == [0x2, 0x2, 0x4, 0x4]
        assert geometry.material_index.tolist() == [0, 0, 1, 1]

    def test_a_level_keeps_only_its_own_triangles(self):
        """Levels overlap in space; exporting two at once doubles every surface."""
        data, _n, _uv = two_level_prop()
        finest = read_geometry(read_model("prop.modelbin", data), lod=0)
        coarser = read_geometry(read_model("prop.modelbin", data), lod=1)
        assert finest.faces.tolist() == [[0, 1, 2], [0, 2, 3]]
        assert coarser.faces.tolist() == [[4, 5, 6], [4, 6, 7]]

    def test_a_longer_record_moves_every_field_by_its_extra_length(self):
        """
        A submesh with several material variants - a billboard's five prints -
        opens with a longer table, and every field after it moves back by the
        extra length. Read at the standard offsets it gives nonsense.
        """
        prints = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11)]
        data, _n, _uv = two_level_prop(variants=prints)
        geometry = read_geometry(read_model("prop.modelbin", data), lod=None)
        assert geometry.lods.tolist() == [0x2, 0x2, 0x4, 0x4]
        # The first variant is the submesh's material; the table is kept whole.
        assert geometry.material_index.tolist() == [1, 1, 1, 1]
        assert geometry.variants[0] == tuple(prints)
        assert geometry.variants[1] == ((1, 0xFFFF),)
        assert geometry.submesh.tolist() == [0, 0, 1, 1]

    def test_a_standard_record_has_one_variant_with_its_winter_material(self):
        data, _n, _uv = two_level_prop(variants=[(0, 6)])
        geometry = read_geometry(read_model("prop.modelbin", data), lod=None)
        assert geometry.variants[0] == ((0, 6),)


class TestNormalsAndUvs:
    def test_the_normal_x_is_the_positions_fourth_short(self):
        data, normals, _uv = two_level_prop()
        geometry = read_geometry(read_model("prop.modelbin", data), lod=None)
        assert np.allclose(geometry.normals, normals, atol=1e-3)

    def test_uvs_are_unsigned_shorts_over_the_whole_range(self):
        data, _n, uvs = two_level_prop()
        geometry = read_geometry(read_model("prop.modelbin", data), lod=None)
        assert np.allclose(geometry.uvs[0], uvs, atol=1e-4)

    def test_positions_are_still_in_metres(self):
        data, _n, _uv = two_level_prop()
        geometry = read_geometry(read_model("prop.modelbin", data), lod=None)
        assert np.allclose(geometry.positions[6], [3, 1, 0], atol=1e-3)
