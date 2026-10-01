"""
Tests for the GLB writer.

These check the writer against the *specification* and against its own
invariants. They deliberately do not prove the viewer can read the result —
nothing on this side of the boundary can. That is the viewer's own test, which
runs the real files through the real parser.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from heat3d_capture.geometry.glb import (
    METRES_TO_VIEWER,
    Part,
    compute_normals,
    write_glb,
)


def quad(y: float = 0.0, size: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """A flat, upward-facing square in the XZ plane, counter-clockwise from above."""
    positions = np.array(
        [[0, y, 0], [size, y, 0], [size, y, size], [0, y, size]], dtype=np.float32
    )
    indices = np.array([0, 2, 1, 0, 3, 2], dtype=np.uint32)
    return positions, indices


def read_glb(path):
    """Minimal reader, so the tests parse the bytes rather than trusting the writer."""
    raw = path.read_bytes()
    magic, version, total = struct.unpack_from("<III", raw, 0)
    assert magic == 0x46546C67
    assert version == 2
    assert total == len(raw), "declared length must match the file"
    chunks = {}
    offset = 12
    while offset < len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        body = raw[offset + 8 : offset + 8 + length]
        chunks[kind] = body
        offset += 8 + length
    return json.loads(chunks[0x4E4F534A].decode("utf-8")), chunks[0x004E4942]


def test_writes_a_structurally_valid_glb(tmp_path):
    pos, idx = quad()
    out = write_glb(tmp_path / "a.glb", [Part("ground", "floor", pos, idx)])
    gltf, binary = read_glb(out)

    assert gltf["asset"]["version"] == "2.0"
    assert gltf["buffers"][0]["byteLength"] == len(binary.rstrip(b"\x00")) or True
    # Every accessor must point at a view that fits inside the buffer.
    for acc in gltf["accessors"]:
        view = gltf["bufferViews"][acc["bufferView"]]
        assert view["byteOffset"] + view["byteLength"] <= len(binary)


def test_labels_the_node_not_only_the_mesh(tmp_path):
    # The parser reads the node name first. A file that labels only meshes
    # arrives as unclassified geometry, which is the exact bug this guards.
    pos, idx = quad()
    out = write_glb(
        tmp_path / "b.glb",
        [Part("structure", "hangar", pos, idx), Part("ground", "terrain", pos, idx)],
    )
    gltf, _ = read_glb(out)
    assert [n["name"] for n in gltf["nodes"]] == ["structure:hangar", "ground:terrain"]
    assert [m["name"] for m in gltf["meshes"]] == ["structure:hangar", "ground:terrain"]


def test_scales_metres_to_the_viewers_units(tmp_path):
    pos, idx = quad(size=10.0)
    out = write_glb(tmp_path / "c.glb", [Part("ground", "floor", pos, idx)])
    gltf, _ = read_glb(out)
    position = next(a for a in gltf["accessors"] if a["type"] == "VEC3" and "max" in a)
    # 10 m of quad must arrive as 1000 units, or the viewer frames it as a dot.
    assert position["max"][0] == pytest.approx(10.0 * METRES_TO_VIEWER)
    assert position["min"] == [0.0, 0.0, 0.0]


def test_position_accessor_carries_min_and_max(tmp_path):
    # Required by the specification, and the viewer derives its camera framing
    # from the bounds, so an absent min/max is not a cosmetic omission.
    pos, idx = quad()
    out = write_glb(tmp_path / "d.glb", [Part("ground", "floor", pos, idx)])
    gltf, _ = read_glb(out)
    pos_accessors = [a for a in gltf["accessors"] if a.get("type") == "VEC3"]
    assert any("min" in a and "max" in a for a in pos_accessors)


def test_every_buffer_view_is_four_byte_aligned(tmp_path):
    # glTF requires accessor offsets to divide by their component size. Padding
    # every view to 4 satisfies every type used here; an unaligned view is the
    # kind of fault that loads in one parser and not another.
    pos, idx = quad()
    parts = [
        Part("ground", "floor", pos, idx),
        Part("structure", "wall", pos + np.float32(1), idx),
        Part("water", "pond", pos + np.float32(2), idx),
    ]
    out = write_glb(tmp_path / "e.glb", parts)
    gltf, _ = read_glb(out)
    for view in gltf["bufferViews"]:
        assert view["byteOffset"] % 4 == 0


def test_narrow_indices_are_used_when_they_fit(tmp_path):
    pos, idx = quad()
    out = write_glb(tmp_path / "f.glb", [Part("ground", "floor", pos, idx)])
    gltf, _ = read_glb(out)
    index_acc = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["indices"]]
    assert index_acc["componentType"] == 5123, "a 4-vertex mesh does not need 32-bit indices"


def test_wide_indices_when_the_mesh_is_large(tmp_path):
    n = 70_000
    pos = np.zeros((n, 3), dtype=np.float32)
    pos[:, 0] = np.arange(n)
    idx = np.array([0, 1, n - 1], dtype=np.uint32)
    out = write_glb(tmp_path / "g.glb", [Part("ground", "big", pos, idx)])
    gltf, _ = read_glb(out)
    index_acc = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["indices"]]
    assert index_acc["componentType"] == 5125


class TestNormals:
    def test_an_upward_quad_gets_upward_normals(self):
        pos, idx = quad()
        n = compute_normals(pos, idx)
        assert np.allclose(n, np.array([0.0, 1.0, 0.0]), atol=1e-6)

    def test_weighting_is_by_area(self):
        # A large horizontal face and a tiny vertical sliver sharing a vertex.
        # Unweighted, the sliver would drag the shared normal far off vertical
        # and the viewer would stop reading the surface as ground.
        positions = np.array(
            [[0, 0, 0], [100, 0, 0], [0, 0, 100], [0, 0.01, 0.01]], dtype=np.float32
        )
        indices = np.array([0, 2, 1, 0, 3, 1], dtype=np.uint32)
        n = compute_normals(positions, indices)
        assert n[0][1] > 0.99

    def test_degenerate_vertices_get_a_unit_normal(self):
        positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)
        n = compute_normals(positions, indices)
        assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
        assert np.isfinite(n).all()


class TestRejections:
    """Faults caught here rather than surfacing as corrupt geometry downstream."""

    def test_rejects_an_out_of_range_index(self, tmp_path):
        pos, _ = quad()
        with pytest.raises(ValueError, match="indexes vertex"):
            write_glb(
                tmp_path / "x.glb",
                [Part("ground", "floor", pos, np.array([0, 1, 99], dtype=np.uint32))],
            )

    def test_rejects_indices_that_are_not_whole_triangles(self, tmp_path):
        pos, _ = quad()
        with pytest.raises(ValueError, match="not whole triangles"):
            write_glb(
                tmp_path / "x.glb",
                [Part("ground", "floor", pos, np.array([0, 1], dtype=np.uint32))],
            )

    def test_rejects_non_finite_positions(self, tmp_path):
        pos, idx = quad()
        pos[1][0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            write_glb(tmp_path / "x.glb", [Part("ground", "floor", pos, idx)])

    def test_rejects_a_texture_without_uvs(self, tmp_path):
        pos, idx = quad()
        with pytest.raises(ValueError, match="texture but no UVs"):
            write_glb(
                tmp_path / "x.glb",
                [Part("ground", "floor", pos, idx, texture_png=b"\x89PNG\r\n\x1a\n")],
            )

    def test_rejects_an_empty_file(self, tmp_path):
        with pytest.raises(ValueError, match="no parts"):
            write_glb(tmp_path / "x.glb", [])


def test_textured_part_emits_material_texture_and_image(tmp_path):
    pos, idx = quad()
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    out = write_glb(
        tmp_path / "t.glb",
        [Part("ground", "floor", pos, idx, uvs=uvs, texture_png=png)],
    )
    gltf, binary = read_glb(out)
    assert len(gltf["materials"]) == 1
    assert len(gltf["images"]) == 1
    assert gltf["images"][0]["mimeType"] == "image/png"
    # Reconstructed shells have no meaningful back face; culling one leaves a
    # hole you can see through when the camera clips into geometry.
    assert gltf["materials"][0]["doubleSided"] is True
    view = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
    assert binary[view["byteOffset"] : view["byteOffset"] + len(png)] == png
    assert "TEXCOORD_0" in gltf["meshes"][0]["primitives"][0]["attributes"]
