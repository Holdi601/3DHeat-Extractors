"""
Writing a GLB the viewer will take at its word.

Two things decide whether extracted terrain arrives as terrain or as an
unclassified grey blob, and neither is visible in the file's geometry: the
**node** carries the label, and the label has to match the viewer's pattern.
The rest of this is the glTF container's own arithmetic, which fails silently -
a wrong byte offset produces a file that opens and shows noise.
"""

from __future__ import annotations

import json
import re
import struct

import numpy as np
import pytest

from heat3d_gamefiles.glb import write_glb


def read_back(path):
    raw = path.read_bytes()
    magic, version, total = struct.unpack_from("<III", raw, 0)
    assert magic == 0x46546C67
    assert version == 2
    assert total == len(raw), "the declared length is not the file's"
    json_length, json_tag = struct.unpack_from("<II", raw, 12)
    assert json_tag == 0x4E4F534A
    document = json.loads(raw[20 : 20 + json_length])
    binary_length, binary_tag = struct.unpack_from("<II", raw, 20 + json_length)
    assert binary_tag == 0x004E4942
    binary = raw[28 + json_length : 28 + json_length + binary_length]
    return document, binary


#: glTF component type -> numpy type and the divisor that normalises it.
COMPONENTS = {
    5120: ("<i1", 127.0),
    5121: ("<u1", 255.0),
    5122: ("<i2", 32767.0),
    5123: ("<u2", 65535.0),
    5125: ("<u4", None),
    5126: ("<f4", None),
}


def accessor(document, binary, index, columns, dtype=None):
    """
    Read one accessor, found through the primitive that uses it, the way a
    reader has to: by its component type, its byte stride, and whether it is
    normalised - quantised normals are bytes four to a vertex.

    Not by position: the file grew normals and colour, and every accessor after
    the first moved. A test that assumes accessor 1 is the index buffer passes
    for exactly as long as nothing is ever added.
    """
    spec = document["accessors"][index]
    view = document["bufferViews"][spec["bufferView"]]
    kind, scale = COMPONENTS[spec["componentType"]]
    size = np.dtype(kind).itemsize
    stride = view.get("byteStride", size * columns)
    rows = np.frombuffer(
        binary, dtype=np.uint8, count=stride * spec["count"], offset=view["byteOffset"]
    ).reshape(spec["count"], stride)
    values = rows[:, : size * columns].copy().view(kind).reshape(spec["count"], columns)
    if spec.get("normalized"):
        values = np.maximum(values.astype(np.float64) / scale, -1.0)
    return values if columns > 1 else values.reshape(-1)


def attribute(document, binary, part, name, columns=3):
    primitive = document["meshes"][part]["primitives"][0]
    return accessor(document, binary, primitive["attributes"][name], columns)


def indices_of(document, binary, part):
    primitive = document["meshes"][part]["primitives"][0]
    return accessor(document, binary, primitive["indices"], 1).astype(np.int64).reshape(-1, 3)


def quad():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.5, 1.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int32)
    return positions, faces


class TestTheFileIsWhatItSaysItIs:
    def test_the_mesh_comes_back_out(self, tmp_path):
        positions, faces = quad()
        path = write_glb(tmp_path / "a.glb", positions, faces)

        document, binary = read_back(path)

        assert np.array_equal(attribute(document, binary, 0, "POSITION"), positions)
        assert np.array_equal(indices_of(document, binary, 0), faces)

    def test_the_accessor_bounds_are_the_mesh_bounds(self, tmp_path):
        """
        glTF requires min and max on a POSITION accessor, and viewers use them
        to frame the camera. Wrong ones load a model that is there and cannot
        be found.
        """
        positions, faces = quad()
        path = write_glb(tmp_path / "a.glb", positions, faces)

        document, _binary = read_back(path)

        spot = document["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
        assert document["accessors"][spot]["min"] == positions.min(axis=0).tolist()
        assert document["accessors"][spot]["max"] == positions.max(axis=0).tolist()

    def test_world_coordinates_are_left_alone(self, tmp_path):
        """
        The reconstruction exporter scales to centimetres to match Unreal.
        This must not: the whole value of reading the game's own archive is
        that the numbers mean something, and a lap of telemetry has to line up
        against them.
        """
        positions, faces = quad()
        positions = positions + np.array([-4096.0, 150.0, 6656.0], dtype=np.float32)
        path = write_glb(tmp_path / "a.glb", positions, faces)

        document, binary = read_back(path)

        spot = document["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
        assert document["accessors"][spot]["min"][0] == pytest.approx(-4096.0)


class TestTheViewerCanClassifyIt:
    def test_the_node_carries_the_label(self, tmp_path):
        """
        `modelParse.ts` reads a part's class from the node name, falling back to
        the mesh. Naming only the mesh loses the classification.
        """
        positions, faces = quad()
        path = write_glb(tmp_path / "a.glb", positions, faces, name="ground:terrain")

        document, _binary = read_back(path)

        assert document["nodes"][0]["name"] == "ground:terrain"

    def test_the_default_label_matches_the_pattern_the_viewer_enforces(self, tmp_path):
        positions, faces = quad()
        path = write_glb(tmp_path / "a.glb", positions, faces)

        document, _binary = read_back(path)

        assert re.match(r"^(ground|structure|water)\s*[:|]", document["nodes"][0]["name"])


class TestItRefusesToWriteSomethingBroken:
    def test_an_index_past_the_vertices(self, tmp_path):
        positions, _faces = quad()

        with pytest.raises(ValueError, match="reaches vertex"):
            write_glb(tmp_path / "a.glb", positions, np.array([[0, 1, 99]]))

    def test_no_triangles_at_all(self, tmp_path):
        positions, _faces = quad()

        with pytest.raises(ValueError, match="no triangles"):
            write_glb(tmp_path / "a.glb", positions, np.zeros((0, 3), dtype=np.int32))

    def test_a_vertex_that_is_not_finite(self, tmp_path):
        """
        One NaN makes a whole model vanish in most viewers, because the bounds
        it computes stop being comparable. Cheaper to catch here than to
        explain later.
        """
        positions, faces = quad()
        positions[1, 1] = np.nan

        with pytest.raises(ValueError, match="not finite"):
            write_glb(tmp_path / "a.glb", positions, faces)

    def test_the_wrong_shape(self, tmp_path):
        with pytest.raises(ValueError, match="N by 3"):
            write_glb(tmp_path / "a.glb", np.zeros((4, 2)), np.zeros((1, 3), dtype=int))


class TestQuantised:
    """The compact form every export uses by default."""

    def write(self, tmp_path, **options):
        positions, faces = quad()
        normals = np.array([[0.0, 1.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.8, -0.6], [-1.0, 0.0, 0.0]])
        uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.25, 1.0], [0.5, 0.5]], dtype=np.float32)
        texture = np.zeros((4, 4, 3), dtype=np.uint8)
        from heat3d_gamefiles.glb import Part

        path = write_glb(
            tmp_path / "q.glb",
            positions,
            parts=[Part("structure:sign", faces, texture=texture)],
            normals=normals,
            uvs=uvs,
            **options,
        )
        return read_back(path), normals, uvs, faces

    def test_normals_are_bytes_and_come_back_within_a_degree(self, tmp_path):
        (document, binary), normals, _uvs, _faces = self.write(tmp_path)
        back = attribute(document, binary, 0, "NORMAL")
        cosine = (back * normals).sum(axis=1) / np.linalg.norm(back, axis=1)
        assert np.degrees(np.arccos(np.clip(cosine, -1, 1))).max() < 1.0
        spec = document["accessors"][document["meshes"][0]["primitives"][0]["attributes"]["NORMAL"]]
        assert spec["componentType"] == 5120 and spec["normalized"]
        assert "KHR_mesh_quantization" in document["extensionsRequired"]

    def test_uvs_are_shorts(self, tmp_path):
        (document, binary), _normals, uvs, _faces = self.write(tmp_path)
        assert np.allclose(attribute(document, binary, 0, "TEXCOORD_0", 2), uvs, atol=1e-4)

    def test_a_small_part_indexes_in_shorts(self, tmp_path):
        (document, binary), _normals, _uvs, faces = self.write(tmp_path)
        spec = document["accessors"][document["meshes"][0]["primitives"][0]["indices"]]
        assert spec["componentType"] == 5123
        assert np.array_equal(indices_of(document, binary, 0), faces)

    def test_the_full_precision_form_is_still_there(self, tmp_path):
        (document, binary), normals, _uvs, _faces = self.write(tmp_path, quantize=False)
        assert "extensionsRequired" not in document
        assert np.allclose(attribute(document, binary, 0, "NORMAL"), normals, atol=1e-6)
