"""
Minimal glTF 2.0 binary writer.

Deliberately hand-rolled rather than routed through Unreal's own glTF exporter.
That exporter is built for material-accurate scene interchange: it bakes
materials and textures, may emit `KHR_mesh_quantization`, and gives no control
over how parts are named. What the viewer wants is the opposite — positions and
triangles, float32, uncompressed, in two parts named so the classifier can tell
the ground from the things standing on it. That is about a hundred lines, and
owning them means the file always matches what the viewer reads.

The output is one .glb: a JSON chunk and a binary chunk, no external files. The
viewer refuses external buffers, and rightly so — a .gltf with a sidecar .bin is
two files to lose instead of one.
"""
import json
import struct

GLB_MAGIC = 0x46546C67
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942

COMPONENT_FLOAT = 5126
COMPONENT_UINT32 = 5125
TARGET_ARRAY_BUFFER = 34962
TARGET_ELEMENT_ARRAY_BUFFER = 34963
MODE_TRIANGLES = 4


class Part(object):
    """One named run of geometry: what becomes a node and a mesh in the file.

    `positions` and `normals` are flat float sequences (x, y, z, ...) already in
    viewer space; `indices` is a flat sequence of triangle corners. Anything
    supporting `tobytes()` works — numpy arrays in practice, since the caller
    reads millions of these out of Unreal.
    """

    def __init__(self, name, positions, indices, normals=None):
        self.name = name
        self.positions = positions
        self.indices = indices
        self.normals = normals

    @property
    def vertex_count(self):
        return len(self.positions) // 3

    @property
    def triangle_count(self):
        return len(self.indices) // 3


def _pad(data, fill):
    """glTF wants every chunk and buffer view 4-byte aligned."""
    rem = len(data) % 4
    return data if rem == 0 else data + fill * (4 - rem)


def _bounds(positions):
    """Component-wise min/max, required on the POSITION accessor.

    Not optional in the spec, and viewers use it for frustum culling and to
    size the scene — a file without it loads as if it were a point at the
    origin in some readers.
    """
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for i in range(0, len(positions), 3):
        for c in range(3):
            v = positions[i + c]
            if v < lo[c]:
                lo[c] = v
            if v > hi[c]:
                hi[c] = v
    if lo[0] == float("inf"):
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    return [float(v) for v in lo], [float(v) for v in hi]


def write_glb(path, parts, extras=None):
    """Write `parts` to a .glb, returning what it cost.

    Parts with no triangles are dropped rather than written as empty meshes:
    an empty primitive is legal but every reader treats it differently, and one
    of the two parts being empty is the normal case for a level with no
    structures.
    """
    parts = [p for p in parts if p.triangle_count > 0]

    buffer = bytearray()
    views = []
    accessors = []
    meshes = []
    nodes = []

    def add_view(data, target):
        offset = len(buffer)
        buffer.extend(data)
        # Pad between views so the next one starts aligned.
        buffer.extend(b"\x00" * ((4 - len(buffer) % 4) % 4))
        views.append(
            {"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target}
        )
        return len(views) - 1

    for part in parts:
        pos_bytes = part.positions.tobytes() if hasattr(part.positions, "tobytes") else struct.pack(
            "<{}f".format(len(part.positions)), *part.positions
        )
        idx_bytes = part.indices.tobytes() if hasattr(part.indices, "tobytes") else struct.pack(
            "<{}I".format(len(part.indices)), *part.indices
        )

        lo, hi = _bounds(part.positions)
        pos_view = add_view(pos_bytes, TARGET_ARRAY_BUFFER)
        accessors.append(
            {
                "bufferView": pos_view,
                "componentType": COMPONENT_FLOAT,
                "count": part.vertex_count,
                "type": "VEC3",
                "min": lo,
                "max": hi,
            }
        )
        attributes = {"POSITION": len(accessors) - 1}

        if part.normals is not None and len(part.normals) == len(part.positions):
            nrm_bytes = (
                part.normals.tobytes()
                if hasattr(part.normals, "tobytes")
                else struct.pack("<{}f".format(len(part.normals)), *part.normals)
            )
            nrm_view = add_view(nrm_bytes, TARGET_ARRAY_BUFFER)
            accessors.append(
                {
                    "bufferView": nrm_view,
                    "componentType": COMPONENT_FLOAT,
                    "count": part.vertex_count,
                    "type": "VEC3",
                }
            )
            attributes["NORMAL"] = len(accessors) - 1

        idx_view = add_view(idx_bytes, TARGET_ELEMENT_ARRAY_BUFFER)
        accessors.append(
            {
                "bufferView": idx_view,
                "componentType": COMPONENT_UINT32,
                "count": len(part.indices),
                "type": "SCALAR",
            }
        )

        meshes.append(
            {
                "name": part.name,
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": len(accessors) - 1,
                        "mode": MODE_TRIANGLES,
                    }
                ],
            }
        )
        # The node carries the name the viewer classifies on. Mesh names are
        # often generic in other exporters, so the viewer reads the node first;
        # both are set to the same thing here to leave no ambiguity.
        nodes.append({"name": part.name, "mesh": len(meshes) - 1})

    gltf = {
        "asset": {"version": "2.0", "generator": "3DHeat Unreal level exporter"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(buffer)}],
    }
    if extras:
        gltf["extras"] = extras

    json_bytes = _pad(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    bin_bytes = _pad(bytes(buffer), b"\x00")

    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", GLB_MAGIC, 2, total))
        f.write(struct.pack("<II", len(json_bytes), JSON_CHUNK))
        f.write(json_bytes)
        f.write(struct.pack("<II", len(bin_bytes), BIN_CHUNK))
        f.write(bin_bytes)

    return {
        "path": path,
        "bytes": total,
        "parts": [
            {"name": p.name, "vertices": p.vertex_count, "triangles": p.triangle_count}
            for p in parts
        ],
    }
