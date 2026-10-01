"""
Just enough glTF binary to read Source2Viewer's export and write the level the
viewer loads: meshes as positions and triangles, one node per part, named so
the viewer knows ground from structure (`ground:<what>`, `structure:<what>`).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read(path: str | Path) -> tuple[dict, bytes]:
    """The JSON document and the binary chunk of a .glb."""
    data = Path(path).read_bytes()
    magic, _, length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"{path} is not a .glb")
    doc, blob, at = None, b"", 12
    while at < length:
        size, kind = struct.unpack_from("<II", data, at)
        chunk = data[at + 8 : at + 8 + size]
        if kind == 0x4E4F534A:
            doc = json.loads(chunk)
        elif kind == 0x004E4942:
            blob = chunk
        at += 8 + size
    if doc is None:
        raise ValueError(f"{path} has no JSON chunk")
    return doc, blob


def _accessor(doc: dict, blob: bytes, index: int) -> np.ndarray:
    a = doc["accessors"][index]
    view = doc["bufferViews"][a["bufferView"]]
    kind = np.dtype(_COMPONENT[a["componentType"]])
    width = _WIDTH[a["type"]]
    start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
    count = a["count"]
    item = kind.itemsize * width
    stride = view.get("byteStride", 0) or item
    if stride == item:
        return np.frombuffer(blob, kind, count=count * width, offset=start).reshape(count, width)
    raw = np.frombuffer(blob, np.uint8, count=stride * (count - 1) + item, offset=start)
    rows = raw[np.arange(count)[:, None] * stride + np.arange(item)[None, :]]
    return rows.copy().view(kind).reshape(count, width)


def _local(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=float).reshape(4, 4).T
    x, y, z, w = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    m = np.eye(4)
    m[:3, :3] = rotation * np.array(node.get("scale", [1.0, 1.0, 1.0]))
    m[:3, 3] = node.get("translation", [0.0, 0.0, 0.0])
    return m


def meshes(doc: dict, blob: bytes) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Every mesh in the scene as (node name, world positions, triangles)."""
    out: list[tuple[str, np.ndarray, np.ndarray]] = []

    def walk(index: int, parent: np.ndarray) -> None:
        node = doc["nodes"][index]
        world = parent @ _local(node)
        if "mesh" in node:
            for prim in doc["meshes"][node["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4:
                    continue
                p = _accessor(doc, blob, prim["attributes"]["POSITION"]).astype(float)
                tris = (
                    _accessor(doc, blob, prim["indices"]).reshape(-1, 3).astype(np.int64)
                    if "indices" in prim
                    else np.arange(len(p)).reshape(-1, 3)
                )
                out.append((node.get("name", ""), (np.c_[p, np.ones(len(p))] @ world.T)[:, :3], tris))
        for child in node.get("children", []):
            walk(child, world)

    scene = doc.get("scenes", [{}])[doc.get("scene", 0)]
    for root in scene.get("nodes", []):
        walk(root, np.eye(4))
    return out


def write(path: str | Path, parts: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    """A .glb of (node name, positions, triangles) parts, flat-shaded."""
    blob = bytearray()
    views, accessors, mesh_list, nodes = [], [], [], []

    def add(array: np.ndarray, target: int, kind: str, component: int, bounds: bool = False) -> int:
        while len(blob) % 4:
            blob.append(0)
        raw = array.tobytes()
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(raw), "target": target})
        blob.extend(raw)
        acc = {"bufferView": len(views) - 1, "componentType": component, "count": len(array), "type": kind}
        if bounds:
            acc["min"] = array.min(axis=0).tolist()
            acc["max"] = array.max(axis=0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    for name, positions, triangles in parts:
        if not len(triangles):
            continue
        # One vertex per corner, so each triangle carries its own normal.
        corners = positions[triangles].reshape(-1, 3).astype(np.float32)
        a, b, c = corners[0::3], corners[1::3], corners[2::3]
        n = np.cross(b - a, c - a)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        normals = np.repeat(n, 3, axis=0).astype(np.float32)
        index = np.arange(len(corners), dtype=np.uint32)
        prim = {
            "attributes": {
                "POSITION": add(corners, 34962, "VEC3", 5126, bounds=True),
                "NORMAL": add(normals, 34962, "VEC3", 5126),
            },
            "indices": add(index, 34963, "SCALAR", 5125),
        }
        mesh_list.append({"name": name, "primitives": [prim]})
        nodes.append({"name": name, "mesh": len(mesh_list) - 1})
    doc = {
        "asset": {"version": "2.0", "generator": "heat3d_cs2"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": mesh_list,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    while len(blob) % 4:
        blob.append(0)
    out = bytearray(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(text) + 8 + len(blob)))
    out += struct.pack("<II", len(text), 0x4E4F534A) + text
    out += struct.pack("<II", len(blob), 0x004E4942) + blob
    Path(path).write_bytes(bytes(out))
