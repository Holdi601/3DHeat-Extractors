"""
Collada shapes, for the BeamNG models that have no compiled `.cdae`.

The game compiles every `.dae` it loads into a `.cdae` and ships both, so this
is the fallback: a mod level that ships only `.dae` files, and the odd shape
whose cache is in an older format. It reads what Torque's importer reads and
turns it into the same thing `beamngshape` returns, so the export cannot tell
the two apart:

- **Levels of detail** from Torque's naming: a mesh node's trailing number is
  the pixel size it is drawn from (`barrier75`, `barrier40`), a negative one
  or a `collision`/`Colmesh`/`LOS` name is never drawn. A shape without the
  convention is one level.
- **Units and the up axis** applied as the importer does: metres from
  `<unit meter>`, and Y-up turned to Z-up.
- **UVs** flipped to the cache's convention, where v runs down the image.

Checked against the `.cdae` the game compiled from the same files - see the
tests - rather than against a reading of the Collada specification.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from .beamngshape import Piece, _merge

_NS = "{http://www.collada.org/2005/11/COLLADASchema}"
_HIDDEN = re.compile(r"^(colmesh|collision|los|bb_|billboard|autobillboard|imposter|nulldetail)", re.I)
_SIZE = re.compile(r"^(.*?)(-?\d+)$")


def _floats(text: str | None) -> np.ndarray:
    return np.array((text or "").split(), dtype=np.float64)


def _ints(text: str | None) -> np.ndarray:
    return np.array((text or "").split(), dtype=np.int64)


def _strip(tag: str) -> str:
    return tag.split("}", 1)[-1]


@dataclass
class _Mesh:
    """One node's geometry: its world matrix and its triangle sets."""

    name: str
    size: float
    matrix: np.ndarray
    # (material name, positions, normals or None, uvs or None, faces)
    sets: list = field(default_factory=list)


@dataclass
class ColladaShape:
    meshes: list[_Mesh]
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]

    def levels(self) -> list[int]:
        sizes = sorted({m.size for m in self.meshes if m.size >= 0}, reverse=True)
        return list(range(len(sizes)))

    def _sizes(self) -> list[float]:
        return sorted({m.size for m in self.meshes if m.size >= 0}, reverse=True)

    def triangle_count(self, level: int) -> int:
        size = self._sizes()[level]
        return sum(len(s[4]) for m in self.meshes if m.size == size for s in m.sets)

    def pieces(self, level: int | None = None) -> list[Piece]:
        sizes = self._sizes()
        if not sizes:
            return []
        size = sizes[0 if level is None else level]
        grouped: dict[str, list] = {}
        for mesh in self.meshes:
            if mesh.size != size:
                continue
            for material, points, normals, uvs, faces in mesh.sets:
                world = np.c_[points, np.ones(len(points))] @ mesh.matrix.T
                n = None
                if normals is not None:
                    n = normals @ np.linalg.inv(mesh.matrix[:3, :3])
                grouped.setdefault(material, []).append((world[:, :3], faces, uvs, n))
        return [_merge(name, parts) for name, parts in grouped.items()]


def read_collada(data: bytes) -> ColladaShape:
    root = ET.fromstring(data)
    unit = root.find(f"{_NS}asset/{_NS}unit")
    scale = float(unit.get("meter", "1")) if unit is not None else 1.0
    up = root.find(f"{_NS}asset/{_NS}up_axis")
    axis = (up.text or "Z_UP").strip().upper() if up is not None else "Z_UP"
    to_z = np.eye(4)
    if axis == "Y_UP":
        to_z[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    elif axis == "X_UP":
        to_z[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    to_z[:3, :3] *= scale

    materials: dict[str, str] = {}
    for material in root.iter(f"{_NS}material"):
        materials[material.get("id", "")] = material.get("name") or material.get("id") or ""

    geometries: dict[str, list] = {}
    for geometry in root.iter(f"{_NS}geometry"):
        mesh = geometry.find(f"{_NS}mesh")
        if mesh is not None:
            geometries[geometry.get("id", "")] = _read_mesh(mesh)

    meshes: list[_Mesh] = []

    def walk(node, parent: np.ndarray) -> None:
        local = np.eye(4)
        for child in node:
            tag = _strip(child.tag)
            if tag == "matrix":
                local = local @ _floats(child.text).reshape(4, 4)
            elif tag == "translate":
                m = np.eye(4)
                m[:3, 3] = _floats(child.text)[:3]
                local = local @ m
            elif tag == "rotate":
                v = _floats(child.text)
                if len(v) == 4 and np.linalg.norm(v[:3]) > 1e-12:
                    u = v[:3] / np.linalg.norm(v[:3])
                    a = np.radians(v[3])
                    k = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
                    m = np.eye(4)
                    m[:3, :3] = np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)
                    local = local @ m
            elif tag == "scale":
                v = _floats(child.text)
                if len(v) == 3:
                    local = local @ np.diag([*v, 1.0])
        world = parent @ local
        name = node.get("name") or node.get("id") or ""
        for instance in node.findall(f"{_NS}instance_geometry"):
            sets = geometries.get((instance.get("url") or "").lstrip("#"))
            if not sets:
                continue
            bound = {}
            for im in instance.iter(f"{_NS}instance_material"):
                bound[im.get("symbol", "")] = materials.get((im.get("target") or "").lstrip("#"), "")
            match = _SIZE.match(name)
            size = float(match.group(2)) if match else 2.0
            hidden = bool(_HIDDEN.match(name)) or size < 0
            if hidden:
                continue
            mesh = _Mesh(name=name, size=size, matrix=to_z @ world)
            for symbol, points, normals, uvs, faces in sets:
                mesh.sets.append((bound.get(symbol, materials.get(symbol, symbol)), points, normals, uvs, faces))
            meshes.append(mesh)
        for child in node.findall(f"{_NS}node"):
            walk(child, world)

    for scene in root.iter(f"{_NS}visual_scene"):
        for node in scene.findall(f"{_NS}node"):
            walk(node, np.eye(4))

    corners = []
    for mesh in meshes:
        for _, points, _, _, _ in mesh.sets:
            if len(points):
                world = np.c_[points, np.ones(len(points))] @ mesh.matrix.T
                corners += [world[:, :3].min(axis=0), world[:, :3].max(axis=0)]
    if corners:
        stack = np.array(corners)
        bounds = (tuple(stack.min(axis=0).tolist()), tuple(stack.max(axis=0).tolist()))
    else:
        bounds = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    return ColladaShape(meshes=meshes, bounds=bounds)  # type: ignore[arg-type]


def _source(mesh, ref: str) -> np.ndarray | None:
    ref = ref.lstrip("#")
    vertices = mesh.find(f"{_NS}vertices[@id='{ref}']")
    if vertices is not None:
        position = vertices.find(f"{_NS}input[@semantic='POSITION']")
        if position is None:
            return None
        return _source(mesh, position.get("source", ""))
    source = mesh.find(f"{_NS}source[@id='{ref}']")
    if source is None:
        return None
    values = _floats(source.findtext(f"{_NS}float_array"))
    accessor = source.find(f"{_NS}technique_common/{_NS}accessor")
    stride = int(accessor.get("stride", "1")) if accessor is not None else 1
    if stride <= 0 or len(values) % stride:
        return None
    return values.reshape(-1, stride)


def _read_mesh(mesh) -> list:
    """Each triangle set: (material symbol, positions, normals, uvs, faces), one vertex per corner."""
    out = []
    for primitive in mesh:
        kind = _strip(primitive.tag)
        if kind not in ("triangles", "polylist", "polygons"):
            continue
        inputs = primitive.findall(f"{_NS}input")
        if not inputs:
            continue
        stride = max(int(i.get("offset", "0")) for i in inputs) + 1
        if kind == "polygons":
            rows = [_ints(p.text) for p in primitive.findall(f"{_NS}p")]
            counts = np.array([len(r) // stride for r in rows], dtype=np.int64)
            indices = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        else:
            indices = _ints(primitive.findtext(f"{_NS}p"))
            if kind == "polylist":
                counts = _ints(primitive.findtext(f"{_NS}vcount"))
            else:
                counts = np.full(len(indices) // (3 * stride), 3, dtype=np.int64)
        if not len(indices) or len(indices) % stride:
            continue
        corners = indices.reshape(-1, stride)
        # Fan every polygon into triangles.
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        tri = []
        for start, count in zip(starts.tolist(), counts.tolist()):
            for k in range(1, count - 1):
                tri.append((start, start + k, start + k + 1))
        if not tri:
            continue
        faces = np.array(tri, dtype=np.int64)
        attribute = {}
        for item in inputs:
            semantic = item.get("semantic")
            if semantic == "TEXCOORD" and item.get("set", "0") not in ("0", None) and "TEXCOORD" in attribute:
                continue
            data = _source(mesh, item.get("source", ""))
            if data is None:
                continue
            attribute.setdefault(semantic, (int(item.get("offset", "0")), data))
        if "VERTEX" not in attribute:
            continue
        offset, data = attribute["VERTEX"]
        positions = data[corners[:, offset], :3]
        normals = None
        if "NORMAL" in attribute:
            offset, data = attribute["NORMAL"]
            normals = data[corners[:, offset], :3]
        uvs = None
        if "TEXCOORD" in attribute:
            offset, data = attribute["TEXCOORD"]
            uvs = data[corners[:, offset], :2].copy()
            uvs[:, 1] = 1.0 - uvs[:, 1]
        out.append((primitive.get("material", ""), positions, normals, uvs, faces))
    return out
