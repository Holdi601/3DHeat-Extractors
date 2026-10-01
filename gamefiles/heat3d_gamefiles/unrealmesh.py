"""
Cooked Unreal 5 static meshes: the vertex and index buffers of each level of
detail.

A `UStaticMesh` export is its properties (unversioned, see `unversioned.py`)
and then its native data. The properties are not needed for geometry and are
not read: the native part is found by its fixed start - a GUID flag of 0,
two bytes of strip flags, `bCooked` = 1, the index of the mesh's own
`BodySetup` export - and read from there in the order `UStaticMesh::Serialize`
and `FStaticMeshRenderData::Serialize` write it for a cooked, non-Nanite mesh:
per level of detail its sections, bounds, then (when inlined) the position
buffer, the tangent and UV buffer, the colour buffer and the index buffers.

Positions are Unreal's: centimetres, X forward, Y right, Z up. Checked on a
stage's road tile, whose decoded extent matches the bounds the mesh stores
for itself.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from .zen import Package, Packages


class MeshError(Exception):
    """A static mesh this cannot read, and why."""


@dataclass
class Section:
    material: int
    first_index: int
    triangles: int


@dataclass
class Lod:
    positions: np.ndarray  # (n, 3) float32, centimetres
    faces: np.ndarray  # (m, 3) int64
    sections: list[Section]
    uvs: np.ndarray | None = None  # (n, 2) of the first channel
    #: Every UV channel, (n, channels, 2).
    channels: np.ndarray | None = None

    def section_faces(self, section: Section) -> np.ndarray:
        start = section.first_index // 3
        return self.faces[start : start + section.triangles]


@dataclass
class StaticMesh:
    name: str
    lods: list[Lod] = field(default_factory=list)
    #: The bounds the mesh keeps for itself: origin, extent, radius (cm).
    bounds: tuple = ()

    def triangle_count(self, level: int) -> int:
        return len(self.lods[level].faces)


class _Reader:
    def __init__(self, data: bytes, at: int) -> None:
        self.data, self.at = data, at

    def take(self, fmt: str):
        values = struct.unpack_from(fmt, self.data, self.at)
        self.at += struct.calcsize(fmt)
        return values if len(values) > 1 else values[0]

    def skip(self, size: int) -> None:
        if size < 0 or self.at + size > len(self.data):
            raise MeshError("ran past the end of the mesh data")
        self.at += size

    def bulk(self) -> tuple[int, int, bytes]:
        """`TResourceArray::BulkSerialize`: element size, count, the bytes."""
        size, count = self.take("<ii")
        if size < 0 or count < 0 or self.at + size * count > len(self.data):
            raise MeshError("a buffer runs past the end of the mesh data")
        raw = self.data[self.at : self.at + size * count]
        self.at += size * count
        return size, count, raw


def _native_start(data: bytes, body_setup: int) -> int | None:
    """Where the native data starts: `[0 u32][strip u16][1 u32][BodySetup index i32]`."""
    pattern = struct.pack("<Ii", 1, body_setup)
    at = 0
    while True:
        at = data.find(pattern, at)
        if at < 0:
            return None
        start = at - 6
        if start >= 0 and data[start : start + 4] == b"\0\0\0\0":
            return start
        at += 1


def read_static_mesh(package: Package, *, all_levels: bool = True) -> StaticMesh:
    """The static mesh a package holds."""
    mesh_export = next((e for e in package.exports if package.class_name(e).endswith(".StaticMesh")), None)
    if mesh_export is None:
        raise MeshError(f"{package.path} holds no static mesh")
    data = package.export_data(mesh_export)
    body = next((e for e in package.exports if package.class_name(e).endswith(".BodySetup")), None)
    start = _native_start(data, body.index + 1 if body is not None else 0)
    if start is None:
        raise MeshError(f"{package.path}: the native data was not found")
    r = _Reader(data, start)
    r.take("<I")  # GUID flag
    r.take("<H")  # strip flags
    if r.take("<I") != 1:
        raise MeshError("not a cooked mesh")
    r.take("<ii")  # BodySetup, NavCollision
    r.skip(16)  # lighting GUID
    sockets = r.take("<i")
    r.skip(4 * max(0, sockets))
    count = r.take("<i")
    if not 0 < count <= 16:
        raise MeshError(f"{count} levels of detail")
    out = StaticMesh(name=mesh_export.name)
    for level in range(count):
        lod = _read_lod(r)
        if lod is None:
            break
        out.lods.append(lod)
        if not all_levels:
            break
    if not out.lods:
        raise MeshError(f"{package.path}: no level of detail is stored inline")
    return out


def _read_lod(r: _Reader) -> Lod | None:
    strip = r.take("<H")
    count = r.take("<i")
    if not 0 <= count <= 4096:
        raise MeshError(f"{count} sections")
    sections = []
    for _ in range(count):
        material, first, triangles, _lo, _hi = r.take("<iIIII")
        r.skip(20)  # five bool32 flags
        sections.append(Section(material, first, triangles))
    r.skip(56)  # bounds: origin, extent, radius, as doubles
    r.take("<f")  # max deviation
    cooked_out, inlined = r.take("<II")
    if cooked_out or not inlined:
        return None
    r.take("<I")  # ray tracing geometry flag
    r.take("<H")  # buffer strip flags
    _stride, vertices = r.take("<II")
    size, n, raw = r.bulk()
    if size != 12 or n != vertices:
        raise MeshError(f"position buffer of {n} x {size} for {vertices} vertices")
    positions = np.frombuffer(raw, dtype="<f4").reshape(-1, 3)
    r.take("<H")
    channels, _n2, full_uvs, high_precision = r.take("<IIII")
    _size, _n, _tangents = r.bulk()
    uv_size, uv_count, uv_raw = r.bulk()
    uvs = None
    table = None
    # One element per UV pair, every channel of a vertex together: a road
    # tile of 14,305 vertices with five channels stores 71,525 of 8 bytes.
    width = 8 if full_uvs else 4
    if channels and uv_size == width and uv_count == vertices * channels:
        table = np.frombuffer(uv_raw, dtype="<f4" if full_uvs else "<f2").reshape(vertices, channels, 2).astype(np.float32)
        uvs = table[:, 0, :]
    r.take("<H")
    _colour_stride, colours = r.take("<II")
    if colours:
        r.bulk()
    wide = r.take("<I")
    index_size, index_count, index_raw = r.bulk()
    r.take("<I")  # expand to 32 bits
    indices = np.frombuffer(index_raw, dtype="<u4" if wide else "<u2").astype(np.int64)
    if len(indices) % 3 or (len(indices) and indices.max() >= vertices):
        raise MeshError("index buffer does not fit the vertices")
    # What follows - reversed and depth-only index buffers, ray tracing data,
    # area-weighted samplers - is stepped over to reach the next level.
    strip_class = strip >> 8

    def index_buffer() -> None:
        r.take("<I")
        r.bulk()
        r.take("<I")

    if not strip_class & 4:
        index_buffer()
    index_buffer()
    if not strip_class & 4:
        index_buffer()
    if not strip_class & 8:
        r.skip(24)
        r.bulk()
    for _ in range(len(sections) + 1):
        n = r.take("<i")
        r.skip(4 * n)
        n = r.take("<i")
        r.skip(4 * n)
        r.take("<f")
    r.skip(12)
    return Lod(positions=positions, faces=indices.reshape(-1, 3), sections=sections, uvs=uvs, channels=table)


class MeshShelf:
    """Static meshes by package path, each read once."""

    def __init__(self, packages: Packages) -> None:
        self.packages = packages
        self._cache: dict[str, StaticMesh | None] = {}
        self.failed: dict[str, str] = {}

    def get(self, path: str) -> StaticMesh | None:
        key = path.lower()
        if key not in self._cache:
            mesh = None
            package = self.packages.open(path)
            if package is None:
                self.failed[key] = "not found"
            else:
                try:
                    mesh = read_static_mesh(package)
                except (MeshError, struct.error, ValueError) as exc:
                    self.failed[key] = str(exc)
            self._cache[key] = mesh
        return self._cache[key]
