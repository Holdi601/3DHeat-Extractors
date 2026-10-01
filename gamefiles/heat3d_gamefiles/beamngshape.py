"""
BeamNG's compiled shapes: `.cdae`, the game's own cache of every Collada file.

Every `.dae` a level places has a `.cdae` beside it, and some shapes ship only
as `.cdae`, so this is the one mesh format the BeamNG export reads. It is
documented by BeamNG (documentation.beamng.com/modding/file_formats/cdae/):
a version word, a MessagePack header, then a MessagePack stream - Torque's
`TSShape` written field by field, each array as `[count, element size, bytes]`.

Two things the documentation does not spell out, both settled against the
`.dae` files the same shapes were built from (every shape with a rotated node
in the game's shared archive and two levels, compared vertex bound for vertex
bound):

- **Node rotations are Torque's**: a `Quat16` read the textbook way rotates the
  wrong way round. Its matrix has to be transposed; with that, the world bounds
  of every node match the Collada to the millimetre, and without it they are
  out by 38 cm in the median.
- **Units and axes are already applied.** A Collada file authored in
  centimetres or Y-up comes out of the cache in metres, Z-up; the cache is what
  the game draws, so nothing here converts.

Levels of detail are `details`, each naming an object level (`objectDetailNum`)
and the pixel size it is drawn from. The largest is the full model; negative
sizes are collision and line-of-sight meshes, never drawn.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

#: `(version & 0xFF)` of every shape this reads.
SHAPE_VERSION = 31

#: Mesh types. Only the first and the sorted kind carry drawable triangles for
#: a static model; skinned meshes are characters and vehicles, decals are
#: projected at run time.
STANDARD, SKIN, DECAL, SORTED, NULL = range(5)

#: `TSDrawPrimitive.matIndex` flags.
_TRIANGLES, _STRIP, _FAN = 0x00000000, 0x40000000, 0x80000000
_TYPE_MASK = 0xC0000000
_NO_MATERIAL = 0x10000000
_MATERIAL_MASK = 0x0FFFFFFF

_NODE = np.dtype([("name", "<i4"), ("parent", "<i4"), ("first_object", "<i4"), ("first_child", "<i4"), ("next", "<i4")])
_OBJECT = np.dtype(
    [("name", "<i4"), ("meshes", "<i4"), ("start", "<i4"), ("node", "<i4"), ("next", "<i4"), ("first_decal", "<i4")]
)
_DETAIL = np.dtype(
    [
        ("name", "<i4"), ("subshape", "<i4"), ("object_detail", "<i4"), ("size", "<f4"),
        ("average_error", "<f4"), ("max_error", "<f4"), ("polys", "<i4"),
        ("bb_dimension", "<i4"), ("bb_detail", "<i4"), ("bb_equator", "<u4"),
        ("bb_polar", "<u4"), ("bb_polar_angle", "<f4"), ("bb_poles", "<u4"),
    ]
)
_PRIMITIVE = np.dtype([("start", "<i4"), ("count", "<i4"), ("material", "<u4")])

#: Object names that are never drawn, whatever level they sit in.
_HIDDEN = ("colmesh", "collision", "los", "bb_", "billboard", "autobillboard", "imposter")

_SHAPE_VECTORS = (
    "nodes", "objects", "subShapeFirstNode", "subShapeFirstObject", "subShapeNumNodes",
    "subShapeNumObjects", "defaultRotations", "defaultTranslations", "nodeRotations",
    "nodeTranslations", "nodeUniformScales", "nodeAlignedScales", "nodeArbitraryScaleFactors",
    "nodeArbitraryScaleRots", "groundTranslations", "groundRotations", "objectStates",
    "triggers", "details",
)
_MESH_VECTORS = ("verts", "tverts", "tverts2", "colors", "norms", "encodedNorms", "primitives", "indices", "tangents")


class UnreadableShape(Exception):
    """Not a `.cdae` this can read, and the message says why."""


@dataclass
class Vector:
    """One `[count, element size, bytes]` array from the stream."""

    count: int
    size: int
    data: bytes

    def array(self, dtype, columns: int | None = None) -> np.ndarray:
        values = np.frombuffer(self.data, dtype=dtype)
        if columns is not None:
            return values.reshape(-1, columns)
        return values


@dataclass
class Mesh:
    kind: int
    verts: np.ndarray | None = None
    tverts: np.ndarray | None = None
    norms: np.ndarray | None = None
    #: RGBA bytes per vertex, where the model is coloured by vertex.
    colors: np.ndarray | None = None
    primitives: np.ndarray | None = None
    indices: np.ndarray | None = None
    verts_per_frame: int = 0

    def triangles(self) -> tuple[np.ndarray, np.ndarray]:
        """Faces and the material slot of each, strips and fans unrolled."""
        faces: list[np.ndarray] = []
        slots: list[np.ndarray] = []
        if self.primitives is None or self.indices is None:
            return np.zeros((0, 3), np.int64), np.zeros(0, np.int64)
        for primitive in self.primitives:
            start, count, material = int(primitive["start"]), int(primitive["count"]), int(primitive["material"])
            run = self.indices[start : start + count].astype(np.int64)
            kind = material & _TYPE_MASK
            if kind == _TRIANGLES:
                tris = run[: len(run) // 3 * 3].reshape(-1, 3)
            elif kind == _STRIP:
                n = len(run) - 2
                if n <= 0:
                    continue
                i = np.arange(n)
                even = (i % 2) == 0
                a, b, c = run[i], run[i + 1], run[i + 2]
                tris = np.stack([a, np.where(even, b, c), np.where(even, c, b)], axis=1)
            elif kind == _FAN:
                n = len(run) - 2
                if n <= 0:
                    continue
                tris = np.stack([np.full(n, run[0]), run[1:-1], run[2:]], axis=1)
            else:
                continue
            # Degenerate triangles are how strips join; they draw nothing.
            keep = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
            tris = tris[keep]
            slot = -1 if material & _NO_MATERIAL else material & _MATERIAL_MASK
            faces.append(tris)
            slots.append(np.full(len(tris), slot, np.int64))
        if not faces:
            return np.zeros((0, 3), np.int64), np.zeros(0, np.int64)
        return np.concatenate(faces), np.concatenate(slots)


@dataclass
class Piece:
    """The triangles of one material in one level of detail, in shape space."""

    material: str
    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray | None
    faces: np.ndarray
    #: Per-vertex colour, 0..1 RGB, where the model carries one.
    colours: np.ndarray | None = None


@dataclass
class Shape:
    names: list[str]
    nodes: np.ndarray
    objects: np.ndarray
    details: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    meshes: list[Mesh]
    materials: list[str]
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    _world: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    def levels(self) -> list[int]:
        """The drawn levels of detail, the full model first."""
        drawn = [i for i, d in enumerate(self.details) if d["size"] >= 0]
        return sorted(drawn, key=lambda i: -float(self.details[i]["size"]))

    def node_matrix(self, index: int) -> np.ndarray:
        """A node's transform to shape space, as a 4x4 for column vectors."""
        cached = self._world.get(index)
        if cached is not None:
            return cached
        local = np.eye(4)
        local[:3, :3] = _torque_rotation(self.rotations[index])
        local[:3, 3] = self.translations[index]
        parent = int(self.nodes[index]["parent"])
        world = self.node_matrix(parent) @ local if parent >= 0 else local
        self._world[index] = world
        return world

    def triangle_count(self, level: int) -> int:
        """What a level of detail costs, without building it."""
        detail = self.details[level]
        k = int(detail["object_detail"])
        total = 0
        for obj in self.objects:
            if k >= int(obj["meshes"]) or self._hidden(obj):
                continue
            mesh = self.meshes[int(obj["start"]) + k]
            if mesh.kind in (STANDARD, SORTED) and mesh.indices is not None:
                total += len(mesh.indices) // 3
        return total

    def _hidden(self, obj) -> bool:
        name = self.names[int(obj["name"])].lower() if 0 <= int(obj["name"]) < len(self.names) else ""
        return name.startswith(_HIDDEN)

    def pieces(self, level: int | None = None) -> list[Piece]:
        """
        One level of detail as triangles per material, in shape space.

        `None` is the full model. Faces are wound so that their geometric
        normal agrees with the stored vertex normals, which is what a viewer
        that culls back faces needs and what Torque's own winding does not
        promise.
        """
        drawn = self.levels()
        if not drawn:
            return []
        if level is None:
            level = drawn[0]
        k = int(self.details[level]["object_detail"])
        grouped: dict[str, list[tuple]] = {}
        for obj in self.objects:
            if k >= int(obj["meshes"]) or self._hidden(obj):
                continue
            mesh = self.meshes[int(obj["start"]) + k]
            if mesh.kind not in (STANDARD, SORTED) or mesh.verts is None:
                continue
            count = mesh.verts_per_frame or len(mesh.verts)
            verts = mesh.verts[:count].astype(np.float64)
            faces, slots = mesh.triangles()
            if not len(faces):
                continue
            ok = (faces < count).all(axis=1)
            faces, slots = faces[ok], slots[ok]
            node = int(obj["node"])
            matrix = self.node_matrix(node) if node >= 0 else np.eye(4)
            positions = verts @ matrix[:3, :3].T + matrix[:3, 3]
            normals = None
            if mesh.norms is not None and len(mesh.norms) >= count:
                normals = mesh.norms[:count].astype(np.float64) @ matrix[:3, :3].T
            uvs = mesh.tverts[:count].astype(np.float64) if mesh.tverts is not None and len(mesh.tverts) >= count else None
            colours = None
            if mesh.colors is not None and len(mesh.colors) >= count:
                colours = mesh.colors[:count, :3].astype(np.float64) / 255.0
            for slot in np.unique(slots):
                chosen = faces[slots == slot]
                name = self.materials[slot] if 0 <= slot < len(self.materials) else ""
                grouped.setdefault(name, []).append((positions, chosen, uvs, normals, colours))
        pieces = []
        for name, parts in grouped.items():
            pieces.append(_merge(name, parts))
        return pieces


def _merge(name, parts) -> Piece:
    """Pieces of one material as one: (positions, faces, uvs, normals[, colours]) each."""
    positions, normals, uvs, faces, colours = [], [], [], [], []
    offset = 0
    textured = all(p[2] is not None for p in parts)
    coloured = all(len(p) > 4 and p[4] is not None for p in parts)
    for part in parts:
        points, chosen, uv, norm = part[:4]
        colour = part[4] if len(part) > 4 else None
        used, local = np.unique(chosen, return_inverse=True)
        local = local.reshape(-1, 3)
        p = points[used]
        if norm is not None:
            n = norm[used]
        else:
            n = _smooth_normals(p, local)
        # Wind each triangle the way its vertices face.
        edge = np.cross(p[local[:, 1]] - p[local[:, 0]], p[local[:, 2]] - p[local[:, 0]])
        facing = (n[local[:, 0]] + n[local[:, 1]] + n[local[:, 2]])
        flip = np.einsum("ij,ij->i", edge, facing) < 0
        local[flip] = local[flip][:, [0, 2, 1]]
        positions.append(p)
        normals.append(n)
        if textured:
            uvs.append(uv[used])
        if coloured:
            colours.append(colour[used])
        faces.append(local + offset)
        offset += len(p)
    normal = np.concatenate(normals)
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.where(length > 1e-9, normal / np.maximum(length, 1e-9), [0.0, 0.0, 1.0])
    return Piece(
        material=name,
        positions=np.concatenate(positions),
        normals=normal,
        uvs=np.concatenate(uvs) if textured else None,
        faces=np.concatenate(faces),
        colours=np.concatenate(colours) if coloured else None,
    )


def _smooth_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(points)
    face = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
    for k in range(3):
        np.add.at(normals, faces[:, k], face)
    return normals


def _torque_rotation(q16: np.ndarray) -> np.ndarray:
    """A `Quat16` as a rotation matrix for column vectors, Torque's way round."""
    x, y, z, w = (float(v) / 32767.0 for v in q16)
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    if norm < 1e-9:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    standard = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return standard.T


def read_shape(data: bytes) -> Shape:
    """Parse a `.cdae`."""
    try:
        import msgpack
    except ImportError as exc:  # pragma: no cover - a missing dependency, said plainly
        raise UnreadableShape("reading BeamNG shapes needs `msgpack` (pip install -r requirements.txt)") from exc
    if len(data) < 8:
        raise UnreadableShape("too short to be a shape")
    version, header_size = struct.unpack_from("<II", data, 0)
    if version & 0xFF != SHAPE_VERSION:
        raise UnreadableShape(f"shape version {version & 0xFF}, not {SHAPE_VERSION}")
    unpacker = msgpack.Unpacker(raw=False, strict_map_key=False)
    unpacker.feed(data[8 : 8 + header_size])
    try:
        header = next(unpacker)
    except StopIteration:
        raise UnreadableShape("no header") from None
    body = data[8 + header_size :]
    if header.get("compression"):
        try:
            import zstandard
        except ImportError as exc:  # pragma: no cover
            raise UnreadableShape("this shape is zstd-compressed; `pip install zstandard`") from exc
        size = int(header.get("bodysize") or 0)
        body = zstandard.ZstdDecompressor().decompress(body, max_output_size=max(size, len(body) * 64))
    stream = msgpack.Unpacker(raw=False, strict_map_key=False, max_bin_len=2**31 - 1, max_array_len=2**31 - 1)
    stream.feed(body)
    it = iter(stream)

    def take():
        try:
            return next(it)
        except StopIteration:
            raise UnreadableShape("the shape ends early") from None

    def vector() -> Vector:
        count, size, blob = take(), take(), take()
        return Vector(int(count), int(size), bytes(blob) if blob else b"")

    for _ in range(4):  # smallest visible size and level, radius, tube radius
        take()
    take()  # centre
    bounds = take()
    vectors = {name: vector() for name in _SHAPE_VECTORS}
    names = [str(take()) for _ in range(int(take()))]
    meshes: list[Mesh] = []
    for _ in range(int(take())):
        kind = int(take())
        if kind == NULL:
            meshes.append(Mesh(NULL))
            continue
        take(); take()  # frames, material frames
        parent = int(take())
        take(); take(); take()  # bounds, centre, radius
        mv = {name: vector() for name in _MESH_VECTORS}
        per_frame = int(take())
        take()  # flags
        if kind == SKIN:
            if parent < 0:
                vector(); vector()  # initial verts and normals
            vector()  # initial transforms
            if parent < 0:
                for _ in range(4):
                    vector()
        mesh = Mesh(kind, verts_per_frame=per_frame)
        if mv["verts"].count:
            mesh.verts = mv["verts"].array("<f4", 3)
        if mv["tverts"].count:
            mesh.tverts = mv["tverts"].array("<f4", 2)
        if mv["norms"].count:
            mesh.norms = mv["norms"].array("<f4", 3)
        if mv["colors"].count and mv["colors"].size == 4:
            mesh.colors = mv["colors"].array(np.uint8, 4)
        if mv["primitives"].count:
            mesh.primitives = np.frombuffer(mv["primitives"].data, dtype=_PRIMITIVE)
        if mv["indices"].count:
            width = mv["indices"].size
            mesh.indices = mv["indices"].array("<u4" if width == 4 else "<u2")
        meshes.append(mesh)
    # Sequences, then the materials. A static shape has no sequences, but an
    # animated prop does, and how their integer sets are written is not
    # documented - so the material list is found from the end instead, where
    # it is last: a count, then that many records of a name and six numbers.
    materials = _material_list(list(it))
    low = tuple(float(v) for v in bounds[:3]) if len(bounds) >= 6 else (0.0, 0.0, 0.0)
    high = tuple(float(v) for v in bounds[3:6]) if len(bounds) >= 6 else (0.0, 0.0, 0.0)
    return Shape(
        names=names,
        nodes=np.frombuffer(vectors["nodes"].data, dtype=_NODE),
        objects=np.frombuffer(vectors["objects"].data, dtype=_OBJECT),
        details=np.frombuffer(vectors["details"].data, dtype=_DETAIL),
        rotations=vectors["defaultRotations"].array("<i2", 4).astype(np.float64)
        if vectors["defaultRotations"].count
        else np.zeros((0, 4)),
        translations=vectors["defaultTranslations"].array("<f4", 3).astype(np.float64)
        if vectors["defaultTranslations"].count
        else np.zeros((0, 3)),
        meshes=meshes,
        materials=materials,
        bounds=(low, high),
    )


def _material_list(tail: list) -> list[str]:
    """The material names at the end of a shape's stream, or none."""
    for count in range(len(tail) // 7, 0, -1):
        at = len(tail) - 7 * count - 1
        if at < 0 or tail[at] != count or isinstance(tail[at], bool):
            continue
        names = tail[at + 1 :: 7]
        if len(names) == count and all(isinstance(n, str) for n in names):
            return [str(n) for n in names]
    return []
