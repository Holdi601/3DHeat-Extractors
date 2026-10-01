"""
Turning a Unity `Mesh` object into vertices and triangles.

`unity.py` finds the objects; this reads one. It is a separate module because it
is a different kind of problem: the container is self-describing and can be
checked against itself, whereas a shipped build carries **no type tree** — the
field layout lives in the compiled player, not in the file — so the only way
through is to know the layout for the engine version and read it in order.

Which means the parse has to be checked against something
---------------------------------------------------------
Reading a struct in the wrong order does not raise; it yields plausible numbers.
So `read_mesh` does not trust itself. Unity stores each mesh's local bounding box
as its own field, computed by the editor from the real vertices, and
`MeshData.aabb_matches()` compares it with the box of the vertices actually
decoded. A layout error moves the vertices, and the two boxes stop agreeing. That
check is what makes it reasonable to point this at a game nobody has tried.

Vertices are not in one place
-----------------------------
A mesh either carries its vertex bytes inline or points at a `.resS` file beside
the container, which is where most of a shipped game's geometry lives. Channels
describe how to read those bytes: which stream, at what offset, in what format
and how many components. Streams are packed one after another with each padded to
a 16-byte boundary, so a stream's start cannot be assumed — it has to be
accumulated from the strides before it.

Scope: Unity 2019 and later, which is every version in the layout below. Earlier
builds moved several of these fields and are refused rather than guessed at.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .unity import SerializedFile, UnityObject, UnsupportedUnity

#: Channel slots, in the order Unity assigns them.
CH_POSITION = 0
CH_NORMAL = 1
CH_TANGENT = 2
CH_COLOUR = 3
CH_UV0 = 4

#: Vertex component formats, as (numpy dtype, bytes per component).
FORMATS = {
    0: (np.float32, 4),
    1: (np.float16, 2),
    2: (np.uint8, 1),  # unorm8
    3: (np.int8, 1),  # snorm8
    4: (np.uint16, 2),  # unorm16
    5: (np.int16, 2),  # snorm16
    6: (np.uint8, 1),
    7: (np.int8, 1),
    8: (np.uint16, 2),
    9: (np.int16, 2),
    10: (np.uint32, 4),
    11: (np.int32, 4),
}

#: Formats whose integers are a fraction of their full range rather than a count.
NORMALISED = {2: 255.0, 3: 127.0, 4: 65535.0, 5: 32767.0}

#: Streams are padded to this boundary.
STREAM_ALIGN = 16

#: Topology 0 is triangles. Anything else (strips, quads, lines) is left alone
#: rather than converted, because a wrong conversion looks like real geometry.
TOPOLOGY_TRIANGLES = 0


@dataclass
class SubMesh:
    """One draw range within a mesh."""

    first_byte: int
    index_count: int
    topology: int
    base_vertex: int
    first_vertex: int
    vertex_count: int
    centre: tuple[float, float, float]
    extent: tuple[float, float, float]


@dataclass
class MeshData:
    """A decoded mesh."""

    name: str
    #: (N, 3) float32 positions.
    vertices: np.ndarray
    #: (M, 3) uint32 triangle indices.
    triangles: np.ndarray
    #: (N, 3) float32 normals, when the mesh carried them.
    normals: np.ndarray | None
    #: (N, 2) float32 texture coordinates, when present.
    uvs: np.ndarray | None
    submeshes: list[SubMesh]
    #: The bounding box Unity stored, as (centre, extent).
    stored_aabb: tuple[tuple[float, float, float], tuple[float, float, float]]

    def aabb_matches(self, tolerance: float = 0.02) -> bool:
        """
        Whether the decoded vertices reproduce the bounding box Unity recorded.

        This is the parse's own check. `tolerance` is relative to the box size,
        loose enough for half-precision positions and normalised integers, tight
        enough that a field read in the wrong order fails it.
        """
        if len(self.vertices) == 0:
            return False
        centre, extent = self.stored_aabb
        stored_min = np.array(centre) - np.array(extent)
        stored_max = np.array(centre) + np.array(extent)
        got_min = self.vertices.min(axis=0)
        got_max = self.vertices.max(axis=0)
        scale = float(np.abs(stored_max - stored_min).max()) or 1.0
        return bool(
            np.abs(got_min - stored_min).max() <= tolerance * scale
            and np.abs(got_max - stored_max).max() <= tolerance * scale
        )


class _Reader:
    """A little-endian cursor with Unity's alignment rules."""

    def __init__(self, data: bytes):
        self.data = data
        self.at = 0

    def _take(self, code: str, width: int):
        if self.at + width > len(self.data):
            raise UnsupportedUnity("mesh ended mid-field")
        (value,) = struct.unpack_from("<" + code, self.data, self.at)
        self.at += width
        return value

    def u8(self) -> int:
        return self._take("B", 1)

    def i32(self) -> int:
        return self._take("i", 4)

    def u32(self) -> int:
        return self._take("I", 4)

    def i64(self) -> int:
        return self._take("q", 8)

    def f32(self) -> float:
        return self._take("f", 4)

    def align(self) -> None:
        self.at = (self.at + 3) // 4 * 4

    def take(self, count: int) -> bytes:
        if self.at + count > len(self.data):
            raise UnsupportedUnity("field ran past the end of the object")
        out = self.data[self.at : self.at + count]
        self.at += count
        return out

    def string(self) -> str:
        length = self.i32()
        if not 0 <= length <= len(self.data):
            raise UnsupportedUnity(f"string length {length} is not credible")
        out = self.take(length).decode("utf-8", "replace")
        self.align()
        return out

    def blob(self) -> bytes:
        size = self.i32()
        if not 0 <= size <= len(self.data):
            raise UnsupportedUnity(f"byte array length {size} is not credible")
        out = self.take(size)
        self.align()
        return out

    def vector3(self) -> tuple[float, float, float]:
        return (self.f32(), self.f32(), self.f32())


def _skip_blend_shapes(reader: _Reader) -> None:
    """
    Step over `m_Shapes`, which no static level geometry uses but every mesh has.

    Four arrays in order: per-vertex deltas, shape ranges, named channels and
    weights. They are stepped rather than kept, but they must be stepped
    *exactly*, because everything after them — including the vertices — is
    positioned relative to where they end.
    """
    reader.take(40 * reader.i32())  # BlendShapeVertex: 3 vectors and an index
    reader.take(12 * reader.i32())  # MeshBlendShape: two counts and two flags
    for _ in range(reader.i32()):  # MeshBlendShapeChannel
        reader.string()
        reader.u32()  # name hash
        reader.i32()  # frame index
        reader.i32()  # frame count
    reader.take(4 * reader.i32())  # full weights
    reader.align()


def read_mesh(
    container: SerializedFile, obj: UnityObject, *, resource_dir: Path | None = None
) -> MeshData:
    """
    Decode one `Mesh` object.

    `resource_dir` is where the `.resS` companion files live — the container's own
    directory unless the caller says otherwise. A mesh whose vertices are external
    and whose resource file is missing raises rather than returning an empty mesh.
    """
    if obj.class_name != "Mesh":
        raise ValueError(f"object {obj.path_id} is a {obj.class_name}, not a Mesh")

    reader = _Reader(container.raw(obj))
    name = reader.string()

    submeshes = []
    for _ in range(reader.i32()):
        submeshes.append(
            SubMesh(
                first_byte=reader.u32(),
                index_count=reader.u32(),
                topology=reader.i32(),
                base_vertex=reader.u32(),
                first_vertex=reader.u32(),
                vertex_count=reader.u32(),
                centre=reader.vector3(),
                extent=reader.vector3(),
            )
        )

    _skip_blend_shapes(reader)

    reader.take(64 * reader.i32())  # m_BindPose, one 4x4 matrix each
    reader.take(4 * reader.i32())  # m_BoneNameHashes
    reader.u32()  # m_RootBoneNameHash
    reader.take(24 * reader.i32())  # m_BonesAABB, min and max
    reader.take(4 * reader.i32())  # m_VariableBoneCountWeights
    reader.align()

    reader.u8()  # m_MeshCompression
    reader.u8()  # m_IsReadable
    reader.u8()  # m_KeepVertices
    reader.u8()  # m_KeepIndices
    reader.align()

    index_format = reader.i32()
    index_bytes = reader.blob()

    vertex_count = reader.u32()
    channels = []
    for _ in range(reader.i32()):
        stream = reader.u8()
        offset = reader.u8()
        fmt = reader.u8()
        dimension = reader.u8() & 0x0F  # the high nibble is a flags field
        channels.append((stream, offset, fmt, dimension))
    vertex_bytes = reader.blob()

    # A compressed mesh stores its vertices as quantised deltas instead. That is
    # a different decoder, not a variation of this one.
    compressed_vertex_count = _peek_compressed(reader)

    local_centre, local_extent = _read_local_aabb(reader)

    stream_data = _read_stream_data(reader)
    if not vertex_bytes and stream_data is not None:
        vertex_bytes = _read_resource(container, stream_data, resource_dir)

    if not vertex_bytes:
        if compressed_vertex_count:
            raise UnsupportedUnity(
                f"mesh {name!r} is stored compressed; that encoding is not implemented"
            )
        raise UnsupportedUnity(f"mesh {name!r} has no vertex data")

    positions = _read_channel(vertex_bytes, channels, CH_POSITION, vertex_count)
    if positions is None:
        raise UnsupportedUnity(f"mesh {name!r} declares no position channel")
    normals = _read_channel(vertex_bytes, channels, CH_NORMAL, vertex_count)
    uvs = _read_channel(vertex_bytes, channels, CH_UV0, vertex_count)

    triangles = _read_triangles(index_bytes, index_format, submeshes)

    return MeshData(
        name=name,
        vertices=positions[:, :3].astype(np.float32),
        triangles=triangles,
        normals=None if normals is None else normals[:, :3].astype(np.float32),
        uvs=None if uvs is None else uvs[:, :2].astype(np.float32),
        submeshes=submeshes,
        stored_aabb=(local_centre, local_extent),
    )


def _packed_float(reader: _Reader) -> int:
    """`PackedFloatVector`: a count, a range, a start, the bits, and their width."""
    count = reader.u32()
    reader.f32()  # range
    reader.f32()  # start
    data = reader.blob()
    reader.u8()  # bit size
    reader.align()
    return count if data else 0


def _packed_int(reader: _Reader) -> int:
    """`PackedIntVector`: a count, the bits, and their width."""
    count = reader.u32()
    data = reader.blob()
    reader.u8()  # bit size
    reader.align()
    return count if data else 0


def _peek_compressed(reader: _Reader) -> int:
    """
    Step over `m_CompressedMesh`, reporting how much it actually holds.

    Eleven members in a fixed order. Empty — which is the normal case for a
    shipped static mesh — they occupy exactly 164 bytes, and that number is what
    puts the bounding box and the streaming info that follow in the right place.
    Miss it and the box still parses, as six plausible floats of nothing.
    """
    held = 0
    held += _packed_float(reader)  # m_Vertices
    held += _packed_float(reader)  # m_UV
    held += _packed_float(reader)  # m_Normals
    held += _packed_float(reader)  # m_Tangents
    _packed_int(reader)  # m_Weights
    _packed_int(reader)  # m_NormalSigns
    _packed_int(reader)  # m_TangentSigns
    _packed_float(reader)  # m_FloatColors
    _packed_int(reader)  # m_BoneIndices
    held += _packed_int(reader)  # m_Triangles
    reader.u32()  # m_UVInfo
    return held


def _read_local_aabb(reader: _Reader):
    centre = reader.vector3()
    extent = reader.vector3()
    return centre, extent


@dataclass(frozen=True)
class _StreamData:
    offset: int
    size: int
    path: str


def _read_stream_data(reader: _Reader) -> _StreamData | None:
    """
    Where the vertices live, when they are not inline.

    The order here was pinned against a real mesh rather than assumed: the
    recorded size came out as 11088 for a 308-vertex mesh of 36-byte stride,
    which is only true if `offset` is the 64-bit field starting immediately
    after the two mesh metrics.
    """
    try:
        reader.i32()  # m_MeshUsageFlags
        reader.i32()  # m_CookingOptions — a flags word, typically 30
        reader.blob()  # baked convex collision mesh
        reader.blob()  # baked triangle collision mesh
        reader.f32()  # m_MeshMetrics[0]
        reader.f32()  # m_MeshMetrics[1]
        offset = reader.i64()
        size = reader.u32()
        path = reader.string()
    except UnsupportedUnity:
        return None
    return _StreamData(offset, size, path) if path else None


def _read_resource(
    container: SerializedFile, stream: _StreamData, resource_dir: Path | None
) -> bytes:
    """Pull the vertex bytes out of the `.resS` file beside the container."""
    directory = resource_dir or container.path.parent
    # The stored path is relative to the build's data folder and carries a
    # prefix Unity uses internally; only the file name is meaningful here.
    candidate = directory / Path(stream.path.replace("archive:/", "")).name
    if not candidate.is_file():
        raise UnsupportedUnity(
            f"mesh vertices live in {candidate.name}, which is not beside the "
            f"container in {directory}"
        )
    with open(candidate, "rb") as handle:
        handle.seek(stream.offset)
        return handle.read(stream.size)


def _stream_layout(channels, vertex_count: int) -> dict[int, tuple[int, int]]:
    """Each stream's (start offset, stride), accumulated in order."""
    strides: dict[int, int] = {}
    for stream, offset, fmt, dimension in channels:
        if dimension == 0:
            continue
        _, width = FORMATS.get(fmt, (None, 0))
        strides[stream] = max(strides.get(stream, 0), offset + width * dimension)

    layout: dict[int, tuple[int, int]] = {}
    at = 0
    for stream in sorted(strides):
        layout[stream] = (at, strides[stream])
        span = strides[stream] * vertex_count
        at += -(-span // STREAM_ALIGN) * STREAM_ALIGN
    return layout


def _read_channel(
    data: bytes, channels, which: int, vertex_count: int
) -> np.ndarray | None:
    """One channel, de-interleaved from its stream and scaled if normalised."""
    if which >= len(channels):
        return None
    stream, offset, fmt, dimension = channels[which]
    if dimension == 0:
        return None
    if fmt not in FORMATS:
        raise UnsupportedUnity(f"vertex format {fmt} is not known")

    dtype, width = FORMATS[fmt]
    start, stride = _stream_layout(channels, vertex_count)[stream]
    needed = start + stride * vertex_count
    if needed > len(data):
        raise UnsupportedUnity(
            f"channel {which} needs {needed} bytes of vertex data but only "
            f"{len(data)} are present"
        )

    raw = np.frombuffer(data, dtype=np.uint8, count=stride * vertex_count, offset=start)
    raw = raw.reshape(vertex_count, stride)[:, offset : offset + width * dimension]
    values = np.ascontiguousarray(raw).view(dtype).reshape(vertex_count, dimension)
    out = values.astype(np.float32)
    if fmt in NORMALISED:
        out = out / NORMALISED[fmt]
    return out


def _read_triangles(index_bytes: bytes, index_format: int, submeshes) -> np.ndarray:
    """
    Triangles from the index buffer, with each submesh's base vertex applied.

    Unity stores indices relative to a submesh's own first vertex, so a mesh with
    more than one submesh comes out as overlapping garbage if the base is not
    added back.
    """
    dtype = np.uint16 if index_format == 0 else np.uint32
    width = 2 if index_format == 0 else 4

    parts = []
    for sub in submeshes:
        if sub.topology != TOPOLOGY_TRIANGLES:
            continue
        count = sub.index_count - sub.index_count % 3
        if count <= 0:
            continue
        end = sub.first_byte + count * width
        if end > len(index_bytes):
            raise UnsupportedUnity(
                f"submesh wants indices up to {end} of {len(index_bytes)} bytes"
            )
        chunk = np.frombuffer(index_bytes, dtype=dtype, count=count, offset=sub.first_byte)
        parts.append(chunk.astype(np.uint32) + sub.base_vertex)

    if not parts:
        return np.zeros((0, 3), dtype=np.uint32)
    return np.concatenate(parts).reshape(-1, 3)
