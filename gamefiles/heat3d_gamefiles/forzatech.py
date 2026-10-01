"""
Reading ForzaTech containers.

Forza is neither Unreal, Unity nor Frostbite — it is Turn 10's own engine — and
it comes up because a racing capture is the case the reconstruction exporter was
built for, so the obvious question is whether the same level can be had exactly
instead of estimated.

The answer: **yes, for terrain.**

What is here
------------
Forza keeps most of its content in ordinary ZIP archives under `media/`. Those
need nothing from this project — `zipfile` opens them — and what matters is
knowing that, rather than writing a reader for a format the standard library
already has.

Inside them are ForzaTech's own files, and `.modelbin` is the one this reads. It
is a tagged chunk container: a short header, a table of records, a block of
names, then the data. Each record gives an offset and a length, and the offsets
form a chain — each chunk starts exactly where the last one ended — which is what
makes the layout checkable without documentation. Reading the table wrong breaks
the chain.

The geometry
------------
`VLay` says which buffer holds which attribute, `VerB` holds the vertices and
`IndB` the indices, each behind a sixteen-byte header giving its count, stride
and format. Terrain positions are four signed shorts scaled into a box, and the
box is the last thirty-two bytes of the `Mesh` chunk: a scale and a bias, both
padded out to four floats.

That bias is the useful part. It is in the **game's own world coordinates**, so
a terrain tile read out of the archive lands where the game puts it without any
instance transform, and a lap of telemetry — which is world coordinates too —
picks out which tiles to read. Everything else in a track is placed by instance
tables that are a separate problem; terrain is not.

The archives themselves are `heat3d_gamefiles.minizip`.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

#: `.modelbin` opens with "Grub" — the four characters stored back to front,
#: which is what a FourCC written as a little-endian integer looks like.
MODEL_MAGIC = b"burG"

#: Bytes before the chunk table: magic, version, table offset, file size, count.
MODEL_HEADER = 20

#: One table record: tag, flags, name offset, data offset, size, stored size.
MODEL_RECORD = 24

#: The streamed geometry containers carry this instead. Recognised so a caller
#: is told what the file is rather than that it is nothing.
GEO_MAGIC = b"PGZP"


class UnsupportedForza(Exception):
    """Not a ForzaTech container this reader can read."""


@dataclass(frozen=True)
class Chunk:
    """One tagged chunk of a `.modelbin`."""

    tag: str
    flags: int
    offset: int
    size: int
    stored: int

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass
class ModelFile:
    """An opened `.modelbin`: its chunks, and the bytes of any one of them."""

    path: Path
    version: int
    chunks: list[Chunk]
    _data: bytes = b""

    def __len__(self) -> int:
        return len(self.chunks)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for chunk in self.chunks:
            out[chunk.tag] = out.get(chunk.tag, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def raw(self, chunk: Chunk) -> bytes:
        return self._data[chunk.offset : chunk.end]


def read_model(path: str | Path, data: bytes | None = None) -> ModelFile:
    """
    Parse a `.modelbin` chunk table.

    `data` lets a file read out of a ZIP be parsed without being written to disk
    first, which is how they normally arrive.
    """
    path = Path(path)
    if data is None:
        data = path.read_bytes()
    if len(data) < MODEL_HEADER:
        raise UnsupportedForza(f"{path.name} is too small to be a ForzaTech model")
    if data[:4] != MODEL_MAGIC:
        if data[:4] == GEO_MAGIC:
            raise UnsupportedForza(
                f"{path.name} is a ForzaTech streamed geometry container, not a "
                "model; that format is not implemented"
            )
        raise UnsupportedForza(f"{path.name} does not begin with {MODEL_MAGIC!r}")

    version, names_at, declared, count = struct.unpack_from("<4I", data, 4)
    if declared != len(data):
        raise UnsupportedForza(
            f"{path.name} declares {declared} bytes but is {len(data)}; the header "
            "did not parse as claimed"
        )
    if MODEL_HEADER + count * MODEL_RECORD > len(data):
        raise UnsupportedForza(
            f"{path.name} claims {count} chunks, which will not fit in it"
        )

    chunks: list[Chunk] = []
    for i in range(count):
        at = MODEL_HEADER + i * MODEL_RECORD
        tag = data[at : at + 4][::-1].decode("ascii", "replace")
        flags, _name, offset, size, stored = struct.unpack_from("<5I", data, at + 4)
        if offset + size > len(data):
            raise UnsupportedForza(
                f"{path.name}: chunk {i} runs to {offset + size} of {len(data)} bytes"
            )
        chunks.append(Chunk(tag=tag, flags=flags, offset=offset, size=size, stored=stored))

    # The chain: each chunk begins where the last one ended, rounded up to four
    # bytes. Nothing in the file states this, so it is evidence rather than a
    # rule — a record read at the wrong stride does not produce it. It holds for
    # all 252 models sampled from a shipped Forza install.
    for before, after in zip(chunks, chunks[1:]):
        expected = -(-before.end // 4) * 4
        if after.offset != expected:
            raise UnsupportedForza(
                f"{path.name}: chunk after {before.tag} starts at {after.offset}, "
                f"but the previous one ends at {before.end} and the next should "
                f"begin at {expected}; the table stride is wrong"
            )

    return ModelFile(path=path, version=version, chunks=chunks, _data=data)


def models_in_archive(archive: str | Path) -> list[str]:
    """Names of the `.modelbin` entries inside one of Forza's ZIP archives."""
    import zipfile

    with zipfile.ZipFile(archive) as bundle:
        return [n for n in bundle.namelist() if n.lower().endswith(".modelbin")]


def read_model_from_archive(archive: str | Path, name: str) -> ModelFile:
    """Parse one `.modelbin` straight out of a ZIP."""
    import zipfile

    with zipfile.ZipFile(archive) as bundle:
        return read_model(Path(archive) / name, bundle.read(name))


#: A buffer's own header: how many elements, how many bytes, the stride, how many
#: attributes each element holds, and the format of the first of them. The bytes
#: after it are the buffer.
BUFFER_HEADER = 16

#: One `VLay` element: a u16 attribute name and a u16 channel, then four u32s.
#: The channel is the reason for the split - it is what tells a second UV set
#: apart from the first, and both name the same string.
LAYOUT_ELEMENT = 20

# Attribute formats. These are DXGI format numbers, which is what made the rest
# of the vertex format fall out: once 13 reads as R16G16B16A16_SNORM - which the
# positions already were - and 57 as R16_UINT - which the indices already were -
# every other code in the layouts names a format directly.

#: R16G16B16A16_SNORM. Positions, scaled into the box the `Mesh` carries. The
#: fourth short is the normal's x - see `read_vertices`.
POSITION_SHORT4 = 13
#: R16G16_SNORM. The normal's y and z.
NORMAL_SHORT2 = 37
#: R16G16_UNORM. A UV pair.
TEXCOORD_USHORT2 = 35
#: R8G8B8A8_UNORM. Colour, or blend weights under a colour's name.
COLOR_UBYTE4 = 28
#: R10G10B10A2_UNORM. A tangent, with the handedness in the two-bit alpha.
TANGENT_1010102 = 24
#: R16_UINT. Indices, three to a triangle.
INDEX_SHORT = 57

#: Every attribute that is not the position is four bytes, which is what lets a
#: buffer's stride and its attribute count be checked against each other.
ATTRIBUTE_BYTES = 4

# The `Mesh` chunk: one submesh, one material's slice of the model's single index
# buffer. Offsets are in bytes from the start of the chunk. Every field here is
# checked against something else in the file rather than taken on trust - the
# checks are in `read_submeshes`, and each was needed, because a misread field
# produces plausible numbers.

#: u16. Index into the model's `MatI` chunks: the material of the first (and
#: in a standard-length record, only) entry of the variant table below.
MESH_MATERIAL = 2
#: u16. Which levels of detail the submesh belongs to, one bit each from bit 1.
#: A terrain model carries all four levels over the same ground, so without this
#: the same square metre is exported four times over.
MESH_LOD_MASK = 10
#: u32s, from the index-buffer slice.
MESH_FIRST_INDEX = 2 + 4 * 8
MESH_INDEX_COUNT = 2 + 4 * 10
MESH_TRIANGLES = 2 + 4 * 11
#: u32s: which `VLay`, which `VerB`, and that buffer's stride. The stride is
#: redundant, and that is its value - it is how the two indices were confirmed.
MESH_LAYOUT = 2 + 4 * 15
MESH_BUFFER = 2 + 4 * 22
MESH_STRIDE = 2 + 4 * 24

#: The size of a `Mesh` chunk whose fields sit at the offsets above. Longer ones
#: carry a longer material table at the front - a tyre stack with six material
#: slots is 282 bytes - and every field after it moves back by the difference.
#: Read that way, every one of 9,300 sampled submeshes of every length from 250
#: to 338 bytes is self-consistent. Shorter ones (218 bytes, some terrain
#: variants) are short at the other end and keep the offsets as they are.
#:
#: The table is the submesh's material *variants*, one per look a placed copy
#: can take - five prints for a trackside billboard, six paint colours for a
#: tyre stack. Each entry is eight bytes, `u16 0xFFFF, u16 summer material,
#: u16 0xFFFF, u16 winter material` (0xFFFF where there is none). A standard
#: record holds one entry and no count; a longer one is `u32 count` then that
#: many entries, which is why it is 238 + 8 * count - 4 bytes long, and why its
#: last entry sits exactly where a standard record's only one does. Which
#: variant a copy uses is in its placement record - see `forzaplacement`.
MESH_STANDARD = 238
MESH_VARIANT = 8

#: A material's path inside a `MatI` chunk. `MatI` is itself a nested chunk
#: container and the path is the only part of it this needs, so it is found by
#: searching rather than by parsing a format within a format.
MATERIAL_PATH = re.compile(rb"Game:[^\x00]{0,240}?\.materialbin")


@dataclass(frozen=True)
class Buffer:
    """A vertex or index buffer, with the header read off it."""

    count: int
    stride: int
    format: int
    data: bytes
    #: How many attributes each element carries. For an attribute buffer this is
    #: the stride over four, and the two are checked against each other.
    attributes: int = 0


@dataclass(frozen=True)
class Element:
    """One attribute in the vertex layout."""

    semantic: str
    #: Which of several, for the attributes that repeat - a second UV set, say.
    channel: int
    buffer: int
    format: int
    offset: int


@dataclass(frozen=True)
class Submesh:
    """One material's slice of a model's index buffer."""

    material: str
    first: int
    count: int
    triangles: int
    #: The bit mask of detail levels it belongs to. Bit 1 is the finest.
    lods: int = 0
    layout: int = -1
    buffer: int = -1
    #: Which `MatI` chunk: the first variant's summer material.
    index: int = 0xFFFF
    #: (summer, winter) `MatI` per variant; 0xFFFF where a season has none.
    variants: tuple[tuple[int, int], ...] = ()

    def at_lod(self, level: int) -> bool:
        return bool(self.lods & (1 << (level + 1)))


@dataclass
class Vertices:
    """Everything a model says about its vertices, where it says it."""

    positions: "np.ndarray"
    #: Unit normals, or None where no layout carried one.
    normals: "np.ndarray | None" = None
    #: UV sets by channel, each (n, 2); NaN for vertices whose group lacks one.
    uvs: dict = field(default_factory=dict)
    #: (n, 4) in 0..1, or None. NaN for vertices whose group lacks it.
    colours: "np.ndarray | None" = None


@dataclass(frozen=True)
class Geometry:
    """A decoded mesh, in the game's own world coordinates."""

    positions: "np.ndarray"
    faces: "np.ndarray"
    #: Material name per triangle, parallel to `faces`. Empty when the model
    #: carried no usable submesh table.
    materials: tuple[str, ...] = ()
    #: Detail-level mask per triangle, parallel to `faces`; None as above.
    lods: "np.ndarray | None" = None
    #: `MatI` index per triangle (0xFFFF unknown), parallel to `faces`.
    material_index: "np.ndarray | None" = None
    #: Which submesh each triangle came from, parallel to `faces`, and each
    #: submesh's material variants as `Submesh.variants` gives them.
    submesh: "np.ndarray | None" = None
    variants: tuple = ()
    normals: "np.ndarray | None" = None
    uvs: dict = field(default_factory=dict)
    colours: "np.ndarray | None" = None

    def __post_init__(self) -> None:
        if len(self.faces) and self.faces.max() >= len(self.positions):
            raise UnsupportedForza(
                f"an index reaches vertex {self.faces.max()} of {len(self.positions)}"
            )

    @property
    def bounds(self) -> tuple["np.ndarray", "np.ndarray"]:
        return self.positions.min(axis=0), self.positions.max(axis=0)


def read_buffer(raw: bytes) -> Buffer:
    """Split a `VerB` or `IndB` chunk into its header and its contents."""
    if len(raw) < BUFFER_HEADER:
        raise UnsupportedForza("a buffer chunk is too short to hold its header")
    count, size, stride, attributes, fmt = struct.unpack_from("<IIHHI", raw, 0)
    if stride and count * stride != size:
        raise UnsupportedForza(
            f"a buffer declares {count} elements of {stride} bytes but {size} bytes"
        )
    if BUFFER_HEADER + size > len(raw):
        raise UnsupportedForza(
            f"a buffer declares {size} bytes but the chunk holds "
            f"{len(raw) - BUFFER_HEADER}"
        )
    return Buffer(
        count=count,
        stride=stride,
        format=fmt,
        data=raw[BUFFER_HEADER : BUFFER_HEADER + size],
        attributes=attributes,
    )


def read_layout(raw: bytes) -> list[Element]:
    """The `VLay` chunk: the attributes a group of vertices carries, in order."""
    if len(raw) < 2:
        raise UnsupportedForza("the vertex layout is empty")
    (names_count,) = struct.unpack_from("<H", raw, 0)
    at = 2
    names: list[str] = []
    for _ in range(names_count):
        (length,) = struct.unpack_from("<I", raw, at)
        at += 4
        names.append(raw[at : at + length].decode("ascii", "replace"))
        at += length
    # A u16, where every other count in the file is a u32. Reading it as four
    # bytes puts the first element two bytes early, which still parses - the
    # padding ahead of it is zeros - and yields an attribute index of 65535 a
    # few elements later, once the error has had something non-zero to slide
    # over.
    (count,) = struct.unpack_from("<H", raw, at)
    at += 2

    elements: list[Element] = []
    for _ in range(count):
        if at + LAYOUT_ELEMENT > len(raw):
            raise UnsupportedForza("the vertex layout ends mid-element")
        name, channel, buffer, fmt, _unknown, offset = struct.unpack_from(
            "<HHIIiI", raw, at
        )
        at += LAYOUT_ELEMENT
        if name >= len(names):
            raise UnsupportedForza(
                f"the layout names attribute {name} of {len(names)}; the element "
                "stride is wrong"
            )
        elements.append(
            Element(
                semantic=names[name],
                channel=channel,
                buffer=buffer,
                format=fmt,
                offset=offset,
            )
        )
    return elements


def _chunks_by_tag(model: ModelFile) -> dict[str, list[Chunk]]:
    out: dict[str, list[Chunk]] = {}
    for chunk in model.chunks:
        out.setdefault(chunk.tag, []).append(chunk)
    return out


def position_box(mesh: bytes):
    """
    The scale and bias that put a mesh's shorts back into the world.

    Taken from the end of the `Mesh` chunk rather than a fixed offset: the chunk
    is 238 bytes for most terrain and 218 for some, and the two float4s are the
    last thing in it either way. Every submesh of a model carries the same box,
    which is checked across 860 of them around one circuit.
    """
    import numpy as np

    if len(mesh) < 32:
        raise UnsupportedForza("the Mesh chunk is too short to hold a position box")
    tail = np.frombuffer(mesh[-32:], dtype="<f4")
    return tail[:3].astype(np.float64), tail[4:7].astype(np.float64)


def material_names(model: ModelFile) -> list[str]:
    """
    The material each `MatI` chunk names, in chunk order.

    Only the leaf name, because the path is a build-machine path whose useful
    part - `Materials\\Road\\...` against `Materials\\Terrain\\...` - is
    carried by the name too, and the name is what someone reads in a viewer.
    """
    out: list[str] = []
    for chunk in model.chunks:
        if chunk.tag != "MatI":
            continue
        found = MATERIAL_PATH.search(model.raw(chunk))
        if not found:
            out.append("")
            continue
        path = found.group(0).decode("ascii", "replace")
        out.append(path.rsplit("\\", 1)[-1].removesuffix(".materialbin"))
    return out


def material_paths(model: ModelFile) -> list[str]:
    """The same, with the whole path - which is what finds the material file."""
    out: list[str] = []
    for chunk in model.chunks:
        if chunk.tag != "MatI":
            continue
        found = MATERIAL_PATH.search(model.raw(chunk))
        out.append(found.group(0).decode("ascii", "replace") if found else "")
    return out


#: The texture slot a standard environment material reads its base colour from.
#: Other shaders name the slot differently; see `material_textures`.
SLOT_BASE_COLOUR = 0x88B483AA


def material_textures(model: ModelFile, known) -> list[list[tuple[int, int]]]:
    """
    The textures each `MatI` chunk binds, as (slot, texture hash) in file order.

    A material does not name its textures. It lists them as `u32 slot, u32,
    u32 hash`, and the hash is the `SourceHash` the track's `AssetManifest.xml`
    gives every texture it ships - so `known`, the set of those hashes, is what
    tells a texture reference from any other four bytes. Found by reading the
    kerb's material: its three hashes are exactly the manifest's hashes for
    `road_gen_rum_round_asan_bclr`, `_nrml` and `_extra`, and the slot before
    the colour one is the same in 1,054 materials sampled across the map.

    The records are not aligned to four bytes (they sit 33 bytes apart), so the
    chunk is searched at every byte offset.
    """
    import numpy as np

    wanted = np.unique(np.asarray(list(known) if not isinstance(known, np.ndarray) else known, dtype=np.uint32))
    out: list[list[tuple[int, int]]] = []
    for chunk in model.chunks:
        if chunk.tag != "MatI":
            continue
        raw = model.raw(chunk)
        found: list[tuple[int, int]] = []
        if len(raw) >= 12:
            buffer = np.frombuffer(raw, dtype=np.uint8)
            for phase in range(4):
                usable = (len(raw) - phase) // 4
                words = buffer[phase : phase + usable * 4].view("<u4")
                slot = np.minimum(np.searchsorted(wanted, words), len(wanted) - 1)
                for i in np.nonzero(wanted[slot] == words)[0]:
                    at = phase + int(i) * 4
                    if at >= 8:
                        found.append((at, struct.unpack_from("<I", raw, at - 8)[0], int(words[i])))
        found.sort()
        out.append([(slot, value) for _at, slot, value in found])
    return out


def read_submeshes(model: ModelFile) -> list[Submesh]:
    """
    The model's submeshes: which stretch of the index buffer each material owns,
    at which level of detail, with which vertex layout.

    A terrain model holds one index buffer and up to forty-seven `Mesh` chunks
    carving it up. Nothing here is trusted on its own:

    - the index ranges have to tile the buffer exactly, nothing twice and nothing
      left over;
    - the stride each submesh names has to be the stride of the buffer it names;
    - its indices have to fall inside that buffer's slice of the vertices.

    Across 5,350 submeshes sampled from the map, every one whose buffer index is
    in range passes all three. A table that fails any of them is refused whole:
    labelling triangles from misread fields is worse than not labelling them.
    """
    import numpy as np

    by = _chunks_by_tag(model)
    if "Mesh" not in by or "IndB" not in by:
        return []
    materials = material_names(model)
    total = read_buffer(model.raw(by["IndB"][0])).count

    out: list[Submesh] = []
    for chunk in by["Mesh"]:
        raw = model.raw(chunk)
        shift = max(0, len(raw) - MESH_STANDARD)
        if len(raw) < MESH_STRIDE + shift + 4:
            return []
        variants = _variants(raw, shift)
        material = variants[0][0]
        (lods,) = struct.unpack_from("<H", raw, MESH_LOD_MASK + shift)
        (first,) = struct.unpack_from("<I", raw, MESH_FIRST_INDEX + shift)
        (count,) = struct.unpack_from("<I", raw, MESH_INDEX_COUNT + shift)
        (triangles,) = struct.unpack_from("<I", raw, MESH_TRIANGLES + shift)
        (layout,) = struct.unpack_from("<I", raw, MESH_LAYOUT + shift)
        (buffer,) = struct.unpack_from("<I", raw, MESH_BUFFER + shift)
        if count != triangles * 3 or first + count > total:
            return []
        out.append(
            Submesh(
                material=materials[material] if material < len(materials) else "",
                index=material,
                variants=variants,
                first=first,
                count=count,
                triangles=triangles,
                lods=lods,
                layout=layout,
                buffer=buffer,
            )
        )

    if sum(s.count for s in out) != total:
        # The ranges are meant to partition the buffer. If they do not, the
        # fields are being read at the wrong offsets.
        return []
    return out


def _variants(raw: bytes, shift: int) -> tuple[tuple[int, int], ...]:
    """
    (summer, winter) material per variant, from the table a `Mesh` record opens
    with. A longer record whose count does not account for its length exactly
    is not trusted, and reports the one variant at the standard place.
    """
    if shift:
        (count,) = struct.unpack_from("<I", raw, 0)
        if 4 + MESH_VARIANT * count == shift + MESH_VARIANT and count <= 64:
            return tuple(
                struct.unpack_from("<2xH2xH", raw, 4 + MESH_VARIANT * i) for i in range(count)
            )
        return (struct.unpack_from("<2xH2xH", raw, shift),)
    return (struct.unpack_from("<2xH2xH", raw, 0),)


def _buffer_layouts(submeshes: list[Submesh], buffers: list[Buffer], layouts: list[list[Element]]) -> dict[int, int]:
    """
    Which layout describes each attribute buffer.

    Taken from the submeshes that draw from it, because the layouts and buffers
    are not in the same order - in the files sampled they are in the same order
    for some models, rotated by one for more, and permuted for the rest - and in
    one model in seven two layouts have the same number of attributes, so
    matching on that is not enough either. A buffer no submesh draws from is
    matched on attribute count, and only when that is unambiguous.
    """
    found: dict[int, int] = {}
    for piece in submeshes:
        if 1 <= piece.buffer < len(buffers) and 0 <= piece.layout < len(layouts):
            found.setdefault(piece.buffer, piece.layout)
    for index in range(1, len(buffers)):
        if index in found:
            continue
        want = buffers[index].stride // ATTRIBUTE_BYTES
        candidates = [
            i for i, layout in enumerate(layouts) if len(layout) - 1 == want
        ]
        if len(candidates) == 1:
            found[index] = candidates[0]
    return found


def read_vertices(model: ModelFile) -> Vertices:
    """
    Positions, and every attribute the layouts describe.

    The shape of it
    ---------------
    One position buffer covers every vertex. The other attributes are split: the
    vertices fall into consecutive groups, each group has a buffer of its own,
    and each buffer has its own layout - the road near a junction carries three
    UV sets and two tangents, the bank beside it only a normal. The attribute
    buffers follow the position buffer in vertex order, so the group a vertex
    belongs to is where its index falls in their running total. Every non-position
    attribute is four bytes, in the order its layout lists it.

    The normal is split too
    -----------------------
    `NORMAL` holds two components, y and z. The third is the *position's* fourth
    short - which looked like padding and is not. Found by measurement rather
    than by reading anything: over 690,000 terrain vertices from 400 models the
    fourth short correlates +0.92 with the x of the normal computed from the
    triangles and nothing else, and with it in place the decoded normal is within
    18 degrees of the computed one for 96% of vertices on sloped ground. Every
    two-component encoding tried without it - octahedral in all 48 axis
    arrangements among them - got y and z right and x wrong.
    """
    import numpy as np

    by = _chunks_by_tag(model)
    for needed in ("VLay", "VerB", "Mesh"):
        if needed not in by:
            raise UnsupportedForza(f"{model.path.name} has no {needed} chunk")
    layouts = [read_layout(model.raw(c)) for c in by["VLay"]]
    buffers = [read_buffer(model.raw(c)) for c in by["VerB"]]

    position = next(
        (e for e in layouts[0] if e.semantic == "POSITION"), None
    )
    if position is None:
        raise UnsupportedForza(f"{model.path.name} has no POSITION attribute")
    if position.format != POSITION_SHORT4:
        raise UnsupportedForza(
            f"{model.path.name} stores positions as format {position.format}; "
            f"only {POSITION_SHORT4} (four shorts, scaled) is implemented"
        )
    if position.buffer >= len(buffers):
        raise UnsupportedForza(
            f"{model.path.name} puts positions in buffer {position.buffer} of "
            f"{len(buffers)}"
        )

    source = buffers[position.buffer]
    if source.stride < position.offset + 8:
        raise UnsupportedForza(
            f"{model.path.name}: a {source.stride}-byte vertex cannot hold four "
            f"shorts at offset {position.offset}"
        )
    raw = np.frombuffer(source.data, dtype=np.uint8).reshape(source.count, source.stride)
    shorts = raw[:, position.offset : position.offset + 8].copy().view("<i2")

    scale, bias = position_box(model.raw(by["Mesh"][0]))
    positions = (shorts[:, :3].astype(np.float64) / 32767.0 * scale + bias).astype(np.float32)
    count = len(positions)
    out = Vertices(positions=positions)

    attribute_buffers = buffers[1:] if position.buffer == 0 else []
    if not attribute_buffers or sum(b.count for b in attribute_buffers) != count:
        # Interleaved, or a shape not yet seen. Positions are still right; the
        # rest is left to be computed rather than guessed at.
        return out

    submeshes = read_submeshes(model)
    owner = _buffer_layouts(submeshes, buffers, layouts)

    normal_yz = np.full((count, 2), np.nan, dtype=np.float64)
    uvs: dict[int, np.ndarray] = {}
    colours = None
    start = 0
    for index, buffer in enumerate(attribute_buffers, start=1):
        stop = start + buffer.count
        layout_index = owner.get(index)
        if layout_index is None or buffer.stride % ATTRIBUTE_BYTES:
            start = stop
            continue
        attributes = [e for e in layouts[layout_index] if e.semantic != "POSITION"]
        if len(attributes) * ATTRIBUTE_BYTES != buffer.stride:
            start = stop
            continue
        rows = np.frombuffer(buffer.data, dtype=np.uint8).reshape(buffer.count, buffer.stride)
        for slot, element in enumerate(attributes):
            cell = rows[:, slot * 4 : slot * 4 + 4].copy()
            if element.semantic == "NORMAL" and element.format == NORMAL_SHORT2:
                normal_yz[start:stop] = cell.view("<i2").astype(np.float64) / 32767.0
            elif element.semantic == "TEXCOORD" and element.format == TEXCOORD_USHORT2:
                target = uvs.setdefault(
                    element.channel, np.full((count, 2), np.nan, dtype=np.float32)
                )
                target[start:stop] = cell.view("<u2").astype(np.float32) / 65535.0
            elif element.semantic == "COLOR" and element.format == COLOR_UBYTE4:
                if colours is None:
                    colours = np.full((count, 4), np.nan, dtype=np.float32)
                colours[start:stop] = cell.astype(np.float32) / 255.0
        start = stop

    known = ~np.isnan(normal_yz[:, 0])
    if known.any():
        normals = np.full((count, 3), np.nan, dtype=np.float64)
        normals[known, 0] = shorts[known, 3].astype(np.float64) / 32767.0
        normals[known, 1:] = normal_yz[known]
        length = np.linalg.norm(normals[known], axis=1, keepdims=True)
        normals[known] /= np.maximum(length, 1e-9)
        out.normals = normals.astype(np.float32)
    out.uvs = uvs
    out.colours = colours
    return out


def read_geometry(model: ModelFile, *, lod: int | None = 0) -> Geometry:
    """
    Positions, triangles and attributes for one `.modelbin`, in world metres.

    `lod` keeps one level of detail, the finest by default. None keeps every
    level - which is only useful for inspecting a file, since the levels cover
    the same ground and overlapping them exports the same square metre four
    times over.

    Only the formats terrain actually uses. Refusing a format outright is the
    point: a decoder that guesses at an unknown vertex format returns a mesh
    rather than an error, and a wrong mesh looks like a right one until someone
    puts it next to the track.
    """
    import numpy as np

    by = _chunks_by_tag(model)
    for needed in ("Mesh", "VLay", "VerB", "IndB"):
        if needed not in by:
            raise UnsupportedForza(f"{model.path.name} has no {needed} chunk")

    vertices = read_vertices(model)

    indices = read_buffer(model.raw(by["IndB"][0]))
    if indices.format != INDEX_SHORT or indices.stride != 2:
        raise UnsupportedForza(
            f"{model.path.name} stores indices as format {indices.format} at "
            f"{indices.stride} bytes; only unsigned shorts are implemented"
        )
    if indices.count % 3:
        raise UnsupportedForza(
            f"{model.path.name} has {indices.count} indices, which is not whole "
            "triangles"
        )
    faces = np.frombuffer(indices.data, dtype="<u2").reshape(-1, 3).astype(np.int32)

    labels: tuple[str, ...] = ()
    masks = None
    indices = None
    owner = None
    submeshes = read_submeshes(model)
    if submeshes:
        per_triangle = np.empty(len(faces), dtype=object)
        masks = np.zeros(len(faces), dtype=np.uint16)
        indices = np.full(len(faces), 0xFFFF, dtype=np.uint16)
        owner = np.zeros(len(faces), dtype=np.int32)
        for number, piece in enumerate(submeshes):
            start = piece.first // 3
            per_triangle[start : start + piece.triangles] = piece.material
            masks[start : start + piece.triangles] = piece.lods
            indices[start : start + piece.triangles] = piece.index
            owner[start : start + piece.triangles] = number
        if lod is not None:
            keep = (masks & (1 << (lod + 1))) != 0
            faces, per_triangle, masks = faces[keep], per_triangle[keep], masks[keep]
            indices, owner = indices[keep], owner[keep]
        labels = tuple(per_triangle.tolist())

    return Geometry(
        positions=vertices.positions,
        faces=faces,
        materials=labels,
        lods=masks,
        material_index=indices,
        submesh=owner,
        variants=tuple(piece.variants for piece in submeshes),
        normals=vertices.normals,
        uvs=vertices.uvs,
        colours=vertices.colours,
    )
