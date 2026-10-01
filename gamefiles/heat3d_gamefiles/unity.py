"""
Reading Unity `SerializedFile` containers and asset bundles.

Unity ships a level as a set of serialised object graphs: `globalgamemanagers`,
`sharedassets*.assets`, `resources.assets`, `level*`, and any number of
`AssetBundle` files. Each holds a table of objects, each object tagged with a
class id, and the geometry is in the ones tagged `Mesh`.

The header, which is where this goes wrong quietly
--------------------------------------------------
The first sixteen bytes are four big-endian `uint32`s kept for compatibility. In
format 22 — Unity 2020 and later — three of them are zero and the real values
follow, because a file may exceed 4 GB::

    [legacy metadata size : 4]  [legacy file size : 4]
    [format version : 4]        [legacy data offset : 4]
    [endianness : 1]            [reserved : 3]
    [metadata size : 4]  [file size : 8]  [data offset : 8]  [unknown : 8]

so the version string begins at offset 48. Note that the second `metadata size`
is a `uint32` while the two beside it are `int64`; assuming all three are 64-bit
puts the version string four bytes late, which still decodes to something
string-shaped and sends everything after it astray.

The header is big-endian. Everything after it follows the endianness byte, which
is 0 — little-endian — on every desktop build.

What is read and what is not
----------------------------
This reads the container and the object table: every object's class, its byte
range, and its path id. That is enough to find the meshes and to report what a
file holds, which is what the exporter needs.

It does **not** carry a general Unity deserialiser. Shipped builds are written
without type trees — the schema is compiled into the player, not the file — so
each class has to be parsed against a layout known from its Unity version. `mesh`
covers the one class this project actually wants; anything else is reported by
class and left alone rather than guessed at.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

#: The class ids worth naming. Unity has several hundred; these are the ones
#: that matter for pulling geometry and its placement out of a scene.
CLASS_NAMES = {
    1: "GameObject",
    4: "Transform",
    23: "MeshRenderer",
    33: "MeshFilter",
    43: "Mesh",
    65: "BoxCollider",
    64: "MeshCollider",
    114: "MonoBehaviour",
    115: "MonoScript",
    142: "AssetBundle",
    224: "RectTransform",
    1001: "PrefabInstance",
}

#: Bundle signatures. `UnityFS` is everything modern; the older two appear in
#: pre-5.3 builds and are recognised so the error can say so.
BUNDLE_SIGNATURES = (b"UnityFS", b"UnityWeb", b"UnityRaw", b"UnityArchive")


class UnsupportedUnity(Exception):
    """Not a Unity container this reader can read; the message says why."""


@dataclass(frozen=True)
class UnityObject:
    """One serialised object: where it is, and what class it is."""

    path_id: int
    offset: int
    size: int
    class_id: int

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.class_id, f"class{self.class_id}")


@dataclass
class SerializedFile:
    """An opened `.assets` / `level` / `globalgamemanagers` file."""

    path: Path
    version: int
    unity_version: str
    platform: int
    has_type_tree: bool
    data_offset: int
    objects: list[UnityObject] = field(default_factory=list)
    #: Other files this one refers to, in load order; index 0 means "this file".
    externals: list[str] = field(default_factory=list)
    _data: bytes = b""

    def __len__(self) -> int:
        return len(self.objects)

    def of_class(self, name: str) -> list[UnityObject]:
        """Every object of a named class, e.g. `of_class("Mesh")`."""
        wanted = {k for k, v in CLASS_NAMES.items() if v == name}
        if not wanted:
            raise KeyError(f"unknown class name {name!r}")
        return [o for o in self.objects if o.class_id in wanted]

    def counts(self) -> dict[str, int]:
        """How many of each class, for reporting what a file holds."""
        out: dict[str, int] = {}
        for o in self.objects:
            out[o.class_name] = out.get(o.class_name, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def raw(self, obj: UnityObject) -> bytes:
        """The bytes of one object, from the data section."""
        start = self.data_offset + obj.offset
        return self._data[start : start + obj.size]


class _Cursor:
    """A cursor that knows its endianness, because the header and body differ."""

    def __init__(self, data: bytes, little: bool = True, at: int = 0):
        self.data = data
        self.at = at
        self.e = "<" if little else ">"

    def _take(self, code: str, width: int):
        if self.at + width > len(self.data):
            raise UnsupportedUnity("file ended mid-record")
        (value,) = struct.unpack_from(self.e + code, self.data, self.at)
        self.at += width
        return value

    def u8(self) -> int:
        return self._take("B", 1)

    def i16(self) -> int:
        return self._take("h", 2)

    def i32(self) -> int:
        return self._take("i", 4)

    def u32(self) -> int:
        return self._take("I", 4)

    def i64(self) -> int:
        return self._take("q", 8)

    def skip(self, count: int) -> None:
        self.at += count

    def align(self, to: int = 4) -> None:
        self.at = (self.at + to - 1) // to * to

    def cstring(self) -> str:
        end = self.data.index(b"\x00", self.at)
        out = self.data[self.at : end].decode("utf-8", "replace")
        self.at = end + 1
        return out


def read_serialized(path: str | Path, data: bytes | None = None) -> SerializedFile:
    """
    Parse a SerializedFile's header and object table.

    `data` lets a file that was unpacked from a bundle in memory be parsed
    without being written out first.
    """
    path = Path(path)
    if data is None:
        data = path.read_bytes()
    if len(data) < 48:
        raise UnsupportedUnity(f"{path.name} is too small to be a SerializedFile")

    if data[:8].startswith(BUNDLE_SIGNATURES):
        raise UnsupportedUnity(
            f"{path.name} is an asset bundle, not a bare SerializedFile — "
            "open it with read_bundle() first"
        )

    head = _Cursor(data, little=False)
    head.skip(8)  # legacy metadata size and file size, zero in format 22
    version = head.u32()
    head.skip(4)  # legacy data offset

    if version < 9 or version > 30:
        raise UnsupportedUnity(
            f"{path.name}: SerializedFile format {version}; this reader handles 9–30"
        )

    little = head.u8() == 0
    head.skip(3)  # reserved

    if version >= 22:
        # A uint32 between two int64s. See the module docstring.
        head.u32()  # metadata size, not needed once the offsets are known
        head.i64()  # file size
        data_offset = head.i64()
        head.i64()  # unknown
    else:
        raise UnsupportedUnity(
            f"{path.name}: format {version} predates Unity 2020 and stores its "
            "offsets in the legacy header; no file here uses it, so it is not "
            "implemented rather than implemented untested"
        )

    body = _Cursor(data, little=little, at=head.at)
    unity_version = body.cstring()
    platform = body.i32()
    has_type_tree = bool(body.u8())

    types = _read_types(body, version, has_type_tree)

    objects: list[UnityObject] = []
    for _ in range(body.i32()):
        body.align(4)
        path_id = body.i64()
        offset = body.i64() if version >= 22 else body.u32()
        size = body.u32()
        type_index = body.i32()
        class_id = types[type_index] if 0 <= type_index < len(types) else -1
        objects.append(UnityObject(path_id, offset, size, class_id))

    externals = _read_externals(body, version)

    return SerializedFile(
        path=path,
        version=version,
        unity_version=unity_version,
        platform=platform,
        has_type_tree=has_type_tree,
        data_offset=data_offset,
        objects=objects,
        externals=externals,
        _data=data,
    )


def _read_types(body: _Cursor, version: int, has_type_tree: bool) -> list[int]:
    """The class id of each type entry, in order; objects index into this."""
    types: list[int] = []
    for _ in range(body.i32()):
        class_id = body.i32()
        body.u8()  # is stripped
        body.i16()  # script type index
        # A MonoBehaviour carries an extra hash identifying its script, because
        # the class id alone does not distinguish two different scripts.
        if class_id == 114:
            body.skip(16)
        body.skip(16)  # old type hash
        if has_type_tree:
            raise UnsupportedUnity(
                "this file embeds a type tree; shipped builds do not, and the "
                "editor-only path is not implemented"
            )
    # The loop above has to run before the count can be returned, so collect as
    # it goes rather than re-reading.
        types.append(class_id)
    return types


def _read_externals(body: _Cursor, version: int) -> list[str]:
    """Files this one refers to. A file id in a pointer indexes into this, from 1."""
    if version >= 14:
        for _ in range(body.i32()):  # script types
            body.align(4)
            body.i32()
            body.i64()
    externals: list[str] = []
    try:
        for _ in range(body.i32()):
            body.cstring()  # temp empty
            body.skip(16)  # GUID
            body.i32()  # type
            externals.append(body.cstring())
    except (UnsupportedUnity, ValueError):
        # The external table is the last thing in the metadata and is only used
        # for cross-file references. A build that ends the section differently
        # should not cost us the object table, which is what matters.
        pass
    return externals


# ---------------------------------------------------------------------------
# Bundles


@dataclass
class BundleEntry:
    """One file inside a `UnityFS` bundle."""

    name: str
    offset: int
    size: int
    flags: int


class UnityBundle:
    """
    A `UnityFS` bundle: a compressed container holding SerializedFiles.

    The block list and directory are themselves compressed, with the method in
    the low bits of a flags word: 0 stored, 1 LZMA, 2 and 3 LZ4HC.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        raw = self.path.read_bytes()
        if not raw.startswith(b"UnityFS"):
            found = raw[:16].split(b"\x00")[0]
            raise UnsupportedUnity(
                f"{self.path.name} is not a UnityFS bundle (signature {found!r})"
            )
        self.entries: dict[str, BundleEntry] = {}
        self._blocks: bytes = b""
        self._parse(raw)

    def __len__(self) -> int:
        return len(self.entries)

    def _parse(self, raw: bytes) -> None:
        head = _Cursor(raw, little=False)
        head.cstring()  # signature
        self.format = head.i32()
        self.player_version = head.cstring()
        self.engine_version = head.cstring()
        head.i64()  # total bundle size
        compressed_blocks_size = head.u32()
        uncompressed_blocks_size = head.u32()
        flags = head.u32()

        if self.format >= 7:
            head.align(16)

        info = raw[head.at : head.at + compressed_blocks_size]
        if flags & 0x80:  # directory sits at the end of the file instead
            info = raw[-compressed_blocks_size:]
        else:
            head.at += compressed_blocks_size

        info = _bundle_decompress(info, flags & 0x3F, uncompressed_blocks_size)

        meta = _Cursor(info, little=False)
        meta.skip(16)  # hash
        block_sizes = []
        for _ in range(meta.i32()):
            block_sizes.append((meta.u32(), meta.u32(), meta.i16()))

        blocks = bytearray()
        at = head.at
        for uncompressed, compressed, block_flags in block_sizes:
            chunk = raw[at : at + compressed]
            at += compressed
            blocks += _bundle_decompress(chunk, block_flags & 0x3F, uncompressed)
        self._blocks = bytes(blocks)

        for _ in range(meta.i32()):
            offset = meta.i64()
            size = meta.i64()
            entry_flags = meta.u32()
            name = meta.cstring()
            self.entries[name] = BundleEntry(name, offset, size, entry_flags)

    def read(self, name: str) -> bytes:
        entry = self.entries[name]
        return self._blocks[entry.offset : entry.offset + entry.size]

    def serialized_files(self) -> list[SerializedFile]:
        """Every entry in the bundle that parses as a SerializedFile."""
        out = []
        for name in self.entries:
            try:
                out.append(read_serialized(self.path / name, self.read(name)))
            except UnsupportedUnity:
                continue
        return out


def _bundle_decompress(payload: bytes, method: int, expected: int) -> bytes:
    if method == 0:
        return payload
    if method == 1:
        import lzma

        # Unity writes a 5-byte LZMA property header and no size field, so the
        # size has to be spliced in before the standard decoder will take it.
        return lzma.decompress(
            payload[:5] + struct.pack("<Q", expected) + payload[5:],
            format=lzma.FORMAT_ALONE,
        )
    if method in (2, 3):
        try:
            import lz4.block
        except ImportError as exc:
            raise UnsupportedUnity(
                "this bundle is LZ4-compressed, which needs the `lz4` package"
            ) from exc
        return lz4.block.decompress(payload, uncompressed_size=expected)
    raise UnsupportedUnity(f"bundle compression method {method} is not known")
