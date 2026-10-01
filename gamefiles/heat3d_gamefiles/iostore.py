"""
Reading Unreal Engine 5 IoStore containers: a `.utoc` table of contents and
the `.ucas` data it indexes.

Unreal 5 ships most content this way rather than in `.pak` files (a `.pak` is
still there, but holds little more than a footer). A container is a table of
chunks - a package's export data, its bulk data, the script objects - each
stored as a run of compression blocks, plus, when the container is indexed, a
directory tree naming the package files.

Like `pak.py`, this reads containers and nothing more: no key material, no key
recovery. An encrypted container is refused with a message saying so. Blocks
compressed with Oodle are decompressed through a library already on the
machine (see `oodle.py`).

Layout of the table of contents, version 8 (Unreal 5.6), all little-endian:

    header (144 bytes): magic, version, header size, entry count, block count,
        block entry size, compression method count and name length, block
        size, directory index size, partition count, container id, key GUID,
        flags, perfect-hash seed count, partition size, chunks without a
        perfect hash
    chunk ids          entries x 12: id u64, index u16 (big-endian), pad, type
    offsets, lengths   entries x 10: two 40-bit big-endian numbers
    hash seeds         seeds x 4, then chunks-without-hash x 4
    blocks             blocks x 12: offset 40 bits, compressed size 24 bits,
                       uncompressed size 24 bits, method u8 (0 = stored)
    method names       count x name length, NUL-padded
    signatures         when signed
    directory index    when indexed
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = b"-==--==--==--==-"

#: Chunk types of Unreal 5's `EIoChunkType`.
EXPORT_BUNDLE_DATA = 1
BULK_DATA = 2
OPTIONAL_BULK_DATA = 3
MEMORY_MAPPED_BULK_DATA = 4
SCRIPT_OBJECTS = 5
CONTAINER_HEADER = 6

_COMPRESSED, _ENCRYPTED, _SIGNED, _INDEXED = 1, 2, 4, 8
_NONE = 0xFFFFFFFF


class IoStoreError(Exception):
    """A container that cannot be read, and why."""


@dataclass(frozen=True)
class Header:
    version: int
    header_size: int
    entries: int
    blocks: int
    block_entry_size: int
    methods: int
    method_length: int
    block_size: int
    directory_size: int
    partitions: int
    container_id: int
    key_guid: bytes
    flags: int
    hash_seeds: int
    partition_size: int
    without_hash: int

    @property
    def encrypted(self) -> bool:
        return bool(self.flags & _ENCRYPTED)


def read_header(data: bytes) -> Header:
    if data[:16] != MAGIC:
        raise IoStoreError("not an IoStore table of contents")
    version = data[16]
    if version < 5:
        raise IoStoreError(f"table of contents version {version} is older than this reads")
    fields = struct.unpack_from("<9I", data, 20)
    (container_id,) = struct.unpack_from("<Q", data, 56)
    (seeds,) = struct.unpack_from("<I", data, 84)
    (partition_size,) = struct.unpack_from("<Q", data, 88)
    (without_hash,) = struct.unpack_from("<I", data, 96)
    return Header(
        version=version,
        header_size=fields[0],
        entries=fields[1],
        blocks=fields[2],
        block_entry_size=fields[3],
        methods=fields[4],
        method_length=fields[5],
        block_size=fields[6],
        directory_size=fields[7],
        partitions=max(1, fields[8]),
        container_id=container_id,
        key_guid=bytes(data[64:80]),
        flags=data[80],
        hash_seeds=seeds,
        partition_size=partition_size,
        without_hash=without_hash,
    )


@dataclass
class Container:
    """One `.utoc` and its `.ucas` partitions."""

    path: Path
    header: Header
    chunk_ids: list[tuple[int, int, int]] = field(default_factory=list)
    spans: list[tuple[int, int]] = field(default_factory=list)
    blocks: list[tuple[int, int, int, int]] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    #: Package file path -> entry, where the container is indexed.
    files: dict[str, int] = field(default_factory=dict)
    mount: str = ""
    by_chunk: dict[tuple[int, int], int] = field(default_factory=dict)

    @classmethod
    def open(cls, utoc: str | Path, *, index: bool = True) -> "Container":
        utoc = Path(utoc)
        data = utoc.read_bytes()
        header = read_header(data)
        if header.encrypted:
            raise IoStoreError(
                f"{utoc.name} is encrypted. This tool holds no keys and recovers none."
            )
        out = cls(utoc, header)
        at = header.header_size
        for i in range(header.entries):
            chunk_id, index_be, _pad, kind = struct.unpack_from("<QHBB", data, at + i * 12)
            index_no = ((index_be & 0xFF) << 8) | (index_be >> 8)
            out.chunk_ids.append((chunk_id, index_no, kind))
            out.by_chunk.setdefault((chunk_id, kind), i)
        at += header.entries * 12
        for i in range(header.entries):
            raw = data[at + i * 10 : at + i * 10 + 10]
            out.spans.append((int.from_bytes(raw[0:5], "big"), int.from_bytes(raw[5:10], "big")))
        at += header.entries * 10
        at += header.hash_seeds * 4 + header.without_hash * 4
        for i in range(header.blocks):
            raw = data[at + i * header.block_entry_size : at + i * header.block_entry_size + 12]
            out.blocks.append(
                (
                    int.from_bytes(raw[0:5], "little"),
                    int.from_bytes(raw[5:8], "little"),
                    int.from_bytes(raw[8:11], "little"),
                    raw[11],
                )
            )
        at += header.blocks * header.block_entry_size
        for i in range(header.methods):
            name = data[at + i * header.method_length : at + (i + 1) * header.method_length]
            out.methods.append(name.split(b"\0")[0].decode("ascii", "replace"))
        at += header.methods * header.method_length
        if header.flags & _SIGNED:
            (hash_size,) = struct.unpack_from("<i", data, at)
            at += 4 + hash_size * 2 + 20 * header.blocks
        if index and header.flags & _INDEXED and header.directory_size:
            out._directory(data[at : at + header.directory_size])
        return out

    def _directory(self, d: bytes) -> None:
        at = 0

        def text(at: int) -> tuple[str, int]:
            (length,) = struct.unpack_from("<i", d, at)
            at += 4
            if length == 0:
                return "", at
            if length < 0:
                return d[at : at - length * 2].decode("utf-16-le").rstrip("\0"), at - length * 2
            return d[at : at + length].decode("utf-8", "replace").rstrip("\0"), at + length

        self.mount, at = text(at)
        (count,) = struct.unpack_from("<i", d, at)
        at += 4
        directories = [struct.unpack_from("<4I", d, at + i * 16) for i in range(count)]
        at += count * 16
        (count,) = struct.unpack_from("<i", d, at)
        at += 4
        files = [struct.unpack_from("<3I", d, at + i * 12) for i in range(count)]
        at += count * 12
        (count,) = struct.unpack_from("<i", d, at)
        at += 4
        strings = []
        for _ in range(count):
            value, at = text(at)
            strings.append(value)
        if not directories:
            return
        stack = [(0, self.mount.rstrip("/"))]
        while stack:
            number, prefix = stack.pop()
            name, first_child, _next, first_file = directories[number]
            here = prefix if name == _NONE else f"{prefix}/{strings[name]}"
            f = first_file
            while f != _NONE:
                file_name, following, entry = files[f]
                self.files[f"{here}/{strings[file_name]}"] = entry
                f = following
            child = first_child
            while child != _NONE:
                stack.append((child, here))
                child = directories[child][2]

    def size(self, entry: int) -> int:
        return self.spans[entry][1]

    def read(self, entry: int, length: int | None = None) -> bytes:
        """One chunk's bytes, decompressed; the first `length` of them if given."""
        offset, size = self.spans[entry]
        if length is not None:
            size = min(size, length)
        return self.read_range(offset, size)

    def read_range(self, offset: int, size: int) -> bytes:
        if size <= 0:
            return b""
        block_size = self.header.block_size
        first = offset // block_size
        last = (offset + size - 1) // block_size
        out = bytearray()
        handles: dict[int, object] = {}
        try:
            for number in range(first, last + 1):
                where, packed, unpacked, method = self.blocks[number]
                partition = where // self.header.partition_size if self.header.partition_size else 0
                handle = handles.get(partition)
                if handle is None:
                    handle = handles[partition] = open(self._partition(partition), "rb")
                handle.seek(where - partition * self.header.partition_size if self.header.partition_size else where)
                raw = handle.read(packed)
                out += self._decompress(raw, unpacked, method)
        finally:
            for handle in handles.values():
                handle.close()
        start = offset - first * block_size
        return bytes(out[start : start + size])

    def _partition(self, number: int) -> Path:
        if number == 0:
            return self.path.with_suffix(".ucas")
        return self.path.with_name(f"{self.path.stem}_s{number}.ucas")

    def _decompress(self, raw: bytes, unpacked: int, method: int) -> bytes:
        if method == 0:
            return raw[:unpacked]
        name = self.methods[method - 1].lower() if method - 1 < len(self.methods) else "?"
        if name == "oodle":
            from .oodle import decompress_oodle

            return decompress_oodle(raw, unpacked)
        if name == "zlib":
            return zlib.decompress(raw)[:unpacked]
        if name == "lz4":
            import lz4.block

            return lz4.block.decompress(raw, uncompressed_size=unpacked)
        raise IoStoreError(f"compression method {name!r} is not supported")

    def chunk(self, chunk_id: int, kind: int) -> int | None:
        return self.by_chunk.get((chunk_id, kind))


class Store:
    """
    Every container of a game, as one place to find a package or a chunk.

    Packages are found by their file path in the directory indexes, or by the
    `/Game/...` name other packages import them under.
    """

    def __init__(self, paks: str | Path) -> None:
        self.paks = Path(paks)
        self.containers: list[Container] = []
        self.files: dict[str, tuple[int, int]] = {}
        #: Lower-cased path -> the path as the container spells it.
        self.names: dict[str, str] = {}
        self.global_container: Container | None = None
        for utoc in sorted(self.paks.glob("*.utoc")):
            try:
                container = Container.open(utoc)
            except (IoStoreError, OSError, struct.error):
                continue
            number = len(self.containers)
            self.containers.append(container)
            if utoc.stem.lower() == "global":
                self.global_container = container
            for path, entry in container.files.items():
                self.files[path.lower()] = (number, entry)
                self.names[path.lower()] = path
        if not self.containers:
            raise IoStoreError(f"no readable IoStore containers in {self.paks}")

    def read_file(self, path: str) -> bytes:
        found = self.files.get(path.lower())
        if found is None:
            raise FileNotFoundError(path)
        return self.containers[found[0]].read(found[1])

    def find(self, contains: str, suffix: str = "") -> list[str]:
        contains, suffix = contains.lower(), suffix.lower()
        return sorted(p for p in self.files if contains in p and p.endswith(suffix))

    def chunk_of(self, path: str) -> tuple[Container, int] | None:
        """The container and chunk id holding a package file's export data."""
        found = self.files.get(path.lower())
        if found is None:
            return None
        container = self.containers[found[0]]
        chunk_id, _index, _kind = container.chunk_ids[found[1]]
        return container, chunk_id

    def bulk(self, path: str, kind: int = BULK_DATA) -> bytes | None:
        """A package's bulk data chunk (`.ubulk`), wherever it is stored."""
        found = self.chunk_of(path)
        if found is None:
            return None
        _container, chunk_id = found
        for container in self.containers:
            entry = container.chunk(chunk_id, kind)
            if entry is not None:
                return container.read(entry)
        return None
