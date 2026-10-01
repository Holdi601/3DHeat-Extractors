"""
Reading Source 2 `.vpk` archives — Counter-Strike 2, Dota, Half-Life: Alyx.

The friendliest of the four container formats here, and the only one Valve
documents. A map is one self-contained `.vpk`, which is convenient: pointing this
at `maps/de_dust2.vpk` gets that map and nothing else.

The directory
-------------
Three nested loops, each ending on an empty string, which is why a reader that
expects a count finds none::

    extension            "vmdl_c", then "vtex_c", then ""
      path               "maps/de_dust2", then ""
        file name        "world_layer0", then ""
          crc, preload size, archive index, offset, length, 0xffff
          [preload bytes]

Paths are stored once per group rather than per file, so the tree is compact and
the full path has to be reassembled from all three parts.

Split archives
--------------
`pak01_dir.vpk` holds only the directory; the data sits in `pak01_000.vpk`,
`pak01_001.vpk` and so on, and an entry's `archive` says which. An index of
0x7fff means the bytes are in the directory file itself, after the tree — that
sentinel is the one detail worth stating, because reading it as a real index
sends the reader looking for `pak01_32767.vpk`.

What this does not do is decode Source 2's compiled assets. A `.vmdl_c` is a
chunked binary of its own, and turning one into geometry is the same kind of job
as the Unreal and Frostbite mesh formats. This gets the files out; the rest is
separate.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path

#: Little-endian at the very start of every VPK.
VPK_MAGIC = 0x55AA1234

#: Versions with a directory this reader understands.
SUPPORTED = (1, 2)

#: Header length, which differs between the two versions.
HEADER_V1 = 12
HEADER_V2 = 28

#: An entry with this archive index keeps its bytes in the directory file,
#: immediately after the tree, rather than in a numbered part.
IN_DIRECTORY = 0x7FFF

#: Every entry's record ends with this. It is the cheapest possible check that
#: the tree is still being read at the right offset.
TERMINATOR = 0xFFFF


class UnsupportedVpk(Exception):
    """Not a VPK this reader can read; the message says why."""


@dataclass(frozen=True)
class VpkEntry:
    """One file inside the archive."""

    path: str
    crc: int
    archive: int
    offset: int
    length: int
    #: Small files are stored inline in the directory, whole.
    preload: bytes = b""

    @property
    def size(self) -> int:
        return self.length + len(self.preload)


class _Reader:
    def __init__(self, data: bytes, at: int = 0):
        self.data = data
        self.at = at

    def u16(self) -> int:
        value = struct.unpack_from("<H", self.data, self.at)[0]
        self.at += 2
        return value

    def u32(self) -> int:
        value = struct.unpack_from("<I", self.data, self.at)[0]
        self.at += 4
        return value

    def cstring(self) -> str:
        end = self.data.find(b"\x00", self.at)
        if end < 0:
            raise UnsupportedVpk("directory ended mid-string")
        out = self.data[self.at : end].decode("utf-8", "replace")
        self.at = end + 1
        return out

    def take(self, count: int) -> bytes:
        out = self.data[self.at : self.at + count]
        if len(out) != count:
            raise UnsupportedVpk("directory ran past the end of the file")
        self.at += count
        return out


class VpkArchive:
    """An opened VPK: what is in it, and the bytes of any one entry."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        data = self.path.read_bytes()
        if len(data) < HEADER_V1:
            raise UnsupportedVpk(f"{self.path.name} is too small to be a VPK")

        magic, version = struct.unpack_from("<II", data, 0)
        if magic != VPK_MAGIC:
            raise UnsupportedVpk(
                f"{self.path.name} does not start with the VPK signature"
            )
        if version not in SUPPORTED:
            raise UnsupportedVpk(f"{self.path.name}: VPK version {version}")

        self.version = version
        tree_size = struct.unpack_from("<I", data, 8)[0]
        header = HEADER_V1 if version == 1 else HEADER_V2
        if header + tree_size > len(data):
            raise UnsupportedVpk(
                f"{self.path.name} declares a {tree_size}-byte directory that does "
                f"not fit in {len(data)} bytes"
            )

        #: Where data stored in this file itself begins.
        self._inline_base = header + tree_size
        self.entries: dict[str, VpkEntry] = {}
        self._read_tree(data, header, tree_size)

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, path: str) -> bool:
        return path in self.entries

    def _read_tree(self, data: bytes, at: int, tree_size: int) -> None:
        reader = _Reader(data, at)
        end = at + tree_size
        while reader.at < end:
            extension = reader.cstring()
            if not extension:
                break
            while True:
                folder = reader.cstring()
                if not folder:
                    break
                while True:
                    name = reader.cstring()
                    if not name:
                        break
                    crc = reader.u32()
                    preload_size = reader.u16()
                    archive = reader.u16()
                    offset = reader.u32()
                    length = reader.u32()
                    terminator = reader.u16()
                    if terminator != TERMINATOR:
                        raise UnsupportedVpk(
                            f"{self.path.name}: entry record ended with "
                            f"0x{terminator:04x}, not 0x{TERMINATOR:04x}; the "
                            "directory is not being read at the right offset"
                        )
                    preload = reader.take(preload_size) if preload_size else b""

                    # " " is how the format spells "no folder" and "no
                    # extension"; joining them literally gives paths like
                    # " /file. ".
                    parts = [p for p in (folder.strip(), name.strip()) if p and p != " "]
                    full = "/".join(parts)
                    if extension.strip() and extension.strip() != " ":
                        full = f"{full}.{extension.strip()}"
                    self.entries[full] = VpkEntry(
                        path=full,
                        crc=crc,
                        archive=archive,
                        offset=offset,
                        length=length,
                        preload=preload,
                    )

    # -- reading ----------------------------------------------------------

    def list(self, pattern: str | None = None) -> list[str]:
        """Paths inside the archive, optionally filtered by a substring."""
        paths = sorted(self.entries)
        if pattern:
            needle = pattern.lower()
            paths = [p for p in paths if needle in p.lower()]
        return paths

    def part_for(self, entry: VpkEntry) -> Path:
        """
        The file an entry's bytes live in.

        A directory-only archive is named `pak01_dir.vpk` and its parts
        `pak01_000.vpk`; a self-contained one keeps everything in itself.
        """
        if entry.archive == IN_DIRECTORY:
            return self.path
        stem = self.path.stem
        if stem.endswith("_dir"):
            stem = stem[: -len("_dir")]
        return self.path.with_name(f"{stem}_{entry.archive:03d}.vpk")

    def read(self, path: str) -> bytes:
        """
        The bytes of one entry.

        A small file may be stored entirely in the directory's preload, in which
        case there is nothing to read from a part at all — and the length field
        is then zero, which is not an error.
        """
        entry = self.entries.get(path)
        if entry is None:
            raise KeyError(path)
        if entry.length == 0:
            return entry.preload

        part = self.part_for(entry)
        if not part.is_file():
            raise UnsupportedVpk(
                f"{path} lives in {part.name}, which is not beside {self.path.name}"
            )
        base = self._inline_base if entry.archive == IN_DIRECTORY else 0
        with open(part, "rb") as handle:
            handle.seek(base + entry.offset)
            body = handle.read(entry.length)
        return entry.preload + body

    def verify(self, path: str) -> bool:
        """
        Whether an entry's bytes match the CRC the directory recorded.

        Source 2 stores one per file, which makes extraction checkable against
        the archive's own claim rather than against a guess.
        """
        import zlib

        entry = self.entries[path]
        return zlib.crc32(self.read(path)) & 0xFFFFFFFF == entry.crc
