"""
ForzaTech's `.minizip`: the container the world is actually stored in.

A Forza track ships four of these, tens of gigabytes each for an open-world
map, and between them they hold every model, texture and placement file the game
streams - hundreds of thousands of entries in the largest alone. Nothing else in the track folder
holds geometry, so anything that wants the map has to come through here.

The awkward part is that the archive holds no names. Not a hash of a name, not a
truncated name: no name table at all, and no per-entry header either. An entry is
addressed by its ordinal and nothing else. The names live outside, in the
`ChunkContentsMiniZip*.txt` that sits beside the archive - one line per entry, in
entry order, which is what makes the pairing work at all. Lose that file and the
archive is 39 GB of anonymous blobs.

The layout
----------
::

    u32   'PGZP'
    u32   version                      (101 here)
    u32   header bytes                 (32)
    u32   entries                      (411,013 - and the manifest's line count)
    u32   bundles                      (81,657)
    u32   entries per segment          (512)
    u32   segments                     (803)
    u32   0
    u32   bundle_start[bundles + 1]    first entry of each bundle
    ...   padding to an eight-byte boundary
    ... then, repeated `segments` times:
    u64   segment base, from the start of the file
    u32   (offset, size, flags) x entries-per-segment
                                       (the last segment holds only what is left)

Two details are worth stating because they are where a reader goes wrong.

**The base is what makes 32-bit offsets work.** A 39.7 GB archive cannot be
addressed by a u32, and it does not try to: each segment carries its own absolute
base and the 512 offsets after it are relative to that. Read the records as one
flat array and the offsets appear to jump backwards every 512 entries, which
looks like a corrupt file rather than the segment boundary it is.

The base is sixty-four bits, and that is worth saying because it does not look
it. Every segment in the first four gigabytes — hundreds of them — has a zero
high word, so reading the base as a u32 followed by a padding field parses
cleanly, extracts correctly, and passes any spot check taken from the start of
the archive. It fails only past the 4 GB mark, and it fails silently: the offset
wraps, the entry decompresses from whatever happens to be there, and what comes
back is noise that is the right length.

**No entry stores its compressed length.** It is the distance to the next
entry's offset - and for the last entry of a segment, the distance to the next
segment's base. So an entry cannot be read without looking at its neighbour,
which is why this reads the whole index (about 5 MB) rather than seeking to one
record.

Payloads are LZ4 blocks, raw deflate, or stored verbatim; the low five bits of
the flags say which.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"PGZP"

#: Bytes before the bundle table.
HEADER = 32

#: The low five bits of an entry's flags name its codec.
CODEC = 0x1F
STORED = 0x00
DEFLATE = 0x08
LZ4 = 0x1F

#: Bits 12 and 13 count the padding bytes after an entry, which is how the next
#: entry starts on a four-byte boundary. They matter more than they look: the
#: compressed length is the distance to the next entry *minus* these, and LZ4
#: refuses a block whose declared input runs past its end, so an entry with one
#: stray byte after it fails outright rather than decoding and ignoring the tail.
PADDING = 0x3000
PADDING_SHIFT = 12


class UnsupportedMiniZip(Exception):
    """Not a `.minizip` this reader can read."""


@dataclass(frozen=True)
class Entry:
    """One stored file: where it is, how big it was, and how it is packed."""

    index: int
    offset: int
    size: int
    stored_size: int
    flags: int

    @property
    def codec(self) -> int:
        return self.flags & CODEC

    @property
    def padding(self) -> int:
        """Alignment bytes after the payload, which are not part of it."""
        return (self.flags & PADDING) >> PADDING_SHIFT


class MiniZip:
    """
    An opened `.minizip`, addressed by entry number.

    The index is about 5 MB and the archive is tens of gigabytes, so this reads
    the index once and then seeks for each payload.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = self.path.open("rb")
        try:
            self._read_index()
        except Exception:
            self._fh.close()
            raise

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "MiniZip":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __len__(self) -> int:
        return self.count

    def _read_index(self) -> None:
        head = self._fh.read(HEADER)
        if len(head) < HEADER or head[:4] != MAGIC:
            raise UnsupportedMiniZip(f"{self.path.name} does not begin with {MAGIC!r}")
        (
            _magic,
            self.version,
            header_bytes,
            self.count,
            self.bundles,
            self.per_segment,
            self.segments,
            _zero,
        ) = struct.unpack("<8I", head)
        if header_bytes != HEADER:
            raise UnsupportedMiniZip(
                f"{self.path.name} declares a {header_bytes}-byte header, not {HEADER}"
            )
        if self.per_segment == 0:
            raise UnsupportedMiniZip(f"{self.path.name} declares no entries per segment")
        expected = -(-self.count // self.per_segment)
        if self.segments != expected:
            raise UnsupportedMiniZip(
                f"{self.path.name}: {self.count} entries at {self.per_segment} per "
                f"segment needs {expected} segments, but it declares {self.segments}"
            )

        # The bundle table is read but not used for addressing: it groups entries
        # into the units the build system zipped together, which matters to the
        # game's streaming and not to reading one file out.
        table = self._fh.read((self.bundles + 1) * 4)
        if len(table) < (self.bundles + 1) * 4:
            raise UnsupportedMiniZip(f"{self.path.name}: the bundle table is truncated")
        self.bundle_start = list(struct.unpack(f"<{self.bundles + 1}I", table))

        # The segment bases are 64-bit and aligned to eight, so an archive with
        # an even bundle count - whose table ends four bytes short of alignment
        # - carries a padding word here and an archive with an odd one does not.
        # Two of the test map's four archives are each way, which is the only reason
        # this was noticed: skipping it reads the base as zero and the first
        # record as a base, and every entry in those two archives then claims a
        # compression method that does not exist.
        at = HEADER + (self.bundles + 1) * 4
        self._fh.seek(at + (-at % 8))

        offsets: list[int] = []
        sizes: list[int] = []
        flags: list[int] = []
        bases: list[int] = []
        left = self.count
        for _ in range(self.segments):
            pair = self._fh.read(8)
            if len(pair) < 8:
                raise UnsupportedMiniZip(f"{self.path.name}: the index is truncated")
            # A u64, and it has to be: the archive is 39.7 GB, so a 32-bit base
            # runs out nine tenths of the way through it. Every segment in the
            # first four gigabytes has a zero high word, which is exactly what
            # makes this easy to read as two u32s and not notice until a tile
            # near the end of the file decompresses to noise.
            (base,) = struct.unpack("<Q", pair)
            bases.append(base)
            # The last segment holds only as many records as there are entries
            # left, rather than a full one padded out.
            here = min(left, self.per_segment)
            raw = self._fh.read(here * 12)
            if len(raw) < here * 12:
                raise UnsupportedMiniZip(f"{self.path.name}: the index is truncated")
            for i in range(here):
                off, size, flag = struct.unpack_from("<3I", raw, i * 12)
                offsets.append(base + off)
                sizes.append(size)
                flags.append(flag)
            left -= here
        self._offsets = offsets
        self._sizes = sizes
        self._flags = flags
        self.bases = bases

        # The stored length is the gap to whatever comes next. Within a segment
        # that is the next entry; at a segment's end it is the next segment's
        # base; at the very end it is the end of the file.
        end = self.path.stat().st_size
        self._next = offsets[1:] + [end]
        for i in range(self.segments - 1):
            last = min((i + 1) * self.per_segment, self.count) - 1
            if 0 <= last < len(self._next):
                self._next[last] = bases[i + 1]

    def entry(self, index: int) -> Entry:
        if not 0 <= index < self.count:
            raise IndexError(f"{index} is outside the archive's {self.count} entries")
        offset = self._offsets[index]
        return Entry(
            index=index,
            offset=offset,
            size=self._sizes[index],
            stored_size=self._next[index] - offset,
            flags=self._flags[index],
        )

    def read(self, index: int) -> bytes:
        """The file at `index`, unpacked."""
        item = self.entry(index)
        self._fh.seek(item.offset)
        raw = self._fh.read(item.stored_size - item.padding)
        if item.codec == STORED:
            # Stored entries are padded out to the next entry's offset, so the
            # declared size is the one to trust.
            return raw[: item.size]
        if item.codec == DEFLATE:
            out = zlib.decompress(raw, -15)
        elif item.codec == LZ4:
            out = lz4_block(raw, item.size)
        else:
            raise UnsupportedMiniZip(
                f"{self.path.name}: entry {index} is packed with method "
                f"{item.codec}, which this reader does not know"
            )
        if len(out) != item.size:
            raise UnsupportedMiniZip(
                f"{self.path.name}: entry {index} unpacked to {len(out)} bytes, "
                f"not the {item.size} it declares"
            )
        return out


def lz4_block(src: bytes, expected: int) -> bytes:
    """
    Decode one LZ4 block of known length.

    The stored bytes run to the next entry's offset, which is usually a little
    past the end of the block — the archive pads. `lz4.block` stops at the
    declared length and ignores the tail, which is what makes that harmless.
    """
    try:
        import lz4.block
    except ImportError:  # pragma: no cover - the package is a declared dependency
        raise UnsupportedMiniZip(
            "this archive is LZ4-compressed, which needs the `lz4` package"
        ) from None
    return lz4.block.decompress(src, uncompressed_size=expected)


class Contents:
    """
    The names that go with an archive, from its `ChunkContentsMiniZip*.txt`.

    The archive stores none, so this is not a convenience - it is the only way to
    ask for a file by name. Line `n` of this file names entry `n`.
    """

    #: Everything before this is the build machine's path and says nothing about
    #: where the file sits in the game.
    ROOT = "zipcache\\pc\\"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.names: list[str] = []
        with self.path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n").rstrip("\r")
                if not line:
                    continue
                name = line.rsplit("|", 1)[0]
                at = name.find(self.ROOT)
                if at >= 0:
                    name = name[at + len(self.ROOT) :]
                self.names.append(name)
        self._by_name = {n.lower(): i for i, n in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        """The entry number for a path, as the manifest spells it."""
        try:
            return self._by_name[name.lower().replace("/", "\\")]
        except KeyError:
            raise KeyError(f"{name} is not in {self.path.name}") from None

    def find(self, fragment: str) -> list[int]:
        """Every entry whose path contains `fragment`, case-insensitively."""
        needle = fragment.lower().replace("/", "\\")
        return [i for i, n in enumerate(self.names) if needle in n.lower()]


def open_chunk(folder: str | Path, number: int) -> tuple[MiniZip, Contents]:
    """Open `GeoChunk{n}.minizip` together with the names that go with it."""
    folder = Path(folder)
    return (
        MiniZip(folder / f"GeoChunk{number}.minizip"),
        Contents(folder / f"ChunkContentsMiniZip{number}.txt"),
    )
