"""
Unreal Engine 5 cooked data small enough to build in a test: an IoStore
container with a directory, a zen package, an unversioned property block.

Written to the layouts the readers were checked against on a shipped game
(see `iostore.py`, `zen.py`, `unversioned.py`), so a test holds the reader to
the format rather than to itself.
"""

from __future__ import annotations

import struct
from pathlib import Path

MAGIC = b"-==--==--==--==-"


def fstring(text: str) -> bytes:
    raw = text.encode("latin-1") + bytes(1)
    return struct.pack("<i", len(raw)) + raw


def write_container(folder: Path, name: str, files: dict[str, bytes], *, mount: str = "../../../acr/Content/", chunk_ids=None) -> Path:
    """A `.utoc` and `.ucas` holding `files` (path under the mount -> bytes), stored uncompressed."""
    folder.mkdir(parents=True, exist_ok=True)
    names = list(files)
    block_size = 0x10000
    ucas = bytearray()
    spans, blocks = [], []
    for path in names:
        data = files[path]
        # A chunk starts on a block of its own in the uncompressed stream.
        spans.append((len(blocks) * block_size, len(data)))
        for at in range(0, max(1, len(data)), block_size):
            piece = data[at : at + block_size]
            blocks.append((len(ucas), len(piece), len(piece), 0))
            ucas += piece

    # Directory: one root directory holding every file.
    strings = names
    directory = fstring(mount) + struct.pack("<i", 1) + struct.pack("<4I", 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0 if names else 0xFFFFFFFF)
    directory += struct.pack("<i", len(names))
    for i in range(len(names)):
        directory += struct.pack("<3I", i, i + 1 if i + 1 < len(names) else 0xFFFFFFFF, i)
    directory += struct.pack("<i", len(strings)) + b"".join(fstring(s) for s in strings)

    entries = len(names)
    header = bytearray(MAGIC)
    header += struct.pack("<BBH", 8, 0, 0)
    header += struct.pack("<9I", 144, entries, len(blocks), 12, 0, 32, block_size, len(directory), 1)
    header += struct.pack("<Q", 1234)
    header += bytes(16)  # key GUID
    header += struct.pack("<BBH", 8, 0, 0)  # flags: indexed
    header += struct.pack("<I", 0)  # hash seeds
    header += struct.pack("<Q", 1 << 40)  # partition size
    header += struct.pack("<I", 0)  # chunks without a perfect hash
    header += bytes(144 - len(header))
    toc = bytearray(header)
    for i in range(entries):
        cid = chunk_ids[i] if chunk_ids else 1000 + i
        toc += struct.pack("<QHBB", cid, 0, 0, 1)
    for start, length in spans:
        toc += start.to_bytes(5, "big") + length.to_bytes(5, "big")
    for where, packed, unpacked, method in blocks:
        toc += where.to_bytes(5, "little") + packed.to_bytes(3, "little") + unpacked.to_bytes(3, "little") + bytes([method])
    toc += directory
    (folder / f"{name}.utoc").write_bytes(bytes(toc))
    (folder / f"{name}.ucas").write_bytes(bytes(ucas))
    return folder / f"{name}.utoc"


def name_batch(names: list[str]) -> bytes:
    strings = b"".join(n.encode("latin-1") for n in names)
    out = struct.pack("<iI", len(names), len(strings)) + struct.pack("<Q", 0xC1640000)
    out += bytes(8 * len(names))
    out += b"".join(bytes([len(n) >> 8, len(n) & 0xFF]) for n in names)
    return out + strings


def zen_package(
    names: list[str],
    exports: list[dict],
    *,
    imports: list[int] = (),
    hashes: list[int] = (),
    imported: list[tuple[str, int]] = (),
    bulk: list[tuple[int, int, int]] = (),
) -> bytes:
    """
    A cooked package in the 5.6 layout. Each export: `name` (index into
    `names`), `cls` (a package object index), `data` (bytes), `public` hash,
    `outer`. `imported` is (package name, number) per imported package.
    """
    body = bytearray(bytes(60))
    body += name_batch(names)
    body += bytes(15)  # what sits between the names and the bulk map in shipped packages
    body += struct.pack("<q", 32 * len(bulk))
    for offset, size, flags in bulk:
        body += struct.pack("<qqqII", offset, -1, size, flags, 0)
    hashes_at = len(body)
    body += b"".join(struct.pack("<Q", h) for h in hashes)
    imports_at = len(body)
    body += b"".join(struct.pack("<Q", i) for i in imports)
    exports_at = len(body)
    offset = 0
    for e in exports:
        body += struct.pack(
            "<QQIIQQQQQIB3x",
            offset, len(e["data"]), e["name"], 0, e.get("outer", 3 << 62), e["cls"], 3 << 62,
            e.get("template", 3 << 62), e.get("public", 0), 0, 0,
        )
        offset += len(e["data"])
    bundles_at = len(body)
    heads_at = len(body)
    body += b"".join(struct.pack("<i4I", 0, 0, 0, 0, 0) for _ in exports)
    entries_at = len(body)
    names_at = len(body)
    body += name_batch([n for n, _ in imported]) if imported else struct.pack("<i", 0)
    body += b"".join(struct.pack("<i", number) for _, number in imported)
    header_size = len(body)
    struct.pack_into("<II", body, 0, 0, header_size)
    struct.pack_into("<III", body, 8, 0, 0, 0)
    # Nine offsets: hashes, imports, exports, bundles, two dependency tables,
    # imported names, and the two cell maps 5.6 added.
    struct.pack_into("<9i", body, 24, hashes_at, imports_at, exports_at, bundles_at, heads_at, entries_at, names_at, header_size, header_size)
    return bytes(body) + b"".join(e["data"] for e in exports)


def unversioned(slots: list[int], zero: set[int] = frozenset()) -> bytes:
    """A property header with `slots` present, `zero` of them zero."""
    words = []
    position = 0
    runs = []
    for slot in sorted(slots):
        if runs and slot == runs[-1][0] + runs[-1][1]:
            runs[-1][1] += 1
        else:
            runs.append([slot, 1])
    zeros = []
    for start, count in runs:
        skip = start - position
        while skip > 127:
            words.append([127, False, 0])
            skip -= 127
        has_zero = any(s in zero for s in range(start, start + count))
        words.append([skip, has_zero, count])
        if has_zero:
            zeros += [s in zero for s in range(start, start + count)]
        position = start + count
    out = bytearray()
    for i, (skip, has_zero, count) in enumerate(words):
        word = skip | (0x80 if has_zero else 0) | (0x100 if i == len(words) - 1 else 0) | (count << 9)
        out += struct.pack("<H", word)
    if zeros:
        size = 1 if len(zeros) <= 8 else 2 if len(zeros) <= 16 else 4 * ((len(zeros) + 31) // 32)
        mask = bytearray(size)
        for i, z in enumerate(zeros):
            if z:
                mask[i // 8] |= 1 << (i % 8)
        out += mask
    return bytes(out)
