"""
Tests for the Unreal pak reader.

Two kinds, and both are needed.

The synthetic ones build an archive byte by byte and read it back. They run
anywhere, and they pin the parts that are easy to get subtly wrong — the footer
field order, the bit-packed entry, the escape for a block size that will not fit
in six bits. Each of those was in fact wrong at some point, and each failed
*quietly*: the encryption flag read one byte late is the first letter of "Zlib",
so every archive claimed to be encrypted; the block-size escape read in the wrong
position gave a 27 GB archive entry offsets in the exabytes.

The real ones open whatever games are installed and verify extraction against the
SHA-1 Unreal writes ahead of every payload. That hash is the ground truth this
reader cannot fake: it covers the bytes as stored, so matching it means the index
decode, the payload offset and every compressed block boundary were all right.
They skip when no game is present rather than failing.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path

import pytest

from heat3d_gamefiles.pak import (
    NO_COMPRESSION,
    PakArchive,
    UnsupportedPak,
    _decode_entry,
    _serialized_size,
    read_footer,
)

from .installed import installed as in_steam


def build_pak(path: Path, files: dict[str, bytes], *, compress: set[str] = frozenset()):
    """
    Write a minimal but real pak: version 11, full directory index, no encryption.

    Small enough to read at a glance, complete enough that the reader cannot tell
    it from one UnrealPak produced — which is the point, because a fixture the
    reader is lenient towards proves nothing.
    """
    blob = bytearray()
    records: dict[str, bytes] = {}

    for name, data in files.items():
        compressed = name in compress
        payload = zlib.compress(data) if compressed else data
        method = 1 if compressed else NO_COMPRESSION
        blocks = 1 if compressed else 0
        offset = len(blob)

        record = bytearray()
        record += struct.pack("<q", 0)  # offset, relative in the repeated record
        record += struct.pack("<q", len(payload))
        record += struct.pack("<q", len(data))
        record += struct.pack("<I", method)
        record += hashlib.sha1(payload).digest()
        if compressed:
            start = _serialized_size(method, blocks)
            record += struct.pack("<i", blocks)
            record += struct.pack("<qq", start, start + len(payload))
        record += bytes([0])  # not encrypted
        record += struct.pack("<I", 65536)  # block size
        assert len(record) == _serialized_size(method, blocks), name

        blob += record
        blob += payload

        # The encoded form the index carries.
        flags = (
            (1 << 31)  # offset fits in 32 bits
            | (1 << 30)  # uncompressed size does
            | (1 << 29)  # size does
            | (method << 23)
            | (blocks << 6)
            | (65536 >> 11)
        )
        encoded = bytearray(struct.pack("<I", flags))
        encoded += struct.pack("<I", offset)
        encoded += struct.pack("<I", len(data))
        if compressed:
            encoded += struct.pack("<I", len(payload))
        records[name] = bytes(encoded)

    encoded_blob = bytearray()
    where: dict[str, int] = {}
    for name, encoded in records.items():
        where[name] = len(encoded_blob)
        encoded_blob += encoded

    def fstring(text: str) -> bytes:
        raw = text.encode("utf-8") + b"\x00"
        return struct.pack("<i", len(raw)) + raw

    # Grouped by folder, the way UnrealPak writes it: a folder carries a
    # trailing slash and no leading one, so it joins onto the mount point
    # without doubling the separator.
    folders: dict[str, list[str]] = {}
    for name in files:
        folder, _, base = name.rpartition("/")
        folders.setdefault(f"{folder}/" if folder else "", []).append(base)

    directory = bytearray(struct.pack("<i", len(folders)))
    for folder, names in folders.items():
        directory += fstring(folder)
        directory += struct.pack("<i", len(names))
        for base in names:
            directory += fstring(base)
            directory += struct.pack("<i", where[f"{folder}{base}"])

    directory_offset = len(blob)
    blob += directory

    index = bytearray()
    index += fstring("../../../")
    index += struct.pack("<i", len(files))
    index += struct.pack("<q", 0)  # path hash seed
    index += struct.pack("<i", 0)  # no path hash index
    index += struct.pack("<i", 1)  # has full directory index
    index += struct.pack("<q", directory_offset)
    index += struct.pack("<q", len(directory))
    index += bytes(20)
    index += struct.pack("<i", len(encoded_blob))
    index += encoded_blob
    index += struct.pack("<i", 0)  # no overflow entries

    index_offset = len(blob)
    blob += index

    footer = bytearray(bytes(16))  # encryption key GUID
    footer += bytes([0])  # index not encrypted
    footer += struct.pack("<I", 0x5A6F12E1)
    footer += struct.pack("<I", 11)
    footer += struct.pack("<q", index_offset)
    footer += struct.pack("<q", len(index))
    footer += bytes(20)
    for name in ("Zlib", "", "", "", ""):
        footer += name.encode("ascii").ljust(32, b"\x00")
    blob += footer

    path.write_bytes(bytes(blob))
    return path


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    return build_pak(
        tmp_path / "sample.pak",
        {
            "Config/Base.ini": b"",
            "Content/notes.txt": b"the quick brown fox" * 40,
            "Content/small.bin": bytes(range(256)),
        },
        compress={"Content/notes.txt"},
    )


def test_footer_reports_an_unencrypted_archive_as_unencrypted(sample: Path):
    """
    The flag sits before the magic, not after it.

    Reading it after lands on the 'Z' of "Zlib", which is non-zero, so every
    archive would report itself encrypted and refuse to open.
    """
    info = read_footer(sample)

    assert info.version == 11
    assert info.encrypted_index is False
    assert info.methods == ("Zlib",)


def test_footer_reads_the_method_names_whole(sample: Path):
    """A cursor four bytes late turns "Zlib" into "lib" and "Oodle" into "odle"."""
    assert read_footer(sample).method_name(1) == "Zlib"
    assert read_footer(sample).method_name(NO_COMPRESSION) == "none"


def test_reads_stored_and_compressed_entries(sample: Path):
    archive = PakArchive(sample)

    assert set(archive.list()) == {
        "../../../Config/Base.ini",
        "../../../Content/notes.txt",
        "../../../Content/small.bin",
    }
    assert archive.read("../../../Content/notes.txt") == b"the quick brown fox" * 40
    assert archive.read("../../../Content/small.bin") == bytes(range(256))
    assert archive.read("../../../Config/Base.ini") == b""


def test_block_size_escape_does_not_shift_the_offset():
    """
    A block size too large for six bits is stored as a following uint32, and it
    comes before the offset rather than after the sizes.

    This is the bug that only shows in big archives: every entry whose block size
    fits in six bits decodes either way, so a small pak reads perfectly while a
    27 GB one gets 7% of its entries shifted four bytes into nonsense.
    """
    method, blocks, block_size = 1, 1, 262144
    flags = (1 << 30) | (1 << 29) | (method << 23) | (blocks << 6) | 0x3F
    blob = struct.pack("<I", flags)
    blob += struct.pack("<I", block_size)
    blob += struct.pack("<q", 23_253_221_376)  # a 64-bit offset
    blob += struct.pack("<I", 24_106_902)
    blob += struct.pack("<I", 3_209_798)

    entry = _decode_entry(blob, 0, "big.uasset")

    assert entry.block_size == block_size
    assert entry.offset == 23_253_221_376
    assert entry.uncompressed_size == 24_106_902
    assert entry.size == 3_209_798


def test_serialized_size_matches_the_repeated_record(sample: Path):
    """
    The payload starts after the record repeated in the data section.

    Wrong by a few bytes, decompression fails — or, for a stored entry, silently
    returns the data shifted.
    """
    archive = PakArchive(sample)
    for name, entry in archive.entries.items():
        with open(sample, "rb") as handle:
            handle.seek(entry.offset)
            record = handle.read(_serialized_size(entry.method, len(entry.blocks)))
        size, uncompressed = struct.unpack_from("<qq", record, 8)
        assert (size, uncompressed) == (entry.size, entry.uncompressed_size), name


def test_refuses_a_file_that_is_not_a_pak(tmp_path: Path):
    junk = tmp_path / "level.utoc"
    junk.write_bytes(b"\x00" * 4096)

    with pytest.raises(UnsupportedPak, match="IoStore"):
        read_footer(junk)


def test_refuses_a_footer_pointing_outside_the_file(sample: Path, tmp_path: Path):
    """A footer that parsed into nonsense must say so, not read wild offsets."""
    data = bytearray(sample.read_bytes())
    at = data.rfind(struct.pack("<I", 0x5A6F12E1))
    struct.pack_into("<q", data, at + 8, 1 << 40)
    broken = tmp_path / "broken.pak"
    broken.write_bytes(bytes(data))

    with pytest.raises(UnsupportedPak, match="outside"):
        read_footer(broken)


# ---------------------------------------------------------------------------
# Against real games, when they happen to be installed.


def installed(limit: int = 4) -> list[Path]:
    """
    Paks of whatever Unreal games are installed, one or two folders below the
    game, and only those whose index is not encrypted: an encrypted one is
    refused by design, which says nothing about the decode.
    """
    found = []
    for path in in_steam(*(f"*/{'*/' * depth}Content/Paks/*.pak" for depth in range(4))):
        try:
            if not read_footer(path).encrypted_index:
                found.append(path)
        except (UnsupportedPak, OSError):
            continue
    return found[:limit]


@pytest.mark.skipif(not installed(), reason="no shipped pak on this machine")
@pytest.mark.parametrize("archive_path", installed(), ids=lambda p: p.name)
def test_real_archive_entries_all_lie_inside_the_file(archive_path: Path):
    """
    An entry offset past the end means the bit-packed decode went wrong.

    This is the check that caught the block-size escape: 11,805 of 167,412
    entries in a 27 GB archive had offsets in the exabytes, while every entry in
    the small archives beside it was fine.
    """
    archive = PakArchive(archive_path)
    size = archive_path.stat().st_size
    stray = [e for e in archive.entries.values() if not 0 <= e.offset < size]

    assert not stray, f"{len(stray)} of {len(archive)} entries point outside the file"


@pytest.mark.skipif(not installed(), reason="no shipped pak on this machine")
@pytest.mark.parametrize("archive_path", installed(), ids=lambda p: p.name)
def test_real_archive_payloads_match_their_recorded_hash(archive_path: Path):
    """
    Unreal writes a SHA-1 of the stored bytes ahead of every payload.

    Matching it over a random sample means the index decode, the payload offset
    and every block boundary were right — which no amount of "it did not crash"
    would establish.
    """
    import random

    archive = PakArchive(archive_path)
    names = [n for n, e in archive.entries.items() if e.uncompressed_size > 0]
    random.Random(7).shuffle(names)
    sample = names[:150]
    assert sample, "archive holds nothing to check"

    for name in sample:
        entry = archive.entries[name]
        with open(archive_path, "rb") as handle:
            handle.seek(entry.offset)
            record = handle.read(53)
            if entry.compressed:
                first, last = entry.blocks[0][0], entry.blocks[-1][1]
                handle.seek(entry.offset + first)
                stored = handle.read(last - first)
            else:
                handle.seek(entry.offset + _serialized_size(entry.method, 0))
                stored = handle.read(entry.uncompressed_size)

        assert hashlib.sha1(stored).digest() == record[28:48], name
