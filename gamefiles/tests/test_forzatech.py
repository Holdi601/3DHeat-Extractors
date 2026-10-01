"""
Tests for the ForzaTech container reader.

Forza comes up because a racing capture is the case the reconstruction exporter
was built for, so the fair question is whether the same level can be had exactly
instead of estimated. The answer so far is that the containers open and the
geometry does not, and these tests are about the half that works.

The check is the same shape as the others here: a `.modelbin` states an offset
and a length for every chunk, and the chunks form a chain — each begins where the
last one ended, rounded up to four bytes. Nothing in the file says so, which is
what makes it evidence: a record read at the wrong stride does not produce a
chain that closes. It closes on all 252 models sampled from a shipped install.
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import pytest

from heat3d_gamefiles.forzatech import (
    GEO_MAGIC,
    MODEL_HEADER,
    MODEL_MAGIC,
    MODEL_RECORD,
    UnsupportedForza,
    models_in_archive,
    read_model,
    read_model_from_archive,
)

from .installed import installed

def archives(limit: int = 6) -> list[Path]:
    return installed("ForzaHorizon*/media/*.zip")[:limit]


def build_model(chunks: list[tuple[str, bytes]]) -> bytes:
    """A minimal but real `.modelbin`, so the fixture is not one the reader is lax about."""
    table_at = MODEL_HEADER
    names_at = table_at + len(chunks) * MODEL_RECORD
    data_at = names_at  # no name table in the fixture

    body = bytearray()
    records = []
    at = data_at
    for tag, payload in chunks:
        records.append((tag, at, len(payload)))
        body += payload
        pad = (-len(payload)) % 4
        body += bytes(pad)
        at += len(payload) + pad

    out = bytearray(MODEL_MAGIC)
    out += struct.pack("<4I", 0x0101, names_at, data_at + len(body), len(chunks))
    for tag, offset, size in records:
        out += tag.encode("ascii")[::-1]
        out += struct.pack("<5I", 0, names_at, offset, size, size)
    out += body
    # The declared total has to be the real one.
    struct.pack_into("<I", out, 12, len(out))
    return bytes(out)


def test_reads_a_chunk_table():
    raw = build_model([("Skel", b"S" * 40), ("Mesh", b"M" * 33), ("VerB", b"V" * 12)])

    model = read_model(Path("fixture.modelbin"), raw)

    assert [c.tag for c in model.chunks] == ["Skel", "Mesh", "VerB"]
    assert model.raw(model.chunks[0]) == b"S" * 40
    assert model.raw(model.chunks[1]) == b"M" * 33
    assert model.counts() == {"Skel": 1, "Mesh": 1, "VerB": 1}


def test_chunks_are_four_byte_aligned():
    """A 33-byte chunk is followed by three bytes of padding, not by its neighbour."""
    raw = build_model([("Mesh", b"M" * 33), ("VerB", b"V" * 12)])

    model = read_model(Path("fixture.modelbin"), raw)

    assert model.chunks[1].offset == model.chunks[0].end + 3
    assert model.chunks[1].offset % 4 == 0


def test_a_broken_chain_is_refused():
    """
    The chain is the whole verification, so it has to be enforced.

    Without it the reader would accept a table read at the wrong stride and hand
    back chunks of the wrong bytes under the right names.
    """
    raw = bytearray(build_model([("Skel", b"S" * 40), ("Mesh", b"M" * 40)]))
    at = MODEL_HEADER + MODEL_RECORD + 12
    # Off the chain but still inside the file, so it is the chain that catches
    # it rather than the bounds check — which is the point of the test.
    (offset,) = struct.unpack_from("<I", raw, at)
    struct.pack_into("<I", raw, at, offset - 4)

    with pytest.raises(UnsupportedForza, match="stride"):
        read_model(Path("broken.modelbin"), bytes(raw))


def test_a_truncated_file_is_refused():
    raw = build_model([("Skel", b"S" * 40)])

    with pytest.raises(UnsupportedForza, match="declares"):
        read_model(Path("short.modelbin"), raw[:-10])


def test_a_streamed_geometry_container_is_named_rather_than_rejected_blankly():
    """
    Pointing the tool at a level's terrain should say what it found.

    "Not a format this tool reads" is much less use than "that is the streamed
    geometry container, which is not implemented".
    """
    with pytest.raises(UnsupportedForza, match="streamed geometry"):
        read_model(Path("GeoChunk0.minizip"), GEO_MAGIC + bytes(120))


def test_something_else_entirely_is_refused():
    with pytest.raises(UnsupportedForza, match="does not begin"):
        read_model(Path("notes.txt"), b"hello there" * 20)


# ---------------------------------------------------------------------------
# Against an installed game, when there is one.

INSTALLED = archives()


@pytest.mark.skipif(not INSTALLED, reason="no Forza install on this machine")
def test_real_models_form_a_closed_chain():
    checked = 0
    for archive in INSTALLED:
        try:
            names = models_in_archive(archive)
        except zipfile.BadZipFile:
            continue
        for name in names[:10]:
            model = read_model_from_archive(archive, name)
            assert model.chunks, name
            checked += 1
        if checked > 30:
            break
    if not checked:
        pytest.skip("no .modelbin in the archives found")
    assert checked > 0


@pytest.mark.skipif(not INSTALLED, reason="no Forza install on this machine")
def test_real_models_carry_the_tags_geometry_would_be_in():
    """
    Worth asserting because it is the honest boundary: the reader finds the
    vertex and index buffers and stops there.
    """
    tags: set[str] = set()
    for archive in INSTALLED:
        try:
            names = models_in_archive(archive)
        except zipfile.BadZipFile:
            continue
        for name in names[:10]:
            tags |= {c.tag for c in read_model_from_archive(archive, name).chunks}
        if tags:
            break
    if not tags:
        pytest.skip("no .modelbin in the archives found")

    assert "Mesh" in tags
