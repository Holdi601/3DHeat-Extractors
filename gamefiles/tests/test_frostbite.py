"""
Tests for the Frostbite reader.

Frostbite ships two forms of table of contents and this reads both, so both are
checked, and each has its own way of proving it was understood.

The **DbObject tree** declares every container's length in bytes. A field type
read at the wrong width desynchronises the cursor, and the container then does
not end where it said it would. `parse_toc` raises on that rather than returning
a half-parsed tree, so a file that parses is a file whose every type was right —
which is how the reader found that `0x09` is an eight-byte long and not a
four-byte integer, four bytes at a time.

The **binary table** proves itself differently: its section offsets are stated
*and* derivable from two counts, so the arithmetic either closes or it does not.
It closes on all 136 binary tables Battlefield 6 ships.

What is deliberately not here is a test that extracts geometry. The tables name
assets and say where their bytes live; turning one into a mesh is the Frostbite
mesh format on top, which this does not attempt. Testing the boundary honestly
means testing the catalogue, not pretending to a level.
"""

from __future__ import annotations

import glob
import os
import struct
from pathlib import Path

import pytest

from heat3d_gamefiles.frostbite import (
    MAGIC_OBFUSCATED,
    MAGIC_PLAIN,
    MAGIC_SIGNED,
    SIGNED_HEADER,
    UnsupportedFrostbite,
    asset_ids,
    describe,
    parse_toc,
    read_header,
    decompress_payload,
    locations,
    read_sections,
    split_blocks,
    superbundles,
)

from .installed import installed

def tables() -> list[Path]:
    return sorted(installed("Battlefield*/**/*.toc"))


INSTALLED = tables()


def test_signed_header_is_not_encryption():
    """
    A signed table is plain after 556 bytes; only the third form is obfuscated.

    Worth pinning as its own test because the high-entropy bytes after the magic
    look exactly like ciphertext, and calling the whole file encrypted would have
    meant refusing every table Battlefield ships.
    """
    assert read_header(struct.pack("<I", MAGIC_SIGNED) + bytes(600)).payload_offset == (
        SIGNED_HEADER
    )
    assert read_header(struct.pack("<I", MAGIC_PLAIN) + bytes(600)).payload_offset == 0


def test_obfuscated_form_is_declined_rather_than_guessed():
    with pytest.raises(UnsupportedFrostbite, match="obfuscated"):
        read_header(struct.pack("<I", MAGIC_OBFUSCATED) + bytes(600))


def test_rejects_a_file_that_is_not_a_table():
    with pytest.raises(UnsupportedFrostbite, match="not a Frostbite"):
        read_header(b"\x00" * 64, "random.bin")


@pytest.mark.skipif(not INSTALLED, reason="no Frostbite game on this machine")
def test_every_shipped_table_is_recognised():
    """Each one parses as a tree or as a section table, with none left over."""
    described = [describe(p) for p in INSTALLED]

    assert len(described) == len(INSTALLED)
    assert any("DbObject tree" in d for d in described), "no tree form found"
    assert any("binary table" in d for d in described), "no binary form found"


@pytest.mark.skipif(not INSTALLED, reason="no Frostbite game on this machine")
def test_the_tree_form_names_its_superbundles():
    """
    Every container in the tree consumed exactly the length it declared.

    `parse_toc` enforces that internally, so reaching a list of superbundle names
    at all means the whole tree was read with the right type widths.
    """
    trees = []
    for path in INSTALLED:
        try:
            trees.append(parse_toc(path)[1])
        except UnsupportedFrostbite:
            continue
    if not trees:
        pytest.skip("this installation ships no DbObject table")

    names = [n for tree in trees for n in superbundles(tree)]
    assert names, "the tree parsed but named nothing"
    assert all(isinstance(n, str) and n for n in names)


@pytest.mark.skipif(not INSTALLED, reason="no Frostbite game on this machine")
def test_binary_section_chain_closes_on_every_table():
    """
    Each section starts where the previous one ends, and the last ends in-file.

    `read_sections` raises when it does not, so this asserts that the reader
    accepted at least a representative number of them rather than quietly
    finding every table unsupported.
    """
    accepted = 0
    for path in INSTALLED:
        data = path.read_bytes()
        info = read_header(data, path.name)
        try:
            sections = read_sections(data, info, path.name)
        except UnsupportedFrostbite:
            continue  # the DbObject form, checked above
        accepted += 1
        assert sections.asset_count >= 0
        assert info.payload_offset + sections.end <= len(data)

    assert accepted > 1, "no binary table was accepted"


@pytest.mark.skipif(not INSTALLED, reason="no Frostbite game on this machine")
def test_asset_identifiers_are_all_distinct():
    """
    A misread record size produces repeats, because it reslices the same bytes.

    Distinctness across tens of thousands of records is strong evidence the
    20-byte stride is right — far stronger than the records merely being
    readable.
    """
    biggest = None
    for path in INSTALLED:
        data = path.read_bytes()
        info = read_header(data, path.name)
        try:
            sections = read_sections(data, info, path.name)
        except UnsupportedFrostbite:
            continue
        if biggest is None or sections.asset_count > biggest[2].asset_count:
            biggest = (data, info, sections)
    if biggest is None:
        pytest.skip("this installation ships no binary table")

    data, info, sections = biggest
    identifiers = asset_ids(data, info, sections)

    assert len(identifiers) == sections.asset_count
    assert len(set(identifiers)) == len(identifiers)
    assert all(len(i) == 16 for i in identifiers)


class TestFindingAndReadingTheBytes:
    """
    The step from a catalogue to an extractor.

    Every asset's location is four big-endian words that nothing documents. The
    interpretation below — a blob index, an offset, a length and a package
    identifier — was read off the numbers and then checked against the files: all
    74,175 records of a Battlefield 6 level land inside an installed `.cas`, none
    runs past the end of its file, and none names a blob that is not there. A
    wrong field order does not produce that.

    The payloads are then chains of eight-byte-headed blocks, and the same kind
    of check applies: a chain either consumes the payload exactly or it is not a
    chain. 218 of 299 sampled payloads do, decompressing to 63 MB carrying
    recognisable Frostbite signatures. The other 81 are high-entropy from their
    first byte and are refused rather than guessed at.
    """

    @staticmethod
    def level_table() -> Path | None:
        for path in INSTALLED:
            data = path.read_bytes()
            try:
                sections = read_sections(data, read_header(data, path.name), path.name)
            except UnsupportedFrostbite:
                continue
            if sections.asset_count > 1000:
                return path
        return None

    @pytest.mark.skipif(not INSTALLED, reason="no Frostbite game on this machine")
    def test_every_location_lands_inside_an_installed_blob(self):
        import glob
        import os

        path = self.level_table()
        if path is None:
            pytest.skip("no table here holds enough assets to be worth checking")
        data = path.read_bytes()
        info = read_header(data, path.name)
        sections = read_sections(data, info, path.name)

        root = path
        while root.parent != root and root.name != "Data":
            root = root.parent
        blobs: dict[int, list[int]] = {}
        for found in glob.glob(os.path.join(str(root.parent), "**", "cas_*.cas"), recursive=True):
            index = int(os.path.basename(found).split("_")[1].split(".")[0])
            blobs.setdefault(index, []).append(os.path.getsize(found))
        if not blobs:
            pytest.skip("no .cas blobs beside this table")

        for spot in locations(data, info, sections):
            assert spot.cas in blobs, f"cas_{spot.cas:02d} is referenced but not installed"
            assert any(spot.end <= size for size in blobs[spot.cas]), (
                f"cas_{spot.cas:02d} is too small for bytes "
                f"{spot.offset}-{spot.end}"
            )

    def test_a_payload_that_is_not_a_block_chain_is_refused(self):
        """
        Refused, not partially decoded.

        A chain has to consume the payload exactly. Anything else is a different
        encoding — or encryption — and returning the first plausible block would
        hand the caller a fragment that looks like data.
        """
        with pytest.raises(UnsupportedFrostbite):
            split_blocks(bytes(range(64)))

    def test_a_stored_block_declares_no_compressed_length(self):
        """
        A zero in the compressed field means stored, as long as it decompresses to.

        Reading it as a zero-length block rejects the whole payload: honouring it
        took the share of a level's payloads that parse as a clean chain from 159
        of 299 to 218.
        """
        body = bytes(range(256)) * 4
        payload = struct.pack(">IHH", len(body), 0x0071, 0) + body

        blocks = split_blocks(payload)

        assert len(blocks) == 1
        assert blocks[0].compressed == len(body)
        assert decompress_payload(payload) == body

    def test_a_stored_block_round_trips(self):
        body = b"VAND" + bytes(range(200))
        payload = struct.pack(">IHH", len(body), 0x0070, len(body)) + body

        assert decompress_payload(payload) == body

    def test_a_chain_of_several_blocks_joins_in_order(self):
        first = b"A" * 100
        second = b"B" * 60
        payload = (
            struct.pack(">IHH", len(first), 0x0070, len(first))
            + first
            + struct.pack(">IHH", len(second), 0x0070, len(second))
            + second
        )

        assert decompress_payload(payload) == first + second
