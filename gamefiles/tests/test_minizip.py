"""
Reading ForzaTech's `.minizip`.

The format has three properties that a reader can get wrong while appearing to
work, and each has a test here because each was got wrong first:

- the segment base is sixty-four bits, and every segment in the first four
  gigabytes has a zero high word, so a 32-bit reader is correct for hundreds of
  segments and then silently wrong;
- an entry's compressed length is the gap to the next entry *minus* the padding
  in its flags, and LZ4 rejects a block whose input runs past its end, so one
  stray byte turns into a hard failure rather than an ignored tail;
- there are three codecs, and the one that is neither LZ4 nor compressed at all
  is deflate, not a corrupt entry.
"""

from __future__ import annotations

import struct

import pytest

from heat3d_gamefiles.minizip import (
    DEFLATE,
    HEADER,
    LZ4,
    MAGIC,
    STORED,
    Contents,
    MiniZip,
    UnsupportedMiniZip,
)

from .forza_fixtures import build_contents, build_minizip


def payloads(n: int) -> list[bytes]:
    """Compressible content, so an LZ4 entry is genuinely shorter than its file."""
    return [
        (f"entry {i}: ".encode() + b"terrain " * (3 + i * 7)) for i in range(n)
    ]


class TestItReadsWhatWasWritten:
    def test_every_entry_round_trips(self, tmp_path):
        want = payloads(10)
        path = build_minizip(tmp_path / "a.minizip", want, per_segment=4)

        with MiniZip(path) as archive:
            assert len(archive) == 10
            assert [archive.read(i) for i in range(10)] == want

    def test_the_three_codecs_all_work(self, tmp_path):
        want = payloads(3)
        path = build_minizip(
            tmp_path / "a.minizip", want, codecs=[STORED, DEFLATE, LZ4]
        )

        with MiniZip(path) as archive:
            assert [archive.entry(i).codec for i in range(3)] == [STORED, DEFLATE, LZ4]
            assert [archive.read(i) for i in range(3)] == want

    def test_padding_is_not_part_of_the_payload(self, tmp_path):
        """
        Entries start on four-byte boundaries, so most carry one to three bytes
        of slack. Handing those to LZ4 does not produce a slightly wrong
        answer - it produces no answer at all.
        """
        want = payloads(6)
        path = build_minizip(tmp_path / "a.minizip", want)

        with MiniZip(path) as archive:
            padded = [i for i in range(6) if archive.entry(i).padding]
            assert padded, "the fixture produced no padded entries to test with"
            for i in padded:
                assert archive.read(i) == want[i]

    def test_an_entry_in_the_last_segment_reads(self, tmp_path):
        """
        The final entry has no next entry to measure against, so its length
        comes from the end of the file. An off-by-one there is invisible until
        the last thing in an archive is the thing you wanted.
        """
        want = payloads(9)
        path = build_minizip(tmp_path / "a.minizip", want, per_segment=4)

        with MiniZip(path) as archive:
            assert archive.read(8) == want[8]


class TestTheSixtyFourBitBase:
    def test_a_base_past_four_gigabytes_is_not_truncated(self, tmp_path):
        """
        Written as an index-level assertion rather than by producing a 4 GB
        file: the offsets are the thing that breaks, and they break in the
        index. Read as two 32-bit fields, this entry's offset comes back
        somewhere near the start of the archive, and whatever happens to live
        there decompresses into something the right length and wholly wrong.
        """
        path = build_minizip(
            tmp_path / "a.minizip", payloads(4), base_offset=5_000_000_000
        )

        with MiniZip(path) as archive:
            assert archive.entry(0).offset > 4 * 1024**3

    def test_the_high_word_is_read_from_the_second_field(self, tmp_path):
        """
        Pinned directly, because the zero high word in every small archive is
        exactly what lets a wrong reader pass every other test in this file.
        """
        path = build_minizip(tmp_path / "a.minizip", payloads(4))
        raw = bytearray(path.read_bytes())
        table_end = HEADER + (4 + 1) * 4
        at = table_end + (-table_end % 8)
        (base,) = struct.unpack_from("<Q", raw, at)
        struct.pack_into("<Q", raw, at, base + (1 << 32))
        path.write_bytes(raw)

        with MiniZip(path) as archive:
            assert archive.entry(0).offset == base + (1 << 32)


    @pytest.mark.parametrize("entries", [3, 4], ids=["odd-table", "even-table"])
    def test_both_alignments_of_the_bundle_table(self, tmp_path, entries):
        """
        The base is eight-byte aligned, so whether a padding word sits before
        the first segment depends on the parity of the bundle count. Two of
        the test map's four archives are each way — and a reader that ignores the
        padding reads a base of zero and the first record as a base, at which
        point every entry in those two archives claims a codec that does not
        exist. Nothing about that looks like an alignment problem.
        """
        want = payloads(entries)
        path = build_minizip(tmp_path / "a.minizip", want)

        with MiniZip(path) as archive:
            assert [archive.read(i) for i in range(entries)] == want


class TestItRefusesRatherThanGuesses:
    def test_a_file_that_is_not_an_archive(self, tmp_path):
        path = tmp_path / "a.minizip"
        path.write_bytes(b"NOPE" + bytes(64))

        with pytest.raises(UnsupportedMiniZip, match="does not begin with"):
            MiniZip(path)

    def test_a_header_of_the_wrong_size(self, tmp_path):
        path = tmp_path / "a.minizip"
        path.write_bytes(struct.pack("<4sIIIIIII", MAGIC, 101, 40, 1, 1, 4, 1, 0))

        with pytest.raises(UnsupportedMiniZip, match="40-byte header"):
            MiniZip(path)

    def test_a_segment_count_that_does_not_match_the_entries(self, tmp_path):
        """
        The one cheap check on the whole index: entries, entries-per-segment
        and segments have to agree. They do in every shipped archive, and if
        they ever do not, the layout is not the one this reads.
        """
        path = tmp_path / "a.minizip"
        path.write_bytes(struct.pack("<4sIIIIIII", MAGIC, 101, HEADER, 100, 1, 4, 3, 0))

        with pytest.raises(UnsupportedMiniZip, match="needs 25 segments"):
            MiniZip(path)

    def test_a_truncated_index(self, tmp_path):
        path = build_minizip(tmp_path / "a.minizip", payloads(8))
        path.write_bytes(path.read_bytes()[: HEADER + 40])

        with pytest.raises(UnsupportedMiniZip):
            MiniZip(path)

    def test_an_index_outside_the_archive(self, tmp_path):
        path = build_minizip(tmp_path / "a.minizip", payloads(4))

        with MiniZip(path) as archive:
            with pytest.raises(IndexError, match="outside the archive"):
                archive.entry(4)


class TestTheNamesBeside:
    def test_a_name_finds_its_entry(self, tmp_path):
        names = [
            "scene\\tbheightfield\\autoterrain_x0_z0_cluster000.i.modelbin",
            "scene\\models\\thing.modelbin",
        ]
        path = build_contents(tmp_path / "c.txt", names)

        contents = Contents(path)

        assert len(contents) == 2
        assert contents.index(names[0]) == 0
        assert contents.index(names[1]) == 1

    def test_the_build_machine_path_is_stripped(self, tmp_path):
        """
        Every line begins with someone's `d:\\scratch\\p4\\...` checkout. Keeping
        it would make every lookup depend on where Turn 10 built the game.
        """
        path = build_contents(tmp_path / "c.txt", ["scene\\models\\thing.modelbin"])

        contents = Contents(path)

        assert contents.names[0] == "scene\\models\\thing.modelbin"

    def test_a_missing_name_says_so(self, tmp_path):
        path = build_contents(tmp_path / "c.txt", ["a\\b.modelbin"])

        with pytest.raises(KeyError, match="is not in"):
            Contents(path).index("nothing\\here.modelbin")

    def test_find_matches_part_of_a_path(self, tmp_path):
        path = build_contents(
            tmp_path / "c.txt",
            ["scene\\tbheightfield\\autoterrain_x0_z0_cluster000.i.modelbin", "x\\y.z"],
        )

        assert Contents(path).find("autoterrain") == [0]
        assert Contents(path).find("nothing") == []
