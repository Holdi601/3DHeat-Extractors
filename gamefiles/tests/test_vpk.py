"""
Tests for the Source 2 VPK reader.

This format is kinder than the others here: Valve documents it, and every entry
carries a CRC of its own bytes. That last part is unusual and worth leaning on —
extraction can be checked against the archive's own claim rather than against a
fixture someone wrote, which is the difference between "it returned something"
and "it returned the right thing".

The synthetic tests build an archive byte by byte and run anywhere. The real ones
open Counter-Strike 2's own maps when it is installed, and verify what comes out.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from heat3d_gamefiles.vpk import (
    HEADER_V2,
    IN_DIRECTORY,
    TERMINATOR,
    VPK_MAGIC,
    UnsupportedVpk,
    VpkArchive,
)

from .installed import installed as in_steam


def installed(limit: int = 6) -> list[Path]:
    return in_steam("Counter-Strike Global Offensive/game/csgo/maps/*.vpk")[:limit]


def build_vpk(path: Path, files: dict[str, bytes], *, preload: set[str] = frozenset()) -> Path:
    """
    A self-contained VPK v2, written the way the format specifies.

    Grouped by extension and folder the way a real one is, because the three
    nested loops that structure is read with are the part a reader gets wrong.
    """
    groups: dict[str, dict[str, list[str]]] = {}
    for name in files:
        head, _, extension = name.rpartition(".")
        folder, _, stem = head.rpartition("/")
        groups.setdefault(extension, {}).setdefault(folder or " ", []).append(stem)

    body = bytearray()
    offsets = {}
    for name, data in files.items():
        if name in preload:
            continue
        offsets[name] = len(body)
        body += data

    tree = bytearray()
    for extension, folders in groups.items():
        tree += extension.encode() + b"\x00"
        for folder, stems in folders.items():
            tree += folder.encode() + b"\x00"
            for stem in stems:
                name = f"{folder}/{stem}.{extension}" if folder != " " else f"{stem}.{extension}"
                data = files[name]
                tree += stem.encode() + b"\x00"
                inline = name in preload
                tree += struct.pack("<I", zlib.crc32(data) & 0xFFFFFFFF)
                tree += struct.pack("<H", len(data) if inline else 0)
                tree += struct.pack("<H", IN_DIRECTORY)
                tree += struct.pack("<I", 0 if inline else offsets[name])
                tree += struct.pack("<I", 0 if inline else len(data))
                tree += struct.pack("<H", TERMINATOR)
                if inline:
                    tree += data
            tree += b"\x00"
        tree += b"\x00"
    tree += b"\x00"

    header = struct.pack(
        "<7I", VPK_MAGIC, 2, len(tree), len(body), 0, 0, 0
    )
    assert len(header) == HEADER_V2
    path.write_bytes(header + bytes(tree) + bytes(body))
    return path


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    return build_vpk(
        tmp_path / "sample.vpk",
        {
            "maps/de_test.nav": b"NAV" + bytes(range(200)),
            "maps/de_test/world.vmdl_c": b"MDL" * 400,
            "readme.txt": b"small enough to live in the directory",
        },
        preload={"readme.txt"},
    )


class TestReadingTheDirectory:
    def test_it_reassembles_the_three_part_paths(self, sample: Path):
        """
        Extension, folder and name are stored in three separate loops, once each
        rather than per file. A reader that mishandles the grouping produces
        plausible paths for the first entry and nonsense after it.
        """
        archive = VpkArchive(sample)

        assert set(archive.list()) == {
            "maps/de_test.nav",
            "maps/de_test/world.vmdl_c",
            "readme.txt",
        }

    def test_it_reads_the_bytes_back(self, sample: Path):
        archive = VpkArchive(sample)

        assert archive.read("maps/de_test.nav") == b"NAV" + bytes(range(200))
        assert archive.read("maps/de_test/world.vmdl_c") == b"MDL" * 400

    def test_a_small_file_stored_in_the_directory_needs_no_part(self, sample: Path):
        """
        Preloaded entries have a length of zero, which is not an error — all the
        bytes are already in hand.
        """
        archive = VpkArchive(sample)
        entry = archive.entries["readme.txt"]

        assert entry.length == 0
        assert archive.read("readme.txt") == b"small enough to live in the directory"

    def test_every_entry_matches_its_own_crc(self, sample: Path):
        archive = VpkArchive(sample)

        assert all(archive.verify(p) for p in archive.list())


class TestRefusingWhatItCannotRead:
    def test_something_that_is_not_a_vpk(self, tmp_path: Path):
        junk = tmp_path / "notes.txt"
        junk.write_bytes(b"hello" * 40)

        with pytest.raises(UnsupportedVpk, match="signature"):
            VpkArchive(junk)

    def test_a_version_it_does_not_know(self, tmp_path: Path):
        odd = tmp_path / "future.vpk"
        odd.write_bytes(struct.pack("<7I", VPK_MAGIC, 9, 0, 0, 0, 0, 0))

        with pytest.raises(UnsupportedVpk, match="version 9"):
            VpkArchive(odd)

    def test_a_directory_that_does_not_fit(self, tmp_path: Path):
        lying = tmp_path / "lying.vpk"
        lying.write_bytes(struct.pack("<7I", VPK_MAGIC, 2, 1 << 24, 0, 0, 0, 0))

        with pytest.raises(UnsupportedVpk, match="does not fit"):
            VpkArchive(lying)

    def test_a_record_that_does_not_end_where_it_should(self, sample: Path, tmp_path: Path):
        """
        Every entry ends with 0xffff, which is the cheapest check that the tree
        is still being read at the right offset. It has to be enforced, or a
        misread directory yields entries pointing at arbitrary bytes.
        """
        data = bytearray(sample.read_bytes())
        at = data.find(struct.pack("<H", TERMINATOR), HEADER_V2)
        struct.pack_into("<H", data, at, 0x1234)
        broken = tmp_path / "broken.vpk"
        broken.write_bytes(bytes(data))

        with pytest.raises(UnsupportedVpk, match="right offset"):
            VpkArchive(broken)


class TestSplitArchives:
    def test_the_directory_sentinel_is_not_read_as_a_part_number(self, sample: Path):
        """
        0x7fff means "in this file". Read as an index it sends the reader
        looking for `sample_32767.vpk`, which does not exist.
        """
        archive = VpkArchive(sample)
        entry = archive.entries["maps/de_test.nav"]

        assert entry.archive == IN_DIRECTORY
        assert archive.part_for(entry) == sample

    def test_a_numbered_part_is_named_from_the_directory(self, tmp_path: Path):
        archive = VpkArchive(build_vpk(tmp_path / "pak01_dir.vpk", {"a.txt": b"x"}))
        entry = archive.entries["a.txt"]
        object.__setattr__(entry, "archive", 3)

        assert archive.part_for(entry).name == "pak01_003.vpk"

    def test_a_missing_part_says_which_one(self, tmp_path: Path):
        archive = VpkArchive(build_vpk(tmp_path / "pak01_dir.vpk", {"a.txt": b"x" * 40}))
        entry = archive.entries["a.txt"]
        object.__setattr__(entry, "archive", 7)

        with pytest.raises(UnsupportedVpk, match="pak01_007.vpk"):
            archive.read("a.txt")


# ---------------------------------------------------------------------------
# Against Counter-Strike 2's own maps, when it is installed.

INSTALLED = installed()


@pytest.mark.skipif(not INSTALLED, reason="Counter-Strike 2 is not installed")
@pytest.mark.parametrize("path", INSTALLED, ids=lambda p: p.stem)
def test_a_shipped_map_opens_and_lists(path: Path):
    archive = VpkArchive(path)

    assert len(archive) > 0
    assert all("\x00" not in name for name in archive.list())


@pytest.mark.skipif(not INSTALLED, reason="Counter-Strike 2 is not installed")
def test_extraction_matches_the_crc_the_archive_recorded(self=None):
    """
    The strongest check available, and it needs no fixture: the archive says
    what each file's CRC is, so matching it means the offsets, the part
    selection and the preload handling were all right.
    """
    import random

    checked = 0
    for path in INSTALLED[:4]:
        archive = VpkArchive(path)
        names = archive.list()
        random.Random(5).shuffle(names)
        for name in names[:25]:
            assert archive.verify(name), f"{name} in {path.name}"
            checked += 1

    assert checked > 50, "too few entries were actually checked"


@pytest.mark.skipif(not INSTALLED, reason="Counter-Strike 2 is not installed")
def test_a_map_carries_its_navigation_mesh():
    """
    Worth asserting because it is the route to map geometry that matters here.

    A heatmap of player positions wants the walkable floor, and the nav mesh is
    exactly that — already computed, already shipped, and far smaller than the
    compiled world geometry it would otherwise have to be derived from.
    """
    for path in INSTALLED:
        archive = VpkArchive(path)
        navs = [n for n in archive.list() if n.endswith(".nav")]
        if not navs:
            continue
        data = archive.read(navs[0])
        assert data[:4] == b"\xce\xfa\xed\xfe", "not a Source navigation mesh"
        return
    pytest.skip("none of the maps examined ships a .nav")
