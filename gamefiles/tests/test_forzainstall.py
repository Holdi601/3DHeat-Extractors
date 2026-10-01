"""
Finding the game, and finding the right track inside it.

Both exist so that exporting a corner of a racetrack does not require anyone to
type a forty-character path into a folder inside a game they already have
installed. That is a convenience, and conveniences are allowed to fail — but not
silently and not by guessing, because the failure mode is a piece of the wrong
map, which looks exactly like a piece of the right one.

So the track is chosen by coordinates rather than by name. A game ships more than
one (the open-world map, a garage or two) and a later title will ship
names nobody has seen; terrain tiles carry their own grid position, so asking
which track actually has ground at a place is both cheap and impossible to get
wrong the way a name match is.
"""

from __future__ import annotations

import pytest

from heat3d_gamefiles.forzainstall import (
    LIBRARY_PATH,
    find_track,
    track_covers,
    track_folders,
)
from heat3d_gamefiles.forzaterrain import NAME_STEP, TILE_METRES


def make_track(root, name, tiles):
    """
    A track folder with a manifest naming `tiles`, and an empty archive beside
    it so it is recognised as a track at all.
    """
    folder = root / "media" / "Tracks" / name
    folder.mkdir(parents=True)
    (folder / "GeoChunk0.minizip").write_bytes(b"")
    lines = [
        "<PREZIPPED>d:\\scratch\\p4\\forte_main\\zipcache\\pc\\tracks\\x\\scene\\"
        f"tbheightfield\\autoterrain_x{x}_z{z}_cluster000.i.modelbin|7\n"
        for x, z in tiles
    ]
    (folder / "ChunkContentsMiniZip0.txt").write_text("".join(lines), encoding="utf-8")
    return folder


def grid(x_from, x_to, z_from, z_to):
    """Tile names for a rectangle of the grid, in the units the names use."""
    return [
        (x * NAME_STEP, z * NAME_STEP)
        for x in range(x_from, x_to + 1)
        for z in range(z_from, z_to + 1)
    ]


class TestFindingTheGame:
    def test_a_folder_counts_as_a_track_by_what_it_holds(self, tmp_path):
        """
        By having streamed geometry, not by being called something. That is what
        the reader needs, and it stays true of a track shipped under a name this
        code has never seen.
        """
        game = tmp_path / "steamapps" / "common" / "SomeRacer"
        make_track(game, "MainMap", grid(0, 1, 0, 1))
        (game / "media" / "Tracks" / "notatrack").mkdir()

        found = track_folders(extra=(game.parent,), defaults=False)

        assert [f.name for f in found] == ["MainMap"]

    def test_the_steam_library_file_is_read_for_its_paths(self, tmp_path):
        text = (
            '"libraryfolders"\n{\n\t"0"\n\t{\n\t\t"path"\t\t"C:\\\\Program Files '
            '(x86)\\\\Steam"\n\t}\n\t"1"\n\t{\n\t\t"path"\t\t"D:\\\\Games"\n\t}\n}\n'
        )

        paths = [m.group(1) for m in LIBRARY_PATH.finditer(text)]

        assert paths == ["C:\\\\Program Files (x86)\\\\Steam", "D:\\\\Games"]

    def test_nothing_installed_says_so_rather_than_guessing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no installed Forza track"):
            find_track((0.0, 0.0), (10.0, 10.0), extra=(tmp_path,), defaults=False)


class TestChoosingTheTrack:
    def test_the_track_with_ground_there_wins(self, tmp_path):
        """
        The one the coordinates belong to, not the first one found. A garage and
        a map both parse; only one of them has the lap on it.
        """
        game = tmp_path / "steamapps" / "common" / "Racer"
        make_track(game, "garage02", grid(0, 0, 0, 0))
        make_track(game, "MainMap", grid(4, 6, 9, 11))

        # Tile 5,10 covers x 2560..3072, z 5120..5632.
        found = find_track((2600.0, 5200.0), (2900.0, 5400.0), extra=(game.parent,), defaults=False)

        assert found.name == "MainMap"

    def test_coordinates_no_track_covers_are_refused(self, tmp_path):
        """
        Refused rather than answered with the closest thing. Cutting a lap
        against the wrong track yields an empty file at best and a plausible
        piece of the wrong map at worst, and nothing about the result says which.
        """
        game = tmp_path / "steamapps" / "common" / "Racer"
        make_track(game, "MainMap", grid(0, 1, 0, 1))

        with pytest.raises(FileNotFoundError, match="none of the installed tracks"):
            find_track((90_000.0, 90_000.0), (91_000.0, 91_000.0), extra=(game.parent,), defaults=False)

    def test_the_bundle_number_on_each_line_is_not_part_of_the_name(self, tmp_path):
        """
        Every manifest line ends `|<bundle>`, and the tile pattern is anchored to
        the end of a file name. Matching the raw line finds nothing at all —
        which reads as a track with no terrain in it, for every track installed.
        """
        game = tmp_path / "steamapps" / "common" / "Racer"
        folder = make_track(game, "MainMap", grid(5, 5, 10, 10))

        assert track_covers(folder, (2600.0, 5200.0), (2900.0, 5400.0)) == 1

    def test_a_tile_is_512_metres_however_its_name_counts(self, tmp_path):
        """
        The names step by 1023 and the tile is 512 metres. Reading the name as
        metres still finds tiles — just the ones from the wrong half of the map.
        """
        game = tmp_path / "steamapps" / "common" / "Racer"
        folder = make_track(game, "MainMap", [(8 * NAME_STEP, 0)])

        assert track_covers(folder, (8 * TILE_METRES + 1, 1), (8 * TILE_METRES + 2, 2)) == 1
        assert track_covers(folder, (8 * NAME_STEP + 1, 1), (8 * NAME_STEP + 2, 2)) == 0

    def test_a_track_with_no_manifest_is_simply_not_covering_anything(self, tmp_path):
        game = tmp_path / "steamapps" / "common" / "Racer"
        folder = game / "media" / "Tracks" / "Bare"
        folder.mkdir(parents=True)

        assert track_covers(folder, (0.0, 0.0), (10.0, 10.0)) == 0
