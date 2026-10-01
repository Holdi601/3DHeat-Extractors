"""
Finding an installed Forza, and the track a set of coordinates belongs to.

Both halves exist so that nobody has to type a path like
`D:\\Games\\steamapps\\common\\ForzaHorizon6\\media\\Tracks\\<map>`. That
path is knowable: Steam records where its libraries are, a Forza install has a
`media/Tracks` folder, and a track folder is one that holds a `GeoChunk*.minizip`.

Which track is knowable too, and worth doing properly rather than by guessing at
a name. A game ships more than one — the open-world map, and a garage or two — and a later title will ship different ones. The coordinates settle it:
terrain tile names carry their own grid position, so the right track is the one
whose tiles actually cover the place in question. That is a lookup in a text
manifest, cheap enough to run across every installed track, and it cannot pick
the wrong one the way a name match can.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

#: A Steam library's `path` entries in `libraryfolders.vdf`. The file is Valve's
#: own key-value format; one regex is a better trade here than a parser for it.
LIBRARY_PATH = re.compile(r'"path"\s*"([^"]+)"')

#: Where Steam itself lives, which is also a library.
STEAM_DEFAULTS = (
    r"C:\Program Files (x86)\Steam",
    r"C:\Program Files\Steam",
)

#: Games install under here on the Microsoft Store, which is how Forza also
#: ships. Read if present; it usually is not.
XBOX_DEFAULTS = (r"C:\XboxGames",)


def steam_libraries() -> list[Path]:
    """Every Steam library folder this machine knows about."""
    roots: list[Path] = []
    seen: set[str] = set()
    for base in STEAM_DEFAULTS:
        config = Path(base) / "config" / "libraryfolders.vdf"
        if not config.exists():
            continue
        try:
            text = config.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for found in LIBRARY_PATH.finditer(text):
            path = Path(found.group(1).replace("\\\\", "\\"))
            key = str(path).lower()
            if key not in seen:
                seen.add(key)
                roots.append(path)
    return roots


def game_folders(
    extra: tuple[str | Path, ...] = (), *, defaults: bool = True
) -> list[Path]:
    """
    Installed games that have a `media/Tracks`, whatever the store.

    `defaults` is there so a caller can search only what it names. Without
    it the machine's own installs are always in the answer, which makes
    anything built on this untestable: a test for "nothing is installed"
    cannot pass on a machine where something is.
    """
    candidates: list[Path] = []
    if defaults:
        for library in steam_libraries():
            candidates.append(Path(library) / "steamapps" / "common")
        for base in XBOX_DEFAULTS:
            candidates.append(Path(base))
    for base in extra:
        candidates.append(Path(base))

    games: list[Path] = []
    for parent in candidates:
        if not parent.is_dir():
            continue
        try:
            children = sorted(parent.iterdir())
        except OSError:
            continue
        for child in children:
            # Xbox installs put the game one level further down, in `Content`.
            for game in (child, child / "Content"):
                if (game / "media" / "Tracks").is_dir():
                    games.append(game)
                    break
    return games


def track_folders(
    extra: tuple[str | Path, ...] = (), *, defaults: bool = True
) -> list[Path]:
    """
    Every track folder that holds streamed geometry.

    A folder qualifies by having a `GeoChunk0.minizip`, not by its name: that is
    what the reader needs, and it is true of a track that ships under a name
    nobody has seen before.
    """
    tracks: list[Path] = []
    for game in game_folders(extra, defaults=defaults):
        root = game / "media" / "Tracks"
        try:
            children = sorted(root.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_dir() and (child / "GeoChunk0.minizip").exists():
                tracks.append(child)
    return tracks


def track_covers(folder: str | Path, low: tuple[float, float], high: tuple[float, float]) -> int:
    """
    How many of a track's terrain tiles fall inside a world-space box.

    Read from the manifest's text rather than by opening the archive, because
    the tile names carry their grid position and the archive does not need to be
    touched to ask this. A whole track answers in well under a second.
    """
    from .forzaterrain import NAME_STEP, TILE, TILE_METRES

    folder = Path(folder)
    hits = 0
    seen: set[tuple[int, int]] = set()
    for number in range(4):
        manifest = folder / f"ChunkContentsMiniZip{number}.txt"
        if not manifest.exists():
            continue
        with manifest.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                # Every line ends `|<bundle>`, and the tile pattern is anchored
                # to the end of a file name. Matching the raw line finds nothing
                # at all, which looks exactly like a track with no terrain in it.
                found = TILE.search(line.rsplit("|", 1)[0])
                if not found:
                    continue
                key = (int(found.group(1)), int(found.group(2)))
                if key in seen:
                    continue
                seen.add(key)
                x0 = key[0] / NAME_STEP * TILE_METRES
                z0 = key[1] / NAME_STEP * TILE_METRES
                if (
                    x0 < high[0]
                    and x0 + TILE_METRES > low[0]
                    and z0 < high[1]
                    and z0 + TILE_METRES > low[1]
                ):
                    hits += 1
    return hits


def find_track(
    low: tuple[float, float],
    high: tuple[float, float],
    *,
    extra: tuple[str | Path, ...] = (),
    defaults: bool = True,
) -> Path:
    """
    The installed track whose terrain covers a box.

    Raises rather than returning the first one found. A lap cut against the
    wrong track produces an empty file at best and, if two games happen to use
    overlapping coordinates, a plausible piece of the wrong map at worst.
    """
    tracks = track_folders(extra, defaults=defaults)
    if not tracks:
        raise FileNotFoundError(
            "no installed Forza track found. Pass the folder directly - it is "
            "the one under media/Tracks that holds GeoChunk0.minizip."
        )
    scored = [(track_covers(t, low, high), t) for t in tracks]
    scored.sort(key=lambda pair: -pair[0])
    if scored[0][0] == 0:
        names = ", ".join(t.name for t in tracks)
        raise FileNotFoundError(
            f"none of the installed tracks ({names}) has terrain at "
            f"x {low[0]:.0f}..{high[0]:.0f}, z {low[1]:.0f}..{high[1]:.0f}. "
            "Those are the game's own metres, as its telemetry reports them."
        )
    return scored[0][1]
