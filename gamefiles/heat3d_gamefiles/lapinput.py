"""
Getting a driven line out of whatever the caller has.

A lap arrives in more than one shape and there is no reason to make anyone
convert between them first. This project alone writes two — the route file
`heat3d_capture route` produces and the one `heat3d_capture laps` produces from a
recorded library — and the library's own per-lap files are a third. They all hold
the same thing: positions, in the game's own world metres.

So this reads positions and nothing else. The channels that make a lap
interesting to analyse — speed, the inputs, the g figures — belong to the
telemetry side of the project, which owns that format properly and turns it into
a table. What the map extractor needs is where the car went, and that is one
field in every one of these.

Recognised, by looking at the file rather than at its name:

- a route file, `{"bounds": ..., "positions": [[x, y, z], ...]}`
- a lap file from the recorded library, `{"Lap": {"samples": [{"X": ..}, ..]}}`
- a `heat3d-lap` document from `heat3d_capture telemetry`, any game:
  `{"format": "heat3d-lap", "game": .., "channels": {"x": [..], ..}}`
- a `.npz` holding `positions`
- a folder of any of the above, which is how a whole course is stored
- a CSV with x, y and z columns, which is what the viewer eats, with or
  without the `# {...}` description line the recorders put first

Which game a lap is from rides along where the file says (`game` in a
`heat3d-lap` document or a CSV's description line), because the `course` verb
reads a different game's files for it. A file that does not say is Forza's:
every format this read before the others existed is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class UnreadableLap(Exception):
    """Nothing in this that looks like a driven line."""


@dataclass
class Driven:
    """Where a car went, and what to call it."""

    name: str
    positions: np.ndarray
    #: How many separate laps went into it, when that is known.
    laps: int = 1
    #: The game it was driven in, as the recorders name it; None where the
    #: file does not say, which is every Forza format.
    game: str | None = None
    #: The track or level, where the file names it.
    track: str | None = None

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.positions.min(axis=0), self.positions.max(axis=0)

    @property
    def metres(self) -> float:
        if len(self.positions) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.positions, axis=0), axis=1).sum())

    def box(self, margin: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        The ground to cut, widened in the plane only.

        Height is deliberately not part of it: a lap is a line, and bounding its
        altitude would cut the ground out from under a hill the road climbs.
        """
        low, high = self.bounds
        return (
            (float(low[0]) - margin, float(low[2]) - margin),
            (float(high[0]) + margin, float(high[2]) + margin),
        )


def _from_json(document: dict, name: str) -> np.ndarray | None:
    channels = document.get("channels")
    if document.get("format") == "heat3d-lap" and isinstance(channels, dict):
        try:
            columns = [np.asarray(channels[k], dtype=np.float64) for k in ("x", "y", "z")]
        except (KeyError, TypeError, ValueError):
            return None
        points = np.stack(columns, axis=1)
        return points[np.isfinite(points).all(axis=1)]

    points = document.get("positions")
    if isinstance(points, list) and points:
        return np.asarray(points, dtype=np.float64)

    samples = (document.get("Lap") or {}).get("samples")
    if isinstance(samples, list) and samples:
        return np.array(
            [[s.get("X", 0.0), s.get("Y", 0.0), s.get("Z", 0.0)] for s in samples],
            dtype=np.float64,
        )

    # A route file written with bounds but no samples still says where to cut;
    # two corners are a degenerate line but a perfectly good box.
    bounds = document.get("bounds")
    if isinstance(bounds, dict) and "min" in bounds and "max" in bounds:
        return np.array([bounds["min"], bounds["max"]], dtype=np.float64)
    return None


def _from_csv(path: Path) -> np.ndarray | None:
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(line for line in fh if not line.startswith("#"))
        if not reader.fieldnames:
            return None
        lookup = {n.strip().lower(): n for n in reader.fieldnames}
        try:
            keys = [lookup["x"], lookup["y"], lookup["z"]]
        except KeyError:
            return None
        points = []
        for row in reader:
            try:
                points.append([float(row[k]) for k in keys])
            except (TypeError, ValueError):
                continue
    return np.asarray(points, dtype=np.float64) if points else None


def read_file(path: str | Path) -> np.ndarray:
    """Positions out of one file, whichever of the shapes it is."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path) as data:
            if "positions" not in data:
                raise UnreadableLap(f"{path.name} has no `positions` array")
            return np.asarray(data["positions"], dtype=np.float64)
    if suffix == ".csv":
        points = _from_csv(path)
        if points is None:
            raise UnreadableLap(f"{path.name} has no x, y and z columns")
        return points
    try:
        # utf-8-sig: the recorded library is written by a .NET tool and its
        # files carry a byte-order mark, which `json` refuses as plain utf-8.
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise UnreadableLap(
            f"{path.name} is not a lap this can read.\n"
            "  Give it a route or lap file (.json), a recording (.npz), a table "
            "with x, y and z\n"
            "  columns (.csv), a folder of any of those, or just a course's name."
        ) from None
    points = _from_json(document, path.name)
    if points is None:
        raise UnreadableLap(
            f"{path.name} is JSON, but holds neither `positions` nor "
            "`Lap.samples` nor `bounds`"
        )
    return points


def describe(path: str | Path) -> dict:
    """
    What a lap file says about itself: `game` and `track` where it names
    them, from a `heat3d-lap` document or a CSV's description line.
    """
    import json

    path = Path(path)
    try:
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig") as fh:
                first = fh.readline()
            if not first.startswith("#"):
                return {}
            meta = json.loads(first[1:].strip())
        elif path.suffix.lower() == ".json":
            meta = json.loads(path.read_text(encoding="utf-8-sig"))
            if meta.get("format") != "heat3d-lap":
                return {}
        else:
            return {}
    except (OSError, ValueError, AttributeError):
        return {}
    return {k: meta.get(k) for k in ("game", "track") if isinstance(meta.get(k), str)}


def read_folder(folder: str | Path) -> tuple[np.ndarray, int]:
    """
    Every lap under a folder, and how many there were.

    A course in the recorded library is a tree — class, then car, then tune,
    then tag — so the search is recursive. `course.json` is metadata rather than
    a lap and is skipped by name; anything else that will not read is skipped
    quietly, because one bad file among hundreds must not cost the rest.
    """
    folder = Path(folder)
    blocks: list[np.ndarray] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.name == "course.json":
            continue
        if path.suffix.lower() not in (".json", ".npz", ".csv"):
            continue
        try:
            points = read_file(path)
        except UnreadableLap:
            continue
        if len(points):
            blocks.append(points)
    if not blocks:
        raise UnreadableLap(f"no readable laps under {folder}")
    return np.concatenate(blocks), len(blocks)


def read(target: str | Path) -> Driven:
    """
    A driven line from a file, a folder, or a course name.

    A bare name is looked up in the recorded-lap library, so `"Lakeside Circuit"`
    works without anyone knowing that it lives in a folder called
    `Lakeside Circuit (course_2800_5000_to_2775_5000)`.
    """
    path = Path(target)
    if path.is_dir():
        points, laps = read_folder(path)
        # A folder of one game's recordings says which; a mix says nothing.
        games = {describe(p).get("game") for p in path.rglob("*") if p.suffix.lower() in (".json", ".csv")}
        game = games.pop() if len(games) == 1 else None
        return Driven(name=_clean(path.name), positions=points, laps=laps, game=game)
    if path.exists():
        about = describe(path)
        return Driven(
            name=_clean(path.stem),
            positions=read_file(path),
            game=about.get("game"),
            track=about.get("track"),
        )

    folder = find_recorded_course(str(target))
    points, laps = read_folder(folder)
    return Driven(name=_clean(folder.name), positions=points, laps=laps)


def _clean(name: str) -> str:
    """Drop the coordinates the recorded library glues onto a course's name."""
    return name.split(" (course_")[0].strip()


def default_library() -> Path:
    """Where FH Companion keeps its recorded laps on Windows."""
    import os

    local = os.environ.get("LOCALAPPDATA")
    if not local:
        raise UnreadableLap("LOCALAPPDATA is not set, so the lap library cannot be found")
    return Path(local) / "FHCompanion" / "laps"


def find_recorded_course(name: str, root: str | Path | None = None) -> Path:
    """One course folder from the recorded library, matched loosely by name."""
    root = Path(root) if root is not None else default_library()
    if not root.is_dir():
        raise UnreadableLap(
            f"{name!r} is not a file, and there is no lap library at {root} to look "
            "it up in"
        )
    folders = [p for p in sorted(root.iterdir()) if p.is_dir()]
    wanted = name.strip().lower()
    exact = [p for p in folders if _clean(p.name).lower() == wanted]
    if exact:
        return exact[0]
    loose = [p for p in folders if wanted in p.name.lower()]
    if len(loose) == 1:
        return loose[0]
    if not loose:
        raise UnreadableLap(
            f"{name!r} is not a file, and no course in {root} matches it"
        )
    raise UnreadableLap(
        f"{name!r} matches {len(loose)} courses: "
        + ", ".join(_clean(p.name) for p in loose)
    )
