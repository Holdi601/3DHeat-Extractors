"""
Reading a library of recorded laps, as FH Companion keeps it.

The scanner records telemetry live. This reads telemetry someone already has —
a folder per course under `%LOCALAPPDATA%/FHCompanion/laps`, a file per lap,
each holding the whole lap at about one sample every five metres. On the machine
this was written against that is 40 courses and 859 laps, and it is a far better
starting point than a fresh drive: the laps are already there, they are already
repeated, and repetition is what turns a route into analysis.

Why it belongs next to the map extractor
----------------------------------------
Every sample carries the game's own world coordinates, so a course's laps say
exactly which stretch of map to cut out of the shipped archive — and once cut,
the geometry and the driving are in the same frame with nothing to align. Lakeside
Circuit's 46 laps sit a median of 0.36 m above the extracted ground, which is
the car's ride height and not an error.

What a lap carries
------------------
Position, and then everything an analysis of a lap actually wants: speed, the
three inputs (throttle, brake, steer), lateral and longitudinal g, gear, clutch,
handbrake, standing water, distance along the course and time since the start.
Fifteen fields, present in every lap file sampled.

The folder name carries the course's start and finish, rounded — `Lakeside Circuit
(course_2800_5000_to_2775_5000)`. `course.json` beside it has them unrounded,
along with the lap count and the shortest and longest lap seen. Both are read;
the folder name is the fallback, because a course that has only ever been driven
once may not have the file.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: The fields every sample carries, and the names they take on the way out. The
#: output names are lower case with units spelled out, because they end up as
#: column headings in front of someone.
SAMPLE_FIELDS: dict[str, str] = {
    "X": "x",
    "Y": "y",
    "Z": "z",
    "Speed": "speed_ms",
    "Throttle": "throttle",
    "Brake": "brake",
    "Steer": "steer",
    "LatG": "lat_g",
    "LongG": "long_g",
    "Gear": "gear",
    "Clutch": "clutch",
    "HandBrake": "handbrake",
    "Puddle": "puddle",
    "Metres": "metres",
    # `elapsed` rather than `seconds` because that is what the viewer looks for
    # in a duration axis, and because it is the more accurate word: it is time
    # since this lap's own start, so every lap lines up at zero instead of being
    # scattered across the fortnight they were driven in.
    "Seconds": "elapsed",
}

#: `<name> (course_<x>_<z>_to_<x>_<z>)`
FOLDER = re.compile(r"^(?P<name>.+?)\s*\(course_(?P<coords>-?\d+_-?\d+_to_-?\d+_-?\d+)\)$")


class NoLapLibrary(Exception):
    """No FH Companion lap library where one was expected."""


def default_library() -> Path:
    """Where FH Companion keeps its laps on Windows."""
    import os

    local = os.environ.get("LOCALAPPDATA")
    if not local:
        raise NoLapLibrary("LOCALAPPDATA is not set, so the library cannot be found")
    return Path(local) / "FHCompanion" / "laps"


@dataclass
class Lap:
    """One recorded lap: what was driven, and by what."""

    course: str
    #: The car's performance class as the game groups them - D through X, plus
    #: R and S1/S2. A string rather than a number: it is a label, and sorting it
    #: numerically would be wrong anyway.
    car_class: str
    car_ordinal: int
    performance_index: int
    seconds: float
    metres: float
    recorded_at: str
    standing_start: bool
    source: Path
    #: (n, 3) world metres, in the game's own frame.
    positions: np.ndarray
    #: Every other per-sample field, by its output name.
    channels: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def label(self) -> str:
        """Something short enough for a legend and unique enough to pick out."""
        return f"{self.recorded_at[:16].replace('T', ' ')} {self.seconds:.3f}s"


@dataclass
class Course:
    """A named course and every lap recorded on it."""

    name: str
    folder: Path
    laps: list[Lap] = field(default_factory=list)
    start: tuple[float, float] | None = None
    finish: tuple[float, float] | None = None

    def __len__(self) -> int:
        return len(self.laps)

    @property
    def samples(self) -> int:
        return sum(len(lap) for lap in self.laps)

    @property
    def closed(self) -> bool:
        """Whether it returns to where it began, within a car's length."""
        if self.start is None or self.finish is None:
            return False
        return float(np.hypot(*(np.array(self.start) - np.array(self.finish)))) < 10.0

    @property
    def positions(self) -> np.ndarray:
        if not self.laps:
            return np.zeros((0, 3))
        return np.concatenate([lap.positions for lap in self.laps])

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        points = self.positions
        if not len(points):
            return np.zeros(3), np.zeros(3)
        return points.min(axis=0), points.max(axis=0)

    @property
    def best(self) -> Lap | None:
        return min(self.laps, key=lambda lap: lap.seconds, default=None)

    def summary(self) -> str:
        low, high = self.bounds
        span = high - low
        best = self.best
        return (
            f"{self.name}: {len(self.laps)} laps, {self.samples:,} samples"
            + (" (a circuit)" if self.closed else "")
            + "\n"
            + (f"  best {best.seconds:.3f}s over {best.metres:.0f} m, class {best.car_class}\n" if best else "")
            + f"  x {low[0]:9.1f} .. {high[0]:9.1f}   ({span[0]:.0f} m)\n"
            f"  y {low[1]:9.1f} .. {high[1]:9.1f}   ({span[1]:.0f} m of climb)\n"
            f"  z {low[2]:9.1f} .. {high[2]:9.1f}   ({span[2]:.0f} m)"
        )

    def save_route(self, path: str | Path, *, margin: float = 0.0) -> Path:
        """
        Write the course's extent in the shape the map extractor reads.

        Deliberately the same file `heat3d_capture route` writes, so a course out
        of this library and a lap driven just now are interchangeable to
        everything downstream.
        """
        path = Path(path).with_suffix(".json")
        low, high = self.bounds
        low = low - np.array([margin, 0.0, margin])
        high = high + np.array([margin, 0.0, margin])
        points = self.positions
        path.write_text(
            json.dumps(
                {
                    "layout": "fhcompanion",
                    "course": self.name,
                    "laps": len(self.laps),
                    "samples": len(points),
                    "bounds": {"min": low.tolist(), "max": high.tolist()},
                    # The fastest lap rather than all of them: the file is a
                    # route, and 46 overlaid laps is a smear rather than a line.
                    "positions": (self.best.positions if self.best is not None else points).tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def save_samples(self, path: str | Path) -> Path:
        """
        Every sample of every lap, as one CSV the viewer can ingest.

        Wide on purpose. The viewer maps columns itself and offers the numeric
        ones as metrics, so writing speed, the inputs and the g figures costs a
        few megabytes and turns one heatmap into every heatmap anyone would want
        of a lap - where the braking happens, where the grip runs out, where the
        throttle goes down.
        """
        path = Path(path).with_suffix(".csv")
        names = list(SAMPLE_FIELDS.values())
        # Positions get fixed decimals, everything else significant figures.
        # Four significant figures is plenty for a throttle position and ruinous
        # for a coordinate: a car at x=2784.97 writes as 2785, the whole course
        # snaps to a one-metre grid, and a racing line drawn from it has stairs
        # in it. Milimetres cost three characters.
        precise = {"x", "y", "z", "metres", "elapsed"}
        with path.open("w", newline="", encoding="utf-8") as fh:
            out = csv.writer(fh)
            # These four names are what the viewer maps itself: `lap` becomes the
            # session, so two laps by one car are never joined into one track;
            # `car` becomes the thing that moves; `car_class` becomes the group
            # to colour by; and `elapsed`, below, becomes the time axis.
            out.writerow(
                ["course", "lap", "car", "car_class", "pi", "lap_seconds", "speed_kmh"]
                + names
            )
            for number, lap in enumerate(self.laps):
                rows = len(lap)
                speed = lap.channels.get("speed_ms", np.zeros(rows))
                columns = [(n, lap.channels.get(n)) for n in names]
                for i in range(rows):
                    out.writerow(
                        [
                            self.name,
                            lap.label,
                            lap.car_ordinal,
                            lap.car_class,
                            lap.performance_index,
                            f"{lap.seconds:.3f}",
                            f"{speed[i] * 3.6:.2f}",
                        ]
                        + [
                            ""
                            if c is None
                            else (f"{c[i]:.3f}" if n in precise else f"{c[i]:.4g}")
                            for n, c in columns
                        ]
                    )
        return path


def read_lap(path: str | Path) -> Lap:
    """One lap file."""
    path = Path(path)
    # utf-8-sig: the files are written by a .NET tool and carry a byte-order
    # mark, which `json` will not accept as plain utf-8.
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    body = document.get("Lap") or {}
    samples = body.get("samples") or []
    if not samples:
        raise ValueError(f"{path.name} holds no samples")

    columns: dict[str, np.ndarray] = {}
    for source, name in SAMPLE_FIELDS.items():
        columns[name] = np.array(
            [s.get(source, 0.0) for s in samples], dtype=np.float64
        )
    positions = np.stack([columns["x"], columns["y"], columns["z"]], axis=1)
    return Lap(
        course=body.get("track") or document.get("Course") or path.parent.name,
        car_class=str(document.get("Class") or ""),
        car_ordinal=int(body.get("carOrdinal") or 0),
        performance_index=int(body.get("performanceIndex") or 0),
        seconds=float(body.get("lapSeconds") or 0.0),
        metres=float(body.get("lengthMetres") or 0.0),
        recorded_at=str(body.get("recordedAt") or ""),
        standing_start=bool(body.get("StandingStart") or False),
        source=path,
        positions=positions,
        channels=columns,
    )


def read_course(folder: str | Path) -> Course:
    """
    Every lap under one course folder.

    Laps sit several directories down - by class, then car, then tune, then tag -
    and that tree is the tool's business rather than this one's, so the search is
    simply recursive.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NoLapLibrary(f"{folder} is not a course folder")

    name = folder.name
    start = finish = None
    found = FOLDER.match(folder.name)
    if found:
        name = found.group("name")
    meta = folder / "course.json"
    if meta.exists():
        try:
            info = json.loads(meta.read_text(encoding="utf-8-sig"))
            name = info.get("Name") or name
            start = (float(info["StartX"]), float(info["StartZ"]))
            finish = (float(info["FinishX"]), float(info["FinishZ"]))
        except (ValueError, KeyError, TypeError):
            # A half-written metadata file must not cost the laps beside it;
            # the folder name carries the same two points, rounded.
            pass
    if start is None and found:
        numbers = [float(n) for n in found.group("coords").replace("_to_", "_").split("_")]
        start, finish = (numbers[0], numbers[1]), (numbers[2], numbers[3])

    laps: list[Lap] = []
    for path in sorted(folder.rglob("*.json")):
        if path.name == "course.json":
            continue
        try:
            laps.append(read_lap(path))
        except (ValueError, KeyError, json.JSONDecodeError):
            # One unreadable lap out of hundreds must not cost the course.
            continue
    laps.sort(key=lambda lap: lap.recorded_at)
    return Course(name=name, folder=folder, laps=laps, start=start, finish=finish)


def courses(root: str | Path | None = None) -> list[Path]:
    """Course folders in a library, without reading any of the laps."""
    root = Path(root) if root is not None else default_library()
    if not root.is_dir():
        raise NoLapLibrary(f"no lap library at {root}")
    return sorted(p for p in root.iterdir() if p.is_dir())


def find_course(name: str, root: str | Path | None = None) -> Path:
    """
    A course folder by name, matched loosely.

    Loosely because the folder name has the start and finish coordinates glued
    on - nobody is going to type `Lakeside Circuit (course_2800_5000_to_2775_5000)`.
    """
    wanted = name.strip().lower()
    folders = courses(root)
    exact = [p for p in folders if p.name.lower() == wanted]
    if exact:
        return exact[0]
    named = [p for p in folders if (FOLDER.match(p.name) or None) and FOLDER.match(p.name).group("name").lower() == wanted]
    if named:
        return named[0]
    loose = [p for p in folders if wanted in p.name.lower()]
    if len(loose) == 1:
        return loose[0]
    if not loose:
        raise KeyError(f"no course matching {name!r}")
    raise KeyError(
        f"{name!r} matches {len(loose)} courses: "
        + ", ".join(sorted(p.name.split(" (")[0] for p in loose))
    )
