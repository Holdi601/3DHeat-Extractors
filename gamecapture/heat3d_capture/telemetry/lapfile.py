"""
One lap or stage as a `heat3d-lap` file, for any game.

Every recorder decodes its game's packets into frames - flat dicts under the
canonical channel names (`docs/lap-format.md`), per wheel as
`name.FL` - and hands a finished run here. What the file says about the game
beyond the numbers goes in `meta`, which the viewer reads to interpret them:

- `slip`: what the slip channels hold, where the game sends any - `peak`
  (normalised so past 1 the tyre lets go), `scaled` (a physical slip divided
  by a typical tarmac peak), `physical` (a fraction and radians), `index`
  (a 0..1 sliding measure);
- `discipline`: `circuit`, `rally` or `arcade`;
- `surface`: `tarmac`, `loose`, `snow`, `ice` or `mixed`, when known.

Channels a game does not send are simply absent; the viewer derives what it
can (drift angle, slip from wheel speeds, airborne) from what is there.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

WHEELS = ("FL", "FR", "RL", "RR")

#: Units the viewer's channel table stores, for the file's own record.
UNITS = {
    "time": "s",
    "distance": "m",
    "x": "m",
    "y": "m",
    "z": "m",
    "speed": "m/s",
    "latVel": "m/s, car frame, to the right",
    "longVel": "m/s, car frame, forwards",
    "vertVel": "m/s, car frame, up",
    "latAcc": "m/s2",
    "longAcc": "m/s2",
    "vertAcc": "m/s2",
    "yaw": "rad",
    "throttle": "0..1",
    "brake": "0..1",
    "clutch": "0..1",
    "handbrake": "0..1",
    "steer": "-1..1, positive right",
    "wheelSurfaceSpeed": "m/s",
    "wheelSpeed": "rad/s",
    "suspensionM": "m",
    "suspension": "0..1 of the lap's travel",
    "brakeTemp": "degC",
    "tireTemp": "degC",
}


@dataclass
class Run:
    """One lap or stage of frames."""

    number: int
    frames: list[dict] = field(default_factory=list)
    #: The game's own time for it, when the game said.
    seconds: float | None = None
    complete: bool = False

    @property
    def metres(self) -> float:
        total = 0.0
        for a, b in zip(self.frames, self.frames[1:]):
            total += math.dist((a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"]))
        return total


def document(run: Run, *, game: str, meta: dict, track: str | None = None, car: dict | None = None) -> dict:
    """A run as a `heat3d-lap` document."""
    frames = run.frames
    t0 = frames[0]["time"]
    names = sorted({k for f in frames for k in f} - {"time"})
    channels: dict[str, list] = {"time": [], "distance": []}
    for name in names:
        if name != "distance":
            channels[name] = []
    travelled = 0.0
    previous = None
    for f in frames:
        if previous is not None:
            travelled += math.dist((previous["x"], previous["y"], previous["z"]), (f["x"], f["y"], f["z"]))
        previous = f
        channels["time"].append(round(f["time"] - t0, 4))
        channels["distance"].append(round(travelled, 3))
        for name in names:
            if name == "distance":
                continue
            v = f.get(name, float("nan"))
            channels[name].append(round(v, 5) if isinstance(v, float) and math.isfinite(v) else v)
    # Suspension as 0..1 of the travel this run used, for the car's bars and
    # for telling a wheel hanging in the air.
    for wheel in WHEELS:
        travel = channels.get(f"suspensionM.{wheel}")
        if travel and f"suspension.{wheel}" not in channels:
            finite = [v for v in travel if isinstance(v, (int, float)) and math.isfinite(v)]
            if not finite:
                continue
            lo, hi = min(finite), max(finite)
            span = hi - lo if hi > lo else 1.0
            channels[f"suspension.{wheel}"] = [round((v - lo) / span, 4) if math.isfinite(v) else v for v in travel]
    seconds = run.seconds if run.seconds is not None else channels["time"][-1]
    return {
        "format": "heat3d-lap",
        "version": 1,
        "game": game,
        "track": track,
        "recordedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "car": car or {},
        "lap": {"number": run.number, "seconds": round(seconds, 4), "complete": run.complete, "metres": round(run.metres, 1)},
        "meta": meta,
        "units": {k: v for k, v in UNITS.items() if k in channels or any(c.startswith(k + ".") for c in channels)},
        "channels": channels,
    }


def write(run: Run, folder: str | Path, *, game: str, meta: dict, track: str | None = None, car: dict | None = None) -> Path:
    """Write a run; the file name carries the game, the time and the run's length."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    doc = document(run, game=game, meta=meta, track=track, car=car)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    kind = "stage" if meta.get("discipline") == "rally" else "lap"
    tag = "" if run.complete else "_partial"
    path = folder / f"{game}_{stamp}_{kind}{run.number}_{doc['lap']['seconds']:.3f}s{tag}.json"
    path.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    return path


def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
