"""
Assetto Corsa, Assetto Corsa Competizione, AC EVO and AC Rally, recorded a lap
at a time for the Race tab.

Kunos's games publish their telemetry for apps the documented way: named
shared-memory pages that the game writes and anyone may read. Nothing is
injected into the game and nothing it does not publish is read.

- `Local\\acpmf_physics`: AC, ACC and AC Rally. `Local\\acevo_pmf_physics`: AC EVO.
  The first 800 bytes are one layout in all four - pedals, speed, the car's
  accelerations and attitude, and per wheel the load, pressure, core and
  surface temperatures, suspension travel, camber, slip ratio and slip angle -
  and they include each tyre's contact point in world coordinates. The mean of
  the four is the car's position, from the physics page alone.
- `Local\\acpmf_graphics`: AC and ACC count laps here (`completedLaps`, the
  last lap's time). AC EVO's graphics page is a different, larger layout, and
  AC Rally drives stages rather than laps, so for those two a lap is found from
  the positions instead: it closes when the car comes back past the point it
  started from, and a stage closes when the car stops at the end of it.

References: the AC1 shared memory SDK; the ACC shared memory documentation
(v1.8); AC EVO's layout as measured in `albertowd/live-telemetry-evo`
(docs/SHARED_MEMORY.md); AC Rally's as read by
`LuizZak/AssettoCorsaRallyTelemetryReader`.

Slip is the one channel whose meaning differs between games. Forza sends slip
normalised so that 1 is the edge of grip; Kunos sends the physical slip ratio
and slip angle. They are divided by a typical peak (0.12 for slip ratio, 0.14
rad - 8 degrees - for slip angle) so the viewer's "past 1 means sliding"
reading holds for both, approximately; the raw values are kept too.
"""

from __future__ import annotations

import json
import math
import struct
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

WHEELS = ("FL", "FR", "RL", "RR")

#: Physics page, offsets shared by AC (1.7+), ACC, AC EVO and AC Rally.
PHYSICS_SIZE = 800
#: How much of it AC (the first game) is guaranteed to have.
PHYSICS_AC1_SIZE = 580

G = 9.80665
SLIP_RATIO_PEAK = 0.12
SLIP_ANGLE_PEAK = 0.14

# (offset, struct format, name)
_SCALARS = [
    (0, "i", "packetId"),
    (4, "f", "gas"),
    (8, "f", "brake"),
    (16, "i", "gear"),
    (20, "i", "rpms"),
    (24, "f", "steerAngle"),
    (28, "f", "speedKmh"),
    (208, "f", "heading"),
    (212, "f", "pitch"),
    (216, "f", "roll"),
    (200, "f", "drs"),
    (248, "i", "pitLimiterOn"),
    (276, "f", "turboBoost"),
    (364, "f", "clutch"),
]
# Only in the 800-byte page (ACC, AC EVO, AC Rally).
_SCALARS_FULL = [
    (564, "f", "brakeBias"),
    (672, "i", "tcInAction"),
    (676, "i", "absInAction"),
]
_VECTORS = [
    (32, 3, "velocity"),
    (44, 3, "accG"),
    (296, 3, "localAngularVel"),
]
#: The car's velocity in its own frame, from AC 1.12 on and in the others: the
#: drift angle is the angle between it and the car's nose.
LOCAL_VELOCITY = 568
#: ACC, AC Rally and AC EVO report the aids acting here, 0..1 (AC's own page
#: holds its settings in the same slots, so it is read only for those).
TC_ACTING = 204
ABS_ACTING = 252
# Per wheel, FL FR RL RR.
_WHEELS = [
    (56, "wheelSlip"),
    (72, "wheelLoad"),
    (88, "wheelsPressure"),
    (104, "wheelAngularSpeed"),
    (152, "tyreCoreTemperature"),
    (168, "camberRAD"),
    (184, "suspensionTravel"),
    (348, "brakeTemp"),
    (368, "tyreTempI"),
    (384, "tyreTempM"),
    (400, "tyreTempO"),
]
_WHEELS_FULL = [
    (640, "slipRatio"),
    (656, "slipAngle"),
    (740, "padLife"),
    (756, "discLife"),
]
TYRE_CONTACT_POINT = 420  # float[4][3], world XYZ

#: Graphics page of AC and ACC: the lap counter and the last lap's time.
GRAPHICS_SIZE = 256
G_COMPLETED_LAPS = 132
G_LAST_LAP_MS = 144
G_NORMALISED_POSITION = 248

MAPPINGS = {
    "ac": ("Local\\acpmf_physics", "Local\\acpmf_graphics", "Local\\acpmf_static"),
    "acc": ("Local\\acpmf_physics", "Local\\acpmf_graphics", "Local\\acpmf_static"),
    "acrally": ("Local\\acpmf_physics", None, "Local\\acpmf_static"),
    "acevo": ("Local\\acevo_pmf_physics", None, "Local\\acevo_pmf_static"),
}
GAME_NAMES = {
    "ac": "assetto-corsa",
    "acc": "assetto-corsa-competizione",
    "acrally": "assetto-corsa-rally",
    "acevo": "assetto-corsa-evo",
}


def decode_physics(page: bytes) -> dict:
    """One physics page to a flat dict of the raw fields, per wheel as `name.FL`."""
    out: dict = {}
    scalars = list(_SCALARS) + (list(_SCALARS_FULL) if len(page) >= PHYSICS_SIZE else [])
    for offset, fmt, name in scalars:
        out[name] = struct.unpack_from("<" + fmt, page, offset)[0]
    for offset, n, name in _VECTORS:
        out[name] = struct.unpack_from(f"<{n}f", page, offset)
    wheels = list(_WHEELS) + (list(_WHEELS_FULL) if len(page) >= PHYSICS_SIZE else [])
    for offset, name in wheels:
        values = struct.unpack_from("<4f", page, offset)
        for wheel, value in zip(WHEELS, values):
            out[f"{name}.{wheel}"] = value
    if len(page) >= TYRE_CONTACT_POINT + 48:
        points = struct.unpack_from("<12f", page, TYRE_CONTACT_POINT)
        out["contact"] = [points[i * 3 : i * 3 + 3] for i in range(4)]
    if len(page) >= LOCAL_VELOCITY + 12:
        out["localVelocity"] = struct.unpack_from("<3f", page, LOCAL_VELOCITY)
    out["tcActing"] = struct.unpack_from("<f", page, TC_ACTING)[0]
    out["absActing"] = struct.unpack_from("<f", page, ABS_ACTING)[0]
    return out


#: Temperatures AC Rally sends in kelvin where the others send Celsius.
KELVIN_FIELDS = ("tyreCoreTemperature", "tyreTempI", "tyreTempM", "tyreTempO", "brakeTemp")


def frame_from(raw: dict, clock: float, game: str = "ac") -> dict | None:
    """
    The raw fields as the viewer's canonical channels. None when the page holds
    no position - the game is in a menu, or the car is in the garage.
    """
    contact = raw.get("contact")
    if not contact:
        return None
    x = sum(p[0] for p in contact) / 4
    y = sum(p[1] for p in contact) / 4
    z = sum(p[2] for p in contact) / 4
    if x == 0 and y == 0 and z == 0:
        return None
    acc = raw["accG"]
    frame = {
        "time": clock,
        "x": x,
        "y": y,
        "z": z,
        "speed": raw["speedKmh"] / 3.6,
        "throttle": raw["gas"],
        "brake": raw["brake"],
        "clutch": raw["clutch"],
        "steerRaw": raw["steerAngle"],
        # 0 reverse, 1 neutral, 2 first: the viewer counts -1, 0, 1.
        "gear": raw["gear"] - 1,
        "rpm": raw["rpms"],
        "latAcc": acc[0] * G,
        "vertAcc": acc[1] * G,
        "longAcc": acc[2] * G,
        "yaw": raw["heading"],
        "pitch": raw["pitch"],
        "roll": raw["roll"],
        "yawRate": raw["localAngularVel"][1],
        "drs": raw["drs"],
        "pitLimiter": raw["pitLimiterOn"],
        "boost": raw["turboBoost"],
    }
    # Channels the first game's page does not have; named as the viewer's
    # channel table names them, so its input panel and car pick them up.
    for source, name in (("brakeBias", "brakeBias"), ("tcInAction", "tcActive"), ("absInAction", "absActive")):
        if source in raw:
            frame[name] = raw[source]
    if game in ("acc", "acrally", "acevo"):
        # The aids acting, as ACC documents these slots (its own flags are unused).
        frame["tcActive"] = 1.0 if raw.get("tcActing", 0.0) > 0.0 else 0.0
        frame["absActive"] = 1.0 if raw.get("absActing", 0.0) > 0.0 else 0.0
    local = raw.get("localVelocity")
    if local and any(local):
        # Car frame: x sideways, y up, z forwards. Which side x points to is
        # documented both ways; the viewer reads the drift angle's size.
        frame["latVel"] = local[0]
        frame["vertVel"] = local[1]
        frame["longVel"] = local[2]
    renames = {
        "tyreCoreTemperature": ("tireTemp", 1.0),
        "tyreTempI": ("tireTempInner", 1.0),
        "tyreTempM": ("tireTempMiddle", 1.0),
        "tyreTempO": ("tireTempOuter", 1.0),
        "wheelsPressure": ("tirePressure", 1.0),
        "wheelLoad": ("wheelLoad", 1.0),
        "wheelAngularSpeed": ("wheelSpeed", 1.0),
        "camberRAD": ("camber", 1.0),
        "suspensionTravel": ("suspensionM", 1.0),
        "brakeTemp": ("brakeTemp", 1.0),
        "slipRatio": ("slipRatio", 1.0 / SLIP_RATIO_PEAK),
        "slipAngle": ("slipAngle", 1.0 / SLIP_ANGLE_PEAK),
        "padLife": ("padLife", 1.0),
        "discLife": ("discLife", 1.0),
    }
    for source, (name, scale) in renames.items():
        for wheel in WHEELS:
            key = f"{source}.{wheel}"
            if key in raw:
                value = raw[key]
                if game == "acrally" and source in KELVIN_FIELDS and value > 200:
                    value -= 273.15
                frame[f"{name}.{wheel}"] = value * scale
                if source in ("slipRatio", "slipAngle"):
                    frame[f"{name}Raw.{wheel}"] = raw[key]
    return frame


@dataclass
class KLap:
    """One lap (or stage) of frames."""

    number: int
    frames: list[dict] = field(default_factory=list)
    seconds: float | None = None
    complete: bool = False

    @property
    def metres(self) -> float:
        total = 0.0
        for a, b in zip(self.frames, self.frames[1:]):
            total += math.hypot(b["x"] - a["x"], b["y"] - a["y"], b["z"] - a["z"])
        return total


def document(lap: KLap, *, game: str, track: str | None, car: str | None) -> dict:
    """A lap as a `heat3d-lap` document, as `fulllap` writes Forza's."""
    frames = lap.frames
    t0 = frames[0]["time"]
    names = [k for k in frames[0] if k not in ("time", "steerRaw")]
    channels: dict[str, list] = {"time": [], "distance": [], "steer": []}
    for name in names:
        channels[name] = []
    # Steering: whatever the game's units, -1..1 by the most the lap used,
    # when that is beyond the -1..1 an input would already be in.
    most = max((abs(f["steerRaw"]) for f in frames), default=1.0)
    steer_scale = 1.0 / most if most > 1.05 else 1.0
    travelled = 0.0
    previous = None
    for f in frames:
        if previous is not None:
            travelled += math.hypot(f["x"] - previous["x"], f["y"] - previous["y"], f["z"] - previous["z"])
        previous = f
        channels["time"].append(round(f["time"] - t0, 4))
        channels["distance"].append(round(travelled, 3))
        channels["steer"].append(round(f["steerRaw"] * steer_scale, 4))
        for name in names:
            v = f[name]
            channels[name].append(round(v, 5) if isinstance(v, float) else v)
    # Suspension as 0..1 of the travel this lap used, for the car's bars.
    for wheel in WHEELS:
        travel = channels.get(f"suspensionM.{wheel}")
        if travel:
            lo, hi = min(travel), max(travel)
            span = hi - lo if hi > lo else 1.0
            channels[f"suspension.{wheel}"] = [round((v - lo) / span, 4) for v in travel]
    seconds = lap.seconds if lap.seconds is not None else channels["time"][-1]
    return {
        "format": "heat3d-lap",
        "version": 1,
        "game": game,
        "track": track,
        "recordedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "car": {"name": car} if car else {},
        "lap": {"number": lap.number, "seconds": round(seconds, 4), "complete": lap.complete, "metres": round(lap.metres, 1)},
        # What the numbers mean where games differ: see lapfile.py.
        "meta": {"slip": "scaled", "discipline": "rally" if "rally" in game else "circuit"},
        "units": {
            "time": "s",
            "distance": "m",
            "speed": "m/s",
            "tireTemp": "degC",
            "tirePressure": "psi",
            "wheelLoad": "N",
            "suspensionM": "m",
            "suspension": "0..1 of the lap's travel",
            "slipRatio": f"normalised: raw / {SLIP_RATIO_PEAK}",
            "slipAngle": f"normalised: raw rad / {SLIP_ANGLE_PEAK}",
            "brakeTemp": "degC",
        },
        "channels": channels,
    }


class CounterSplitter:
    """Laps by the game's own counter (AC, ACC): the lap time is the game's."""

    def __init__(self, *, minimum_metres: float = 300.0):
        self.minimum = minimum_metres
        self.current: KLap | None = None

    def feed(self, frame: dict, laps: int, last_lap_ms: int) -> list[KLap]:
        done: list[KLap] = []
        if self.current is None:
            self.current = KLap(number=laps)
        elif laps != self.current.number:
            closing = self.current
            if laps == closing.number + 1:
                closing.complete = True
                if last_lap_ms > 0:
                    closing.seconds = last_lap_ms / 1000.0
            if closing.frames and closing.metres >= self.minimum:
                done.append(closing)
            self.current = KLap(number=laps)
        self.current.frames.append(frame)
        return done

    def flush(self) -> list[KLap]:
        lap, self.current = self.current, None
        return [lap] if lap and len(lap.frames) > 1 and lap.metres >= self.minimum else []


class LineSplitter:
    """
    Laps without a counter (AC EVO, AC Rally): the point the recording started
    at is the line. A lap closes where the car next passes closest to it,
    heading the same way, having gone at least `minimum_metres`; a stage (or
    any run) closes when the car stands still for `stop_seconds`.
    """

    def __init__(self, *, radius: float = 15.0, minimum_metres: float = 300.0, stop_seconds: float = 8.0):
        self.radius = radius
        self.minimum = minimum_metres
        self.stop = stop_seconds
        self.line: tuple[float, float] | None = None
        self.direction: tuple[float, float] | None = None
        self.current: KLap | None = None
        self.number = 0
        self._travelled = 0.0
        self._near: list[int] = []
        self._still_since: float | None = None

    def _dist(self, f: dict) -> float:
        return math.hypot(f["x"] - self.line[0], f["z"] - self.line[1])

    def feed(self, frame: dict) -> list[KLap]:
        done: list[KLap] = []
        if self.current is None:
            self.current = KLap(number=self.number)
            self._travelled = 0.0
            self._near = []
        cur = self.current
        if cur.frames:
            prev = cur.frames[-1]
            self._travelled += math.hypot(frame["x"] - prev["x"], frame["z"] - prev["z"])
        cur.frames.append(frame)
        if self.line is None:
            self.line = (frame["x"], frame["z"])
        if self.direction is None and self._travelled > 5:
            dx, dz = frame["x"] - self.line[0], frame["z"] - self.line[1]
            n = math.hypot(dx, dz) or 1.0
            self.direction = (dx / n, dz / n)

        # Standing still at the end of a run.
        if frame["speed"] < 1.0:
            self._still_since = self._still_since if self._still_since is not None else frame["time"]
            if frame["time"] - self._still_since >= self.stop:
                done += self.flush()
                return done
        else:
            self._still_since = None

        # Back past the line: the closest sample within the radius closes the lap.
        if self.direction is not None and self._travelled > self.minimum:
            if self._dist(frame) < self.radius:
                self._near.append(len(cur.frames) - 1)
            elif self._near:
                best = min(self._near, key=lambda i: self._dist(cur.frames[i]))
                f = cur.frames
                j = min(len(f) - 1, best + 1)
                heading = (f[j]["x"] - f[best]["x"], f[j]["z"] - f[best]["z"])
                same_way = heading[0] * self.direction[0] + heading[1] * self.direction[1] > 0
                self._near = []
                if same_way:
                    # The line is wherever the recording began, so every lap
                    # runs line to line - but one begun from standstill is a
                    # launch, not a lap to compare.
                    closing = KLap(number=cur.number, frames=f[: best + 1], complete=f[0]["speed"] > 5.0)
                    closing.seconds = f[best]["time"] - f[0]["time"]
                    if closing.metres >= self.minimum:
                        done.append(closing)
                    self.number += 1
                    rest = f[best:]
                    self.current = KLap(number=self.number, frames=rest)
                    self._travelled = KLap(0, rest).metres
        return done

    def flush(self) -> list[KLap]:
        lap, self.current = self.current, None
        self.number += 1
        if lap is None or len(lap.frames) < 2 or lap.metres < self.minimum:
            return []
        return [lap]


def read_static(page: bytes, game: str) -> tuple[str | None, str | None]:
    """Track and car from the static page, where the layout is known."""

    def wide(offset: int, chars: int) -> str:
        raw = page[offset : offset + chars * 2]
        text = raw.decode("utf-16-le", errors="ignore")
        return text.split("\x00", 1)[0].strip()

    def narrow(offset: int, chars: int) -> str:
        return page[offset : offset + chars].split(b"\x00", 1)[0].decode("latin-1", errors="ignore").strip()

    try:
        if game == "acevo":
            track = narrow(136, 33)
            config = narrow(169, 33)
            return (f"{track} {config}".strip() or None, None)
        car = wide(68, 33)
        track = wide(134, 33)
        return (track or None, car or None)
    except Exception:
        return (None, None)


def lap_filename(lap: KLap, game: str) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    seconds = lap.seconds if lap.seconds is not None else lap.frames[-1]["time"] - lap.frames[0]["time"]
    kind = "lap" if lap.complete else "run"
    return f"{stamp}_{game}_{kind}{lap.number}_{seconds:.3f}s.lap.json"


def write(lap: KLap, folder: Path, *, game: str, track: str | None, car: str | None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / lap_filename(lap, game)
    path.write_text(json.dumps(document(lap, game=GAME_NAMES[game], track=track, car=car), separators=(",", ":")), encoding="utf-8")
    return path


def _open(name: str, size: int):
    """A read-only view of a named mapping the game created. Windows only."""
    import mmap

    return mmap.mmap(-1, size, tagname=name, access=mmap.ACCESS_READ)


def run(out: str | Path, *, game: str, hz: float = 200.0) -> list[Path]:
    """Record until Ctrl+C; returns the files written."""
    if sys.platform != "win32":
        raise SystemExit("The Assetto Corsa games publish their telemetry as Windows shared memory: record on the PC the game runs on.")
    if game not in MAPPINGS:
        raise SystemExit(f"unknown game {game!r}; one of {', '.join(MAPPINGS)}")
    physics_name, graphics_name, static_name = MAPPINGS[game]
    out = Path(out)
    size = PHYSICS_SIZE
    try:
        physics = _open(physics_name, size)
    except OSError:
        size = PHYSICS_AC1_SIZE
        physics = _open(physics_name, size)
    graphics = _open(graphics_name, GRAPHICS_SIZE) if graphics_name else None
    static = _open(static_name, 256)
    splitter: CounterSplitter | LineSplitter = CounterSplitter() if graphics else LineSplitter()
    written: list[Path] = []
    last_packet = None
    track = car = None
    start = time.perf_counter()
    period = 1.0 / hz
    try:
        while True:
            page = bytes(physics[:size])
            packet = struct.unpack_from("<i", page, 0)[0]
            if packet != last_packet and packet != 0:
                last_packet = packet
                if track is None:
                    track, car = read_static(bytes(static[:256]), game)
                frame = frame_from(decode_physics(page), time.perf_counter() - start, game)
                if frame is not None:
                    if graphics is not None:
                        g = bytes(graphics[:GRAPHICS_SIZE])
                        laps = struct.unpack_from("<i", g, G_COMPLETED_LAPS)[0]
                        last_ms = struct.unpack_from("<i", g, G_LAST_LAP_MS)[0]
                        done = splitter.feed(frame, laps, last_ms)  # type: ignore[call-arg]
                    else:
                        done = splitter.feed(frame)  # type: ignore[call-arg]
                    for lap in done:
                        written.append(write(lap, out, game=game, track=track, car=car))
                        print(f"  {written[-1].name}")
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    for lap in splitter.flush():
        written.append(write(lap, out, game=game, track=track, car=car))
    return written
