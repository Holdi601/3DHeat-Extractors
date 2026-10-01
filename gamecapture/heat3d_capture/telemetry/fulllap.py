"""
Every channel Forza sends, recorded a lap at a time, for the Race tab.

FH Companion keeps a lap as thirteen channels sampled every five metres: enough
to see a line and a speed trace, not enough to say why a lap was slow. The game
sends a great deal more, sixty times a second - every wheel's slip ratio, slip
angle, combined slip, temperature, suspension travel and rotation speed, the
engine's RPM, power and torque, the car's yaw rate - and this keeps all of it.

The game sends Data Out to one address. So this listens on that address and
passes every packet on unchanged (`forward`), which is how FH Companion keeps
working alongside it: point the game at this, and FH Companion at the forward
port.

The lap file (`heat3d-lap`)
---------------------------
One JSON document per lap, channels as columns::

    {"format": "heat3d-lap", "version": 1, "game": "forza",
     "car": {...}, "lap": {"number": 3, "seconds": 52.103, "complete": true},
     "channels": {"time": [...], "distance": [...], "speed": [...],
                  "tireTemp.FL": [...], ...},
     "units": {"speed": "m/s", "tireTemp": "degC", ...}}

Channel names are the canonical ones (`docs/lap-format.md`), so the
same file shape carries any game: a later Assetto Corsa or iRacing recorder
writes the same document with more channels, and the viewer draws what it finds.
"""

from __future__ import annotations

import json
import math
import socket
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .forza import (
    DASH_BLOCK_LENGTH,
    FM_DASH_LENGTH,
    SLED_LENGTH,
    Layout,
    LayoutUndetermined,
    detect_layout,
)

WHEELS = ("FL", "FR", "RL", "RR")

#: Sled block: (name, format, offset). Offsets are absolute and the same in
#: every variant; per-wheel fields are four values in FL, FR, RL, RR order.
SLED = [
    ("isRaceOn", "<i", 0),
    ("timestampMs", "<I", 4),
    ("rpmMax", "<f", 8),
    ("rpmIdle", "<f", 12),
    ("rpm", "<f", 16),
    # Car-local acceleration and velocity: x right, y up, z forward.
    ("accelX", "<f", 20),
    ("accelY", "<f", 24),
    ("accelZ", "<f", 28),
    ("velX", "<f", 32),
    ("velY", "<f", 36),
    ("velZ", "<f", 40),
    ("angVelX", "<f", 44),
    ("angVelY", "<f", 48),
    ("angVelZ", "<f", 52),
    ("yaw", "<f", 56),
    ("pitch", "<f", 60),
    ("roll", "<f", 64),
    ("carOrdinal", "<i", 212),
    ("carClass", "<i", 216),
    ("carPI", "<i", 220),
    ("drivetrain", "<i", 224),
    ("cylinders", "<i", 228),
]
SLED_WHEELS = [
    ("suspension", "<4f", 68),  # normalised: 0 fully stretched, 1 fully compressed
    ("slipRatio", "<4f", 84),  # normalised: 0 full grip, |x| > 1 grip lost
    ("wheelSpeed", "<4f", 100),  # rad/s
    ("rumble", "<4i", 116),  # on a rumble strip: 1
    ("puddle", "<4f", 132),  # puddle depth, 0..1
    ("surfaceRumble", "<4f", 148),
    ("slipAngle", "<4f", 164),  # normalised like slipRatio
    ("combinedSlip", "<4f", 180),  # normalised like slipRatio
    ("suspensionM", "<4f", 196),  # metres
]

#: Dash block, relative to its own start (which `detect_layout` finds).
DASH = [
    ("x", "<f", 0),
    ("y", "<f", 4),
    ("z", "<f", 8),
    ("speed", "<f", 12),
    ("power", "<f", 16),  # watts
    ("torque", "<f", 20),  # newton metres
    ("boost", "<f", 40),
    ("fuel", "<f", 44),
    ("distanceTravelled", "<f", 48),
    ("bestLap", "<f", 52),
    ("lastLap", "<f", 56),
    ("currentLap", "<f", 60),
    ("raceTime", "<f", 64),
    ("lapNumber", "<H", 68),
    ("racePosition", "<B", 70),
    ("throttle", "<B", 71),
    ("brake", "<B", 72),
    ("clutch", "<B", 73),
    ("handbrake", "<B", 74),
    ("gear", "<B", 75),
    ("steer", "<b", 76),
    ("drivingLine", "<b", 77),
    ("aiBrakeDifference", "<b", 78),
]
DASH_TIRE_TEMP = 24  # four floats, Fahrenheit

#: Forza Motorsport (2023) appends tyre wear and the track after the dash block.
FM2023_LENGTH = FM_DASH_LENGTH + 20


def decode_full(packet: bytes, layout: Layout) -> dict:
    """
    One packet, every field, in the canonical channel names and SI units.

    Pedals come out 0..1, steering -1..1, temperatures in Celsius (the game
    sends Fahrenheit), and the car-local accelerations are also given under the
    names the analysis uses: `latAcc` (positive to the right), `longAcc`
    (positive forwards) and `yawRate`.
    """
    out: dict = {}
    for name, fmt, at in SLED:
        out[name] = struct.unpack_from(fmt, packet, at)[0]
    for name, fmt, at in SLED_WHEELS:
        for wheel, value in zip(WHEELS, struct.unpack_from(fmt, packet, at)):
            out[f"{name}.{wheel}"] = value
    base = layout.dash_base
    for name, fmt, at in DASH:
        out[name] = struct.unpack_from(fmt, packet, base + at)[0]
    for wheel, value in zip(WHEELS, struct.unpack_from("<4f", packet, base + DASH_TIRE_TEMP)):
        out[f"tireTemp.{wheel}"] = (value - 32.0) * 5.0 / 9.0
    if len(packet) >= base + DASH_BLOCK_LENGTH + 20:
        tail = base + DASH_BLOCK_LENGTH
        for wheel, value in zip(WHEELS, struct.unpack_from("<4f", packet, tail)):
            out[f"tireWear.{wheel}"] = value
        out["trackOrdinal"] = struct.unpack_from("<i", packet, tail + 16)[0]
    for pedal in ("throttle", "brake", "clutch", "handbrake"):
        out[pedal] /= 255.0
    out["steer"] /= 127.0
    out["drivingLine"] /= 127.0
    out["aiBrakeDifference"] /= 127.0
    out["latAcc"] = out["accelX"]
    out["longAcc"] = out["accelZ"]
    out["vertAcc"] = out["accelY"]
    out["yawRate"] = out["angVelY"]
    return out


#: Per-lap facts, taken once rather than written into every sample.
CAR_FIELDS = ("carOrdinal", "carClass", "carPI", "drivetrain", "cylinders", "rpmMax", "rpmIdle")
#: Not worth a column: constant through a lap, or bookkeeping.
DROPPED = set(CAR_FIELDS) | {"isRaceOn", "timestampMs", "bestLap", "lastLap", "raceTime", "trackOrdinal"}

UNITS = {
    "time": "s",
    "distance": "m",
    "x": "m",
    "y": "m",
    "z": "m",
    "speed": "m/s",
    "rpm": "rpm",
    "power": "W",
    "torque": "N m",
    "latAcc": "m/s2",
    "longAcc": "m/s2",
    "vertAcc": "m/s2",
    "yawRate": "rad/s",
    "yaw": "rad",
    "pitch": "rad",
    "roll": "rad",
    "tireTemp": "degC",
    "suspension": "0..1",
    "suspensionM": "m",
    "slipRatio": "normalised",
    "slipAngle": "normalised",
    "combinedSlip": "normalised",
    "wheelSpeed": "rad/s",
    "throttle": "0..1",
    "brake": "0..1",
    "steer": "-1..1",
}


@dataclass
class Lap:
    """One lap's samples, as decoded packets."""

    number: int
    frames: list[dict] = field(default_factory=list)
    #: The game's own lap time, when the lap was closed by the game counting it.
    seconds: float | None = None
    complete: bool = False

    @property
    def distance(self) -> float:
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1]["distanceTravelled"] - self.frames[0]["distanceTravelled"]

    def document(self, *, game: str = "forza", layout: str = "") -> dict:
        """The lap as a `heat3d-lap` document."""
        first = self.frames[0]
        start_time = first["currentLap"]
        start_distance = first["distanceTravelled"]
        names = [k for k in first if k not in DROPPED]
        channels: dict[str, list] = {"time": [], "distance": []}
        for name in names:
            channels.setdefault(name, [])
        for frame in self.frames:
            channels["time"].append(round(frame["currentLap"] - start_time, 4))
            channels["distance"].append(round(frame["distanceTravelled"] - start_distance, 3))
            for name in names:
                value = frame[name]
                channels[name].append(round(value, 5) if isinstance(value, float) else value)
        seconds = self.seconds if self.seconds is not None else channels["time"][-1]
        return {
            "format": "heat3d-lap",
            "version": 1,
            "game": game,
            "layout": layout,
            "recordedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
            "car": {name: first[name] for name in CAR_FIELDS},
            "lap": {
                "number": self.number,
                "seconds": round(seconds, 4),
                "complete": self.complete,
                "metres": round(self.distance, 1),
            },
            "units": {name: unit for name, unit in UNITS.items()},
            "channels": channels,
        }


class LapSplitter:
    """
    Frames in, finished laps out.

    A lap closes when the game's lap counter moves on - its time is then the
    game's own `lastLap`, which is what the game shows - or when the race stops
    (the game leaves a race, or the packets stop for two seconds), in which case
    it is kept as a partial run if it covered at least `minimum_metres`: a
    point-to-point race never counts a second lap.
    """

    def __init__(self, *, minimum_metres: float = 400.0, gap_seconds: float = 2.0):
        self.minimum_metres = minimum_metres
        self.gap_ms = gap_seconds * 1000.0
        self.current: Lap | None = None
        self._last_ms: int | None = None

    def feed(self, frame: dict) -> list[Lap]:
        done: list[Lap] = []
        stamp = frame["timestampMs"]
        if self._last_ms is not None and (stamp - self._last_ms > self.gap_ms or stamp < self._last_ms):
            done += self.flush()
        self._last_ms = stamp
        if not frame["isRaceOn"]:
            done += self.flush()
            return done
        if self.current is None:
            self.current = Lap(number=frame["lapNumber"])
        elif frame["lapNumber"] != self.current.number:
            closing = self.current
            closing.complete = True
            if frame["lastLap"] > 0:
                closing.seconds = frame["lastLap"]
            if closing.frames:
                done.append(closing)
            self.current = Lap(number=frame["lapNumber"])
        self.current.frames.append(frame)
        return done

    def flush(self) -> list[Lap]:
        """Close whatever is open, if it is worth keeping."""
        lap, self.current = self.current, None
        if lap is None or len(lap.frames) < 2 or lap.distance < self.minimum_metres:
            return []
        return [lap]


def lap_filename(lap: Lap) -> str:
    first = lap.frames[0]
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    seconds = lap.seconds if lap.seconds is not None else lap.frames[-1]["currentLap"] - first["currentLap"]
    kind = "lap" if lap.complete else "run"
    return f"{stamp}_car{first['carOrdinal']}_{kind}{lap.number}_{seconds:.3f}s.lap.json"


def game_for(layout: str) -> str:
    """Which Forza, from the packet layout: the two send different lengths."""
    if "Horizon" in layout:
        return "forza-horizon"
    if "Motorsport" in layout or layout.startswith(f"{FM2023_LENGTH}-byte"):
        return "forza-motorsport"
    return "forza"


def write_lap(lap: Lap, folder: str | Path, *, layout: str = "") -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / lap_filename(lap)
    path.write_text(json.dumps(lap.document(game=game_for(layout), layout=layout), separators=(",", ":")), encoding="utf-8")
    return path


class FullRecorder:
    """
    Listen, relay, decode, split and write, one packet at a time.

    Layout detection is the listener's (`forza.detect_layout`): packets are held
    until the car has moved and the dash block's position is proven, and only
    then decoded. Relaying does not wait for it.
    """

    def __init__(
        self,
        folder: str | Path,
        *,
        host: str = "127.0.0.1",
        port: int = 5300,
        forward: tuple[str, int] | None = None,
        timeout: float = 0.5,
    ):
        self.folder = Path(folder)
        self.forward = forward
        self.layout: Layout | None = None
        self.splitter = LapSplitter()
        self.packets = 0
        self.written: list[Path] = []
        self._pending: list[bytes] = []
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(timeout)
        self._sock.bind((host, port))
        self._out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if forward else None
        self._heard = time.monotonic()

    def close(self) -> None:
        for lap in self.splitter.flush():
            self._write(lap)
        self._sock.close()
        if self._out is not None:
            self._out.close()

    def __enter__(self) -> "FullRecorder":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def poll(self) -> list[Path]:
        """Take one packet if there is one. Returns laps written by it."""
        try:
            packet, _addr = self._sock.recvfrom(2048)
        except socket.timeout:
            # Silence is a gap too: the game paused, or was closed mid-lap.
            if time.monotonic() - self._heard > self.splitter.gap_ms / 1000.0:
                return [self._write(lap) for lap in self.splitter.flush()]
            return []
        self._heard = time.monotonic()
        return self.feed(packet)

    def feed(self, packet: bytes) -> list[Path]:
        """Offer one packet; separate from `poll` so tests can drive it."""
        self.packets += 1
        if self._out is not None and self.forward is not None:
            self._out.sendto(packet, self.forward)
        if len(packet) < SLED_LENGTH + DASH_BLOCK_LENGTH:
            return []
        if self.layout is None:
            self._pending.append(packet)
            if len(self._pending) > 600:
                self._pending = self._pending[-300:]
            try:
                self.layout = detect_layout(self._pending)
            except LayoutUndetermined:
                return []
            backlog, self._pending = self._pending, []
            written = []
            for held in backlog:
                written += self._frames(held)
            return written
        return self._frames(packet)

    def _frames(self, packet: bytes) -> list[Path]:
        if len(packet) != self.layout.packet_length:
            return []
        frame = decode_full(packet, self.layout)
        if not all(math.isfinite(v) for v in (frame["x"], frame["z"], frame["speed"])):
            return []
        return [self._write(lap) for lap in self.splitter.feed(frame)]

    def _write(self, lap: Lap) -> Path:
        path = write_lap(lap, self.folder, layout=self.layout.name if self.layout else "")
        self.written.append(path)
        return path


def run(folder: str | Path, *, port: int, forward: tuple[str, int] | None, report=print) -> list[Path]:
    """Record until interrupted; returns the files written."""
    started = time.monotonic()
    last = 0.0
    with FullRecorder(folder, port=port, forward=forward) as recorder:
        try:
            while True:
                for path in recorder.poll():
                    report(f"\nwrote {path.name}")
                now = time.monotonic() - started
                if now - last >= 1.0:
                    last = now
                    lap = recorder.splitter.current
                    state = (
                        f"lap {lap.number}, {len(lap.frames)} samples, {lap.distance:,.0f} m"
                        if lap is not None
                        else "waiting for a race"
                    )
                    report(f"\r[{now:5.0f}s] {recorder.packets} packets - {state}   ", end="", flush=True)
        except KeyboardInterrupt:
            pass
    return recorder.written


def parse_forward(text: str | None) -> tuple[str, int] | None:
    if not text:
        return None
    host, _, port = text.rpartition(":")
    return (host or "127.0.0.1", int(port))


def frames_from(packets: Iterable[bytes], layout: Layout) -> list[dict]:
    """Decode a batch of packets with a known layout (for tests and tools)."""
    return [decode_full(p, layout) for p in packets if len(p) == layout.packet_length]
