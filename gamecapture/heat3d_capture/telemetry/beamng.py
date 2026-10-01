"""
BeamNG.drive, recorded a lap at a time for the Race tab.

BeamNG sends its telemetry through "protocols": Lua modules in the vehicle
that the game's own `lua/vehicle/protocols.lua` packs and sends over UDP from
the car the player drives (documentation.beamng.com/modding/protocols). The
stock ones - OutGauge (no position) and MotionSim (no wheels) - miss what a
car-control analysis needs, so this ships its own, `mods/beamng/heat3d`, which
sends what the vehicle's Lua has: position, world velocity, the car's forward
and up vectors, yaw rate, pedals and steering, rpm, gear, the ABS and TC lamps,
and per wheel its surface speed (m/s), rotation (rad/s), load (N) and the
ground material under it (-1: off the ground). Switch on Options > Other >
Protocols > "others".

BeamNG's world has Z up; the viewer's has Y up, so positions and vectors are
turned about X - (x, y, z) -> (x, z, -y) - which keeps the world's handedness.
The car's own velocities come from projecting the world velocity onto its
axes: forwards, up, and right = forwards x up. Accelerations are taken from the
velocity rather than from the car's sensors, whose frame is not documented.

Laps close where the car passes the point the recording started at again,
heading the same way (as for AC EVO); a run closes when the car stands still.
"""

from __future__ import annotations

import math
import os
import shutil
import socket
import struct
from pathlib import Path

from .kunos import LineSplitter
from .lapfile import Run, write

PORT = 4460
MAGIC = b"H3D1"
#: 4-byte magic, then floats: time, position 3, velocity 3, forward 3, up 3,
#: yaw rate, 9 inputs, then 4 per wheel of surface speed, rotation, load, material.
FLOATS = 1 + 3 + 3 + 3 + 3 + 1 + 9 + 16
LENGTH = 4 + 4 * FLOATS
WHEELS = ("FL", "FR", "RL", "RR")

MOD = Path(__file__).resolve().parents[2] / "mods" / "beamng" / "heat3d"


def to_viewer(v: tuple[float, float, float]) -> tuple[float, float, float]:
    """BeamNG's Z-up world to the viewer's Y-up one, turned about X."""
    return (v[0], v[2], -v[1])


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def decode(packet: bytes) -> dict | None:
    if len(packet) < LENGTH or packet[:4] != MAGIC:
        return None
    f = struct.unpack_from(f"<{FLOATS}f", packet, 4)
    k = iter(f)
    take = lambda n: tuple(next(k) for _ in range(n))  # noqa: E731
    raw = {
        "time": next(k),
        "position": take(3),
        "velocity": take(3),
        "forward": take(3),
        "up": take(3),
        "yawRate": next(k),
    }
    for name in ("throttle", "brake", "clutch", "handbrake", "steer", "rpm", "gear", "abs", "tcs"):
        raw[name] = next(k)
    for name in ("wheelSpeed", "wheelAngular", "wheelLoad", "wheelMaterial"):
        raw[name] = take(4)
    return raw


def frame_from(raw: dict) -> dict:
    p = to_viewer(raw["position"])
    v = to_viewer(raw["velocity"])
    forward = to_viewer(raw["forward"])
    up = to_viewer(raw["up"])
    right = cross(forward, up)
    frame = {
        "time": raw["time"],
        "x": p[0],
        "y": p[1],
        "z": p[2],
        "speed": math.sqrt(dot(v, v)),
        "longVel": dot(v, forward),
        "latVel": dot(v, right),
        "vertVel": dot(v, up),
        "yaw": math.atan2(forward[0], forward[2]),
        "yawRate": raw["yawRate"],
        "throttle": raw["throttle"],
        "brake": raw["brake"],
        "clutch": raw["clutch"],
        "handbrake": raw["handbrake"],
        "steer": raw["steer"],
        "rpm": raw["rpm"],
        "gear": raw["gear"],
        "absActive": raw["abs"],
        "tcActive": raw["tcs"],
        "_v": v,
        "_f": forward,
        "_r": right,
        "_u": up,
    }
    for k, wheel in enumerate(WHEELS):
        frame[f"wheelSurfaceSpeed.{wheel}"] = abs(raw["wheelSpeed"][k])
        frame[f"wheelSpeed.{wheel}"] = raw["wheelAngular"][k]
        frame[f"wheelLoad.{wheel}"] = raw["wheelLoad"][k]
        frame[f"groundContact.{wheel}"] = 0.0 if raw["wheelMaterial"][k] < 0 else 1.0
    return frame


def add_accelerations(frames: list[dict]) -> None:
    """
    The car-frame accelerations from the world velocity, over +-2 samples. The
    frame where one lap ends and the next begins belongs to both, and is done
    once: what the first lap worked out, the second keeps.
    """
    n = len(frames)
    vectors = [(f.get("_v"), f.get("_f"), f.get("_r"), f.get("_u")) for f in frames]
    for i, f in enumerate(frames):
        if vectors[i][0] is None:
            continue
        a, b = max(0, i - 2), min(n - 1, i + 2)
        while vectors[a][0] is None and a < i:
            a += 1
        while vectors[b][0] is None and b > i:
            b -= 1
        dt = frames[b]["time"] - frames[a]["time"]
        if dt <= 0:
            acc = (0.0, 0.0, 0.0)
        else:
            va, vb = vectors[a][0], vectors[b][0]
            acc = tuple((vb[j] - va[j]) / dt for j in range(3))
        _, fwd, right, up = vectors[i]
        f["longAcc"] = dot(acc, fwd)
        f["latAcc"] = dot(acc, right)
        f["vertAcc"] = dot(acc, up)
    for f in frames:
        for key in ("_v", "_f", "_r", "_u"):
            f.pop(key, None)


class Recorder:
    def __init__(self, out: str | Path):
        self.out = Path(out)
        self.splitter = LineSplitter()
        self.written: list[Path] = []
        self.last_time = -1.0

    def feed(self, packet: bytes) -> list[Path]:
        raw = decode(packet)
        if raw is None:
            return []
        frame = frame_from(raw)
        # The car was reset: the clock starts again.
        if frame["time"] < self.last_time - 0.5:
            self.splitter = LineSplitter()
        self.last_time = frame["time"]
        out = []
        for lap in self.splitter.feed(frame):
            out.append(self._write(Run(number=lap.number, frames=lap.frames, seconds=lap.seconds, complete=lap.complete)))
        return out

    def _write(self, run: Run) -> Path:
        add_accelerations(run.frames)
        path = write(run, self.out, game="beamng", meta={"discipline": "circuit"})
        self.written.append(path)
        return path

    def close(self) -> list[Path]:
        return [
            self._write(Run(number=lap.number, frames=lap.frames, seconds=lap.seconds, complete=lap.complete))
            for lap in self.splitter.flush()
        ]


def user_mods() -> Path:
    """BeamNG's user folder for unpacked mods (0.30 on: under LOCALAPPDATA)."""
    base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "BeamNG" / "BeamNG.drive" / "current" / "mods" / "unpacked"
    return base


def install(target: Path | None = None) -> Path:
    """Copy the protocol mod into BeamNG's unpacked mods; returns where it went."""
    target = (target or user_mods()) / "heat3d"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(MOD, target)
    return target


def run(out: str | Path, *, port: int = PORT, report=print) -> list[Path]:
    """Listen until Ctrl+C; returns the files written."""
    recorder = Recorder(out)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(0.5)
    report(f"listening for BeamNG on port {port}. Laps go to {out}. Ctrl+C to stop.")
    try:
        while True:
            try:
                packet, _ = sock.recvfrom(2048)
            except socket.timeout:
                continue
            for path in recorder.feed(packet):
                report(f"lap written: {path.name}")
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        for path in recorder.close():
            report(f"run written: {path.name}")
    return recorder.written
