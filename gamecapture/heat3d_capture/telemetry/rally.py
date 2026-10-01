"""
Rally games' telemetry, recorded a stage at a time for the Race tab.

All three send documented UDP telemetry that the game switches on from its own
settings files; nothing is injected and nothing unpublished is read.

EA SPORTS WRC
    `Documents/My Games/WRC/telemetry/config.json`: enable the `wrc` structure's
    `session_update` packet (`bEnabled: true`) to this machine and port. The
    default `wrc` structure is 237 bytes, little-endian, no header: positions,
    velocity and acceleration in the world (X left, Y up, Z forwards), the car's
    left/forward/up unit vectors, per wheel (BL BR FL FR) the hub's height and
    vertical speed and the contact patch's forward speed, brake temperatures,
    pedals after assists, steering (-1 left, +1 right), handbrake, and the
    stage clock, distance and length. `configure_wrc()` switches the packet on.
    Sources: EA's UDP Telemetry Guide v1.3; the channel list the game writes to
    `readme/channels.json`; offsets as decoded by nobonobo/obs-codemasters-
    telemetry and SpaceMonkey.

DiRT Rally 2.0 (and DiRT Rally, DiRT 4)
    `Documents/My Games/DiRT Rally 2.0/hardwaresettings/hardware_settings_config.xml`:
    `<udp enabled="true" extradata="3" ip="127.0.0.1" port="20777" delay="1" />`.
    66 floats (DiRT Rally: 64): stage clock and distance, world position and
    velocity, the car's side and forward vectors, per wheel (RL RR FL FR) the
    suspension position and speed and the contact patch speed, pedals,
    steering, gear, lateral and longitudinal g, rpm / 10, and the stage length.
    No handbrake. Sources: Codemasters' DiRT 4 UDP document; ErlerPhilipp/
    dr2_logger; soong-construction/dirt-rally-time-recorder.

Richard Burns Rally (RSF / NGP plugin)
    `RichardBurnsRally.ini`, `[NGP]`: `udpTelemetry=1` and the endpoint. 664
    bytes (`Plugins/NGP/sdk/rbr.telemetry.data.TelemetryData.h`): the stage
    clock and distance to the end, pedals, steering, gear, world position,
    attitude, the car's own velocities (surge forwards, sway sideways), per
    wheel (LF RF LB RB) spring deflection and force. No wheel speeds. Source:
    mika-n/RBRUDPTelemetryLogger; groybe/rbr-udp-telem. Its position axes and
    speed units are not documented: up is taken as the axis the stage varies
    least along, and speed from the positions.

None of the three sends tyre slip. The viewer derives it from the contact
patch speeds against the car's speed, and the drift angle from the car's own
velocity - see `docs/lap-format.md`.
"""

from __future__ import annotations

import json
import math
import socket
import struct
import time
from pathlib import Path

from .lapfile import Run, dot, write

G = 9.80665

# ---------------------------------------------------------------- EA SPORTS WRC

WRC_LENGTH = 237
#: (offset, format, name) in the default `wrc` session_update packet.
WRC_FIELDS = [
    (0, "Q", "packet_uid"),
    (8, "f", "game_total_time"),
    (12, "f", "game_delta_time"),
    (37, "B", "gear_index"),
    (38, "B", "gear_neutral"),
    (39, "B", "gear_reverse"),
    (41, "f", "speed"),
    (45, "f", "transmission_speed"),
    (49, "3f", "position"),
    (61, "3f", "velocity"),
    (73, "3f", "acceleration"),
    (85, "3f", "left"),
    (97, "3f", "forward"),
    (109, "3f", "up"),
    (121, "4f", "hub_position"),
    (137, "4f", "hub_velocity"),
    (153, "4f", "cp_forward_speed"),
    (169, "4f", "brake_temperature"),
    (185, "f", "rpm_max"),
    (189, "f", "rpm_idle"),
    (193, "f", "rpm"),
    (197, "f", "throttle"),
    (201, "f", "brake"),
    (205, "f", "clutch"),
    (209, "f", "steering"),
    (213, "f", "handbrake"),
    (217, "f", "stage_time"),
    (221, "d", "stage_distance"),
    (229, "d", "stage_length"),
]
#: EA WRC orders the wheels back-left, back-right, front-left, front-right.
WRC_WHEELS = ("RL", "RR", "FL", "FR")


def decode_wrc(packet: bytes) -> dict | None:
    if len(packet) < WRC_LENGTH:
        return None
    raw: dict = {}
    for offset, fmt, name in WRC_FIELDS:
        values = struct.unpack_from("<" + fmt, packet, offset)
        raw[name] = values[0] if len(values) == 1 else values
    return raw


def frame_wrc(raw: dict) -> dict:
    """World vectors projected onto the car's axes: x right, y up, z forwards."""
    left, forward, up = raw["left"], raw["forward"], raw["up"]
    v, a = raw["velocity"], raw["acceleration"]
    gear = raw["gear_index"]
    if gear == raw["gear_reverse"]:
        gear = -1
    elif gear == raw["gear_neutral"]:
        gear = 0
    frame = {
        "x": raw["position"][0],
        "y": raw["position"][1],
        "z": raw["position"][2],
        "speed": raw["speed"],
        "longVel": dot(v, forward),
        "latVel": -dot(v, left),
        "vertVel": dot(v, up),
        "longAcc": dot(a, forward),
        "latAcc": -dot(a, left),
        "vertAcc": dot(a, up),
        "yaw": math.atan2(forward[0], forward[2]),
        "throttle": raw["throttle"],
        "brake": raw["brake"],
        "clutch": raw["clutch"],
        "handbrake": raw["handbrake"],
        "steer": raw["steering"],
        "gear": gear,
        "rpm": raw["rpm"],
        "stageDistance": raw["stage_distance"],
    }
    for wheel, hub, speed, temp in zip(WRC_WHEELS, raw["hub_position"], raw["cp_forward_speed"], raw["brake_temperature"]):
        frame[f"suspensionM.{wheel}"] = hub
        frame[f"wheelSurfaceSpeed.{wheel}"] = speed
        frame[f"brakeTemp.{wheel}"] = temp
    return frame


def configure_wrc(config: Path, *, port: int) -> str:
    """
    Switch on the default `wrc` session_update packet to this machine in the
    game's telemetry config, keeping everything else; the file is backed up
    first. Returns what it did.
    """
    text = config.read_text(encoding="utf-8")
    doc = json.loads(text)
    packets = doc.setdefault("udp", {}).setdefault("packets", [])
    for p in packets:
        if p.get("structure") == "wrc" and p.get("packet") == "session_update":
            if p.get("bEnabled") and p.get("port") == port and p.get("ip") in ("127.0.0.1", "localhost"):
                return "already on"
            p.update({"ip": "127.0.0.1", "port": port, "bEnabled": True, "frequencyHz": p.get("frequencyHz", -1) or -1})
            break
    else:
        packets.append({"structure": "wrc", "packet": "session_update", "ip": "127.0.0.1", "port": port, "frequencyHz": -1, "bEnabled": True})
    backup = config.with_suffix(".json.bak")
    backup.write_text(text, encoding="utf-8")
    config.write_text(json.dumps(doc, indent=4), encoding="utf-8")
    return f"switched on, the old file kept as {backup.name}"


# ---------------------------------------------------------------- DiRT Rally 2.0

#: Float indices in the extradata=3 packet.
DR = {
    "total_time": 0,
    "lap_time": 1,
    "lap_distance": 2,
    "position": 4,
    "speed": 7,
    "velocity": 8,
    "side": 11,
    "forward": 14,
    "suspension_position": 17,
    "suspension_velocity": 21,
    "wheel_patch_speed": 25,
    "throttle": 29,
    "steer": 30,
    "brake": 31,
    "clutch": 32,
    "gear": 33,
    "g_lat": 34,
    "g_lon": 35,
    "lap": 36,
    "rpm10": 37,
    "brake_temp": 51,
    "laps_completed": 59,
    "total_laps": 60,
    "track_length": 61,
    "last_lap_time": 62,
}
#: DiRT orders the wheels rear-left, rear-right, front-left, front-right.
DR_WHEELS = ("RL", "RR", "FL", "FR")


def decode_dirt(packet: bytes) -> list[float] | None:
    if len(packet) < 64 * 4:
        return None
    n = min(66, len(packet) // 4)
    values = list(struct.unpack_from(f"<{n}f", packet, 0))
    return values + [0.0] * (66 - n)


def frame_dirt(f: list[float]) -> dict:
    v = tuple(f[DR["velocity"] : DR["velocity"] + 3])
    forward = tuple(f[DR["forward"] : DR["forward"] + 3])
    side = tuple(f[DR["side"] : DR["side"] + 3])
    frame = {
        "x": f[DR["position"]],
        "y": f[DR["position"] + 1],
        "z": f[DR["position"] + 2],
        "speed": f[DR["speed"]],
        "longVel": dot(v, forward),
        # The side vector's sense is documented both ways; the viewer reads
        # the drift angle's size, and the steering's sense from the path.
        "latVel": dot(v, side),
        "latAcc": f[DR["g_lat"]] * G,
        "longAcc": f[DR["g_lon"]] * G,
        "yaw": math.atan2(forward[0], forward[2]),
        "throttle": f[DR["throttle"]],
        "brake": f[DR["brake"]],
        "clutch": f[DR["clutch"]],
        "steer": f[DR["steer"]],
        "gear": f[DR["gear"]] if f[DR["gear"]] < 9.5 else -1.0,
        "rpm": f[DR["rpm10"]] * 10.0,
        "stageDistance": f[DR["lap_distance"]],
    }
    for k, wheel in enumerate(DR_WHEELS):
        frame[f"suspensionM.{wheel}"] = f[DR["suspension_position"] + k]
        frame[f"wheelSurfaceSpeed.{wheel}"] = f[DR["wheel_patch_speed"] + k]
        frame[f"brakeTemp.{wheel}"] = f[DR["brake_temp"] + k]
    return frame


# ---------------------------------------------------------------- Richard Burns Rally

RBR_LENGTH = 664
RBR_FIELDS = [
    (0, "I", "total_steps"),
    (4, "i", "stage_index"),
    (8, "f", "progress"),
    (12, "f", "race_time"),
    (16, "f", "drive_line"),
    (20, "f", "distance_to_end"),
    (24, "f", "steering"),
    (28, "f", "throttle"),
    (32, "f", "brake"),
    (36, "f", "handbrake"),
    (40, "f", "clutch"),
    (44, "i", "gear"),
    (60, "f", "speed"),
    (64, "3f", "position"),
    (76, "3f", "attitude"),
    (88, "3f", "velocity"),
    (112, "3f", "acceleration"),
    (136, "f", "rpm"),
]
#: LF RF LB RB blocks of 128 bytes from 152; spring deflection at +0, force at +8.
RBR_WHEELS = ("FL", "FR", "RL", "RR")


def decode_rbr(packet: bytes) -> dict | None:
    if len(packet) < RBR_LENGTH:
        return None
    raw: dict = {}
    for offset, fmt, name in RBR_FIELDS:
        values = struct.unpack_from("<" + fmt, packet, offset)
        raw[name] = values[0] if len(values) == 1 else values
    raw["spring_deflection"] = [struct.unpack_from("<f", packet, 152 + 128 * k)[0] for k in range(4)]
    raw["spring_force"] = [struct.unpack_from("<f", packet, 152 + 128 * k + 8)[0] for k in range(4)]
    return raw


def frame_rbr(raw: dict) -> dict:
    surge, sway, heave = raw["velocity"]
    ax, ay, az = raw["acceleration"]
    px, py, pz = raw["position"]
    frame = {
        # Axes as sent; `orient_up` puts the up axis in y once the stage is known.
        "x": px,
        "y": py,
        "z": pz,
        "longVel": surge,
        "latVel": sway,
        "vertVel": heave,
        "longAcc": ax,
        "latAcc": ay,
        "throttle": raw["throttle"],
        "brake": raw["brake"],
        "handbrake": raw["handbrake"],
        "clutch": raw["clutch"],
        "steer": raw["steering"],
        # 0 reverse, 1 neutral, 2 first.
        "gear": raw["gear"] - 1,
        "rpm": raw["rpm"],
        "stageDistance": raw["drive_line"],
    }
    for wheel, deflection, force in zip(RBR_WHEELS, raw["spring_deflection"], raw["spring_force"]):
        frame[f"suspensionM.{wheel}"] = deflection
        frame[f"wheelLoad.{wheel}"] = force
    return frame


def orient_up(frames: list[dict]) -> None:
    """
    Put the axis the run varies least along in `y`: a stage covers kilometres
    across the map and tens of metres in height. RBR does not document its axes.
    """
    if len(frames) < 10:
        return
    ranges = {k: max(f[k] for f in frames) - min(f[k] for f in frames) for k in ("x", "y", "z")}
    up = min(ranges, key=ranges.get)
    if up == "y":
        return
    for f in frames:
        f["y"], f[up] = f[up], f["y"]


def speed_from_positions(frames: list[dict]) -> None:
    """Speed from the positions, for a game that sends it in a unit the settings choose."""
    n = len(frames)
    for i, f in enumerate(frames):
        a, b = max(0, i - 2), min(n - 1, i + 2)
        dt = frames[b]["time"] - frames[a]["time"]
        d = math.dist((frames[a]["x"], frames[a]["y"], frames[a]["z"]), (frames[b]["x"], frames[b]["y"], frames[b]["z"]))
        f["speed"] = d / dt if dt > 0 else 0.0


# ---------------------------------------------------------------- stages

class StageSplitter:
    """
    Frames into stages by the game's stage clock: a stage opens when the clock
    starts, closes complete when the game says the car has finished (distance
    reached the stage's length, or the game's finish flag), and is dropped when
    the clock goes back (a restart) before then.
    """

    def __init__(self, *, minimum_metres: float = 300.0):
        self.minimum = minimum_metres
        self.current: Run | None = None
        self.number = 0
        self.last_clock = -1.0

    def feed(self, frame: dict, clock: float, finished: bool, seconds: float | None = None) -> list[Run]:
        done: list[Run] = []
        if clock <= 0 or (self.current is not None and clock < self.last_clock - 0.5):
            # Back at the start: a restart, or the next stage. A run cut short
            # is kept as a partial one, if it went anywhere.
            done.extend(self._abandon())
            self.last_clock = clock
            if clock <= 0:
                return done
        if self.current is None:
            self.number += 1
            self.current = Run(number=self.number)
        self.last_clock = clock
        if self.current.complete:
            return done
        self.current.frames.append({**frame, "time": clock})
        if finished:
            self.current.complete = True
            self.current.seconds = seconds if seconds and seconds > 0 else clock
            if self.current.metres >= self.minimum:
                done.append(self.current)
        return done

    def _abandon(self) -> list[Run]:
        run, self.current = self.current, None
        if run and not run.complete and len(run.frames) > 10 and run.metres >= self.minimum:
            return [run]
        return []

    def flush(self) -> list[Run]:
        return self._abandon()


GAMES = {
    "wrc": "ea-sports-wrc",
    "dirtrally2": "dirt-rally-2",
    "dirtrally": "dirt-rally",
    "dirt4": "dirt-4",
    "rbr": "richard-burns-rally",
}
DEFAULT_PORTS = {"wrc": 20777, "dirtrally2": 20777, "dirtrally": 20777, "dirt4": 20777, "rbr": 6776}


class RallyRecorder:
    """Packets in, stage files out, for one of the games above."""

    def __init__(self, out: str | Path, *, game: str, surface: str | None = None):
        if game not in GAMES:
            raise ValueError(f"unknown rally game {game!r}; one of {', '.join(GAMES)}")
        self.out = Path(out)
        self.game = game
        self.splitter = StageSplitter()
        self.meta = {"discipline": "rally", "slip": None}
        if surface:
            self.meta["surface"] = surface
        self.written: list[Path] = []
        self.warned: set[str] = set()

    def _warn(self, key: str, text: str) -> None:
        if key not in self.warned:
            self.warned.add(key)
            print(f"note: {text}")

    def feed(self, packet: bytes) -> list[Path]:
        if self.game == "wrc":
            raw = decode_wrc(packet)
            if raw is None:
                self._warn("size", f"a {len(packet)}-byte packet: expected the {WRC_LENGTH}-byte default `wrc` structure")
                return []
            frame = frame_wrc(raw)
            length = raw["stage_length"]
            finished = length > 0 and raw["stage_distance"] >= length - 0.5
            runs = self.splitter.feed(frame, raw["stage_time"], finished)
            # Sanity: the world velocity and the speed are the same quantity.
            v = math.sqrt(dot(raw["velocity"], raw["velocity"]))
            if raw["speed"] > 10 and abs(v - raw["speed"]) > 0.2 * raw["speed"]:
                self._warn("wrc-speed", "velocity and speed disagree - the packet layout may not be the default `wrc` structure")
        elif self.game in ("dirtrally2", "dirtrally", "dirt4"):
            f = decode_dirt(packet)
            if f is None:
                return []
            frame = frame_dirt(f)
            length = f[DR["track_length"]]
            laps = f[DR["laps_completed"]]
            if self.splitter.current is None or not self.splitter.current.frames:
                self.laps_at_start = laps
            # A stage finishes as its lap count goes up; a rallycross lap, each time it does.
            finished = laps > getattr(self, "laps_at_start", laps) or (
                length > 0 and f[DR["total_laps"]] <= 1 and f[DR["lap_distance"]] >= 0.999 * length
            )
            last = f[DR["last_lap_time"]]
            runs = self.splitter.feed(frame, f[DR["lap_time"]], finished, last if last > 0 else None)
            v = math.sqrt(sum(c * c for c in f[DR["velocity"] : DR["velocity"] + 3]))
            if f[DR["speed"]] > 10 and abs(v - f[DR["speed"]]) > 0.2 * f[DR["speed"]]:
                self._warn("dirt-speed", "velocity and speed disagree - is extradata set to 3?")
        else:
            raw = decode_rbr(packet)
            if raw is None:
                return []
            frame = frame_rbr(raw)
            finished = raw["race_time"] > 0 and raw["distance_to_end"] <= 0
            runs = self.splitter.feed(frame, raw["race_time"], finished)
        out = []
        for run in runs:
            out.append(self._write(run))
        return out

    def _write(self, run: Run) -> Path:
        if self.game == "rbr":
            orient_up(run.frames)
            speed_from_positions(run.frames)
        path = write(run, self.out, game=GAMES[self.game], meta={k: v for k, v in self.meta.items() if v})
        self.written.append(path)
        return path

    def close(self) -> list[Path]:
        return [self._write(run) for run in self.splitter.flush()]


def run(out: str | Path, *, game: str, port: int | None = None, surface: str | None = None, report=print) -> list[Path]:
    """Listen until Ctrl+C; returns the files written."""
    port = port or DEFAULT_PORTS[game]
    recorder = RallyRecorder(out, game=game, surface=surface)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(0.5)
    report(f"listening for {GAMES[game]} on port {port}. Stages go to {out}. Ctrl+C to stop.")
    try:
        while True:
            try:
                packet, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue
            for path in recorder.feed(packet):
                report(f"stage written: {path.name}")
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        for path in recorder.close():
            report(f"partial stage written: {path.name}")
    return recorder.written
