"""
Racing games' own telemetry: Forza, the Assetto Corsa games, the rally games
and BeamNG, recorded as routes, laps and stages.

Every game here publishes its telemetry on purpose, for dashboards and motion
rigs; nothing is injected and nothing the game does not hand out is read. The
decoding is not written twice: it is gamecapture's lap recorders in
`gamecapture/heat3d_capture/telemetry` (tested there against packets built from
the published layouts), and this module turns their frames into the watcher's
rows - the car's position a few times a second with its speed, gear and pedals,
an event per lap or stage, and the numbers a driver keeps: laps, best lap,
distance, top speed.

Forza (Horizon 4/5/6, Motorsport)
    "Data Out" UDP, switched on in the game's HUD and Gameplay settings, to
    127.0.0.1 port 5300, format Dash. One packet per simulation tick; the dash
    block's offset is found from the packets (`forza.detect_layout`). World
    metres, Y up; the world is left-handed and is drawn as sent, as the lap recorder
    does (Forza Motorsport's and Forza Horizon's "Data Out" documentation).
    The game sends to one port, and FH Companion and other Data Out apps listen
    on the same one. Two programs cannot share it: on Windows the socket bound
    to 127.0.0.1 gets the packets and one bound to all addresses on the same
    port gets none, whichever came first. So the watcher waits a little after
    the game starts before it takes the port, never takes it from an app that
    holds it, and lets go of it as soon as another app starts listening there.
    While it holds the port it passes every packet on unchanged to
    127.0.0.1:5301 - set FH Companion's port box (its telemetry tab) to 5301 and
    both record.

Assetto Corsa, ACC, AC EVO, AC Rally
    Shared memory the game writes for apps: `Local\\acpmf_physics`,
    `Local\\acpmf_graphics`, `Local\\acpmf_static` (AC EVO: `Local\\acevo_pmf_*`).
    Opened read-only and only once the game has made it (`simracing_shm`). The
    position is the mean of the four tyre contact points, world metres, Y up.
    Laps by the game's counter (AC, ACC) or by passing the starting point again
    (AC EVO, AC Rally), as `kunos.py` does for the lap recorder.

EA SPORTS WRC, DiRT Rally 2.0, Richard Burns Rally
    UDP: WRC's `Documents/My Games/WRC/telemetry/config.json` (the `wrc`
    structure's `session_update` packet), DiRT's `hardware_settings_config.xml`
    (`<udp enabled="true" extradata="3" port="20777">`), RBR's NGP plugin
    (`udpTelemetry=1`, port 6776). Stages by the game's stage clock
    (`rally.StageSplitter`). RBR does not document its axes: the up axis is the
    one the stage varies least along, decided when the session closes.

BeamNG.drive
    The `heat3d` protocol mod gamecapture ships (UDP 4460, Options > Other >
    Protocols > "others"); BeamNG's Z-up world turned to Y up as `beamng.py` does.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import socket
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..session import Axes, Session
from .base import Adapter, Context

log = logging.getLogger("heat3d_watch")

try:  # The recorders live beside this package; __main__ and the tests put them on the path.
    from heat3d_capture.telemetry import beamng, forza, fulllap, kunos, rally
except ImportError as e:  # pragma: no cover - only without the repository around it
    beamng = forza = fulllap = kunos = rally = None  # type: ignore[assignment]
    _IMPORT_ERROR: str | None = f"gamecapture's lap recorders are not importable: {e}"
else:
    _IMPORT_ERROR = None

KMH = 3.6


# ------------------------------------------------------------------ where things are


def documents() -> Path:
    """The user's Documents folder, where games keep `My Games`; OneDrive's when it holds them."""
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    for candidate in (home / "Documents", home / "OneDrive" / "Documents", home / "OneDrive" / "Dokumente"):
        if (candidate / "My Games").is_dir():
            return candidate
    return home / "Documents"


def backup_once(path: Path) -> Path:
    """`<file>.heat3d-backup`, made the first time a file is changed and never overwritten."""
    backup = path.with_name(path.name + ".heat3d-backup")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def set_xml_attributes(text: str, tag: str, wanted: dict[str, str]) -> tuple[str, list[str]]:
    """
    Set attributes on the first `<tag .../>` in a game's settings XML, touching
    nothing else in the file. Returns the new text and what changed.
    """
    found = re.search(rf"<{tag}\b[^>]*>", text)
    if not found:
        return text, []
    element = found.group(0)
    changes = []
    for name, value in wanted.items():
        attr = re.search(rf'\b{name}\s*=\s*["”“]([^"”“]*)["”“]', element)
        if attr and attr.group(1) == value:
            continue
        if attr:
            element = element[: attr.start()] + f'{name}="{value}"' + element[attr.end() :]
            changes.append(f'{name}="{attr.group(1)}" -> "{value}"')
        else:
            close = element.rstrip(">").rstrip("/").rstrip()
            tail = element[len(close) :]
            element = f'{close} {name}="{value}"{tail}'
            changes.append(f'{name}="{value}" added')
    return text[: found.start()] + element + text[found.end() :], changes


# ------------------------------------------------------------------ UDP


def port_holder(port: int) -> str | None:
    """Which other program listens on this UDP port, if one does: its name and pid."""
    try:
        import psutil

        for c in psutil.net_connections(kind="udp"):
            if c.laddr and c.laddr.port == port and c.pid and c.pid != os.getpid():
                try:
                    name = psutil.Process(c.pid).name()
                except Exception:  # noqa: BLE001
                    name = "another program"
                return f"{name} (pid {c.pid})"
    except Exception:  # noqa: BLE001 - not knowing is not a reason to stop
        return None
    return None


class UdpPort:
    """
    A game's telemetry port on localhost, taken only when it is free, with every
    packet optionally passed on unchanged to another port.

    No SO_REUSEADDR on Windows: there it lets a second socket take over a port
    another program holds, and the packets would stop reaching that program.
    And because a socket bound to 127.0.0.1 gets loopback packets ahead of a
    program that binds the same port on all addresses later - which binds
    without an error and then hears nothing - the port is checked every few
    seconds while held, and given up the moment another program listens on it.
    """

    def __init__(
        self,
        port: int,
        *,
        host: str = "127.0.0.1",
        forward: tuple[str, int] | None = None,
        grace: float = 0.0,
        recheck: float = 3.0,
    ):
        self.port = port
        self.host = host
        self.forward = forward
        self.recheck = recheck
        self._sock: socket.socket | None = None
        self._out: socket.socket | None = None
        self._tried = 0.0
        self._checked = 0.0
        self._not_before = time.monotonic() + grace
        self._said: str | None = None
        self.relayed = 0
        self.yielded = 0

    @property
    def ready(self) -> bool:
        return self._sock is not None

    def open(self) -> str | None:
        """Take the port. Returns why it could not be taken, or None."""
        holder = port_holder(self.port)
        if holder:
            return f"UDP {self.port} is in use by {holder}; not taking it from that program"
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            if sys.platform != "win32":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            except OSError:
                pass
            sock.bind((self.host, self.port))
            sock.setblocking(False)
        except OSError as e:
            sock.close()
            return f"UDP {self.port} could not be opened: {e}"
        self._sock = sock
        if self.forward:
            self._out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return None

    def try_open(self, who: str, every: float = 5.0) -> bool:
        """Open the port if it is time to try again; say why not, once per reason. Give it up to a newcomer."""
        now = time.monotonic()
        if self.ready:
            if self.recheck and now - self._checked >= self.recheck:
                self._checked = now
                holder = port_holder(self.port)
                if holder:
                    self.close()
                    self.yielded += 1
                    self._tried = now
                    reason = f"{holder} started listening on UDP {self.port}; leaving the port to it"
                    log.info("%s: %s", who, reason)
                    self._said = reason
                    return False
            return True
        if now < self._not_before or now - self._tried < every:
            return False
        self._tried = now
        self._checked = now
        reason = self.open()
        if reason and reason != self._said:
            log.info("%s: %s", who, reason)
        self._said = reason
        return reason is None

    def receive(self, limit: int = 4000) -> list[bytes]:
        """Everything waiting, without blocking."""
        out: list[bytes] = []
        if self._sock is None:
            return out
        while len(out) < limit:
            try:
                packet, _ = self._sock.recvfrom(65535)
            except (BlockingIOError, socket.timeout):
                break
            except ConnectionResetError:
                continue
            except OSError:
                break
            out.append(packet)
            if self._out is not None and self.forward is not None:
                try:
                    self._out.sendto(packet, self.forward)
                    self.relayed += 1
                except OSError:
                    pass
        return out

    def close(self) -> None:
        for s in (self._sock, self._out):
            if s is not None:
                s.close()
        self._sock = self._out = None


# ------------------------------------------------------------------ driving


@dataclass
class _Mover:
    x: float
    y: float
    z: float
    t: float
    emitted: float = -1e9
    metres: float = 0.0
    speed: float = 0.0
    clock: float | None = None


class Driving:
    """
    What a racing game says in one sitting, as the watcher's rows: positions a
    few times a second, events with the car's position, and the statistics.

    The session opens with the first thing worth recording, so a game left in
    its menus leaves no folder behind. A jump further than the car could have
    driven - a rewind, a reset to the road, a fast travel - is a `teleport`
    row, so the viewer's line heat does not draw a line across the map.
    """

    def __init__(self, adapter: Adapter, ctx: Context, *, hz: float = 10.0, jump: float = 50.0):
        self.adapter = adapter
        self.ctx = ctx
        self.period = 1.0 / hz
        self.jump = jump
        self.session: Session | None = None
        self.movers: dict[str, _Mover] = {}
        self.laps = 0
        #: The player's own driving, over every car and stint of the sitting.
        self.metres = 0.0
        self.pending: dict[str, float] = {}
        self.ranges: dict[str, list[float]] = {}

    def open(self, **meta: Any) -> Session:
        if self.session is None:
            self.session = self.ctx.open(self.adapter)
            for key, value in self.pending.items():
                self.session.stat(key, value, "add")
            self.pending.clear()
        if meta:
            self.session.meta.update({k: v for k, v in meta.items() if v not in (None, "")})
        return self.session

    def count(self, key: str, n: float = 1) -> None:
        """A tally that is kept even before the session opens (bad packets in a menu)."""
        if self.session is None:
            self.pending[key] = self.pending.get(key, 0) + n
        else:
            self.session.stat(key, n, "add")

    def sample(
        self,
        x: float,
        y: float,
        z: float,
        *,
        player: str = "me",
        key: str | None = None,
        own: bool = True,
        speed: float | None = None,
        force: bool = False,
        clock: float | None = None,
        team: str | None = None,
        session: str | None = None,
        **attrs: Any,
    ) -> bool:
        """
        One position from the game. Every sample counts for the distance; one
        every `1/hz` seconds (or `force`) becomes a row. `speed` is m/s, from the
        game (packets arrive in batches, so the watcher's own clock cannot time
        them) - and `clock`, the game's own time in seconds, is what a jump is
        timed by when the game gives one. `key` tells cars apart when
        their names can change (the names arrive later); the player by default.
        Returns whether a row was written.
        """
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, y, z)):
            return False
        s = self.open()
        now = s.now()
        key = key or player
        m = self.movers.get(key)
        if m is None:
            m = self.movers[key] = _Mover(x, y, z, now)
        else:
            # Capped: a car that sat still through a pause has not been driving for that long.
            if clock is not None and m.clock is not None:
                dt = min(1.0, max(0.0, clock - m.clock))
            else:
                dt = min(1.0, max(0.0, now - m.t))
            d = math.dist((m.x, m.y, m.z), (x, y, z))
            fastest = max(speed or 0.0, m.speed, 100.0)
            if d > self.jump + fastest * dt * 1.5:
                s.event("teleport", x=x, y=y, z=z, player=player, team=team, session=session,
                        from_x=round(m.x, 3), from_y=round(m.y, 3), from_z=round(m.z, 3), metres=round(d, 1))
                m.emitted = now
            else:
                m.metres += d
                if own:
                    self.metres += d
            m.x, m.y, m.z, m.t = x, y, z, now
        m.clock = clock
        if speed is not None and math.isfinite(speed):
            m.speed = speed
        if own:
            s.stat("distance_km", round(self.metres / 1000.0, 3))
            if speed is not None and math.isfinite(speed):
                s.stat("top_speed_kmh", round(speed * KMH, 1), "max")
            for axis, v in (("x", x), ("y", y), ("z", z)):
                r = self.ranges.setdefault(axis, [v, v])
                r[0], r[1] = min(r[0], v), max(r[1], v)
        if not force and now - m.emitted < self.period:
            return False
        m.emitted = now
        if speed is not None and math.isfinite(speed):
            attrs["speed_kmh"] = round(speed * KMH, 1)
        for k, v in list(attrs.items()):
            if isinstance(v, float):
                attrs[k] = round(v, 3)
        s.position(x, y, z, player=player, team=team, session=session, **attrs)
        return True

    def event(self, name: str, *, player: str | None = "me", key: str | None = None, where: bool = True, **attrs: Any) -> None:
        """An event, at the player's last known position unless told otherwise."""
        s = self.open()
        m = self.movers.get(key or player) if (key or player) else None
        if where and m is not None and "x" not in attrs:
            attrs.update(x=m.x, y=m.y, z=m.z)
        s.event(name, player=player, **attrs)

    def lap(self, seconds: float | None, *, player: str = "me", key: str | None = None, own: bool = True, **attrs: Any) -> None:
        """A lap closed: an event, and for the player's own car the count and the best."""
        good = seconds is not None and math.isfinite(seconds) and seconds > 0
        self.event("lap", player=player, key=key, lap_time=round(seconds, 3) if good else None, **attrs)
        if own:
            self.laps += 1
            s = self.open()
            s.stat("laps", self.laps)
            if good:
                s.stat("best_lap", round(seconds, 3), "min")

    def close(self) -> None:
        self.ctx.close(self.session)
        self.session = None
        self.movers.clear()
        self.laps = 0
        self.metres = 0.0
        self.ranges.clear()


# ------------------------------------------------------------------ Forza


def _forza_frame(packet: bytes, layout: Any) -> dict | None:
    if len(packet) != layout.packet_length:
        return None
    frame = fulllap.decode_full(packet, layout)
    if not all(math.isfinite(frame[k]) for k in ("x", "y", "z", "speed")):
        return None
    return frame


@dataclass
class ForzaAdapter(Adapter):
    """Forza's Data Out: the car's position at 10 Hz, laps, races; relayed on to FH Companion."""

    port: int = 5300
    #: Where every packet is passed on, unchanged: FH Companion or any other Data Out app.
    forward: tuple[str, int] | None = ("127.0.0.1", 5301)
    hz: float = 10.0
    #: Seconds without a packet after which the drive counts as stopped (game paused, closed).
    silence: float = 3.0
    #: Seconds after the game starts before the port is taken, so an app that starts with the game binds first.
    grace: float = 15.0

    def missing(self) -> str | None:
        return _IMPORT_ERROR

    def start(self, ctx: Context) -> None:
        self._udp = UdpPort(self.port, forward=self.forward, grace=self.grace)
        self._drive = Driving(self, ctx, hz=self.hz)
        self._layout = None
        self._pending: list[bytes] = []
        self._on = False
        self._lap: int | None = None
        self._race_position = 0
        self._stint = 0
        self._heard = time.monotonic()

    def poll(self, ctx: Context) -> None:
        if not self._udp.try_open(self.id):
            return
        packets = self._udp.receive()
        if packets:
            self._heard = time.monotonic()
        elif self._on and time.monotonic() - self._heard > self.silence:
            self._off()
        for packet in packets:
            try:
                self.feed(packet)
            except Exception:  # noqa: BLE001 - one bad packet must not end the recording
                self._drive.count("bad_packets")

    def feed(self, packet: bytes) -> None:
        """One packet. Separate from `poll` so tests can drive it without a socket."""
        d = self._drive
        if len(packet) < forza.SLED_LENGTH + forza.DASH_BLOCK_LENGTH:
            if len(packet) == forza.SLED_LENGTH:
                d.count("sled_packets")
            else:
                d.count("bad_packets")
            return
        if self._layout is None:
            self._pending.append(packet)
            if len(self._pending) > 600:
                self._pending = self._pending[-300:]
            try:
                self._layout = forza.detect_layout(self._pending)
            except forza.LayoutUndetermined:
                return
            self._pending.clear()
        frame = _forza_frame(packet, self._layout)
        if frame is None:
            d.count("bad_packets")
            return
        if not frame["isRaceOn"]:
            self._off()
            return
        x, y, z = frame["x"], frame["y"], frame["z"]
        if not self._on:
            self._on = True
            self._stint += 1
            self._lap = frame["lapNumber"]
            d.open(layout=self._layout.name)
            d.sample(x, y, z, speed=frame["speed"], force=True, session=str(self._stint), clock=frame["timestampMs"] / 1000.0)
            d.event("session_start", session=str(self._stint), car_ordinal=frame["carOrdinal"],
                    car_class=frame["carClass"], car_pi=frame["carPI"], drivetrain=frame["drivetrain"])
            return
        d.sample(
            x, y, z,
            speed=frame["speed"],
            clock=frame["timestampMs"] / 1000.0,
            session=str(self._stint),
            gear=frame["gear"],
            throttle=frame["throttle"],
            brake=frame["brake"],
            lap=frame["lapNumber"],
        )
        position = frame["racePosition"]
        if frame["lapNumber"] != self._lap:
            if self._lap is not None and frame["lapNumber"] == self._lap + 1:
                d.lap(frame["lastLap"], lap=self._lap, session=str(self._stint), race_position=position or None)
            self._lap = frame["lapNumber"]
        if position and not self._race_position:
            d.event("race_start", session=str(self._stint), race_position=position)
        elif self._race_position and not position:
            # The game shows no position any more: the race is over (or was left).
            d.event("finish", session=str(self._stint), race_position=self._race_position)
            d.open().stat("races", 1, "add")
        self._race_position = position

    def _off(self) -> None:
        if self._on:
            self._on = False
            self._drive.event("session_end", session=str(self._stint))

    def stop(self, ctx: Context) -> None:
        self._off()
        if self._udp.relayed and self._drive.session is not None:
            self._drive.session.meta["relayed_to"] = f"{self.forward[0]}:{self.forward[1]}" if self.forward else None
        self._udp.close()
        self._drive.close()


_FORZA_COMMON = dict(
    kind="telemetry",
    multiplayer=True,
    source="Forza Data Out, UDP 127.0.0.1:5300 (Dash format); every packet relayed unchanged to 127.0.0.1:5301",
    records=(
        "own car's position (world XYZ, m) at 10 Hz with speed, gear, throttle, brake, lap",
        "drive start and end (session_start/session_end), laps with the game's lap time",
        "race start and finish with the race position",
        "teleports (rewind, fast travel) so routes break there",
        "laps, best lap, distance, top speed, races",
    ),
    enable=(
        "In the game: Settings > HUD and Gameplay > Data Out On, IP 127.0.0.1, port 5300 (Motorsport: format Dash). "
        "FH Companion or another Data Out app: set its port to 5301 - the watcher passes every packet on; "
        "while another app listens on 5300 the watcher leaves the port to it and records nothing"
    ),
    axes=Axes(up="y", units="metres; Forza's world is left-handed, drawn as sent like the lap recorder does"),
    verified=(
        "decoding is gamecapture's lap recorder's (tested on packets built from the Data Out layout); the adapter tested on "
        "synthetic packets and on real FH Companion lap trajectories replayed as packets; untested against the game itself"
    ),
)

FORZA = [
    ForzaAdapter(id="forza-horizon-6", game="Forza Horizon 6", exes=("forzahorizon6.exe",), **_FORZA_COMMON),
    ForzaAdapter(id="forza-horizon-5", game="Forza Horizon 5", exes=("ForzaHorizon5.exe",), **_FORZA_COMMON),
    ForzaAdapter(id="forza-horizon-4", game="Forza Horizon 4", exes=("ForzaHorizon4.exe",), **_FORZA_COMMON),
    ForzaAdapter(
        id="forza-motorsport",
        game="Forza Motorsport",
        exes=("forza_steamworks_release_final.exe", "forza_gaming.desktop.x64_release_final.exe", "ForzaMotorsport7.exe"),
        **_FORZA_COMMON,
    ),
]


# ------------------------------------------------------------------ Assetto Corsa games

#: Graphics page (AC, ACC): status and session at the start, the lap counter and times.
G_STATUS = 4  # 0 off, 1 replay, 2 live, 3 pause
G_SESSION = 8
G_POSITION = 136
G_BEST_LAP_MS = 148
AC_LIVE = 2
STATIC_SIZE = 256


@dataclass
class KunosAdapter(Adapter):
    """One of the Assetto Corsa games' shared memory: positions, laps, the track and car it names."""

    #: `kunos.MAPPINGS` key: "ac", "acc", "acevo", "acrally".
    flavour: str = "ac"
    hz: float = 10.0
    #: Override of the mapping names, for tests.
    names: tuple[str, str | None, str] | None = None

    def missing(self) -> str | None:
        if _IMPORT_ERROR:
            return _IMPORT_ERROR
        if sys.platform != "win32":
            return "shared memory telemetry is Windows only"
        return None

    def start(self, ctx: Context) -> None:
        from .simracing_shm import SharedView

        self._view = SharedView
        self._drive = Driving(self, ctx, hz=self.hz)
        self._physics = self._graphics = self._static = None
        self._tried = 0.0
        self._last_packet: int | None = None
        self._clock0 = time.perf_counter()
        self._splitter = kunos.CounterSplitter() if self._has_graphics else kunos.LineSplitter()
        self._live = False
        self._run = 0
        self._still_since: float | None = None

    @property
    def _has_graphics(self) -> bool:
        return (self.names or kunos.MAPPINGS[self.flavour])[1] is not None

    def _open(self) -> bool:
        if self._physics is not None:
            return True
        now = time.monotonic()
        if now - self._tried < 2.0:
            return False
        self._tried = now
        physics, graphics, static = self.names or kunos.MAPPINGS[self.flavour]
        views = [self._view.open(name) if name else None for name in (physics, graphics, static)]
        if any(v is None for v, name in zip(views, (physics, graphics, static)) if name):
            # The game makes its pages one by one: wait until all of them are there.
            for v in views:
                if v is not None:
                    v.close()
            return False
        self._physics, self._graphics, self._static = views
        return True

    def poll(self, ctx: Context) -> None:
        if not self._open():
            return
        try:
            self._step()
        except Exception:  # noqa: BLE001
            self._drive.count("bad_reads")

    def _step(self) -> None:
        size = kunos.PHYSICS_SIZE if self._physics.size >= kunos.PHYSICS_SIZE else kunos.PHYSICS_AC1_SIZE
        page = self._physics.read(0, size)
        if len(page) < kunos.PHYSICS_AC1_SIZE:
            return
        packet = struct.unpack_from("<i", page, 0)[0]
        g = self._graphics.read(0, kunos.GRAPHICS_SIZE) if self._graphics is not None else b""
        status = struct.unpack_from("<i", g, G_STATUS)[0] if len(g) >= kunos.GRAPHICS_SIZE else AC_LIVE
        if packet == self._last_packet or packet == 0:
            return
        self._last_packet = packet
        if status != AC_LIVE:
            # A replay, a pause, the menus: not driving.
            self._end_live()
            return
        clock = time.perf_counter() - self._clock0
        frame = kunos.frame_from(kunos.decode_physics(page), clock, self.flavour)
        if frame is None:
            self._end_live()
            return
        d = self._drive
        x, y, z = frame["x"], frame["y"], frame["z"]
        if not self._live:
            self._live = True
            self._run += 1
            track, car = kunos.read_static(self._static.read(0, STATIC_SIZE), self.flavour) if self._static else (None, None)
            d.open(track=track, car=car)
            d.sample(x, y, z, speed=frame["speed"], force=True, session=str(self._run), clock=clock)
            session_type = struct.unpack_from("<i", g, G_SESSION)[0] if g else None
            d.event("session_start", session=str(self._run), track=track, car=car, session_type=session_type)
        d.sample(
            x, y, z,
            speed=frame["speed"],
            clock=clock,
            session=str(self._run),
            gear=frame["gear"],
            throttle=frame["throttle"],
            brake=frame["brake"],
        )
        slim = {"time": clock, "x": x, "y": y, "z": z, "speed": frame["speed"]}
        if isinstance(self._splitter, kunos.CounterSplitter):
            laps = struct.unpack_from("<i", g, kunos.G_COMPLETED_LAPS)[0]
            last_ms = struct.unpack_from("<i", g, kunos.G_LAST_LAP_MS)[0]
            done = self._splitter.feed(slim, laps, last_ms)
            position = struct.unpack_from("<i", g, G_POSITION)[0]
            if position > 0:
                d.open().stat("race_position", position)
        else:
            done = self._splitter.feed(slim)
        for lap in done:
            self._closed(lap)

    def _closed(self, lap: Any, *, stopped: bool = True) -> None:
        """A lap or run the splitter closed; `stopped`: by the car standing still, not by the drive ending."""
        d = self._drive
        seconds = lap.seconds if lap.seconds is not None else lap.frames[-1]["time"] - lap.frames[0]["time"]
        if lap.complete:
            d.lap(seconds, lap=lap.number, session=str(self._run), metres=round(lap.metres, 1))
        elif self.flavour == "acrally" and stopped:
            # A stage ends where the car stops after it.
            d.event("finish", session=str(self._run), stage_time=round(seconds, 3), metres=round(lap.metres, 1))
            s = d.open()
            s.stat("stages", 1, "add")
            s.stat("best_stage", round(seconds, 3), "min")
        else:
            d.event("run_end", session=str(self._run), seconds=round(seconds, 3), metres=round(lap.metres, 1))

    def _end_live(self) -> None:
        if self._live:
            self._live = False
            for lap in self._splitter.flush():
                self._closed(lap, stopped=False)
            self._drive.event("session_end", session=str(self._run))

    def stop(self, ctx: Context) -> None:
        self._end_live()
        for v in (self._physics, self._graphics, self._static):
            if v is not None:
                v.close()
        self._physics = self._graphics = self._static = None
        self._drive.close()


def _kunos_records(laps: str) -> tuple[str, ...]:
    return (
        "own car's position (world XYZ, m: mean of the tyre contact points) at 10 Hz with speed, gear, throttle, brake",
        "session start and end with the track and car the game names",
        laps,
        "teleports (back to the pits, a restart) so routes break there",
        "laps, best lap, distance, top speed",
    )


_KUNOS_AXES = Axes(up="y", units="metres")
_KUNOS_VERIFIED = (
    "decoding is gamecapture's lap recorder's (tested on synthetic pages); the adapter tested on shared memory built in the "
    "test from the documented layout; untested against the game"
)

KUNOS = [
    KunosAdapter(
        id="assetto-corsa",
        game="Assetto Corsa",
        exes=("acs.exe", "acs_x86.exe"),
        flavour="ac",
        source="shared memory Local\\acpmf_physics, Local\\acpmf_graphics, Local\\acpmf_static",
        records=_kunos_records("laps by the game's lap counter, with its lap time"),
        enable="",
        axes=_KUNOS_AXES,
        verified=_KUNOS_VERIFIED,
    ),
    KunosAdapter(
        id="assetto-corsa-competizione",
        game="Assetto Corsa Competizione",
        exes=("AC2-Win64-Shipping.exe",),
        flavour="acc",
        source="shared memory Local\\acpmf_physics, Local\\acpmf_graphics, Local\\acpmf_static",
        records=_kunos_records("laps by the game's lap counter, with its lap time; race position"),
        enable="",
        axes=_KUNOS_AXES,
        verified=_KUNOS_VERIFIED,
    ),
    KunosAdapter(
        id="assetto-corsa-evo",
        game="Assetto Corsa EVO",
        exes=("AssettoCorsaEVO.exe",),
        flavour="acevo",
        source="shared memory Local\\acevo_pmf_physics, Local\\acevo_pmf_static",
        records=_kunos_records("laps when the car passes where the drive began, heading the same way"),
        enable="",
        axes=_KUNOS_AXES,
        verified=_KUNOS_VERIFIED + "; the executable name is from player reports",
    ),
    KunosAdapter(
        id="assetto-corsa-rally",
        game="Assetto Corsa Rally",
        # Unreal 5; the executable's name is not published - these are the likely ones.
        exes=("acr.exe", "ACR-Win64-Shipping.exe", "AssettoCorsaRally.exe", "acrally-Win64-Shipping.exe"),
        flavour="acrally",
        source="shared memory Local\\acpmf_physics, Local\\acpmf_static (the same pages as AC and ACC)",
        records=_kunos_records("stages: one ends where the car stops after it, with the time it took"),
        enable="",
        axes=_KUNOS_AXES,
        verified=_KUNOS_VERIFIED + "; the executable name is unconfirmed",
    ),
]


# ------------------------------------------------------------------ rally games


@dataclass
class RallyAdapter(Adapter):
    """EA WRC, DiRT Rally 2.0 or RBR over UDP: positions, stages by the game's clock."""

    #: `rally.GAMES` key: "wrc", "dirtrally2", "rbr".
    flavour: str = "wrc"
    port: int = 20777
    forward: tuple[str, int] | None = None
    hz: float = 10.0

    def missing(self) -> str | None:
        return _IMPORT_ERROR

    def start(self, ctx: Context) -> None:
        self._udp = UdpPort(self.port, forward=self.forward)
        self._drive = Driving(self, ctx, hz=self.hz)
        self._splitter = rally.StageSplitter()
        self._stage = 0
        self._laps_at_start: float | None = None
        self._last: tuple[float, float, float, float] | None = None

    def poll(self, ctx: Context) -> None:
        if not self._udp.try_open(self.id):
            return
        for packet in self._udp.receive():
            try:
                self.feed(packet)
            except Exception:  # noqa: BLE001
                self._drive.count("bad_packets")

    def _decode(self, packet: bytes) -> tuple[dict, float, bool, float | None, float | None] | None:
        """(frame, stage clock, finished, the game's time for it, speed m/s) - as `rally.RallyRecorder` reads them."""
        if self.flavour == "wrc":
            raw = rally.decode_wrc(packet)
            if raw is None:
                return None
            frame = rally.frame_wrc(raw)
            length = raw["stage_length"]
            finished = length > 0 and raw["stage_distance"] >= length - 0.5
            return frame, raw["stage_time"], finished, None, raw["speed"]
        if self.flavour == "dirtrally2":
            f = rally.decode_dirt(packet)
            if f is None:
                return None
            frame = rally.frame_dirt(f)
            DR = rally.DR
            length = f[DR["track_length"]]
            laps = f[DR["laps_completed"]]
            if self._splitter.current is None or not self._splitter.current.frames:
                self._laps_at_start = laps
            start = self._laps_at_start if self._laps_at_start is not None else laps
            finished = laps > start or (length > 0 and f[DR["total_laps"]] <= 1 and f[DR["lap_distance"]] >= 0.999 * length)
            last = f[DR["last_lap_time"]]
            return frame, f[DR["lap_time"]], finished, (last if last > 0 else None), f[DR["speed"]]
        raw = rally.decode_rbr(packet)
        if raw is None:
            return None
        frame = rally.frame_rbr(raw)
        finished = raw["race_time"] > 0 and raw["distance_to_end"] <= 0
        # RBR's speed unit is not documented: taken from the positions instead.
        return frame, raw["race_time"], finished, None, None

    def feed(self, packet: bytes) -> None:
        d = self._drive
        decoded = self._decode(packet)
        if decoded is None:
            d.count("bad_packets")
            return
        frame, clock, finished, seconds, speed = decoded
        x, y, z = frame["x"], frame["y"], frame["z"]
        if not all(math.isfinite(v) for v in (x, y, z, clock)):
            d.count("bad_packets")
            return
        if speed is None:
            # Over the game's own stage clock, from one packet to the next.
            last, self._last = self._last, (x, y, z, clock)
            if last is not None and 0 < clock - last[3] < 1.0:
                speed = math.dist(last[:3], (x, y, z)) / (clock - last[3])
        before = self._splitter.number
        current = self._splitter.current
        runs = self._splitter.feed({"x": x, "y": y, "z": z}, clock, finished, seconds)
        for run in runs:
            self._closed(run)
        if current is not None and current is not self._splitter.current and not current.complete and current not in runs:
            # The clock went back before the finish, too early for the run to be kept.
            d.event("restart", session=str(current.number))
        if self._splitter.current is None:
            return
        if self._splitter.number != before:
            self._stage = self._splitter.number
            d.open()
            d.sample(x, y, z, speed=speed, force=True, session=str(self._stage), clock=clock)
            d.event("session_start", session=str(self._stage))
        if self._splitter.current.complete:
            return
        d.sample(
            x, y, z,
            speed=speed,
            clock=clock,
            session=str(self._stage),
            gear=frame.get("gear"),
            throttle=frame.get("throttle"),
            brake=frame.get("brake"),
            stage_distance=frame.get("stageDistance"),
        )

    def _closed(self, run: Any) -> None:
        d = self._drive
        s = d.open()
        if run.complete:
            seconds = run.seconds
            d.event("finish", session=str(run.number), stage_time=round(seconds, 3) if seconds else None,
                    metres=round(run.metres, 1))
            s.stat("stages", 1, "add")
            if seconds:
                s.stat("best_stage", round(seconds, 3), "min")
        else:
            d.event("restart", session=str(run.number), metres=round(run.metres, 1))

    def stop(self, ctx: Context) -> None:
        for run in self._splitter.flush():
            self._closed(run)
        s = self._drive.session
        if s is not None and self.flavour == "rbr" and self._drive.ranges:
            # RBR does not say which way is up: the axis the drive varied least along.
            spans = {k: hi - lo for k, (lo, hi) in self._drive.ranges.items()}
            if spans.get("z", 1e9) < spans.get("y", 1e9) and spans["z"] < spans.get("x", 1e9):
                s.axes = Axes(up="z", units=s.axes.units)
            s.meta["up_from"] = "the axis the drive varied least along"
        self._udp.close()
        self._drive.close()

    # -- switching the game's output on ---------------------------------------------

    def config_file(self) -> Path | None:
        docs = documents() / "My Games"
        if self.flavour == "wrc":
            return docs / "WRC" / "telemetry" / "config.json"
        if self.flavour == "dirtrally2":
            return docs / "DiRT Rally 2.0" / "hardwaresettings" / "hardware_settings_config.xml"
        return None

    def setup(self, ctx: Context, *, apply: bool) -> list[str]:
        path = self.config_file()
        if path is None or not path.exists():
            return [f"{path} not found - start the game once first"] if path else []
        text = path.read_text(encoding="utf-8")
        if self.flavour == "wrc":
            new, changes = _wrc_config(text, self.port)
        else:
            new, changes = set_xml_attributes(text, "udp", {"enabled": "true", "extradata": "3", "ip": "127.0.0.1", "port": str(self.port)})
            changes = [f"{path.name} <udp>: {c}" for c in changes]
        if changes and apply:
            backup = backup_once(path)
            path.write_text(new, encoding="utf-8")
            changes.append(f"the original is kept as {backup.name}")
        return changes


def _wrc_config(text: str, port: int) -> tuple[str, list[str]]:
    """EA WRC's telemetry config with the `wrc` session_update packet on to this machine."""
    doc = json.loads(text)
    packets = doc.setdefault("udp", {}).setdefault("packets", [])
    want = {"ip": "127.0.0.1", "port": port, "bEnabled": True}
    for p in packets:
        if p.get("structure") == "wrc" and p.get("packet") == "session_update":
            changes = [f"wrc session_update {k}: {p.get(k)!r} -> {v!r}" for k, v in want.items() if p.get(k) != v]
            if not changes:
                return text, []
            p.update(want)
            break
    else:
        packets.append({"structure": "wrc", "packet": "session_update", "frequencyHz": -1, **want})
        changes = [f"wrc session_update packet added, on, to 127.0.0.1:{port}"]
    return json.dumps(doc, indent=4), changes


_RALLY_RECORDS = (
    "own car's position (world XYZ, m) at 10 Hz with speed, gear, throttle, brake, stage distance",
    "stage start (the stage clock starting), finish with the stage time, restarts",
    "teleports (a reset to the road) so routes break there",
    "stages, best stage time, distance, top speed",
)
_RALLY_VERIFIED = (
    "decoding is gamecapture's lap recorder's (tested on packets built from the documented layouts); the adapter tested on "
    "packets built in the test, sent over UDP; untested against the game"
)

RALLY = [
    RallyAdapter(
        id="ea-wrc",
        game="EA SPORTS WRC",
        exes=("WRC.exe",),
        flavour="wrc",
        port=20777,
        source="UDP 127.0.0.1:20777, the default `wrc` structure's session_update packet (Documents\\My Games\\WRC\\telemetry\\config.json)",
        records=_RALLY_RECORDS,
        enable="`python -m heat3d_watch setup ea-wrc --apply` switches the session_update packet on in config.json (backed up first)",
        axes=Axes(up="y", units="metres"),
        verified=_RALLY_VERIFIED,
    ),
    RallyAdapter(
        id="dirt-rally-2",
        game="DiRT Rally 2.0",
        exes=("dirtrally2.exe",),
        flavour="dirtrally2",
        port=20777,
        source="UDP 127.0.0.1:20777, extradata=3 (Documents\\My Games\\DiRT Rally 2.0\\hardwaresettings\\hardware_settings_config.xml)",
        records=_RALLY_RECORDS,
        enable="`python -m heat3d_watch setup dirt-rally-2 --apply` sets <udp enabled=\"true\" extradata=\"3\"> (backed up first)",
        axes=Axes(up="y", units="metres"),
        verified=_RALLY_VERIFIED,
    ),
    RallyAdapter(
        id="richard-burns-rally",
        game="Richard Burns Rally",
        exes=("RichardBurnsRally_SSE.exe", "RichardBurnsRally.exe"),
        flavour="rbr",
        port=6776,
        multiplayer=False,
        source="NGP plugin UDP telemetry, 127.0.0.1:6776 (RichardBurnsRally.ini [NGP] udpTelemetry=1)",
        records=_RALLY_RECORDS,
        enable="RSF/NGP: in RichardBurnsRally.ini, section [NGP], set udpTelemetry=1 and the endpoint to 127.0.0.1:6776",
        axes=Axes(up="y", units="metres; up axis decided from the drive (RBR does not document its axes)"),
        verified=_RALLY_VERIFIED,
    ),
]


# ------------------------------------------------------------------ BeamNG


@dataclass
class BeamNGAdapter(Adapter):
    """BeamNG.drive through gamecapture's protocol mod: positions, laps, resets."""

    port: int = 4460
    hz: float = 10.0

    def missing(self) -> str | None:
        return _IMPORT_ERROR

    def start(self, ctx: Context) -> None:
        self._udp = UdpPort(self.port)
        self._drive = Driving(self, ctx, hz=self.hz)
        self._splitter = kunos.LineSplitter()
        self._last_time = -1.0
        self._runs = 0

    def poll(self, ctx: Context) -> None:
        if not self._udp.try_open(self.id):
            return
        for packet in self._udp.receive():
            try:
                self.feed(packet)
            except Exception:  # noqa: BLE001
                self._drive.count("bad_packets")

    def feed(self, packet: bytes) -> None:
        d = self._drive
        raw = beamng.decode(packet)
        if raw is None:
            d.count("bad_packets")
            return
        frame = beamng.frame_from(raw)
        x, y, z = frame["x"], frame["y"], frame["z"]
        if frame["time"] < self._last_time - 0.5:
            # The car was reset: its clock starts again.
            for lap in self._splitter.flush():
                self._closed(lap)
            self._splitter = kunos.LineSplitter()
            d.sample(x, y, z, speed=frame["speed"], force=True, session=str(self._runs + 1))
            d.event("respawn", session=str(self._runs + 1))
            self._runs += 1
        elif self._last_time < 0:
            self._runs = 1
            d.sample(x, y, z, speed=frame["speed"], force=True, session="1")
            d.event("session_start", session="1")
        self._last_time = frame["time"]
        d.sample(x, y, z, speed=frame["speed"], clock=frame["time"], session=str(self._runs), gear=frame["gear"],
                 throttle=frame["throttle"], brake=frame["brake"])
        slim = {"time": frame["time"], "x": x, "y": y, "z": z, "speed": frame["speed"]}
        for lap in self._splitter.feed(slim):
            self._closed(lap)

    def _closed(self, lap: Any) -> None:
        seconds = lap.seconds if lap.seconds is not None else lap.frames[-1]["time"] - lap.frames[0]["time"]
        if lap.complete:
            self._drive.lap(seconds, lap=lap.number, session=str(self._runs), metres=round(lap.metres, 1))
        else:
            self._drive.event("run_end", session=str(self._runs), seconds=round(seconds, 3), metres=round(lap.metres, 1))

    def stop(self, ctx: Context) -> None:
        for lap in self._splitter.flush():
            self._closed(lap)
        if self._drive.session is not None:
            self._drive.event("session_end", session=str(self._runs))
        self._udp.close()
        self._drive.close()

    def setup(self, ctx: Context, *, apply: bool) -> list[str]:
        target = beamng.user_mods() / "heat3d"
        if target.exists():
            return []
        lines = [f"install the heat3d protocol mod into {target}", "then in the game: Options > Other > Protocols > \"others\" on"]
        if apply:
            beamng.install()
        return lines


BEAMNG = [
    BeamNGAdapter(
        id="beamng",
        game="BeamNG.drive",
        exes=("BeamNG.drive.x64.exe", "BeamNG.drive.exe"),
        kind="telemetry",
        multiplayer=False,
        source="UDP 127.0.0.1:4460 from the heat3d protocol mod (gamecapture/mods/beamng/heat3d)",
        records=(
            "own car's position (world, m, turned to Y up) at 10 Hz with speed, gear, throttle, brake",
            "drive start, resets (respawn), laps past where the drive began",
            "laps, best lap, distance, top speed",
        ),
        enable="`python -m heat3d_watch setup beamng --apply` installs the mod; then Options > Other > Protocols > \"others\"",
        axes=Axes(up="y", units="metres (BeamNG's Z-up world turned about X, as the lap recorder does)"),
        verified="decoding is gamecapture's lap recorder's (tested on packets built from the mod's struct); adapter tested on such packets; the mod is untested in the game",
    ),
]


ADAPTERS = [*FORZA, *KUNOS, *RALLY, *BEAMNG]
