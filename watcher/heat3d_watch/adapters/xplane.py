"""
X-Plane 11 and 12: the sim's own UDP interface on port 49000.

X-Plane answers anyone who asks on 127.0.0.1:49000, at the rate asked for, to
the address and port the request came from (X-Plane's "Exchanging Data with
X-Plane" document, shipped in `<X-Plane>\\Instructions\\`):

    RREF  "RREF\\0" + int32 freq + int32 index + char[400] dataref path (413 bytes).
          The answer is "RREF" + one internal byte + (int32 index, float32 value)
          pairs; freq 0 stops it.
    RPOS  "RPOS\\0" + the rate as a NUL-terminated ASCII string ("10\\0"). The
          answer is "RPOS" + one internal byte + double lon, lat, elevation (m
          MSL) + float AGL (m), pitch, true heading, roll (deg), vx east, vy up,
          vz south (m/s), P, Q, R (rad/s), tightly packed.

RPOS gives latitude and longitude as doubles (RREF only as floats, half a metre
apart), so it drives the position; RREF brings the rest and stands in when
RPOS does not come. Datarefs (names as in X-Plane 12's `Resources\\plugins\\DataRefs.txt`,
https://developer.x-plane.com/datarefs/):

    sim/flightmodel/position/latitude, longitude, elevation  deg, deg, m MSL
    sim/flightmodel/position/true_theta, true_phi, true_psi  pitch, roll, true heading (deg)
    sim/flightmodel/position/indicated_airspeed              knots
    sim/flightmodel/position/groundspeed                     m/s
    sim/flightmodel/position/y_agl                           m above ground
    sim/flightmodel/forces/g_nrml                            g, normal to the wings
    sim/flightmodel/failures/onground_any                    1 on the ground
    sim/flightmodel2/misc/has_crashed                        1 crashed
    sim/time/total_flight_time_sec                           since the flight was reset

X-Plane 12 accepts these only with Settings > Network > "Accept incoming
connections" on. Positions go into a flat map around the sitting's first fix
(`Tangent`): x east, z north in metres, y the altitude above sea level - so a
level flight stays level however far it goes, which a true tangent plane would
bend down with the earth.

This module also holds what the other flight sims share: the flat map, the
flight bookkeeping (takeoffs, landings, crashes, distance), the UDP socket and
the Steam library list.
"""

from __future__ import annotations

import logging
import math
import os
import re
import socket
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..session import Axes, Session
from .base import Adapter, Context

log = logging.getLogger("heat3d_watch")

# ---------------------------------------------------------------------------- shared by the flight sims

#: WGS-84, the ellipsoid every sim here gives latitude and longitude on.
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3

#: The flat map's axes: x east, y up (altitude), z north. East-up-north is left-handed and the
#: viewer's world is right-handed, so z is mirrored for drawing - north ends up where a map has it.
TANGENT_AXES = Axes(up="y", flip_z=True, units="metres (east, altitude, north from the first fix)")


def ecef(lat: float, lon: float, alt: float) -> tuple[float, float, float]:
    """Latitude, longitude (degrees) and height (m) as earth-centred metres."""
    p, l = math.radians(lat), math.radians(lon)
    n = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(p) ** 2)
    return (
        (n + alt) * math.cos(p) * math.cos(l),
        (n + alt) * math.cos(p) * math.sin(l),
        (n * (1 - WGS84_E2) + alt) * math.sin(p),
    )


@dataclass
class Tangent:
    """A flat map around one point: east and north metres from it, the altitude as height.

    East and north are the tangent plane's (exact near the origin, off by a few
    metres per hundred kilometres further out); the height is the altitude above
    sea level rather than the height above the plane, so the earth's curvature
    does not pull distant flights down.
    """

    lat: float
    lon: float

    def __post_init__(self) -> None:
        self._origin = ecef(self.lat, self.lon, 0.0)
        p, l = math.radians(self.lat), math.radians(self.lon)
        self._sp, self._cp, self._sl, self._cl = math.sin(p), math.cos(p), math.sin(l), math.cos(l)

    def local(self, lat: float, lon: float, alt: float) -> tuple[float, float, float]:
        """(x east, y altitude, z north) in metres."""
        x, y, z = ecef(lat, lon, 0.0)
        dx, dy, dz = x - self._origin[0], y - self._origin[1], z - self._origin[2]
        east = -self._sl * dx + self._cl * dy
        north = -self._sp * self._cl * dx - self._sp * self._sl * dy + self._cp * dz
        return east, alt, north

    def meta(self) -> dict[str, float]:
        return {"lat": self.lat, "lon": self.lon}


def ground_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle metres between two (lat, lon), for jumps too far for the flat map."""
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371000.0 * math.asin(min(1.0, math.sqrt(h)))


@dataclass
class FlightLog:
    """One sitting's flights from position fixes: routes, takeoffs, landings, crashes, the numbers.

    Every flight is a route for the viewer's line heat: it begins with
    `session_start` (or `spawn`) and ends with `session_end`, `crash` or
    `death`; a jump no aircraft could fly is cut with a `teleport` where it
    left and a `spawn` where it arrived. The `session` column names the flight.
    """

    session: Session
    player: str = "me"
    #: The label's prefix: "flight-1", "life-1", ...
    word: str = "flight"
    #: The statistic counting them.
    plural: str = "flights"
    flight: int = 0
    live: bool = False
    on_ground: bool | None = None
    crashed: bool = False
    last: tuple[float, float, float] | None = None
    last_at: float | None = None
    #: Faster than this between two fixes is a jump (a new position, a slew, a reload), not flight.
    max_speed: float = 1500.0

    @property
    def label(self) -> str:
        return f"{self.word}-{self.flight}"

    def _at(self, at: tuple[float, float, float] | None) -> dict[str, float]:
        p = at or self.last
        return {} if p is None else {"x": p[0], "y": p[1], "z": p[2]}

    def begin(self, at: tuple[float, float, float] | None = None, event: str = "session_start", **attrs: Any) -> None:
        if self.live:
            self.end()
        self.flight += 1
        self.live = True
        self.on_ground = None
        self.crashed = False
        self.last = None
        self.last_at = None
        self.session.stat(self.plural, self.flight)
        self.session.event(event, player=self.player, session=self.label, **self._at(at), **attrs)

    def end(self, event: str = "session_end", **attrs: Any) -> None:
        if not self.live:
            return
        self.live = False
        self.session.event(event, player=self.player, session=self.label, **self._at(None), **attrs)

    def mark(self, event: str, at: tuple[float, float, float] | None = None, **attrs: Any) -> None:
        """Something that happened to the aircraft, where it was."""
        if event in ("takeoff", "landing"):
            self.session.stat(f"{event}s", 1, "add")
        elif event == "crash":
            self.session.stat("crashes", 1, "add")
        label = self.label if self.flight else None
        self.session.event(event, player=self.player, session=label, **self._at(at), **attrs)

    def fix(
        self,
        x: float,
        y: float,
        z: float,
        *,
        on_ground: bool | None = None,
        crashed: bool | None = None,
        g: float | None = None,
        **attrs: Any,
    ) -> None:
        """Where the aircraft is now; records the row and whatever changed since the last one."""
        if not all(math.isfinite(v) for v in (x, y, z)):
            return
        s = self.session
        now = time.monotonic()
        here = (x, y, z)
        if not self.live:
            self.begin(here)
        elif self.last is not None and self.last_at is not None:
            step = math.dist(self.last, here)
            dt = max(now - self.last_at, 1e-3)
            if step > 50.0 and step / dt > self.max_speed:
                # Relocated in the sim's map, a slew, a saved flight loaded: not a line to draw.
                s.event("teleport", player=self.player, session=self.label, **self._at(None), metres=round(step))
                self.on_ground = None
                s.event("spawn", player=self.player, session=self.label, x=x, y=y, z=z)
            else:
                s.stat("distance_km", round(step / 1000.0, 4), "add")
                if dt < 2.0:
                    s.stat("flight_time", round(dt, 3), "add")
                    if on_ground is False:
                        s.stat("airborne_time", round(dt, 3), "add")
        row = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in attrs.items()}
        if on_ground is not None:
            row["on_ground"] = int(on_ground)
        if g is not None and math.isfinite(g):
            row["g"] = round(g, 3)
            s.stat("max_g", round(g, 2), "max")
        s.position(x, y, z, player=self.player, session=self.label, **row)
        s.stat("max_altitude_m", round(y, 1), "max")
        if on_ground is not None:
            if self.on_ground is not None and on_ground != self.on_ground:
                self.mark("landing" if on_ground else "takeoff", here)
            self.on_ground = on_ground
        if crashed is not None:
            if crashed and not self.crashed:
                self.mark("crash", here)
            self.crashed = crashed
        self.last, self.last_at = here, now


def udp_socket(port: int = 0, host: str = "127.0.0.1") -> socket.socket:
    """A non-blocking UDP socket on localhost that survives a peer not listening yet."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SIO_UDP_CONNRESET"):
        # Windows turns "nobody on that port" into an error on the next receive; a sim still loading is not one.
        try:
            sock.ioctl(socket.SIO_UDP_CONNRESET, False)  # type: ignore[attr-defined]
        except OSError:
            pass
    sock.bind((host, port))
    sock.setblocking(False)
    return sock


def drain(sock: socket.socket | None, limit: int = 5000) -> list[tuple[bytes, Any]]:
    """Every datagram waiting, without blocking."""
    out: list[tuple[bytes, Any]] = []
    if sock is None:
        return out
    while len(out) < limit:
        try:
            out.append(sock.recvfrom(65535))
        except (BlockingIOError, socket.timeout):
            break
        except ConnectionResetError:
            continue
        except OSError:
            break
    return out


def steam_libraries() -> list[Path]:
    """Steam's library folders on this machine (from `libraryfolders.vdf`), the main one first."""
    roots: list[Path] = []
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
        if base:
            roots.append(Path(base) / "Steam")
    out: list[Path] = []
    for root in roots:
        vdf = root / "steamapps" / "libraryfolders.vdf"
        if not vdf.is_file():
            continue
        if root not in out:
            out.append(root)
        try:
            text = vdf.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in re.finditer(r'"path"\s+"([^"]+)"', text):
            p = Path(m.group(1).replace("\\\\", "\\"))
            if p not in out:
                out.append(p)
    return out


# ---------------------------------------------------------------------------- X-Plane's packets

XPLANE_PORT = 49000

#: (index, dataref, name). The index is ours: X-Plane sends it back with the value.
DATAREFS: tuple[tuple[int, str, str], ...] = (
    (1, "sim/flightmodel/position/latitude", "lat"),
    (2, "sim/flightmodel/position/longitude", "lon"),
    (3, "sim/flightmodel/position/elevation", "alt"),
    (4, "sim/flightmodel/position/true_theta", "pitch"),
    (5, "sim/flightmodel/position/true_phi", "roll"),
    (6, "sim/flightmodel/position/true_psi", "heading"),
    (7, "sim/flightmodel/position/indicated_airspeed", "ias_kt"),
    (8, "sim/flightmodel/position/groundspeed", "groundspeed"),
    (9, "sim/flightmodel/forces/g_nrml", "g"),
    (10, "sim/flightmodel/position/y_agl", "agl"),
    (11, "sim/flightmodel2/misc/has_crashed", "crashed"),
    (12, "sim/flightmodel/failures/onground_any", "on_ground"),
    (13, "sim/time/total_flight_time_sec", "flight_time"),
)
NAMES = {i: name for i, _, name in DATAREFS}

RPOS_FORMAT = "<dddffffffffff"
RPOS_FIELDS = ("lon", "lat", "alt", "agl", "pitch", "heading", "roll", "vx", "vy", "vz", "p", "q", "r")


def rref_request(freq: int, index: int, path: str) -> bytes:
    """Ask for one dataref `freq` times a second (0: stop), answered with `index`."""
    raw = path.encode("ascii")[:399]
    return b"RREF\0" + struct.pack("<ii", freq, index) + raw.ljust(400, b"\0")


def rpos_request(hz: int) -> bytes:
    return b"RPOS\0" + str(int(hz)).encode("ascii") + b"\0"


def decode_rref(packet: bytes) -> dict[int, float]:
    """An RREF answer: index -> value. Raises ValueError on anything else."""
    if packet[:4] != b"RREF" or len(packet) < 5 or (len(packet) - 5) % 8:
        raise ValueError("not an RREF answer")
    return {i: v for i, v in struct.iter_unpack("<if", packet[5:])}


def decode_rpos(packet: bytes) -> dict[str, float]:
    """An RPOS answer, by field name. Raises ValueError on anything else."""
    size = struct.calcsize(RPOS_FORMAT)
    if packet[:4] != b"RPOS" or len(packet) < 5 + size:
        raise ValueError("not an RPOS answer")
    return dict(zip(RPOS_FIELDS, struct.unpack_from(RPOS_FORMAT, packet, 5)))


# ---------------------------------------------------------------------------- the adapter


@dataclass
class XPlaneAdapter(Adapter):
    #: Where X-Plane listens: changeable for tests and for a sim on another port.
    host: str = "127.0.0.1"
    port: int = XPLANE_PORT
    hz: int = 10
    #: A new flight this far from the map's origin starts a new session with its own map.
    new_map_km: float = 300.0
    _sock: socket.socket | None = None
    _session: Session | None = None
    _log: FlightLog | None = None
    _tangent: Tangent | None = None
    _values: dict[str, float] = field(default_factory=dict)
    _rpos: dict[str, float] | None = None
    _rpos_at: float = 0.0
    _heard: float = 0.0
    _asked: float = 0.0
    _flight_time: float | None = None
    _new_flight: bool = False

    def start(self, ctx: Context) -> None:
        self.stop(ctx)
        self._sock = udp_socket()
        self._values, self._rpos, self._heard, self._flight_time = {}, None, 0.0, None
        self._ask(self.hz)

    def _ask(self, hz: int) -> None:
        """Subscribe (or with 0, unsubscribe) - again whenever X-Plane went quiet, it forgets on restart."""
        if self._sock is None:
            return
        target = (self.host, self.port)
        try:
            for index, path, _ in DATAREFS:
                self._sock.sendto(rref_request(hz, index, path), target)
            self._sock.sendto(rpos_request(hz), target)
        except OSError as e:
            log.debug("xplane: request not sent: %s", e)
        self._asked = time.monotonic()

    def poll(self, ctx: Context) -> None:
        now = time.monotonic()
        fresh = False
        for packet, _ in drain(self._sock):
            try:
                if packet[:4] == b"RREF":
                    for index, value in decode_rref(packet).items():
                        if index in NAMES:
                            self._values[NAMES[index]] = value
                    fresh = True
                elif packet[:4] == b"RPOS":
                    self._rpos, self._rpos_at = decode_rpos(packet), now
                    fresh = True
            except (ValueError, struct.error):
                if self._session is not None:
                    self._session.stat("bad_packets", 1, "add")
        if fresh:
            self._heard = now
        elif now - self._heard > 3.0 and now - self._asked > 5.0:
            self._ask(self.hz)
        if fresh:
            self._step(ctx, now)

    def _fix(self, now: float) -> tuple[float, float, float] | None:
        if self._rpos is not None and now - self._rpos_at < 1.0:
            return self._rpos["lat"], self._rpos["lon"], self._rpos["alt"]
        v = self._values
        if all(k in v for k in ("lat", "lon", "alt")):
            return v["lat"], v["lon"], v["alt"]
        return None

    def _step(self, ctx: Context, now: float) -> None:
        v = self._values
        ft = v.get("flight_time")
        if ft is not None:
            if self._flight_time is not None and ft < self._flight_time - 1.0:
                # The flight was reset: a new start, a new aircraft, a crash cleared.
                if self._log is not None:
                    self._log.end()
                self._new_flight = True
            self._flight_time = ft
        fix = self._fix(now)
        if fix is None:
            return
        lat, lon, alt = fix
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0) or (lat == 0.0 and lon == 0.0):
            return
        if self._session is not None and self._tangent is not None and self._new_flight:
            if ground_distance((self._tangent.lat, self._tangent.lon), (lat, lon)) > self.new_map_km * 1000:
                self._close(ctx)
        self._new_flight = False
        if self._session is None:
            self._tangent = Tangent(lat, lon)
            self._session = ctx.open(self, meta={"origin": self._tangent.meta()})
            self._log = FlightLog(self._session)
        assert self._tangent is not None and self._log is not None
        x, y, z = self._tangent.local(lat, lon, alt)
        src = self._rpos if self._rpos is not None and now - self._rpos_at < 1.0 else {}
        on_ground = v.get("on_ground")
        crashed = v.get("crashed")
        self._log.fix(
            x,
            y,
            z,
            on_ground=None if on_ground is None else on_ground >= 0.5,
            crashed=None if crashed is None else crashed >= 0.5,
            g=v.get("g"),
            lat=lat,
            lon=lon,
            alt=alt,
            heading=src.get("heading", v.get("heading")),
            pitch=src.get("pitch", v.get("pitch")),
            roll=src.get("roll", v.get("roll")),
            ias_kt=v.get("ias_kt"),
            groundspeed=v.get("groundspeed"),
            agl=src.get("agl", v.get("agl")),
            flight_time=ft,
        )

    def _close(self, ctx: Context) -> None:
        if self._log is not None:
            self._log.end()
        ctx.close(self._session)
        self._session, self._log, self._tangent = None, None, None

    def stop(self, ctx: Context) -> None:
        if self._sock is not None:
            self._ask(0)
            self._sock.close()
            self._sock = None
        self._close(ctx)


ADAPTERS = [
    XPlaneAdapter(
        id="xplane",
        game="X-Plane",
        kind="telemetry",
        exes=("X-Plane.exe",),
        multiplayer=False,
        source="X-Plane's UDP interface on 127.0.0.1:49000 (RREF datarefs and RPOS position, answered to the watcher)",
        records=(
            "own position at 10 Hz (latitude/longitude/altitude as metres east/up/north of the first fix; "
            "lat/lon/alt kept as columns)",
            "attitude, indicated airspeed, groundspeed, AGL, normal g, on-ground",
            "takeoffs, landings, crashes, flight resets (session_start/session_end per flight)",
            "flight time, airborne time, distance, highest altitude, highest g, landings",
        ),
        enable="X-Plane 12: Settings > Network > Accept incoming connections (X-Plane 11 answers by default)",
        interval=0.1,
        axes=TANGENT_AXES,
        verified=(
            "tested against a fake X-Plane answering RREF/RPOS per X-Plane 12's 'Exchanging Data with X-Plane'; "
            "dataref names checked against X-Plane 12's DataRefs.txt; untested against the running sim"
        ),
    )
]
