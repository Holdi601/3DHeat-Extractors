"""
Euro Truck Simulator 2 and American Truck Simulator: the SCS telemetry SDK, through the RenCloud plugin.

SCS publishes a telemetry SDK (https://modding.scssoft.com/wiki/Documentation/Engine/SDK/Telemetry)
for plugins the player puts in `<game>\\bin\\win_x64\\plugins\\`. The game itself
publishes nothing; the widely used plugin from https://github.com/RenCloud/scs-sdk-plugin
(MIT) copies the SDK's channels into shared memory `Local\\SCSTelemetry`
(32 KiB), laid out in `scs-telemetry/inc/scs-telemetry-common.hpp`. This reads
revision 12 of that layout (PLUGIN_REVID 12) and nothing else: the plugin
writes its revision at offset 40, and any other revision is left unread rather
than misread. Offsets used (zones as the header marks them; natural alignment):

       0  bool sdkActive          4  bool paused
      40  u32  telemetry_plugin_revision, version_major, version_minor, game (1 ETS2, 2 ATS)
      64  u32  common_ui.time_abs (in-game minutes)
     640  i32  gameplay_i.jobDeliveredEarnedXp
     948  f32  truck_f.speed (m/s); 1056 truckOdometer (km)
    1460  f32  gameplay_f.jobDeliveredDistanceKm
    2200  f64  truck_dp.coordinateX/Y/Z (m), rotationX/Y/Z (heading, pitch, roll in turns, 0..1)
    2300  char[64] config_s: truckBrand +64, truckName +192, cargo +320, cityDst +448, compDst +576,
          citySrc +704, compSrc +832; 3436 gameplay_s.fineOffence[32]
    4200  i64  gameplay_ll: jobCancelledPenalty, jobDeliveredRevenue, fineAmount, tollgatePayAmount,
          ferryPayAmount, trainPayAmount
    4300  bool special_b: onJob, jobFinished, jobCancelled, jobDelivered, fined, tollgate, ferry,
          train, refuel, refuelPayed

The truck's world placement is SCS's world frame: x east, y up, z south, metres
(right-handed, y up - the viewer draws it as it is). Ferries, trains and loading
a save move the truck without driving: the route is cut there.
"""

from __future__ import annotations

import ctypes
import logging
import math
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from ..session import Axes, Session
from .base import Adapter, Context

log = logging.getLogger("heat3d_watch")

NAME = "Local\\SCSTelemetry"
REVISION = 12
#: Bytes read: everything up to the end of the special events zone; trailers are not read.
READ = 4400

SPECIAL = ("on_job", "job_finished", "job_cancelled", "job_delivered", "fined", "tollgate", "ferry", "train",
           "refuel", "refuel_paid")
STRINGS = {"truck_brand": 2364, "truck_name": 2492, "cargo": 2620, "city_dst": 2748, "company_dst": 2876,
           "city_src": 3004, "company_src": 3132}


def _s(buf: bytes, offset: int, size: int = 64) -> str:
    return buf[offset : offset + size].split(b"\0", 1)[0].decode("utf-8", errors="replace")


def decode(buf: bytes) -> dict[str, Any]:
    """The values used here, from a copy of the shared memory. Raises ValueError for another layout."""
    if len(buf) < READ:
        raise ValueError("shorter than the layout")
    revision, vmaj, vmin, game = struct.unpack_from("<4I", buf, 40)
    if revision != REVISION:
        raise ValueError(f"plugin revision {revision}, this reads {REVISION}")
    d: dict[str, Any] = {
        "active": bool(buf[0]),
        "paused": bool(buf[4]),
        "revision": revision,
        "game_version": f"{vmaj}.{vmin}",
        "game": {1: "ets2", 2: "ats"}.get(game, ""),
        "time_abs": struct.unpack_from("<I", buf, 64)[0],
        "xp": struct.unpack_from("<i", buf, 640)[0],
        "speed": struct.unpack_from("<f", buf, 948)[0],
        "odometer": struct.unpack_from("<f", buf, 1056)[0],
        "delivered_km": struct.unpack_from("<f", buf, 1460)[0],
        "fine_offence": _s(buf, 3436, 32),
    }
    x, y, z, heading, pitch, roll = struct.unpack_from("<6d", buf, 2200)
    d.update(x=x, y=y, z=z, heading=heading, pitch=pitch, roll=roll)
    for k, off in STRINGS.items():
        d[k] = _s(buf, off)
    keys = ("cancel_penalty", "revenue", "fine_amount", "toll_amount", "ferry_amount", "train_amount")
    d.update(zip(keys, struct.unpack_from("<6q", buf, 4200)))
    d.update(zip(SPECIAL, (bool(b) for b in buf[4300 : 4300 + len(SPECIAL)])))
    return d


class SharedMemory:
    """A read-only view of a named mapping someone else made; never creates one."""

    FILE_MAP_READ = 0x0004

    def __init__(self, name: str, size: int):
        k = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        k.OpenFileMappingW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
        k.OpenFileMappingW.restype = ctypes.c_void_p
        k.MapViewOfFile.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_size_t]
        k.MapViewOfFile.restype = ctypes.c_void_p
        k.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
        k.CloseHandle.argtypes = [ctypes.c_void_p]
        self.k, self.size = k, size
        self.handle = k.OpenFileMappingW(self.FILE_MAP_READ, 0, name)
        if not self.handle:
            raise FileNotFoundError(name)
        self.view = k.MapViewOfFile(self.handle, self.FILE_MAP_READ, 0, 0, size)
        if not self.view:
            k.CloseHandle(self.handle)
            raise OSError(f"cannot map {name}")

    def read(self) -> bytes:
        return ctypes.string_at(self.view, self.size)

    def close(self) -> None:
        if self.view:
            self.k.UnmapViewOfFile(self.view)
            self.view = None
        if self.handle:
            self.k.CloseHandle(self.handle)
            self.handle = None


@dataclass
class ScsAdapter(Adapter):
    name: str = NAME
    #: A step longer than this between two samples is a ferry, a train, a loaded save: cut, not driven.
    jump_m: float = 200.0
    #: While standing, a row only this often.
    idle_seconds: float = 5.0
    _mem: Any = None
    _next_try: float = 0.0
    _session: Session | None = None
    _last: tuple[float, float, float] | None = None
    _last_row: float = 0.0
    _flags: dict[str, bool] = field(default_factory=dict)
    _odo: float | None = None
    _stint: int = 0
    _warned: bool = False

    def missing(self) -> str | None:
        return None if sys.platform == "win32" else "Windows only (shared memory)"

    def start(self, ctx: Context) -> None:
        self.stop(ctx)
        self._next_try = 0.0

    def _open(self) -> bool:
        if self._mem is not None:
            return True
        now = time.monotonic()
        if now < self._next_try:
            return False
        self._next_try = now + 2.0
        try:
            self._mem = SharedMemory(self.name, READ)
        except OSError:
            return False  # the plugin is not installed, or the game has not loaded it yet
        return True

    def poll(self, ctx: Context) -> None:
        if not self._open():
            return
        try:
            d = decode(self._mem.read())
        except ValueError as e:
            if not self._warned:
                log.warning("%s: %s; not reading it", self.id, e)
                self._warned = True
            return
        self.sample(ctx, d)

    @property
    def _label(self) -> str:
        return f"drive-{self._stint}"

    def sample(self, ctx: Context, d: dict[str, Any]) -> None:
        """One look at the telemetry: the truck's place, and what the special flags say happened."""
        if not d["active"]:
            return
        if self._session is None:
            self._session = ctx.open(self, meta={"plugin_revision": d["revision"], "game_version": d["game_version"]})
        s = self._session
        here = (d["x"], d["y"], d["z"])
        if not all(math.isfinite(v) for v in here):
            return
        now = time.monotonic()
        common = dict(player="me", session=self._label if self._stint else None)
        if self._last is None:
            self._stint += 1
            common["session"] = self._label
            s.event("spawn", x=here[0], y=here[1], z=here[2], **common, truck=d["truck_name"] or None)
            self._last, self._last_row = here, now
        step = math.dist(self._last, here)
        if step > self.jump_m:
            s.event("teleport", x=self._last[0], y=self._last[1], z=self._last[2], **common, metres=round(step))
            self._stint += 1
            common["session"] = self._label
            s.event("spawn", x=here[0], y=here[1], z=here[2], **common)
            self._last, self._last_row = here, now
        elif not d["paused"] and (step > 0.5 or now - self._last_row >= self.idle_seconds):
            s.position(
                *here,
                **common,
                heading=round(d["heading"] * 360.0, 2),
                speed_kmh=round(d["speed"] * 3.6, 1),
                odometer=round(d["odometer"], 2),
            )
            s.stat("max_speed_kmh", round(abs(d["speed"]) * 3.6, 1), "max")
            s.stat("distance_km", round(step / 1000.0, 4), "add")
            self._last, self._last_row = here, now
        self._special(s, d, here, common)

    def _special(self, s: Session, d: dict[str, Any], at: tuple[float, float, float], common: dict[str, Any]) -> None:
        rose = [k for k in SPECIAL if d[k] and not self._flags.get(k, False)]
        first = not self._flags
        self._flags = {k: d[k] for k in SPECIAL}
        if first:
            rose = [k for k in rose if k == "on_job"]  # a job already running at start is still worth its row
        pos = dict(x=at[0], y=at[1], z=at[2], **common)
        for k in rose:
            if k == "on_job":
                s.event("job_start", **pos, cargo=d["cargo"] or None, from_city=d["city_src"] or None,
                        from_company=d["company_src"] or None, to_city=d["city_dst"] or None,
                        to_company=d["company_dst"] or None)
            elif k == "job_delivered":
                s.event("delivered", **pos, revenue=d["revenue"], xp=d["xp"], km=round(d["delivered_km"], 1),
                        cargo=d["cargo"] or None, to_city=d["city_dst"] or None)
                s.stat("deliveries", 1, "add")
                s.stat("revenue", d["revenue"], "add")
                s.stat("xp", d["xp"], "add")
            elif k == "job_cancelled":
                s.event("job_cancelled", **pos, penalty=d["cancel_penalty"])
                s.stat("jobs_cancelled", 1, "add")
            elif k == "fined":
                s.event("fine", **pos, offence=d["fine_offence"] or None, amount=d["fine_amount"])
                s.stat("fines", 1, "add")
                s.stat("fines_total", d["fine_amount"], "add")
            elif k == "tollgate":
                s.event("tollgate", **pos, amount=d["toll_amount"])
                s.stat("tolls_total", d["toll_amount"], "add")
            elif k in ("ferry", "train"):
                s.event(k, **pos, amount=d[f"{k}_amount"])
            elif k == "refuel":
                s.event("refuel", **pos)
                s.stat("refuels", 1, "add")

    def stop(self, ctx: Context) -> None:
        if self._session is not None and self._last is not None:
            self._session.event("session_end", x=self._last[0], y=self._last[1], z=self._last[2], player="me",
                                session=self._label)
        ctx.close(self._session)
        self._session, self._last, self._flags, self._stint, self._warned = None, None, {}, 0, False
        if self._mem is not None:
            self._mem.close()
            self._mem = None


def _adapter(id: str, game: str, exe: str) -> ScsAdapter:
    return ScsAdapter(
        id=id,
        game=game,
        kind="telemetry",
        exes=(exe,),
        multiplayer=True,
        source=f"shared memory {NAME} written by the RenCloud scs-sdk-plugin (revision {REVISION}) in "
        "<game>\\bin\\win_x64\\plugins",
        records=(
            "the truck's world position (x east, y up, z south, m) at 10 Hz while moving, every 5 s while standing; "
            "heading, speed, odometer",
            "jobs started (cargo, from, to), delivered (revenue, XP, km), cancelled; fines, tolls, ferries, trains, "
            "refuels",
            "ferries, trains and loaded saves cut the route (teleport, spawn)",
            "distance, top speed, deliveries, revenue, XP, fines",
        ),
        enable=(
            "install scs-telemetry.dll (revision 12) from github.com/RenCloud/scs-sdk-plugin into "
            "<game>\\bin\\win_x64\\plugins; the game asks once per start to allow the SDK"
        ),
        interval=0.1,
        axes=Axes(up="y", units="metres (SCS world: x east, y up, z south)"),
        verified=(
            "tested on shared memory built from the plugin's revision 12 header layout; untested against the game "
            "(not installed on the test machine)"
        ),
    )


ADAPTERS = [
    _adapter("ets2", "Euro Truck Simulator 2", "eurotrucks2.exe"),
    _adapter("ats", "American Truck Simulator", "amtrucks.exe"),
]
