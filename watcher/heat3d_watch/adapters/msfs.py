"""
Microsoft Flight Simulator 2020 and 2024: SimConnect, through the player's own SimConnect.dll.

SimConnect is the sim's published client API (SDK documentation:
https://docs.flightsimulator.com/html/Programming_Tools/SimConnect/SimConnect_SDK.htm,
https://docs.flightsimulator.com/msfs2024/html/6_Programming_APIs/SimConnect/SimConnect_SDK.htm).
The watcher does not ship the DLL: it loads the one that came with the sim or
its SDK - MSFS 2020 puts `SimConnect.dll` in its install folder; MSFS 2024 on
Steam ships only its internal one, so 2024 players install the MSFS 2024 SDK
(`<SDK>\\SimConnect SDK\\lib\\SimConnect.dll`; the 2024 sim also accepts 2020 clients).

The calls, from SimConnect.h, all returning an HRESULT (0 = S_OK):

    SimConnect_Open(HANDLE*, LPCSTR name, HWND, DWORD, HANDLE, DWORD config)
    SimConnect_AddToDataDefinition(HANDLE, DWORD define, LPCSTR var, LPCSTR units,
                                   SIMCONNECT_DATATYPE (FLOAT64 = 4), float epsilon, DWORD datum)
    SimConnect_RequestDataOnSimObject(HANDLE, DWORD request, DWORD define, DWORD object (0 = user),
                                      SIMCONNECT_PERIOD (SIM_FRAME = 3), DWORD flags, DWORD origin,
                                      DWORD interval, DWORD limit)
    SimConnect_SubscribeToSystemEvent(HANDLE, DWORD event, LPCSTR name)
    SimConnect_GetNextDispatch(HANDLE, SIMCONNECT_RECV**, DWORD*)   E_FAIL when nothing waits
    SimConnect_Close(HANDLE)

Every message starts with SIMCONNECT_RECV {DWORD dwSize, dwVersion, dwID}
(#pragma pack 1). The ones read here:

    1  EXCEPTION       dwException, dwSendID, dwIndex
    2  OPEN            szApplicationName[256], then version DWORDs
    3  QUIT            the sim is closing
    4  EVENT           uGroupID, uEventID, dwData
    6  EVENT_FILENAME  EVENT + szFileName[260], dwFlags        (FlightLoaded)
    8  SIMOBJECT_DATA  dwRequestID, dwObjectID, dwDefineID, dwFlags, dwentrynumber,
                       dwoutof, dwDefineCount, then the values (FLOAT64 each here)

Simulation variables (units asked for so no conversion is needed): PLANE
LATITUDE/LONGITUDE (degrees), PLANE ALTITUDE (meters, above sea level), PLANE
ALT ABOVE GROUND (meters), PLANE PITCH/BANK DEGREES and PLANE HEADING DEGREES
TRUE (degrees; SimConnect's own signs - nose up is negative pitch), AIRSPEED
INDICATED and GROUND VELOCITY (knots), G FORCE (gforce), SIM ON GROUND (bool).
System events: SimStart, SimStop, Crashed, FlightLoaded.

Positions go into the same flat map as X-Plane's (x east, y altitude, z north
in metres from the sitting's first fix).
"""

from __future__ import annotations

import ctypes
import logging
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..session import Session
from .base import Adapter, Context
from .xplane import TANGENT_AXES, FlightLog, Tangent, ground_distance, steam_libraries

log = logging.getLogger("heat3d_watch")

# ---------------------------------------------------------------------------- the messages

RECV_EXCEPTION = 1
RECV_OPEN = 2
RECV_QUIT = 3
RECV_EVENT = 4
RECV_EVENT_FILENAME = 6
RECV_SIMOBJECT_DATA = 8

DATATYPE_FLOAT64 = 4
PERIOD_SIM_FRAME = 3
OBJECT_USER = 0
UNUSED = 0xFFFFFFFF
MAX_PATH = 260

#: (simulation variable, units, our name). Order is the order of the values in a data message.
SIMVARS: tuple[tuple[str, str, str], ...] = (
    ("PLANE LATITUDE", "degrees", "lat"),
    ("PLANE LONGITUDE", "degrees", "lon"),
    ("PLANE ALTITUDE", "meters", "alt"),
    ("PLANE ALT ABOVE GROUND", "meters", "agl"),
    ("PLANE PITCH DEGREES", "degrees", "pitch"),
    ("PLANE BANK DEGREES", "degrees", "bank"),
    ("PLANE HEADING DEGREES TRUE", "degrees", "heading"),
    ("AIRSPEED INDICATED", "knots", "ias_kt"),
    ("GROUND VELOCITY", "knots", "groundspeed_kt"),
    ("G FORCE", "gforce", "g"),
    ("SIM ON GROUND", "bool", "on_ground"),
)

#: Our ids for the system events subscribed to.
EVENTS = {1: "SimStart", 2: "SimStop", 3: "Crashed", 4: "FlightLoaded"}
DEFINE_ID = 1
REQUEST_ID = 1


def _text(buf: bytes, offset: int, length: int) -> str:
    raw = buf[offset : offset + length].split(b"\0", 1)[0]
    return raw.decode("utf-8", errors="replace").strip()


def decode(buf: bytes) -> dict[str, Any]:
    """One SimConnect message as a dict with its `kind`. Raises ValueError when it is cut short."""
    if len(buf) < 12:
        raise ValueError("shorter than SIMCONNECT_RECV")
    size, version, rid = struct.unpack_from("<III", buf, 0)
    if size > len(buf):
        raise ValueError("message cut short")
    out: dict[str, Any] = {"id": rid, "version": version}
    if rid == RECV_EXCEPTION:
        exc, send_id, index = struct.unpack_from("<III", buf, 12)
        out.update(kind="exception", exception=exc, send_id=send_id, index=index)
    elif rid == RECV_OPEN:
        out.update(kind="open", application=_text(buf, 12, 256))
        if len(buf) >= 12 + 256 + 16:
            out["version_major"], out["version_minor"] = struct.unpack_from("<II", buf, 268)
    elif rid == RECV_QUIT:
        out.update(kind="quit")
    elif rid in (RECV_EVENT, RECV_EVENT_FILENAME):
        group, event, data = struct.unpack_from("<III", buf, 12)
        out.update(kind="event", group=group, event=event, data=data, name=EVENTS.get(event, ""))
        if rid == RECV_EVENT_FILENAME:
            out["filename"] = _text(buf, 24, MAX_PATH)
    elif rid == RECV_SIMOBJECT_DATA:
        request, obj, define, flags, entry, outof, count = struct.unpack_from("<7I", buf, 12)
        n = (len(buf) - 40) // 8
        values = struct.unpack_from(f"<{n}d", buf, 40) if n > 0 else ()
        out.update(kind="data", request=request, object=obj, define=define, values=values)
        if define == DEFINE_ID and len(values) >= len(SIMVARS):
            out["vars"] = {name: values[i] for i, (_, _, name) in enumerate(SIMVARS)}
    else:
        out.update(kind="other")
    return out


# ---------------------------------------------------------------------------- finding the DLL

DLL = "SimConnect.dll"


def install_folders() -> list[Path]:
    """Where the two sims are installed through Steam or the Xbox app (both may be absent)."""
    out: list[Path] = []
    for lib in steam_libraries():
        out.append(lib / "steamapps" / "common" / "MicrosoftFlightSimulator")
        out.append(lib / "steamapps" / "common" / "MSFS2024")
    for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        root = Path(f"{letter}:\\XboxGames")
        try:
            if not root.is_dir():
                continue
        except OSError:
            continue
        out.append(root / "Microsoft Flight Simulator" / "Content")
        out.append(root / "Microsoft Flight Simulator 2024" / "Content")
    return out


def _sdk_folders(is2024: bool) -> list[Path]:
    out = []
    env = os.environ.get("MSFS2024_SDK" if is2024 else "MSFS_SDK")
    if env:
        out.append(Path(env.strip().strip('"')))
    out.append(Path(r"C:\MSFS 2024 SDK" if is2024 else r"C:\MSFS SDK"))
    return out


def candidates(is2024: bool, exe_folder: Path | None = None) -> list[Path]:
    """Every place a SimConnect.dll for this sim may be, in the order they are tried.

    The sim's own folder first, then its SDK; for 2024 the 2020 copies after
    (the 2024 sim accepts 2020 clients). Never `SimConnect_internal.dll`: that
    is the sim's in-process library, not a client one.
    """
    own, other = [], []
    for folder in install_folders():
        (own if ("2024" in folder.as_posix()) == is2024 else other).append(folder)
    out = [exe_folder / DLL] if exe_folder else []
    out += [f / DLL for f in own]
    out += [sdk / "SimConnect SDK" / "lib" / DLL for sdk in _sdk_folders(is2024)]
    if is2024:
        out += [f / DLL for f in other]
        out += [sdk / "SimConnect SDK" / "lib" / DLL for sdk in _sdk_folders(False)]
    return out


def find_simconnect(is2024: bool, exe_folder: Path | None = None) -> Path | None:
    for p in candidates(is2024, exe_folder):
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    return None


# ---------------------------------------------------------------------------- the client


class SimConnect:
    """The handful of SimConnect calls the watcher makes, through ctypes on the player's DLL."""

    def __init__(self, path: Path):
        self.dll = ctypes.WinDLL(str(path))  # type: ignore[attr-defined]
        d = self.dll
        H, DW, P = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_char_p
        d.SimConnect_Open.argtypes = [ctypes.POINTER(H), P, H, DW, H, DW]
        d.SimConnect_AddToDataDefinition.argtypes = [H, DW, P, P, DW, ctypes.c_float, DW]
        d.SimConnect_RequestDataOnSimObject.argtypes = [H, DW, DW, DW, DW, DW, DW, DW, DW]
        d.SimConnect_SubscribeToSystemEvent.argtypes = [H, DW, P]
        d.SimConnect_GetNextDispatch.argtypes = [H, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(DW)]
        d.SimConnect_Close.argtypes = [H]
        for name in ("Open", "AddToDataDefinition", "RequestDataOnSimObject", "SubscribeToSystemEvent",
                     "GetNextDispatch", "Close"):
            getattr(d, f"SimConnect_{name}").restype = ctypes.c_long
        self.handle = ctypes.c_void_p()

    @property
    def open(self) -> bool:
        return bool(self.handle.value)

    def connect(self, name: str) -> bool:
        hr = self.dll.SimConnect_Open(ctypes.byref(self.handle), name.encode("ascii"), None, 0, None, 0)
        if hr != 0:
            self.handle = ctypes.c_void_p()
            return False
        h = self.handle
        for var, units, _ in SIMVARS:
            self.dll.SimConnect_AddToDataDefinition(
                h, DEFINE_ID, var.encode(), units.encode(), DATATYPE_FLOAT64, 0.0, UNUSED
            )
        self.dll.SimConnect_RequestDataOnSimObject(h, REQUEST_ID, DEFINE_ID, OBJECT_USER, PERIOD_SIM_FRAME, 0, 0, 0, 0)
        for eid, ev in EVENTS.items():
            self.dll.SimConnect_SubscribeToSystemEvent(h, eid, ev.encode())
        return True

    def messages(self, limit: int = 2000) -> list[bytes]:
        """Everything waiting, copied out of SimConnect's buffer."""
        out: list[bytes] = []
        if not self.open:
            return out
        ptr, size = ctypes.c_void_p(), ctypes.c_uint32()
        while len(out) < limit:
            hr = self.dll.SimConnect_GetNextDispatch(self.handle, ctypes.byref(ptr), ctypes.byref(size))
            if hr != 0 or not ptr.value or size.value < 12:
                break
            out.append(ctypes.string_at(ptr.value, min(size.value, 1 << 20)))
        return out

    def close(self) -> None:
        if self.open:
            try:
                self.dll.SimConnect_Close(self.handle)
            except OSError:
                pass
        self.handle = ctypes.c_void_p()


# ---------------------------------------------------------------------------- the adapter


@dataclass
class MsfsAdapter(Adapter):
    is2024: bool = False
    #: Seconds between attempts to connect while the sim is still loading.
    retry: float = 5.0
    new_map_km: float = 300.0
    _client: Any = None
    _dll: Path | None = None
    _missing: tuple[float, str | None] = (0.0, None)
    _next_try: float = 0.0
    _session: Session | None = None
    _log: FlightLog | None = None
    _tangent: Tangent | None = None
    _running: bool = True
    _new_flight: bool = False
    _flight_file: str = ""

    def missing(self) -> str | None:
        if sys.platform != "win32":
            return "Windows only (SimConnect)"
        at, why = self._missing
        if time.monotonic() - at < 60 and at:
            return why
        found = find_simconnect(self.is2024)
        why = None
        if found is None:
            sdk = "MSFS 2024 SDK" if self.is2024 else "MSFS SDK"
            why = (f"no SimConnect.dll found (the sim folder, the {sdk}'s SimConnect SDK\\lib); "
                   f"install the {sdk} or set {'MSFS2024_SDK' if self.is2024 else 'MSFS_SDK'}")
        self._missing = (time.monotonic(), why)
        return why

    def start(self, ctx: Context) -> None:
        self.stop(ctx)
        self._dll = find_simconnect(self.is2024, ctx.exe.parent if ctx.exe else None)
        if self._dll is None:
            raise RuntimeError(self.missing() or "no SimConnect.dll")
        self._client = SimConnect(self._dll)
        self._next_try = 0.0

    def poll(self, ctx: Context) -> None:
        c = self._client
        if c is None:
            return
        now = time.monotonic()
        if not c.open:
            if now < self._next_try:
                return
            self._next_try = now + self.retry
            if not c.connect("3DHeat Watcher"):
                return
            log.info("%s: connected to SimConnect (%s)", self.id, self._dll)
        self.handle(ctx, c.messages())

    def handle(self, ctx: Context, messages: list[bytes]) -> None:
        """Take a poll's worth of messages: events in order, the last position as this poll's fix."""
        latest: dict[str, float] | None = None
        taken: dict[str, float] | None = None
        for raw in messages:
            try:
                m = decode(raw)
            except (ValueError, struct.error):
                if self._session is not None:
                    self._session.stat("bad_packets", 1, "add")
                continue
            kind = m["kind"]
            if kind == "data" and "vars" in m:
                latest = m["vars"]
                # A touchdown between two polls still counts: on-ground changes are taken from every frame.
                if self._log is not None and self._log.live and self._log.on_ground is not None:
                    og = latest["on_ground"] >= 0.5
                    if og != self._log.on_ground:
                        self._fix(ctx, latest)
                        taken = latest
            elif kind == "event":
                self._event(ctx, m)
            elif kind == "quit":
                self._close(ctx)
                if self._client is not None:
                    self._client.close()
                self._next_try = time.monotonic() + self.retry
            elif kind == "exception":
                log.debug("%s: SimConnect exception %s on send %s", self.id, m["exception"], m["send_id"])
        if latest is not None and latest is not taken and self._running:
            self._fix(ctx, latest)

    def _event(self, ctx: Context, m: dict[str, Any]) -> None:
        name = m.get("name")
        if name == "SimStart":
            self._running = True
        elif name == "SimStop":
            self._running = False
            if self._log is not None:
                self._log.end()
            self._new_flight = True
        elif name == "Crashed":
            if self._log is not None and self._log.live:
                self._log.mark("crash")
        elif name == "FlightLoaded":
            self._flight_file = Path(m.get("filename", "").replace("\\", "/")).name
            if self._log is not None:
                self._log.end()
            self._new_flight = True

    def _fix(self, ctx: Context, v: dict[str, float]) -> None:
        lat, lon, alt = v["lat"], v["lon"], v["alt"]
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0) or (lat == 0.0 and lon == 0.0):
            return  # in the menus SimConnect reports 0/0
        if self._session is not None and self._tangent is not None and self._new_flight:
            if ground_distance((self._tangent.lat, self._tangent.lon), (lat, lon)) > self.new_map_km * 1000:
                self._close(ctx)
        if self._session is None:
            self._tangent = Tangent(lat, lon)
            self._session = ctx.open(self, meta={"origin": self._tangent.meta()})
            self._log = FlightLog(self._session)
        assert self._tangent is not None and self._log is not None
        if self._new_flight and self._log.live:
            self._log.end()
        self._new_flight = False
        x, y, z = self._tangent.local(lat, lon, alt)
        if not self._log.live:
            self._log.begin((x, y, z), flight_file=self._flight_file or None)
        self._log.fix(
            x,
            y,
            z,
            on_ground=v["on_ground"] >= 0.5,
            g=v["g"],
            lat=lat,
            lon=lon,
            alt=alt,
            agl=v["agl"],
            heading=v["heading"],
            pitch=v["pitch"],
            bank=v["bank"],
            ias_kt=v["ias_kt"],
            groundspeed_kt=v["groundspeed_kt"],
        )

    def _close(self, ctx: Context) -> None:
        if self._log is not None:
            self._log.end()
        ctx.close(self._session)
        self._session, self._log, self._tangent = None, None, None

    def stop(self, ctx: Context) -> None:
        self._close(ctx)
        if self._client is not None:
            self._client.close()
            self._client = None


def _adapter(id: str, game: str, exe: str, is2024: bool) -> MsfsAdapter:
    sdk = "MSFS 2024 SDK" if is2024 else "MSFS SDK"
    return MsfsAdapter(
        id=id,
        game=game,
        kind="telemetry",
        exes=(exe,),
        multiplayer=True,
        is2024=is2024,
        source=(
            "SimConnect through the player's own SimConnect.dll (the sim folder, or "
            f"%{'MSFS2024_SDK' if is2024 else 'MSFS_SDK'}%\\SimConnect SDK\\lib)"
        ),
        records=(
            "own aircraft position every sim frame, taken at 10 Hz (latitude/longitude/altitude as metres "
            "east/up/north of the first fix; lat/lon/alt kept as columns)",
            "attitude, indicated airspeed, groundspeed, AGL, g, on-ground",
            "takeoffs, landings, crashes, flights started and stopped (session_start/session_end)",
            "flight time, airborne time, distance, highest altitude, highest g, landings",
        ),
        enable=(
            "nothing in the sim; it needs SimConnect.dll - MSFS 2020 has it in its folder, "
            f"MSFS 2024 players install the {sdk}" if is2024 else
            "nothing; SimConnect.dll comes with MSFS 2020 (or the MSFS SDK)"
        ),
        interval=0.1,
        axes=TANGENT_AXES,
        verified=(
            "message decoding tested on buffers built from SimConnect.h's layouts; the live SimConnect "
            "connection is untested (no SimConnect.dll on the test machine)"
        ),
    )


ADAPTERS = [
    _adapter("msfs2020", "Microsoft Flight Simulator 2020", "FlightSimulator.exe", False),
    _adapter("msfs2024", "Microsoft Flight Simulator 2024", "FlightSimulator2024.exe", True),
]
