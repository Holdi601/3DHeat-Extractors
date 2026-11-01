"""
Shared memory that racing games publish for dashboards, opened read-only - and
never created.

Several sims hand out their telemetry as a named Windows file mapping that the
game creates and anyone may read: Kunos's `Local\\acpmf_*` pages (Assetto Corsa,
ACC, AC Rally, AC EVO's `Local\\acevo_pmf_*`), iRacing's
`Local\\IRSDKMemMapFileName`, RaceRoom's `$R3E`, Automobilista 2 / Project CARS 2's
`$pcars2$`.

Why not `mmap.mmap(-1, size, tagname=...)`, which is the usual Python way: when
the mapping does not exist yet it *creates* one - here, read-only. A game
started after that opens the watcher's mapping instead of making its own and
then cannot write to it, so its telemetry is dead until the watcher lets go.
The watcher starts an adapter as soon as the game's process appears, which is
exactly the moment the game has not made its mapping yet. So this opens with
`OpenFileMappingW`, which only ever finds an existing mapping, maps it with
`FILE_MAP_READ`, and reports "not there yet" otherwise; the adapters try again
a moment later.

Nothing here can write: the view is mapped read-only and is only ever copied
out of with `ctypes.string_at`.

Two games are read here as well; every car on track is in their pages.

RaceRoom Racing Experience - `$R3E`
    Sector3's `r3e.h` (github.com/sector3studios/r3e-api, sample-c/src/r3e.h,
    version 3.6), `#pragma pack(1)`. The header says where the drivers start
    and how big one is (`all_drivers_offset`, `driver_data_size`), so the
    per-driver part - name, slot, place, world position (float x y z, m),
    completed laps, current lap time, finish status, pit lane, speed - is read
    by those. The player's own fields (best and previous lap, race position,
    incident points, gear, pedals, track and layout names) are read at their
    3.6 offsets only when `all_drivers_offset` is the 2008 that layout puts it
    at. Y is taken as up; the header does not say.

Automobilista 2 / Project CARS 2 - `$pcars2$`
    SMS's `SharedMemory.h` (shipped in the game's Support\\SharedMemory folder;
    AMS2: Options > System > Shared Memory "Project CARS 2"), `#pragma pack(4)`.
    `mParticipantInfo[64]` from byte 28, 100 bytes each: mIsActive, mName[64],
    mWorldPosition[3] (m, Y up), mCurrentLapDistance, mRacePosition,
    mLapsCompleted, mCurrentLap, mCurrentSector; mViewedParticipantIndex (20)
    is the player's car; per participant mLastLapTimes (9200), mRaceStates
    (9520), mSpeeds (10800); the player's mBestLapTime (6716), mLastLapTime
    (6720), mSpeed (6848), mBrake, mThrottle, mGear; mSequenceNumber (7320) is
    odd while the game writes. The offsets were worked out from two
    independent copies of the header (the game's own copy is not installed
    here), which agree.
"""

from __future__ import annotations

import ctypes
import math
import struct
import sys
import time
from dataclasses import dataclass

FILE_MAP_READ = 0x0004


class _MBI(ctypes.Structure):
    _fields_ = [
        ("BaseAddress", ctypes.c_void_p),
        ("AllocationBase", ctypes.c_void_p),
        ("AllocationProtect", ctypes.c_ulong),
        ("PartitionId", ctypes.c_ushort),
        ("RegionSize", ctypes.c_size_t),
        ("State", ctypes.c_ulong),
        ("Protect", ctypes.c_ulong),
        ("Type", ctypes.c_ulong),
    ]


def _kernel32():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenFileMappingW.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_wchar_p]
    k.OpenFileMappingW.restype = ctypes.c_void_p
    k.MapViewOfFile.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_size_t]
    k.MapViewOfFile.restype = ctypes.c_void_p
    k.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
    k.UnmapViewOfFile.restype = ctypes.c_int
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    k.CloseHandle.restype = ctypes.c_int
    k.VirtualQuery.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MBI), ctypes.c_size_t]
    k.VirtualQuery.restype = ctypes.c_size_t
    return k


class SharedView:
    """A read-only view of a mapping that already exists. `open()` returns None while it does not."""

    def __init__(self, name: str, handle: int, address: int, size: int):
        self.name = name
        self._handle = handle
        self._address = address
        self.size = size

    @classmethod
    def open(cls, name: str) -> "SharedView | None":
        if sys.platform != "win32":
            return None
        k = _kernel32()
        handle = k.OpenFileMappingW(FILE_MAP_READ, 0, name)
        if not handle:
            return None
        address = k.MapViewOfFile(handle, FILE_MAP_READ, 0, 0, 0)
        if not address:
            k.CloseHandle(handle)
            return None
        info = _MBI()
        size = info.RegionSize if k.VirtualQuery(address, ctypes.byref(info), ctypes.sizeof(info)) else 0
        return cls(name, handle, address, int(size))

    def read(self, offset: int = 0, length: int | None = None) -> bytes:
        """A copy of `length` bytes from `offset`, cut at the end of the view."""
        if self._address is None or offset >= self.size:
            return b""
        n = self.size - offset if length is None else max(0, min(length, self.size - offset))
        return ctypes.string_at(self._address + offset, n)

    def close(self) -> None:
        if self._address is None:
            return
        k = _kernel32()
        k.UnmapViewOfFile(self._address)
        k.CloseHandle(self._handle)
        self._address = None

    def __enter__(self) -> "SharedView":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


# ------------------------------------------------------------------ every car on track


@dataclass
class Car:
    """One car as a game's page has it, in common terms."""

    key: str
    name: str
    x: float
    y: float
    z: float
    laps: int
    own: bool = False
    speed: float | None = None
    place: int | None = None
    #: The game's time for the lap just completed, when it keeps one per car.
    last_lap: float | None = None
    #: The running time of the current lap, when that is all the game keeps.
    lap_clock: float | None = None
    finished: bool = False
    pit: bool | None = None


@dataclass
class Snapshot:
    live: bool
    cars: list[Car]
    track: str | None = None
    session: str | None = None
    best_lap: float | None = None
    own_last_lap: float | None = None
    gear: int | None = None
    throttle: float | None = None
    brake: float | None = None
    incidents: int | None = None
    position: int | None = None


def _text(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()


def _num(v: float) -> float | None:
    return v if isinstance(v, float) and math.isfinite(v) and v > 0 else None


# Imported here, below the view: `racing` uses the view, and this uses `racing`.
from ..session import Axes  # noqa: E402
from .base import Adapter, Context  # noqa: E402
from .racing import Driving  # noqa: E402


@dataclass
class ShmRaceAdapter(Adapter):
    """A sim whose shared memory has every car: positions at 10 Hz, laps, finishes."""

    tagname: str = ""
    hz: float = 10.0

    def missing(self) -> str | None:
        if sys.platform != "win32":
            return "shared memory telemetry is Windows only"
        return None

    def start(self, ctx: Context) -> None:
        self._drive = Driving(self, ctx, hz=self.hz)
        self._view: SharedView | None = None
        self._tried = 0.0
        self._laps: dict[str, int] = {}
        self._clock: dict[str, float] = {}
        self._finished: dict[str, bool] = {}
        self._pit: dict[str, bool] = {}
        self._session: str | None = None
        self._runs = 0

    def snapshot(self, view: SharedView) -> Snapshot | None:  # pragma: no cover - per game
        raise NotImplementedError

    def poll(self, ctx: Context) -> None:
        if self._view is None:
            now = time.monotonic()
            if now - self._tried < 2.0:
                return
            self._tried = now
            self._view = SharedView.open(self.tagname)
            if self._view is None:
                return
        try:
            snap = self.snapshot(self._view)
        except Exception:  # noqa: BLE001
            self._drive.count("bad_reads")
            return
        if snap is None or not snap.live:
            return
        try:
            self.step(snap)
        except Exception:  # noqa: BLE001
            self._drive.count("bad_reads")

    def step(self, snap: Snapshot) -> None:
        d = self._drive
        if snap.session != self._session:
            self._session = snap.session
            self._runs += 1
            self._laps.clear()
            self._clock.clear()
            self._finished.clear()
            self._pit.clear()
            d.open(track=snap.track)
            d.event("session_start", player=None, where=False, session=str(self._runs), track=snap.track, kind=snap.session)
        sid = str(self._runs)
        s = d.open()
        for car in snap.cars:
            if not all(math.isfinite(v) for v in (car.x, car.y, car.z)) or car.x == car.y == car.z == 0:
                continue
            extra = {"gear": snap.gear, "throttle": snap.throttle, "brake": snap.brake} if car.own else {}
            d.sample(
                car.x, car.y, car.z,
                player=car.name, key=car.key, own=car.own, speed=car.speed, session=sid,
                own_car=1 if car.own else 0, lap=car.laps + 1, race_position=car.place, **extra,
            )
            before = self._laps.get(car.key)
            if before is not None and car.laps == before + 1:
                seconds = snap.own_last_lap if car.own and snap.own_last_lap else car.last_lap
                if seconds is None and car.lap_clock is not None:
                    # The game keeps only the running lap time: its last value before it went back.
                    seconds = self._clock.get(car.key)
                d.lap(seconds, player=car.name, key=car.key, own=car.own, session=sid, lap=before + 1,
                      race_position=car.place, own_car=1 if car.own else 0)
            self._laps[car.key] = car.laps
            if car.lap_clock is not None and car.lap_clock > 0:
                self._clock[car.key] = car.lap_clock
            if car.finished and self._finished.get(car.key) is False:
                d.event("finish", player=car.name, key=car.key, session=sid, race_position=car.place,
                        own_car=1 if car.own else 0)
                if car.own:
                    s.stat("finishes", 1, "add")
            self._finished[car.key] = car.finished
            if car.own and car.pit is not None:
                if self._pit.get(car.key) is not None and car.pit != self._pit[car.key]:
                    d.event("pit_in" if car.pit else "pit_out", player=car.name, key=car.key, session=sid)
                    if car.pit:
                        s.stat("pit_stops", 1, "add")
                self._pit[car.key] = car.pit
        if snap.best_lap:
            s.stat("best_lap", round(snap.best_lap, 3), "min")
        if snap.incidents is not None and snap.incidents >= 0:
            s.stat("incidents", snap.incidents)
        if snap.position:
            s.stat("race_position", snap.position)

    def stop(self, ctx: Context) -> None:
        if self._view is not None:
            self._view.close()
            self._view = None
        self._drive.close()


# ------------------------------------------------------------------ RaceRoom

R3E_SIZE = 44024
#: Where 3.6 puts `num_cars`; the player's own fields are read only when the page agrees.
R3E_DRIVERS_AT_36 = 2008
R3E_FINISHED = 1
R3E_SESSIONS = {0: "practice", 1: "qualify", 2: "race", 3: "warmup"}


@dataclass
class RaceRoomAdapter(ShmRaceAdapter):
    def snapshot(self, view: SharedView) -> Snapshot | None:
        raw = view.read(0, R3E_SIZE + 4096)
        if len(raw) < 64:
            return None
        major, _minor, drivers_at, driver_size = struct.unpack_from("<4i", raw, 0)
        paused, menus, replay = struct.unpack_from("<3i", raw, 20)
        if major != 3 or driver_size < 256 or drivers_at <= 0:
            return None
        if menus > 0 or replay > 0 or paused > 0:
            return Snapshot(live=False, cars=[])
        known = drivers_at == R3E_DRIVERS_AT_36
        n = struct.unpack_from("<i", raw, drivers_at)[0]
        own_slot = struct.unpack_from("<i", raw, 1196 + 92)[0] if known else None
        cars = []
        for i in range(max(0, min(n, 128))):
            at = drivers_at + 4 + i * driver_size
            if at + 256 > len(raw):
                break
            slot = struct.unpack_from("<i", raw, at + 92)[0]
            x, y, z = struct.unpack_from("<3f", raw, at + 148)
            cars.append(Car(
                key=str(slot),
                name=_text(raw[at : at + 64]) or f"car {slot}",
                x=x, y=y, z=z,
                laps=struct.unpack_from("<i", raw, at + 164)[0],
                own=own_slot is not None and slot == own_slot,
                speed=struct.unpack_from("<f", raw, at + 252)[0],
                place=struct.unpack_from("<i", raw, at + 132)[0] or None,
                lap_clock=struct.unpack_from("<f", raw, at + 172)[0],
                finished=struct.unpack_from("<i", raw, at + 128)[0] == R3E_FINISHED,
                pit=struct.unpack_from("<i", raw, at + 224)[0] > 0,
            ))
        snap = Snapshot(live=True, cars=cars)
        if known:
            track = _text(raw[600:664])
            layout = _text(raw[664:728])
            kind = struct.unpack_from("<i", raw, 780)[0]
            snap.track = f"{track} {layout}".strip() or None
            snap.session = f"{track}|{layout}|{R3E_SESSIONS.get(kind, kind)}"
            snap.best_lap = _num(struct.unpack_from("<f", raw, 1068)[0])
            snap.own_last_lap = _num(struct.unpack_from("<f", raw, 1084)[0])
            snap.position = struct.unpack_from("<i", raw, 988)[0] or None
            snap.incidents = struct.unpack_from("<i", raw, 1172)[0]
            snap.gear = struct.unpack_from("<i", raw, 1408)[0]
            snap.throttle = struct.unpack_from("<f", raw, 1500)[0]
            snap.brake = struct.unpack_from("<f", raw, 1508)[0]
        return snap


# ------------------------------------------------------------------ Automobilista 2 / Project CARS 2

PC2_READ = 11312
PC2_PARTICIPANTS = 28
PC2_PARTICIPANT = 100
PC2_SEQUENCE = 7320
PC2_PLAYING = 2  # mGameState GAME_INGAME_PLAYING
PC2_FINISHED = 3  # mRaceStates RACESTATE_FINISHED
PC2_SESSIONS = {1: "practice", 2: "test", 3: "qualify", 4: "formation lap", 5: "race", 6: "time attack"}


@dataclass
class PCars2Adapter(ShmRaceAdapter):
    def snapshot(self, view: SharedView) -> Snapshot | None:
        for _ in range(4):
            raw = view.read(0, PC2_READ)
            if len(raw) < PC2_READ:
                return None
            seq = struct.unpack_from("<I", raw, PC2_SEQUENCE)[0]
            # Odd while the game writes; a copy is good when it was even and has not moved since.
            if seq % 2 == 0 and struct.unpack_from("<I", view.read(PC2_SEQUENCE, 4), 0)[0] == seq:
                break
        else:
            return None
        _version, _build, game_state, session_state = struct.unpack_from("<4I", raw, 0)
        if game_state != PC2_PLAYING:
            return Snapshot(live=False, cars=[])
        viewed, count = struct.unpack_from("<2i", raw, 20)
        cars = []
        for i in range(max(0, min(count, 64))):
            at = PC2_PARTICIPANTS + i * PC2_PARTICIPANT
            if not raw[at]:
                continue
            x, y, z = struct.unpack_from("<3f", raw, at + 68)
            race_position, laps = struct.unpack_from("<2I", raw, at + 84)
            cars.append(Car(
                key=str(i),
                name=_text(raw[at + 1 : at + 65]) or f"car {i}",
                x=x, y=y, z=z,
                laps=laps,
                own=i == viewed,
                speed=struct.unpack_from("<f", raw, 10800 + 4 * i)[0],
                place=race_position or None,
                last_lap=_num(struct.unpack_from("<f", raw, 9200 + 4 * i)[0]),
                finished=struct.unpack_from("<I", raw, 9520 + 4 * i)[0] == PC2_FINISHED,
            ))
        track = f"{_text(raw[6576:6640])} {_text(raw[6640:6704])}".strip() or None
        snap = Snapshot(
            live=True,
            cars=cars,
            track=track,
            session=f"{track}|{PC2_SESSIONS.get(session_state, session_state)}",
            best_lap=_num(struct.unpack_from("<f", raw, 6716)[0]),
            own_last_lap=_num(struct.unpack_from("<f", raw, 6720)[0]),
            gear=struct.unpack_from("<i", raw, 6876)[0],
            throttle=struct.unpack_from("<f", raw, 6864)[0],
            brake=struct.unpack_from("<f", raw, 6860)[0],
        )
        own = next((c for c in cars if c.own), None)
        if own is not None:
            snap.position = own.place
            own.speed = struct.unpack_from("<f", raw, 6848)[0]
        return snap


_RECORDS = (
    "every car's position (world XYZ, m) at 10 Hz by driver name, own car marked, with speed and race position; "
    "own gear, throttle, brake",
    "session start with the track the game names; laps per car with lap time; finishes; own pit stops",
    "teleports (back to the pits, a restart) so routes break there",
    "own laps, best lap, race position, distance, top speed",
)
_PC2_VERIFIED = (
    "tested on a page built in the test; offsets from two independent copies of SharedMemory.h that agree, "
    "not from the copy the game ships (not installed here); untested against the game"
)

ADAPTERS = [
    RaceRoomAdapter(
        id="raceroom",
        game="RaceRoom Racing Experience",
        exes=("RRRE64.exe", "RRRE.exe"),
        kind="telemetry",
        multiplayer=True,
        tagname="$R3E",
        source="shared memory $R3E (Sector3's r3e.h, version 3)",
        records=_RECORDS + ("own incident points",),
        enable="",
        axes=Axes(up="y", units="metres; Y up assumed (r3e.h does not say)"),
        verified=(
            "tested on a page built in the test from r3e.h 3.6 (offsets worked out from the official header); "
            "untested against the game; other cars' lap times are the last running lap time seen (to about 0.1 s)"
        ),
    ),
    PCars2Adapter(
        id="automobilista-2",
        game="Automobilista 2",
        exes=("AMS2AVX.exe", "AMS2.exe"),
        kind="telemetry",
        multiplayer=True,
        tagname="$pcars2$",
        source="shared memory $pcars2$ (SMS SharedMemory.h layout)",
        records=_RECORDS,
        enable='In the game: Options > System > Shared Memory: "Project CARS 2"',
        axes=Axes(up="y", units="metres"),
        verified=_PC2_VERIFIED,
    ),
    PCars2Adapter(
        id="project-cars-2",
        game="Project CARS 2",
        exes=("pCARS2AVX.exe", "pCARS2.exe"),
        kind="telemetry",
        multiplayer=True,
        tagname="$pcars2$",
        source="shared memory $pcars2$ (SMS SharedMemory.h)",
        records=_RECORDS,
        enable='In the game: Options > System > Shared Memory: "Project CARS 2"',
        axes=Axes(up="y", units="metres"),
        verified=_PC2_VERIFIED,
    ),
]
