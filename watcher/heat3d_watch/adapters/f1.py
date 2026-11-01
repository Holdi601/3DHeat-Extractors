"""
EA SPORTS F1 23, F1 24 and F1 25: the official UDP telemetry, every car on track.

The games send it themselves (Settings > Telemetry Settings, or
`Documents/My Games/<F1 xx>/hardwaresettings/hardware_settings_config.xml`,
`<udp enabled="true" ip="127.0.0.1" port="20777" sendRate="20" format="2025" ...>`).
Little-endian, packed, a 29-byte header on every packet (the same in the 2023,
2024 and 2025 formats):

    uint16 packetFormat, uint8 gameYear, gameMajorVersion, gameMinorVersion,
    packetVersion, packetId, uint64 sessionUID, float sessionTime,
    uint32 frameIdentifier, overallFrameIdentifier, uint8 playerCarIndex,
    secondaryPlayerCarIndex

What is read, by packet id:

    0 Motion        CarMotionData[22], 60 bytes each: world position x y z
                    (metres; Y is up), world velocity, forward and right
                    vectors (int16 / 32767), g-forces, yaw pitch roll
    1 Session       weather, temperatures, total laps, track length, session
                    type, track id, ... network game
    2 Lap Data      LapData[22]: last lap time, lap distance, race position,
                    current lap number, pit stops, penalties, grid position,
                    driver and result status. 50 bytes per car in 2023; 57 in
                    2024 and 2025, which added the minutes parts of the deltas
                    and the speed trap.
    3 Event         a four-letter code and a union of details: SSTA SEND FTLP
                    RTMT DRSE DRSD TMPT CHQF RCWN PENA SPTP STLG LGOT DTSV SGSV
                    FLBK BUTN RDFL OVTK SCAR COLL
    4 Participants  ParticipantData[22]: AI or human, driver, team, race
                    number, name (48 bytes in 2023 and 2024, 32 in 2025; 58,
                    60 and 57 bytes per car)
    6 Car Telemetry CarTelemetryData[22], 60 bytes: speed km/h, throttle,
                    steer, brake, clutch, gear, rpm ...

Every car's position is in the Motion packet - online too, where the game
itself hands it out (only fuel, ERS, damage and tyre-set data are held back
for players whose "Your Telemetry" is restricted). Each car is recorded at
about 10 Hz under its participant name ("car N" until the names arrive), the
team by name, and the player's own car marked `own`.

Sources: EA's "F1 25 UDP Specification" (forums.ea.com, F1 games info hub),
the F1 23 and F1 24 specifications; the same tables mirrored in
MacManley/f1-25-udp (MIT). Packets in tests are built from these layouts.
"""

from __future__ import annotations

import logging
import struct
import time
from dataclasses import dataclass
from typing import Any

from ..session import Axes
from .base import Adapter, Context
from .racing import Driving, UdpPort, backup_once, documents, set_xml_attributes

log = logging.getLogger("heat3d_watch")

HEADER = struct.Struct("<HBBBBBQfIIBB")
HEADER_SIZE = HEADER.size  # 29
CARS = 22

MOTION, SESSION, LAP_DATA, EVENT, PARTICIPANTS, CAR_TELEMETRY = 0, 1, 2, 3, 4, 6

CAR_MOTION = struct.Struct("<6f6h6f")  # 60
CAR_TELEMETRY_DATA = struct.Struct("<HfffBbHBBH4H4B4BH4f4B")  # 60

#: LapData per format: (struct, field names). Only the fields used are named.
_LAP_2023 = (
    struct.Struct("<IIHBHBHHfffBBBBBBBBBBBBBBBHHB"),  # 50
    ("last_lap_ms", "current_lap_ms", None, None, None, None, None, None, "lap_distance", "total_distance", None,
     "position", "lap", "pit_status", "pit_stops", "sector", "lap_invalid", "penalties", "warnings", None, None,
     None, "grid_position", "driver_status", "result_status", None, None, None, None),
)
_LAP_2024 = (
    struct.Struct("<IIHBHBHBHBfffBBBBBBBBBBBBBBBHHBfB"),  # 57
    ("last_lap_ms", "current_lap_ms", None, None, None, None, None, None, None, None, "lap_distance",
     "total_distance", None, "position", "lap", "pit_status", "pit_stops", "sector", "lap_invalid", "penalties",
     "warnings", None, None, None, "grid_position", "driver_status", "result_status", None, None, None, None,
     None, None),
)
#: Participants per format: (bytes per car, name length).
_PARTICIPANT = {2023: (58, 48), 2024: (60, 48), 2025: (57, 32)}

SESSION_HEAD = struct.Struct("<BbbBHBbBHHBBBBBB")  # from byte 29: weather .. numMarshalZones
NETWORK_GAME = 154

#: Team ids 0-9 by format year; other ids are said by number.
TEAMS = {
    2023: ("Mercedes", "Ferrari", "Red Bull Racing", "Williams", "Aston Martin", "Alpine", "Alpha Tauri", "Haas",
           "McLaren", "Alfa Romeo"),
    2024: ("Mercedes", "Ferrari", "Red Bull Racing", "Williams", "Aston Martin", "Alpine", "RB", "Haas", "McLaren",
           "Sauber"),
}
TEAMS[2025] = TEAMS[2024]

#: The spec's track ids (public circuit names).
TRACKS = {
    0: "Melbourne", 1: "Paul Ricard", 2: "Shanghai", 3: "Sakhir (Bahrain)", 4: "Catalunya", 5: "Monaco",
    6: "Montreal", 7: "Silverstone", 8: "Hockenheim", 9: "Hungaroring", 10: "Spa", 11: "Monza", 12: "Singapore",
    13: "Suzuka", 14: "Abu Dhabi", 15: "Texas", 16: "Brazil", 17: "Austria", 18: "Sochi", 19: "Mexico",
    20: "Baku (Azerbaijan)", 21: "Sakhir Short", 22: "Silverstone Short", 23: "Texas Short", 24: "Suzuka Short",
    25: "Hanoi", 26: "Zandvoort", 27: "Imola", 28: "Portimao", 29: "Jeddah", 30: "Miami", 31: "Las Vegas",
    32: "Losail", 39: "Silverstone (Reverse)", 40: "Austria (Reverse)", 41: "Zandvoort (Reverse)",
}
SESSION_TYPES = {
    0: "unknown", 1: "practice 1", 2: "practice 2", 3: "practice 3", 4: "short practice", 5: "qualifying 1",
    6: "qualifying 2", 7: "qualifying 3", 8: "short qualifying", 9: "one-shot qualifying", 10: "sprint shootout 1",
    11: "sprint shootout 2", 12: "sprint shootout 3", 13: "short sprint shootout", 14: "one-shot sprint shootout",
    15: "race", 16: "race 2", 17: "race 3", 18: "time trial",
}
#: LapData m_resultStatus.
FINISHED = 3
OUT = {4: "did not finish", 5: "disqualified", 6: "not classified", 7: "retired"}


def year_of(packet_format: int) -> int | None:
    """The layout a packet format uses; a newer format is read as the newest known, sizes permitting."""
    if packet_format < 2023:
        return None
    return min(packet_format, 2025)


def header(packet: bytes) -> dict | None:
    if len(packet) < HEADER_SIZE:
        return None
    v = HEADER.unpack_from(packet, 0)
    return {
        "format": v[0], "game_year": v[1], "packet_version": v[4], "id": v[5], "session_uid": v[6],
        "session_time": v[7], "frame": v[8], "player": v[10],
    }


def motion(packet: bytes) -> list[tuple[float, float, float, tuple[float, float, float]]]:
    """Per car: world x, y, z (m) and the world velocity (m/s)."""
    out = []
    for i in range(CARS):
        v = CAR_MOTION.unpack_from(packet, HEADER_SIZE + i * CAR_MOTION.size)
        out.append((v[0], v[1], v[2], (v[3], v[4], v[5])))
    return out


def lap_data(packet: bytes, year: int) -> list[dict]:
    layout, names = _LAP_2023 if year == 2023 else _LAP_2024
    out = []
    for i in range(CARS):
        v = layout.unpack_from(packet, HEADER_SIZE + i * layout.size)
        out.append({n: x for n, x in zip(names, v) if n})
    return out


def car_telemetry(packet: bytes) -> list[dict]:
    out = []
    for i in range(CARS):
        v = CAR_TELEMETRY_DATA.unpack_from(packet, HEADER_SIZE + i * CAR_TELEMETRY_DATA.size)
        out.append({"speed_kmh": v[0], "throttle": v[1], "steer": v[2], "brake": v[3], "gear": v[5], "rpm": v[6]})
    return out


def participants(packet: bytes, year: int) -> tuple[int, list[dict]]:
    size, name_len = _PARTICIPANT[year]
    active = packet[HEADER_SIZE]
    out = []
    for i in range(CARS):
        at = HEADER_SIZE + 1 + i * size
        raw = packet[at + 7 : at + 7 + name_len].split(b"\x00", 1)[0]
        out.append({
            "ai": packet[at],
            "driver_id": packet[at + 1],
            "team_id": packet[at + 3],
            "race_number": packet[at + 5],
            "name": raw.decode("utf-8", errors="replace").strip(),
        })
    return active, out


def session_info(packet: bytes) -> dict:
    v = SESSION_HEAD.unpack_from(packet, HEADER_SIZE)
    return {
        "weather": v[0], "track_temperature": v[1], "air_temperature": v[2], "total_laps": v[3],
        "track_length": v[4], "session_type": v[5], "track_id": v[6], "formula": v[7],
        "network_game": packet[NETWORK_GAME] if len(packet) > NETWORK_GAME else None,
    }


def event(packet: bytes, year: int) -> tuple[str, dict]:
    """The event's code and its details, by the union member the code selects."""
    code = packet[HEADER_SIZE : HEADER_SIZE + 4].decode("ascii", errors="replace")
    at = HEADER_SIZE + 4
    b = packet[at:] + bytes(16)
    d: dict[str, Any] = {}
    if code == "FTLP":
        d = {"vehicle": b[0], "lap_time": struct.unpack_from("<f", b, 1)[0]}
    elif code == "RTMT":
        d = {"vehicle": b[0]}
        if year >= 2025:
            d["reason"] = b[1]
    elif code in ("TMPT", "RCWN", "DTSV"):
        d = {"vehicle": b[0]}
    elif code == "SGSV":
        d = {"vehicle": b[0]}
        if year >= 2025:
            d["stop_time"] = struct.unpack_from("<f", b, 1)[0]
    elif code == "DRSD":
        d = {"reason": b[0]} if year >= 2025 else {}
    elif code == "PENA":
        d = dict(zip(("penalty_type", "infringement", "vehicle", "other_vehicle", "time", "lap", "places_gained"), b[:7]))
    elif code == "SPTP":
        d = {"vehicle": b[0], "speed_kmh": struct.unpack_from("<f", b, 1)[0], "overall_fastest": b[5], "driver_fastest": b[6]}
    elif code == "STLG":
        d = {"lights": b[0]}
    elif code == "FLBK":
        frame, t = struct.unpack_from("<If", b, 0)
        d = {"to_frame": frame, "to_session_time": t}
    elif code == "BUTN":
        d = {"buttons": struct.unpack_from("<I", b, 0)[0]}
    elif code == "OVTK":
        d = {"vehicle": b[0], "other_vehicle": b[1]}
    elif code == "SCAR":
        d = {"safety_car_type": b[0], "event_type": b[1]}
    elif code == "COLL":
        d = {"vehicle": b[0], "other_vehicle": b[1]}
    return code, d


#: Event codes to event names. BUTN (button presses) is not kept.
EVENTS = {
    "SSTA": "session_start", "SEND": "session_end", "FTLP": "fastest_lap", "RTMT": "retirement",
    "DRSE": "drs_enabled", "DRSD": "drs_disabled", "TMPT": "teammate_in_pits", "CHQF": "chequered_flag",
    "RCWN": "race_winner", "PENA": "penalty", "SPTP": "speed_trap", "STLG": "start_lights",
    "LGOT": "lights_out", "DTSV": "drive_through_served", "SGSV": "stop_go_served", "FLBK": "flashback",
    "RDFL": "red_flag", "OVTK": "overtake", "SCAR": "safety_car", "COLL": "collision",
}
#: Minimum packet size per id and format year.
SIZES = {
    MOTION: {2023: 1349, 2024: 1349, 2025: 1349},
    SESSION: {2023: 155, 2024: 155, 2025: 155},
    LAP_DATA: {2023: 1131, 2024: 1285, 2025: 1285},
    EVENT: {2023: 45, 2024: 45, 2025: 45},
    PARTICIPANTS: {2023: 1306, 2024: 1350, 2025: 1284},
    CAR_TELEMETRY: {2023: 1352, 2024: 1352, 2025: 1352},
}


@dataclass
class F1Adapter(Adapter):
    """One F1 game's UDP telemetry: every car's position, laps, the race's events."""

    port: int = 20777
    forward: tuple[str, int] | None = None
    hz: float = 10.0
    #: The folder under `Documents/My Games` the game keeps its settings in.
    folder: str = "F1 25"

    def start(self, ctx: Context) -> None:
        self._udp = UdpPort(self.port, forward=self.forward)
        self._drive = Driving(self, ctx, hz=self.hz)
        self._uid: int | None = None
        self._names: list[str] = [f"car {i}" for i in range(CARS)]
        self._teams: list[str | None] = [None] * CARS
        self._laps: list[dict] = [{} for _ in range(CARS)]
        self._lap_no: list[int | None] = [None] * CARS
        self._result: list[int | None] = [None] * CARS
        self._tele: list[dict] = [{} for _ in range(CARS)]
        self._player = 0
        self._flashback = False
        self._year = 2025

    def poll(self, ctx: Context) -> None:
        if not self._udp.try_open(self.id):
            return
        for packet in self._udp.receive():
            try:
                self.feed(packet)
            except Exception:  # noqa: BLE001
                self._drive.count("bad_packets")

    # -- decoding ---------------------------------------------------------------------

    def _who(self, i: int) -> str:
        if 0 <= i < CARS:
            return self._names[i]
        return f"car {i}"

    def _own(self, i: int) -> bool:
        return i == self._player

    def feed(self, packet: bytes) -> None:
        d = self._drive
        h = header(packet)
        year = year_of(h["format"]) if h else None
        if h is None or year is None:
            d.count("unsupported_packets")
            return
        need = SIZES.get(h["id"], {}).get(year)
        if need is None:
            return  # a packet this does not read: setups, status, damage, history ...
        if len(packet) < need:
            d.count("bad_packets")
            return
        self._year = year
        self._player = h["player"]
        if h["session_uid"] != self._uid:
            self._new_session(h)
        kind = h["id"]
        if kind == MOTION:
            self._motion(packet, h)
        elif kind == LAP_DATA:
            self._lap_data(packet, year)
        elif kind == CAR_TELEMETRY:
            self._tele = car_telemetry(packet)
        elif kind == PARTICIPANTS:
            self._participants(packet, year)
        elif kind == SESSION:
            info = session_info(packet)
            s = d.open()
            sid = self._sid()
            if s.meta.get("sessions", {}).get(sid) is None:
                s.meta.setdefault("sessions", {})[sid] = {
                    "track": TRACKS.get(info["track_id"], f"track {info['track_id']}"),
                    "session_type": SESSION_TYPES.get(info["session_type"], str(info["session_type"])),
                    "total_laps": info["total_laps"],
                    "track_length_m": info["track_length"],
                    "online": bool(info["network_game"]),
                }
        elif kind == EVENT:
            self._event(packet, year)

    def _sid(self) -> str:
        return f"{self._uid:016x}" if self._uid is not None else "0"

    def _new_session(self, h: dict) -> None:
        """A new game session (practice, qualifying, race ...): the per-car bookkeeping starts again."""
        if self._uid is not None and h["session_uid"] == 0:
            return
        self._uid = h["session_uid"]
        self._laps = [{} for _ in range(CARS)]
        self._lap_no = [None] * CARS
        self._result = [None] * CARS
        self._drive.open(packet_format=h["format"])
        self._drive.open().stat("sessions", 1, "add")

    def _motion(self, packet: bytes, h: dict) -> None:
        d = self._drive
        sid = self._sid()
        for i, (x, y, z, v) in enumerate(motion(packet)):
            if x == 0 and y == 0 and z == 0:
                continue
            status = self._result[i]
            if status is not None and status < 2:
                continue  # an empty slot: invalid or inactive
            own = self._own(i)
            t = self._tele[i] if i < len(self._tele) else {}
            speed = t["speed_kmh"] / 3.6 if "speed_kmh" in t else (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
            who = self._who(i)
            key = str(i)
            if own and self._flashback:
                # The game went back in time: the car is where it was, not driven there.
                self._flashback = False
                d.event("teleport", player=who, x=x, y=y, z=z, session=sid, reason="flashback", own_car=1)
                m = d.movers.get(key)
                if m is not None:
                    m.x, m.y, m.z, m.emitted = x, y, z, -1e9
            d.sample(
                x, y, z,
                player=who,
                key=key,
                own=own,
                speed=speed,
                clock=h["session_time"],
                team=self._teams[i],
                session=sid,
                car=i,
                own_car=1 if own else 0,
                gear=t.get("gear"),
                throttle=t.get("throttle"),
                brake=t.get("brake"),
                lap=self._lap_no[i],
                race_position=self._laps[i].get("position"),
                game_time=round(h["session_time"], 3),
            )

    def _lap_data(self, packet: bytes, year: int) -> None:
        d = self._drive
        sid = self._sid()
        for i, lap in enumerate(lap_data(packet, year)):
            status = lap["result_status"]
            if status < 2:
                self._result[i] = status  # an empty slot, or a car not yet out
                continue
            own = self._own(i)
            who = self._who(i)
            before = self._lap_no[i]
            if before is not None and lap["lap"] == before + 1:
                seconds = lap["last_lap_ms"] / 1000.0
                d.lap(seconds, player=who, key=str(i), own=own, session=sid, lap=before, team=self._teams[i],
                      race_position=lap["position"], own_car=1 if own else 0)
            self._lap_no[i] = lap["lap"]
            old = self._result[i]
            if status != old:
                if status == FINISHED and old is not None:
                    d.event("finish", player=who, key=str(i), session=sid, team=self._teams[i], race_position=lap["position"],
                            own_car=1 if own else 0)
                    if own:
                        d.open().stat("finishes", 1, "add")
                elif status in OUT and old is not None and old not in OUT:
                    d.event("out", player=who, key=str(i), session=sid, team=self._teams[i], why=OUT[status],
                            own_car=1 if own else 0)
                self._result[i] = status
            self._laps[i] = lap
            if own:
                s = d.open()
                if lap["position"]:
                    s.stat("race_position", lap["position"])
                if lap["grid_position"]:
                    s.stat("grid_position", lap["grid_position"])
                s.stat("pit_stops", lap["pit_stops"])
                s.stat("penalty_seconds", lap["penalties"])

    def _participants(self, packet: bytes, year: int) -> None:
        active, people = participants(packet, year)
        teams = TEAMS.get(year, TEAMS[2025])
        names: list[str] = []
        for i, p in enumerate(people):
            name = p["name"] or f"car {i}"
            if name in names:
                name = f"{name} #{p['race_number']}"
            names.append(name)
            tid = p["team_id"]
            self._teams[i] = teams[tid] if tid < len(teams) else (None if tid == 255 else f"team {tid}")
        if names != self._names:
            self._names = names
            s = self._drive.open()
            s.meta["cars"] = active
            s.meta["drivers"] = {str(i): n for i, n in enumerate(names[:active])}

    def _event(self, packet: bytes, year: int) -> None:
        code, info = event(packet, year)
        name = EVENTS.get(code)
        if name is None:
            return
        d = self._drive
        sid = self._sid()
        attrs: dict[str, Any] = {"session": sid, "code": code}
        who: str | None = None
        key: str | None = None
        vehicle = info.pop("vehicle", None)
        other = info.pop("other_vehicle", None)
        if vehicle is not None and vehicle < CARS:
            who = self._who(vehicle)
            key = str(vehicle)
            attrs["team"] = self._teams[vehicle]
            attrs["own_car"] = 1 if self._own(vehicle) else 0
        if other is not None and other < CARS:
            attrs["other"] = self._who(other)
            m = d.movers.get(str(other))
            if m is not None:
                attrs.update(other_x=round(m.x, 3), other_y=round(m.y, 3), other_z=round(m.z, 3))
        if who is None:
            # Race-wide: at the player's own car, if it is on track.
            mine = d.movers.get(str(self._player))
            if mine is not None:
                attrs.update(x=mine.x, y=mine.y, z=mine.z)
        for k, v in info.items():
            attrs[k] = round(v, 3) if isinstance(v, float) else v
        d.event(name, player=who, key=key, **attrs)
        s = d.open()
        mine = vehicle is not None and self._own(vehicle)
        theirs = other is not None and self._own(other)
        if code == "FLBK":
            self._flashback = True
            s.stat("flashbacks", 1, "add")
        elif code == "OVTK":
            if mine:
                s.stat("overtakes", 1, "add")
            if theirs:
                s.stat("overtaken", 1, "add")
        elif code == "COLL" and (mine or theirs):
            s.stat("collisions", 1, "add")
        elif code == "PENA" and mine:
            s.stat("penalties", 1, "add")
        elif code == "FTLP" and mine:
            s.stat("fastest_laps", 1, "add")
        elif code == "RCWN" and mine:
            s.stat("wins", 1, "add")

    def stop(self, ctx: Context) -> None:
        self._udp.close()
        self._drive.close()

    # -- switching the output on --------------------------------------------------------

    def setup(self, ctx: Context, *, apply: bool) -> list[str]:
        path = documents() / "My Games" / self.folder / "hardwaresettings" / "hardware_settings_config.xml"
        if not path.exists():
            return [f"{path} not found - start the game once first"]
        text = path.read_text(encoding="utf-8")
        new, changes = set_xml_attributes(text, "udp", {"enabled": "true", "ip": "127.0.0.1", "port": str(self.port)})
        changes = [f"{path.name} <udp>: {c}" for c in changes]
        if changes and apply:
            backup = backup_once(path)
            path.write_text(new, encoding="utf-8")
            changes.append(f"the original is kept as {backup.name}")
        return changes


def _f1(year: int, exe: str) -> F1Adapter:
    return F1Adapter(
        id=f"f1-{year}",
        game=f"EA SPORTS F1 {year}",
        exes=(exe,),
        kind="telemetry",
        multiplayer=True,
        folder=f"F1 {year}",
        source=f"UDP telemetry 127.0.0.1:20777 (Documents\\My Games\\F1 {year}\\hardwaresettings\\hardware_settings_config.xml)",
        records=(
            "every car's position (world XYZ, m) at 10 Hz, by driver name, team, own car marked, with speed, gear, throttle, brake",
            "laps per car with the lap time and race position; finishes and retirements",
            "the game's race events: session start and end, fastest lap, penalties, overtakes, collisions, "
            "safety car, red flag, start lights, chequered flag, winner, flashbacks",
            "track, session type, laps and drivers of each session",
            "own laps, best lap, race and grid position, pit stops, penalties, overtakes, collisions, distance, top speed",
        ),
        enable=(
            "In the game: Settings > Telemetry Settings > UDP Telemetry On, IP 127.0.0.1, port 20777 - or "
            f"`python -m heat3d_watch setup f1-{year} --apply` (edits hardware_settings_config.xml, backed up first; "
            "the game must be closed). While another app holds 20777 the watcher leaves the port to it"
        ),
        axes=Axes(up="y", units="metres"),
        verified="tested on packets in the 2023, 2024 and 2025 formats built from EA's published specification, sent over UDP; untested against the game",
    )


ADAPTERS = [_f1(25, "F1_25.exe"), _f1(24, "F1_24.exe"), _f1(23, "F1_23.exe")]
