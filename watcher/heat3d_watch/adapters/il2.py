"""
IL-2 Sturmovik: Great Battles - the game's telemetry UDP output, and its own Tacview recordings.

1. Telemetry ("Telemetry output of IL2 v4.003", revision 1.2, shipped as
   `<game>\\Doc\\UDP_Output\\Il2_TeleOutput.pdf`). Switched on in
   `<game>\\data\\startup.cfg`, section `[KEY = telemetrydevice]`: `addr`,
   `addr1`, ... = "IP[:port]" (several targets at once), `decimation` (50 Hz /
   n), `enable`, `port`. `setup` adds one target, 127.0.0.1:41995 (JoyPro uses
   41994, the game's default is 4322), and switches the output on. Packets,
   little-endian:

       DWORD  Packet_ID 0x54000101
       WORD   Size
       DWORD  Tick (1/50 s)
       BYTE   Indicators_Count, then SIndicator {WORD id, BYTE n, n x 32-bit float}
       BYTE   Events_Count,     then SEvent     {WORD id, BYTE size, size bytes}

   Indicators: 0 ENG_RPM, 1 ENG_MP, 2/3 engine shake, 4 LGEARS_STATE, 5
   LGEARS_PRESS, 6 EAS (m/s), 7 AOA (rad), 8 ACCELERATION (xyz incl. gravity),
   9 COCKPIT_SHAKE, 10 AGL (m), 11 FLAPS, 12 AIR_BRAKES, 13 AOS (rad), 14
   VERTICAL_SPEED (m/s). Events (2-byte packed): 0 SET_FOCUS (STRING: length
   byte + ASCII), 1-3 SETUP_ENG/GUN/LGEAR, 4 DROP_BOMB and 5 ROCKET_LAUNCH
   {float pos[3], float mass, WORD flags}, 6 HIT and 7 DAMAGE {float pos[3],
   float force[3]}, 8 EXPLOSION {float pos[3], float radius}, 9 GUN_FIRE {BYTE
   gun}, 10 SRV_ADDR, 11 SRV_TITLE, 12 SRS_ADDR (STRING), 13 CLIENT_DAT {long
   client, long server client, char name[32]}, 14 CTRL_DAT {long parent, short
   coalition}. Event positions are local to the player's aircraft; the stream
   has no world position at all - so these are events and numbers, not a track.

2. Tacview recording: `tacviewrecord = 1` in `[KEY = track_record]` makes the
   game write `<game>\\data\\Tracks\\*.acmi` (in multiplayer the server decides).
   A finished file is converted: every aircraft, vehicle and ship with its
   positions, spawn and removal (destroyed or gone), takeoffs and landings when
   the recording has them. ACMI 2.x text format, plain or zipped, per
   https://www.tacview.net/documentation/acmi/en/ - "#t" lines move the clock,
   "id,Key=Value,..." lines set properties (hex ids, commas escaped "\\,", a
   trailing "\\" continues the line), "-id" removes an object, object 0 carries
   the globals (ReferenceLongitude/Latitude, RecordingTime, ...) and the events
   ("Event=Destroyed|id|text"). T = Lon|Lat|Alt[|U|V] or
   Lon|Lat|Alt|Roll|Pitch|Yaw[|U|V|Heading], longitude and latitude relative to
   the reference, empty fields unchanged. Positions go into a flat map around
   the reference point: x east, y altitude, z north, metres.
"""

from __future__ import annotations

import io
import json
import logging
import math
import re
import struct
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..session import Axes, Session
from .base import Adapter, Context
from .xplane import TANGENT_AXES, Tangent, drain, steam_libraries, udp_socket

log = logging.getLogger("heat3d_watch")

PORT = 41995
PACKET_ID = 0x54000101
BACKUP = ".heat3d-backup"
#: Install folder names the game has had on Steam.
GAME_FOLDERS = ("IL-2 Sturmovik Battle of Stalingrad", "IL2Series", "IL-2 Sturmovik Great Battles")
EXES = ("Il-2.exe", "IL2Series.exe")

# ---------------------------------------------------------------------------- telemetry packets

INDICATORS = {
    0: "eng_rpm", 1: "eng_mp", 2: "eng_shake_frq", 3: "eng_shake_amp", 4: "lgears_state", 5: "lgears_press",
    6: "eas", 7: "aoa", 8: "acceleration", 9: "cockpit_shake", 10: "agl", 11: "flaps", 12: "air_brakes",
    13: "aos", 14: "vertical_speed",
}
EVENTS = {
    0: "set_focus", 1: "setup_eng", 2: "setup_gun", 3: "setup_lgear", 4: "drop_bomb", 5: "rocket_launch",
    6: "hit", 7: "damage", 8: "explosion", 9: "gun_fire", 10: "srv_addr", 11: "srv_title", 12: "srs_addr",
    13: "client_dat", 14: "ctrl_dat",
}


@dataclass
class Packet:
    tick: int
    indicators: dict[str, tuple[float, ...]]
    events: list[tuple[str, dict[str, Any]]]


def _string(data: bytes) -> str:
    if not data:
        return ""
    n = data[0]
    return data[1 : 1 + n].split(b"\0", 1)[0].decode("ascii", errors="replace")


def decode_event(eid: int, data: bytes) -> tuple[str, dict[str, Any]]:
    """One SEvent's data, by the spec's structures. Unknown ids keep their bytes' length only."""
    name = EVENTS.get(eid, f"event_{eid}")
    if eid in (0, 10, 11, 12):
        return name, {"text": _string(data)}
    if eid in (4, 5):
        x, y, z, mass = struct.unpack_from("<4f", data, 0)
        flags = struct.unpack_from("<H", data, 16)[0] if len(data) >= 18 else 0
        return name, {"local": (x, y, z), "mass": mass, "flags": flags}
    if eid in (6, 7):
        x, y, z, fx, fy, fz = struct.unpack_from("<6f", data, 0)
        return name, {"local": (x, y, z), "force": (fx, fy, fz)}
    if eid == 8:
        x, y, z, r = struct.unpack_from("<4f", data, 0)
        return name, {"local": (x, y, z), "radius": r}
    if eid == 9:
        return name, {"gun": data[0] if data else 0}
    if eid == 13:
        client, server, raw = struct.unpack_from("<ii32s", data, 0)
        player = raw.split(b"\0", 1)[0].decode("utf-8", "replace")
        return name, {"client": client, "server_client": server, "name": player}
    if eid == 14:
        parent, coalition = struct.unpack_from("<ih", data, 0)
        return name, {"parent": parent, "coalition": coalition}
    return name, {"size": len(data)}


def decode_packet(buf: bytes) -> Packet:
    """A telemetry message. Raises ValueError (or struct.error) on anything that is not one, or is cut short."""
    if len(buf) < 12:
        raise ValueError("too short")
    pid, size, tick, ic = struct.unpack_from("<IHIB", buf, 0)
    if pid != PACKET_ID:
        raise ValueError(f"packet id {pid:#x}")
    off = 11
    indicators: dict[str, tuple[float, ...]] = {}
    for _ in range(ic):
        iid, n = struct.unpack_from("<HB", buf, off)
        off += 3
        indicators[INDICATORS.get(iid, f"indicator_{iid}")] = struct.unpack_from(f"<{n}f", buf, off)
        off += 4 * n
    if off >= len(buf):
        raise ValueError("no events count")
    ec = buf[off]
    off += 1
    events = []
    for _ in range(ec):
        eid, esize = struct.unpack_from("<HB", buf, off)
        off += 3
        data = buf[off : off + esize]
        if len(data) < esize:
            raise ValueError("event cut short")
        off += esize
        events.append(decode_event(eid, data))
    return Packet(tick, indicators, events)


# ---------------------------------------------------------------------------- startup.cfg

SECTION = re.compile(r"^\s*\[KEY\s*=\s*([^\]\s]+)\s*\]\s*$", re.I)
END = re.compile(r"^\s*\[END\]\s*$", re.I)
KEYVAL = re.compile(r"^(\s*)([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$")


def _section(lines: list[str], name: str) -> tuple[int, int] | None:
    for i, line in enumerate(lines):
        m = SECTION.match(line)
        if m and m.group(1).lower() == name:
            for j in range(i + 1, len(lines)):
                if END.match(lines[j]):
                    return i, j
            return None
    return None


def _value(v: str) -> str:
    return v.split("//", 1)[0].strip().strip('"').strip()


def edit_telemetry(text: str, port: int = PORT) -> tuple[str, list[str]]:
    """startup.cfg with the watcher as one more telemetry target and the output on; and what changed."""
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    target = f"127.0.0.1:{port}"
    found = _section(lines, "telemetrydevice")
    if found is None:
        block = ["[KEY = telemetrydevice]", f'\taddr = "{target}"', "\tdecimation = 5", "\tenable = true",
                 "\tport = 4322", "[END]"]
        while lines and not lines[-1].strip():
            lines.pop()
        lines += ["", *block, ""]
        return nl.join(lines) + nl, [f"add [KEY = telemetrydevice] sending to {target} at 10 Hz"]
    start, end = found
    changes: list[str] = []
    used, ours, enable, last_addr, indent = set(), False, None, start, "\t"
    for i in range(start + 1, end):
        m = KEYVAL.match(lines[i])
        if not m:
            continue
        indent, key, value = m.group(1) or "\t", m.group(2), _value(m.group(3))
        if key.lower().startswith("addr"):
            used.add(key.lower())
            last_addr = i
            if value == target:
                ours = True
        elif key.lower() == "enable":
            enable = (i, value)
    if not ours:
        key = "addr" if "addr" not in used else next(f"addr{n}" for n in range(1, 100) if f"addr{n}" not in used)
        lines.insert(last_addr + 1, f'{indent}{key} = "{target}"')
        end += 1
        if enable is not None and enable[0] > last_addr:
            enable = (enable[0] + 1, enable[1])
        changes.append(f"add {key} = \"{target}\" to [KEY = telemetrydevice] (other targets stay)")
    if enable is None:
        lines.insert(end, f"{indent}enable = true")
        changes.append("add enable = true to [KEY = telemetrydevice]")
    elif enable[1].lower() not in ("true", "1"):
        lines[enable[0]] = f"{indent}enable = true"
        changes.append(f"switch the telemetry output on (enable = true, was {enable[1]})")
    if not changes:
        return text, []
    return nl.join(lines) + nl, changes


def edit_tacview(text: str) -> tuple[str, list[str]]:
    """startup.cfg with `tacviewrecord = 1` in [KEY = track_record]; and what changed."""
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    found = _section(lines, "track_record")
    if found is None:
        while lines and not lines[-1].strip():
            lines.pop()
        lines += ["", "[KEY = track_record]", "\ttacviewrecord = 1", "[END]", ""]
        return nl.join(lines) + nl, ["add [KEY = track_record] with tacviewrecord = 1"]
    start, end = found
    indent = "\t"
    for i in range(start + 1, end):
        m = KEYVAL.match(lines[i])
        if not m:
            continue
        indent = m.group(1) or "\t"
        if m.group(2).lower() == "tacviewrecord":
            if _value(m.group(3)) in ("1", "true"):
                return text, []
            lines[i] = f"{indent}tacviewrecord = 1"
            return nl.join(lines) + nl, [f"set tacviewrecord = 1 in [KEY = track_record] (was {_value(m.group(3))})"]
    lines.insert(end, f"{indent}tacviewrecord = 1")
    return nl.join(lines) + nl, ["add tacviewrecord = 1 to [KEY = track_record]"]


def game_folders(ctx: Context | None = None, extra: Iterable[str] = (), *, find: bool = True) -> list[Path]:
    """Install folders of the game: any configured; with `find` also the running exe's
    (<game>\\bin\\game\\Il-2.exe) and the Steam libraries'."""
    out: list[Path] = [Path(p) for p in extra]
    if find:
        if ctx is not None and ctx.exe is not None and len(ctx.exe.parents) >= 3:
            out.append(ctx.exe.parents[2])
        for lib in steam_libraries():
            for name in GAME_FOLDERS:
                out.append(lib / "steamapps" / "common" / name)
    seen, found = set(), []
    for p in out:
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            if (p / "data" / "startup.cfg").is_file():
                found.append(p)
        except OSError:
            continue
    return found


def apply_edit(cfg: Path, edit, *, apply: bool) -> list[str]:
    """Run an edit on one startup.cfg: the backup once, the change, what it was."""
    try:
        raw = cfg.read_bytes()
    except OSError as e:
        return [f"leave {cfg}: cannot read it ({e})"]
    text = raw.decode("utf-8", errors="surrogateescape")
    new, changes = edit(text)
    if not changes:
        return []
    if apply:
        backup = cfg.with_name(cfg.name + BACKUP)
        if not backup.exists():
            backup.write_bytes(raw)
        cfg.write_bytes(new.encode("utf-8", errors="surrogateescape"))
    return [f"{cfg}: {c} (restart IL-2)" for c in changes]


# ---------------------------------------------------------------------------- the telemetry adapter


@dataclass
class Il2TelemetryAdapter(Adapter):
    port: int = PORT
    #: Install folders to set up besides the ones found (tests, unusual installs).
    folders: tuple[str, ...] = ()
    #: Also look for the game next to the running exe and in the Steam libraries.
    find: bool = True
    #: Seconds of silence that end a sortie (menus, mission end).
    quiet: float = 5.0
    _sock: Any = None
    _session: Session | None = None
    _sortie: int = 0
    _live: bool = False
    _aircraft: str = ""
    _tick: int | None = None
    _heard: float = 0.0
    _state_tick: int = -10**9
    _team: str | None = None

    def setup(self, ctx: Context, *, apply: bool) -> list[str]:
        games = game_folders(ctx, self.folders, find=self.find)
        if not games:
            return ["no IL-2 Great Battles install with data\\startup.cfg found"]
        out: list[str] = []
        for g in games:
            out += apply_edit(g / "data" / "startup.cfg", lambda t: edit_telemetry(t, self.port), apply=apply)
        return out

    def start(self, ctx: Context) -> None:
        self.stop(ctx)
        self._sock = udp_socket(self.port)

    def poll(self, ctx: Context) -> None:
        now = time.monotonic()
        rounds = 0
        got = False
        for raw, _ in drain(self._sock):
            try:
                p = decode_packet(raw)
            except (ValueError, struct.error, IndexError):
                if self._session is not None:
                    self._session.stat("bad_packets", 1, "add")
                continue
            got = True
            rounds += self._packet(ctx, p)
        if got:
            self._heard = now
        elif self._live and now - self._heard > self.quiet:
            self._end()
        if rounds and self._session is not None:
            # One row per poll for a burst, not one per bullet.
            self._session.event("gun_fire", player="me", session=self._label, rounds=rounds)
            self._session.stat("rounds_fired", rounds, "add")

    @property
    def _label(self) -> str | None:
        return f"sortie-{self._sortie}" if self._sortie else None

    def _end(self) -> None:
        if self._live and self._session is not None:
            self._session.event("session_end", player="me", session=self._label, aircraft=self._aircraft or None)
        self._live = False

    def _begin(self, aircraft: str) -> None:
        assert self._session is not None
        self._end()
        self._sortie += 1
        self._live = True
        self._aircraft = aircraft
        self._session.stat("sorties", self._sortie)
        self._session.event("spawn", player="me", session=self._label, aircraft=aircraft or None, team=self._team)

    def _packet(self, ctx: Context, p: Packet) -> int:
        if self._session is None:
            self._session = ctx.open(self)
        s = self._session
        if self._tick is not None and p.tick + 50 < self._tick:
            self._end()  # the clock went back: a new mission
        self._tick = p.tick
        rounds = 0
        for name, d in p.events:
            if name == "set_focus":
                if not self._live or d["text"] != self._aircraft:
                    self._begin(d["text"])
            elif name == "gun_fire":
                rounds += 1
            elif name in ("drop_bomb", "rocket_launch"):
                rocket = name == "rocket_launch" or bool(d["flags"] & 1)
                s.stat("rockets_fired" if rocket else "bombs_dropped", 1, "add")
                s.stat("ordnance_kg", round(d["mass"], 2), "add")
                s.event(name, player="me", session=self._label, mass=round(d["mass"], 2), **_local(d))
            elif name in ("hit", "damage"):
                s.stat("hits_taken" if name == "hit" else "damage_taken", 1, "add")
                f = d["force"]
                s.event(name, player="me", session=self._label, force=round(math.hypot(*f), 1), **_local(d))
            elif name == "explosion":
                s.stat("explosions", 1, "add")
                s.event(name, player="me", session=self._label, radius=round(d["radius"], 2), **_local(d))
            elif name in ("srv_title", "srv_addr", "srs_addr"):
                if s.meta.get(name.replace("srv_", "server_")) != d["text"]:
                    s.meta[name.replace("srv_", "server_")] = d["text"]
                    s.multiplayer = True
            elif name == "client_dat":
                s.meta["pilot"] = d["name"]
            elif name == "ctrl_dat":
                self._team = str(d["coalition"])
        ind = p.indicators
        eas, agl, vs = _one(ind, "eas"), _one(ind, "agl"), _one(ind, "vertical_speed")
        acc = ind.get("acceleration")
        g = math.sqrt(sum(a * a for a in acc)) / 9.80665 if acc and len(acc) >= 3 else None
        if eas is not None:
            s.stat("max_eas_kmh", round(eas * 3.6, 1), "max")
        if g is not None:
            s.stat("max_g", round(g, 2), "max")
        if agl is not None:
            s.stat("max_agl_m", round(agl, 1), "max")
        if self._live:
            if p.tick - self._state_tick >= 50 or p.tick < self._state_tick:
                # Once a second of game time: a row of the numbers, and that second flown.
                if 0 < p.tick - self._state_tick < 500:
                    s.stat("flight_time", round((p.tick - self._state_tick) / 50.0, 2), "add")
                self._state_tick = p.tick
                s.event(
                    "state",
                    player="me",
                    session=self._label,
                    tick=p.tick,
                    eas=_r(eas),
                    agl=_r(agl),
                    vertical_speed=_r(vs),
                    g=_r(g, 3),
                    rpm=_r(_one(ind, "eng_rpm")),
                )
        return rounds

    def stop(self, ctx: Context) -> None:
        self._end()
        ctx.close(self._session)
        self._session, self._sortie, self._tick, self._aircraft, self._state_tick = None, 0, None, "", -10**9
        if self._sock is not None:
            self._sock.close()
            self._sock = None


def _one(ind: dict[str, tuple[float, ...]], name: str) -> float | None:
    v = ind.get(name)
    return float(v[0]) if v else None


def _r(v: float | None, digits: int = 2) -> float | None:
    return None if v is None or not math.isfinite(v) else round(v, digits)


def _local(d: dict[str, Any]) -> dict[str, float]:
    x, y, z = d["local"]
    return {"local_x": round(x, 3), "local_y": round(y, 3), "local_z": round(z, 3)}


# ---------------------------------------------------------------------------- ACMI


def acmi_text_lines(raw: io.BufferedIOBase | Any) -> Iterator[str]:
    """Logical lines of an ACMI text stream: BOM gone, "\\"-continued lines joined."""
    text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline=None)
    pending = ""
    for line in text:
        line = line.rstrip("\n")
        if line.endswith("\\") and not line.endswith("\\\\"):
            pending += line[:-1] + "\n"
            continue
        yield pending + line
        pending = ""
    if pending:
        yield pending


def split_fields(line: str) -> list[str]:
    """Split on commas not escaped with a backslash; unescape "\\," and "\\\\"."""
    out, cur, i = [], [], 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            cur.append(line[i + 1])
            i += 2
            continue
        if c == ",":
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    out.append("".join(cur))
    return out


#: T's components by count, as slots of lon, lat, alt, roll, pitch, yaw, u, v, heading.
T_SLOTS = {3: (0, 1, 2), 5: (0, 1, 2, 6, 7), 6: (0, 1, 2, 3, 4, 5), 9: (0, 1, 2, 3, 4, 5, 6, 7, 8)}

#: Object type tags that are not drawn as tracks: shells, missiles in flight, debris, markers.
SKIP_TAGS = {"weapon", "projectile", "shrapnel", "misc", "navaid", "flare", "chaff", "decoy", "bullet", "shell"}


@dataclass
class AcmiObject:
    id: str
    props: dict[str, str] = field(default_factory=dict)
    t: list[float | None] = field(default_factory=lambda: [None] * 9)
    seen: bool = False
    last_row: float = -1e18
    pending: dict[str, Any] | None = None
    destroyed: bool = False

    @property
    def skip(self) -> bool:
        tags = {x.strip().lower() for x in self.props.get("Type", "").split("+")}
        return bool(tags & SKIP_TAGS)

    @property
    def who(self) -> str:
        pilot = self.props.get("Pilot")
        if pilot:
            return pilot
        return f"{self.props.get('Name') or 'object'} #{self.id}"


@dataclass
class AcmiRow:
    event: str
    t: float
    lat: float | None
    lon: float | None
    alt: float | None
    attrs: dict[str, Any]


def read_acmi(lines: Iterable[str], *, sample_seconds: float = 1.0) -> tuple[dict[str, str], list[AcmiRow]]:
    """The globals and the rows of a recording: spawns, positions (thinned per object), removals, events."""
    glob: dict[str, str] = {}
    objects: dict[str, AcmiObject] = {}
    rows: list[AcmiRow] = []
    now = 0.0
    ref_lon = ref_lat = 0.0

    def where(o: AcmiObject) -> tuple[float | None, float | None, float | None]:
        lon, lat, alt = o.t[0], o.t[1], o.t[2]
        if lon is None or lat is None:
            return None, None, alt
        return ref_lat + lat, ref_lon + lon, alt

    def describe(o: AcmiObject) -> dict[str, Any]:
        p = o.props
        d: dict[str, Any] = {"player": o.who, "object": o.id, "name": p.get("Name") or None,
                             "type": p.get("Type") or None}
        team = p.get("Coalition") or p.get("Color")
        if team:
            d["team"] = team
        return d

    def row(event: str, o: AcmiObject, **extra: Any) -> None:
        lat, lon, alt = where(o)
        attrs = describe(o)
        for k, v in zip(("roll", "pitch", "yaw", "u", "v", "heading"), o.t[3:]):
            if v is not None:
                attrs[k] = v
        attrs.update(extra)
        rows.append(AcmiRow(event, now, lat, lon, alt, attrs))

    for line in lines:
        if not line or line.startswith("//"):
            continue
        if line[0] == "#":
            try:
                now = float(line[1:].strip())
            except ValueError:
                continue
            continue
        if line[0] == "-":
            o = objects.pop(line[1:].strip().lower(), None)
            if o is not None and o.seen and not o.skip:
                row("death" if o.destroyed else "left", o)
            continue
        if line.startswith(("FileType=", "FileVersion=")):
            k, _, v = line.partition("=")
            glob[k] = v.strip()
            continue
        parts = split_fields(line)
        oid = parts[0].strip().lower()
        props: dict[str, str] = {}
        for p in parts[1:]:
            k, sep, v = p.partition("=")
            if sep:
                props[k.strip()] = v
        if oid == "0":
            for k, v in props.items():
                if k == "Event":
                    bits = v.split("|")
                    kind, ids = bits[0], [b.strip().lower() for b in bits[1:-1]]
                    text = bits[-1] if len(bits) > 1 else ""
                    for i in ids:
                        o = objects.get(i)
                        if o is None:
                            continue
                        if kind == "Destroyed":
                            o.destroyed = True
                        elif kind in ("TakenOff", "Landed") and not o.skip:
                            row("takeoff" if kind == "TakenOff" else "landing", o, text=text or None)
                    if kind in ("Message", "Bookmark") and text:
                        rows.append(AcmiRow(kind.lower(), now, None, None, None, {"text": text}))
                else:
                    glob[k] = v
                    if k == "ReferenceLongitude":
                        ref_lon = _float(v, 0.0)
                    elif k == "ReferenceLatitude":
                        ref_lat = _float(v, 0.0)
            continue
        o = objects.get(oid)
        if o is None:
            o = objects[oid] = AcmiObject(oid)
        moved = False
        for k, v in props.items():
            if k == "T":
                comps = v.split("|")
                slots = T_SLOTS.get(len(comps))
                if slots is None:
                    continue
                for slot, c in zip(slots, comps):
                    if c.strip():
                        o.t[slot] = _float(c, o.t[slot])
                moved = True
            else:
                o.props[k] = v
        if not moved or o.skip or o.t[0] is None or o.t[1] is None:
            continue
        if not o.seen:
            o.seen = True
            o.last_row = now
            row("spawn", o)
            continue
        if now - o.last_row >= sample_seconds:
            o.last_row = now
            row("position", o)
    for o in objects.values():
        if o.seen and not o.skip:
            row("death" if o.destroyed else "end", o)
    return glob, rows


def _float(v: str, default: Any) -> Any:
    try:
        f = float(v)
    except ValueError:
        return default
    return f if math.isfinite(f) else default


def open_acmi(path: Path) -> Iterator[str]:
    """The lines of a .acmi, whether text or a zip holding one."""
    with path.open("rb") as f:
        head = f.read(4)
        f.seek(0)
        if head == b"PK\x03\x04":
            with zipfile.ZipFile(f) as z:
                names = [n for n in z.namelist() if not n.endswith("/")]
                if not names:
                    return
                with z.open(names[0]) as member:
                    yield from acmi_text_lines(member)
        else:
            yield from acmi_text_lines(f)


def _when(glob: dict[str, str]) -> float | None:
    for key in ("RecordingTime", "ReferenceTime"):
        v = glob.get(key)
        if not v:
            continue
        try:
            d = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            continue
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        if key == "ReferenceTime" and d.year < 2013:
            continue  # a mission's historic date, not when it was flown
        return d.timestamp()
    return None


def convert_acmi(ctx: Context, adapter: Adapter, path: Path, *, sample_seconds: float = 1.0) -> Session | None:
    """One finished recording as one session. Returns it closed, or None when it held nothing."""
    glob, rows = read_acmi(open_acmi(path), sample_seconds=sample_seconds)
    if not rows:
        return None
    lat0 = _float(glob.get("ReferenceLatitude", ""), None)
    lon0 = _float(glob.get("ReferenceLongitude", ""), None)
    if lat0 is None or lon0 is None:
        first = next((r for r in rows if r.lat is not None), None)
        lat0, lon0 = (first.lat, first.lon) if first else (0.0, 0.0)
    tangent = Tangent(lat0, lon0)
    t0 = min(r.t for r in rows)
    duration = max(r.t for r in rows) - t0
    when = _when(glob) or (path.stat().st_mtime - duration)
    meta = {"origin": tangent.meta(), "file": path.name}
    for k in ("DataSource", "DataRecorder", "Title", "ReferenceTime", "RecordingTime"):
        if glob.get(k):
            meta[k[0].lower() + k[1:]] = glob[k]
    s = ctx.open(adapter, when=when, meta=meta)
    counts: dict[str, set[str]] = {"spawn": set(), "death": set()}
    for r in rows:
        t = r.t - t0
        ms = (when + t) * 1000.0
        a = dict(r.attrs)
        player, team = a.pop("player", None), a.pop("team", None)
        pos: dict[str, float] = {}
        if r.lat is not None and r.lon is not None and r.alt is not None:
            x, y, z = tangent.local(r.lat, r.lon, r.alt)
            pos = {"x": x, "y": y, "z": z}
            a.update(lat=round(r.lat, 7), lon=round(r.lon, 7), alt=round(r.alt, 2))
        s.event(r.event, t=t, ms=ms, player=player, team=team, **pos, **a)
        if r.event in counts and a.get("object"):
            counts[r.event].add(a["object"])
        if r.event in ("takeoff", "landing"):
            s.stat(f"{r.event}s", 1, "add")
    s.stat("objects", len(counts["spawn"]))
    s.stat("destroyed", len(counts["death"]))
    s.stat("duration_s", round(duration, 1))
    ctx.close(s)
    return s


# ---------------------------------------------------------------------------- the Tacview adapter


@dataclass
class Il2TacviewAdapter(Adapter):
    #: Install folders besides the ones found; their data\Tracks is watched.
    folders: tuple[str, ...] = ()
    #: Also look for the game next to the running exe and in the Steam libraries.
    find: bool = True
    #: A file untouched this long (and the same size as at the last look) is finished.
    settle: float = 30.0
    sample_seconds: float = 1.0
    _seen: dict[str, tuple[int, float]] = field(default_factory=dict)
    _done: dict[str, list[float]] | None = None
    _dirs: list[Path] = field(default_factory=list)
    _dirs_at: float = 0.0

    def setup(self, ctx: Context, *, apply: bool) -> list[str]:
        games = game_folders(ctx, self.folders, find=self.find)
        if not games:
            return ["no IL-2 Great Battles install with data\\startup.cfg found"]
        out: list[str] = []
        for g in games:
            out += apply_edit(g / "data" / "startup.cfg", edit_tacview, apply=apply)
        return out

    def _state_file(self, ctx: Context) -> Path:
        return ctx.data / "state" / f"{self.id}.json"

    def _load(self, ctx: Context) -> dict[str, list[float]]:
        if self._done is None:
            try:
                self._done = json.loads(self._state_file(ctx).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._done = {}
        return self._done

    def _save(self, ctx: Context) -> None:
        p = self._state_file(ctx)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._done or {}, indent=1), encoding="utf-8")

    def track_dirs(self, ctx: Context) -> list[Path]:
        if time.monotonic() - self._dirs_at > 60 or not self._dirs_at:
            extra = list(self.folders) + list(ctx.settings.replay_folders.get(self.id, []))
            dirs = [g / "data" / "Tracks" for g in game_folders(ctx, self.folders, find=self.find)]
            dirs += [Path(p) for p in ctx.settings.replay_folders.get(self.id, [])]
            dirs += [Path(p) / "data" / "Tracks" for p in extra]
            self._dirs = [d for d in dict.fromkeys(dirs) if d.is_dir()]
            self._dirs_at = time.monotonic()
        return self._dirs

    def poll(self, ctx: Context) -> None:
        done = self._load(ctx)
        now = time.time()
        for d in self.track_dirs(ctx):
            try:
                files = sorted(d.glob("*.acmi"))
            except OSError:
                continue
            for f in files:
                key = str(f)
                try:
                    st = f.stat()
                except OSError:
                    continue
                stamp = [st.st_size, st.st_mtime]
                if done.get(key) == stamp:
                    continue
                last = self._seen.get(key)
                self._seen[key] = (st.st_size, st.st_mtime)
                if now - st.st_mtime < self.settle or last != (st.st_size, st.st_mtime):
                    continue  # still being written, or first look
                try:
                    convert_acmi(ctx, self, f, sample_seconds=self.sample_seconds)
                except (OSError, zipfile.BadZipFile, ValueError, UnicodeError) as e:
                    log.warning("%s: %s not converted: %s", self.id, f.name, e)
                done[key] = stamp
                self._save(ctx)
                return  # one file per poll: a long backlog does not hold the thread for minutes


ADAPTERS = [
    Il2TelemetryAdapter(
        id="il2",
        game="IL-2 Sturmovik: Great Battles",
        kind="telemetry",
        exes=EXES,
        multiplayer=True,
        source=(
            f"the game's telemetry UDP output to 127.0.0.1:{PORT} "
            "([KEY = telemetrydevice] in <game>\\data\\startup.cfg)"
        ),
        records=(
            "sorties: the aircraft taken (spawn) and when the output stops (session_end) - no world position: "
            "the game does not send one",
            "bombs and rockets released (mass), hits and damage taken, explosions, gunfire bursts - "
            "positions local to the aircraft",
            "equivalent airspeed, AGL, vertical speed, g, engine RPM once a second",
            "sorties, rounds fired, bombs, rockets, hits taken, highest speed, highest g, flight time; "
            "server title and pilot name online",
        ),
        enable=(
            f"python -m heat3d_watch setup il2 --apply (adds addrN = \"127.0.0.1:{PORT}\", enable = true), "
            "restart IL-2"
        ),
        interval=0.1,
        axes=Axes(up="y", units="none: the telemetry has no world positions"),
        verified=(
            "tested on packets built from the game's own Il2_TeleOutput spec (rev 1.2); startup.cfg edits "
            "checked on a copy of a real one; untested against the running game"
        ),
    ),
    Il2TacviewAdapter(
        id="il2-tacview",
        game="IL-2 Sturmovik: Great Battles",
        kind="replay",
        exes=(),
        multiplayer=True,
        source="the game's own Tacview recordings <game>\\data\\Tracks\\*.acmi (ACMI 2.x text, plain or zipped)",
        records=(
            "every aircraft, vehicle and ship of the recording: positions once a second "
            "(metres east/up/north of the recording's reference point; lat/lon/alt as columns)",
            "spawns, destructions (death) and departures (left), takeoffs and landings when recorded",
            "objects, destroyed, takeoffs, landings and length per recording",
        ),
        enable="python -m heat3d_watch setup il2-tacview --apply (tacviewrecord = 1 in [KEY = track_record])",
        interval=5.0,
        axes=TANGENT_AXES,
        verified="tested on synthetic ACMI 2.2 files (text and zipped); no IL-2 recording on the test machine",
    ),
]
