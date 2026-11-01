"""
DCS World: a small hook script of the watcher's own, sending UDP lines to 127.0.0.1:41993.

DCS runs Lua files in `Saved Games\\DCS*\\Scripts\\Hooks\\` at start; a hook can
read the export functions (`Export.Lo*`) and receive the simulation's callbacks
(`DCS.setUserCallbacks`: onSimulationStart/Frame/Stop, onGameEvent) - see
`<DCS>\\API\\DCS_ControlAPI.html` and https://wiki.hoggitworld.com/view/DCS_Export_Script.
`setup` writes `heat3d-watch.lua` there (its own file: Export.lua and other
tools' hooks are not touched), which sends ten lines a second and one per game
event, never waiting for an answer. The port is 41993 so JoyPro's hook (41992)
and Tacview (42674) keep working beside it. The lines (fields split by ';'):

    hello;v=1                                       the hook was loaded
    start;mission=..;map=..;me=<player id>;name=..;mp=0|1
    p;t=<model s>;x=;y=;z=;lat=;lon=;alt=;hdg=;pitch=;bank=;ias=;agl=;g=;type=
    ev;t=..;name=<onGameEvent name>;args=a|b|c      every onGameEvent
    noexport;t=..                                   the server forbids ownship export
    stop;t=..                                       the mission ended

`p` is LoGetSelfData: Position x/y/z - DCS's own world metres, x north, y up,
z east - plus LatLongAlt (degrees, m), Heading/Pitch/Bank (radians), with
LoGetIndicatedAirSpeed (m/s), LoGetAltitudeAboveGroundLevel (m) and
LoGetAccelerationUnits().y (g). onGameEvent's arguments, as DCS_ControlAPI
gives them: takeoff/landing (playerID, unit_missionID, airdromeName),
crash/eject/pilot_death (playerID, unit_missionID), kill (killerPlayerID,
killerUnitType, killerSide, victimPlayerID, victimUnitType, victimSide,
weaponName), self_kill (playerID), change_slot (playerID, slotID, prevSide),
mission_end (winner, msg).

A multiplayer server can forbid ownship export (LoIsOwnshipExportAllowed is
false); then there are no positions, only events, and the session says so.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..session import Axes, Session
from .base import Adapter, Context
from .xplane import FlightLog, drain, udp_socket

log = logging.getLogger("heat3d_watch")

PORT = 41993
HOOK_NAME = "heat3d-watch.lua"
MARK = "-- heat3d-watch hook"
BACKUP = ".heat3d-backup"

HOOK = r"""-- heat3d-watch hook 1: sends this flight's position and game events to the 3DHeat Watcher
-- (UDP 127.0.0.1:__PORT__). Written by `python -m heat3d_watch setup dcs --apply`; delete this file to stop it.
-- It only reads (Export.Lo*, DCS.*, net.*), never changes the simulation and never waits for an answer.
-- One line per datagram, fields split by ';':
--   hello;v=1                                    the hook was loaded
--   start;mission=..;map=..;me=..;name=..;mp=0|1 a mission began
--   p;t=..;x=..;y=..;z=..;lat=..;lon=..;alt=..;hdg=..;pitch=..;bank=..;ias=..;agl=..;g=..;type=..
--   ev;t=..;name=..;args=a|b|c                   a game event (onGameEvent)
--   noexport;t=..                                the server forbids ownship export
--   stop;t=..                                    the mission ended
local ok, err = pcall(function()
    package.path = package.path .. ';.\\LuaSocket\\?.lua'
    package.cpath = package.cpath .. ';.\\LuaSocket\\?.dll'
    local socket = require('socket')
    local udp = socket.udp()
    udp:settimeout(0)
    local PORT = __PORT__
    local RATE = 0.1
    local last = -1
    local quiet = -100
    local fmt = string.format

    local function send(text)
        pcall(udp.sendto, udp, text, '127.0.0.1', PORT)
    end

    local function call(f, ...)
        if type(f) ~= 'function' then return nil end
        local good, a, b, c = pcall(f, ...)
        if good then return a, b, c end
        return nil
    end

    local function lo(name)
        return call(Export and Export[name])
    end

    local function clean(v)
        if v == nil then return '' end
        return (string.gsub(tostring(v), '[;|\r\n]', ' '))
    end

    local function n(v, digits)
        if type(v) ~= 'number' or v ~= v or v == math.huge or v == -math.huge then return '' end
        return fmt('%.' .. digits .. 'f', v)
    end

    local function now()
        return call(DCS.getModelTime) or 0
    end

    local function frame()
        local t = now()
        if t < last then last = -1 end
        if t - last < RATE then return end
        last = t
        if lo('LoIsOwnshipExportAllowed') == false then
            if t - quiet > 5 then
                quiet = t
                send('noexport;t=' .. n(t, 2))
            end
            return
        end
        local s = lo('LoGetSelfData')
        if type(s) ~= 'table' then return end
        local p = type(s.Position) == 'table' and s.Position or {}
        local g = type(s.LatLongAlt) == 'table' and s.LatLongAlt or {}
        local acc = lo('LoGetAccelerationUnits')
        send(table.concat({
            'p', 't=' .. n(t, 3), 'x=' .. n(p.x, 2), 'y=' .. n(p.y, 2), 'z=' .. n(p.z, 2),
            'lat=' .. n(g.Lat, 7), 'lon=' .. n(g.Long, 7), 'alt=' .. n(g.Alt, 2),
            'hdg=' .. n(s.Heading, 4), 'pitch=' .. n(s.Pitch, 4), 'bank=' .. n(s.Bank, 4),
            'ias=' .. n(lo('LoGetIndicatedAirSpeed'), 2), 'agl=' .. n(lo('LoGetAltitudeAboveGroundLevel'), 2),
            'g=' .. n(type(acc) == 'table' and acc.y or nil, 3), 'type=' .. clean(s.Name),
        }, ';'))
    end

    local callbacks = {}
    function callbacks.onSimulationStart()
        last = -1
        local theatre = ''
        local m = call(DCS.getCurrentMission)
        if type(m) == 'table' and type(m.mission) == 'table' then theatre = m.mission.theatre or '' end
        local me = net and call(net.get_my_player_id)
        local name = me and net and call(net.get_player_info, me, 'name')
        send(table.concat({
            'start', 'mission=' .. clean(call(DCS.getMissionName)), 'map=' .. clean(theatre),
            'me=' .. clean(me), 'name=' .. clean(name), 'mp=' .. (call(DCS.isMultiplayer) and '1' or '0'),
        }, ';'))
    end
    function callbacks.onSimulationFrame()
        pcall(frame)
    end
    function callbacks.onSimulationStop()
        send('stop;t=' .. n(now(), 3))
    end
    function callbacks.onGameEvent(name, ...)
        pcall(function(...)
            local args = {}
            for i = 1, select('#', ...) do args[#args + 1] = clean((select(i, ...))) end
            send('ev;t=' .. n(now(), 3) .. ';name=' .. clean(name) .. ';args=' .. table.concat(args, '|'))
        end, ...)
    end
    DCS.setUserCallbacks(callbacks)
    send('hello;v=1')
end)
if not ok and log and log.write then
    pcall(log.write, 'HEAT3D', log.WARNING, 'heat3d-watch hook not loaded: ' .. tostring(err))
end
"""


#: onGameEvent names whose first argument is the player they happened to.
PLAYER_EVENTS = ("takeoff", "landing", "crash", "eject", "pilot_death", "self_kill", "change_slot", "friendly_fire",
                 "connect", "disconnect")


def hook_lua(port: int = PORT) -> str:
    return HOOK.replace("__PORT__", str(int(port)))


def parse_line(line: str) -> tuple[str, dict[str, str]]:
    """`kind;k=v;k=v` -> (kind, {k: v}). Values keep everything after the first '='."""
    parts = line.strip().split(";")
    fields: dict[str, str] = {}
    for p in parts[1:]:
        k, sep, v = p.partition("=")
        if sep:
            fields[k.strip()] = v
    return parts[0].strip(), fields


def number(fields: dict[str, str], key: str) -> float | None:
    try:
        v = float(fields.get(key, ""))
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def saved_games() -> Path:
    return Path(os.environ.get("USERPROFILE") or Path.home()) / "Saved Games"


@dataclass
class DcsAdapter(Adapter):
    port: int = PORT
    #: Where DCS's `Saved Games\DCS*` folders are; empty: the player's Saved Games.
    saved_games: str = ""
    #: Seconds without a position after which a dead pilot's next position is a new life.
    respawn_gap: float = 3.0
    _sock: Any = None
    _session: Session | None = None
    _log: FlightLog | None = None
    _me: str = ""
    _mp: bool = False
    _dead: bool = False
    _seen: float = 0.0
    _blocked: bool = False

    # -- setup ---------------------------------------------------------------------------

    def folders(self) -> list[Path]:
        root = Path(self.saved_games) if self.saved_games else saved_games()
        try:
            return sorted(d for d in root.glob("DCS*") if d.is_dir())
        except OSError:
            return []

    def setup(self, ctx: Context, *, apply: bool) -> list[str]:
        folders = self.folders()
        if not folders:
            return ["no Saved Games\\DCS* folder found: start DCS once, then run setup again"]
        text = hook_lua(self.port)
        out: list[str] = []
        for d in folders:
            target = d / "Scripts" / "Hooks" / HOOK_NAME
            if target.exists():
                try:
                    old = target.read_text(encoding="utf-8", errors="replace")
                except OSError as e:
                    out.append(f"leave {target}: cannot read it ({e})")
                    continue
                if old == text:
                    continue
                if not old.startswith(MARK):
                    out.append(f"leave {target}: a file of that name that the watcher did not write")
                    continue
                out.append(f"update {target} (restart DCS)")
                if apply:
                    backup = target.with_name(target.name + BACKUP)
                    if not backup.exists():
                        backup.write_text(old, encoding="utf-8")
                    target.write_text(text, encoding="utf-8")
            else:
                out.append(f"add {target}: sends position and game events to 127.0.0.1:{self.port} (restart DCS)")
                if apply:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(text, encoding="utf-8")
        return out

    # -- running -------------------------------------------------------------------------

    def start(self, ctx: Context) -> None:
        self.stop(ctx)
        self._sock = udp_socket(self.port)

    def poll(self, ctx: Context) -> None:
        for packet, _ in drain(self._sock):
            try:
                text = packet.decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001 - decode with replace does not fail, but a datagram never stops us
                continue
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    self.line(ctx, line)
                except (ValueError, KeyError, TypeError):
                    if self._session is not None:
                        self._session.stat("bad_packets", 1, "add")

    def _open(self, ctx: Context, **meta: Any) -> Session:
        self._close(ctx)
        mp = meta.pop("mp", self._mp)
        self._session = ctx.open(self, multiplayer=bool(mp), meta={k: v for k, v in meta.items() if v})
        self._log = FlightLog(self._session, word="life", plural="lives")
        self._dead = False
        self._blocked = False
        return self._session

    def _close(self, ctx: Context) -> None:
        if self._log is not None:
            self._log.end()
        ctx.close(self._session)
        self._session, self._log = None, None

    def line(self, ctx: Context, line: str) -> None:
        kind, f = parse_line(line)
        if kind == "p":
            self._position(ctx, f)
        elif kind == "ev":
            self._event(ctx, f)
        elif kind == "start":
            self._me = f.get("me", "")
            self._mp = f.get("mp") == "1"
            s = self._open(ctx, map=f.get("map"), mission=f.get("mission"), pilot=f.get("name"), mp=self._mp)
            s.event("mission_start", player="me", mission=f.get("mission") or None, map=f.get("map") or None)
        elif kind == "stop":
            if self._session is not None:
                self._session.event("mission_stop", player="me", model_time=number(f, "t"))
            self._close(ctx)
        elif kind == "noexport":
            s = self._session or self._open(ctx)
            if not self._blocked:
                # A server rule, not a fault: the session says why it has no track.
                self._blocked = True
                s.event("no_export", player="me", model_time=number(f, "t"))
                s.stat("export_blocked", 1)
                s.meta["no_export"] = "the server forbids ownship export: events only, no positions"
        elif kind == "hello":
            pass
        else:
            raise ValueError(f"unknown line {kind!r}")

    def _position(self, ctx: Context, f: dict[str, str]) -> None:
        x, y, z = number(f, "x"), number(f, "y"), number(f, "z")
        if x is None or y is None or z is None:
            raise ValueError("position without x/y/z")
        now = time.monotonic()
        gap = now - self._seen
        self._seen = now
        if self._session is None:
            self._open(ctx)
        assert self._log is not None
        if self._dead:
            if gap < self.respawn_gap:
                return  # the wreck, or the view after death: not a new life yet
            self._dead = False
        if not self._log.live:
            self._log.begin((x, y, z), event="spawn", aircraft=f.get("type") or None)
        self._log.fix(
            x,
            y,
            z,
            g=number(f, "g"),
            lat=number(f, "lat"),
            lon=number(f, "lon"),
            alt=number(f, "alt"),
            heading=number(f, "hdg"),
            pitch=number(f, "pitch"),
            bank=number(f, "bank"),
            ias=number(f, "ias"),
            agl=number(f, "agl"),
            model_time=number(f, "t"),
            aircraft=f.get("type") or None,
        )

    def _mine(self, pid: str) -> bool:
        if self._me:
            return pid == self._me
        return not self._mp  # alone in the mission, every player event is the player's

    def _end_life(self, event: str, **attrs: Any) -> None:
        assert self._log is not None
        if self._log.live:
            self._log.mark(event, **attrs)
            self._log.end()
        self._dead = True

    def _event(self, ctx: Context, f: dict[str, str]) -> None:
        name = f.get("name", "")
        args = f.get("args", "").split("|") if f.get("args") else []
        if self._session is None:
            self._open(ctx)
        s, fl = self._session, self._log
        assert s is not None and fl is not None
        t = number(f, "t")
        arg = lambda i: args[i] if len(args) > i else ""  # noqa: E731
        if name in ("takeoff", "landing") and self._mine(arg(0)):
            fl.mark(name, airfield=arg(2) or None, model_time=t)
        elif name == "crash" and self._mine(arg(0)):
            self._end_life("crash", model_time=t)
        elif name == "eject" and self._mine(arg(0)):
            s.stat("ejections", 1, "add")
            self._end_life("eject", model_time=t)
        elif name in ("pilot_death", "self_kill") and self._mine(arg(0)):
            s.stat("deaths", 1, "add")
            if fl.live:
                fl.end("death", model_time=t, how=name)
            else:
                s.event("death", player="me", model_time=t, how=name)
            self._dead = True
        elif name == "kill":
            killer, victim = arg(0), arg(3)
            detail = dict(killer_type=arg(1) or None, victim_type=arg(4) or None, weapon=arg(6) or None, model_time=t)
            if self._mine(killer) and killer != victim:
                s.stat("kills", 1, "add")
                fl.mark("kill", **detail)
            elif self._mine(victim):
                s.stat("shot_down", 1, "add")
                fl.mark("killed", **detail)
            else:
                s.event("kill", killer_id=killer or None, victim_id=victim or None, **detail)
        elif name == "change_slot" and self._mine(arg(0)):
            if fl.live:
                fl.end()
            self._dead = False
            s.event("change_slot", player="me", slot=arg(1) or None, model_time=t)
        elif name in PLAYER_EVENTS and arg(0) not in ("", "-1"):
            # Someone else's: kept under their id, out of the player's own numbers.
            who = "me" if self._mine(arg(0)) else f"player-{arg(0)}"
            s.event(name, player=who, args=args[1:] or None, model_time=t)
        elif name:
            s.event(name, args=args or None, model_time=t)

    def stop(self, ctx: Context) -> None:
        self._close(ctx)
        if self._sock is not None:
            self._sock.close()
            self._sock = None


ADAPTERS = [
    DcsAdapter(
        id="dcs",
        game="DCS World",
        kind="telemetry",
        exes=("DCS.exe",),
        multiplayer=True,
        source=(
            f"the watcher's hook %USERPROFILE%\\Saved Games\\DCS*\\Scripts\\Hooks\\{HOOK_NAME}, "
            f"UDP lines to 127.0.0.1:{PORT}"
        ),
        records=(
            "own position at 10 Hz in DCS world metres (x north, y up, z east), lat/lon/alt as columns",
            "attitude, indicated airspeed, AGL, g, aircraft type",
            "takeoffs and landings with the airfield; crashes, ejections, pilot deaths",
            "kills and being shot down with both unit types and the weapon; other game events as given",
            "lives per mission (spawn ... session_end), kills, deaths, distance, highest altitude, highest g",
        ),
        enable=f"python -m heat3d_watch setup dcs --apply (adds Scripts\\Hooks\\{HOOK_NAME}), then restart DCS",
        interval=0.1,
        axes=Axes(up="y", units="metres (DCS world: x north, y up, z east)"),
        verified=(
            "line format tested with synthetic datagrams; the Lua hook is untested inside DCS "
            "(written from the DCS hook/export API, not run)"
        ),
    )
]
