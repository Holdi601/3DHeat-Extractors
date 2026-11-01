"""
Elite Dangerous: the Player Journal and Status.json the game writes while it runs.

Where: `%USERPROFILE%\\Saved Games\\Frontier Developments\\Elite Dangerous\\`

* `Journal.<yyyy-mm-ddThhmmss>.<part>.log` - one JSON object per line, one
  file per launch (a long session continues in part .02, .03 ...). Every line
  has `timestamp` (UTC, ISO 8601) and `event`. Galaxy positions are the
  `StarPos` [x, y, z] of the system, in light years from Sol, on `FSDJump`,
  `Location` and `CarrierJump`. Elite's galactic frame is y up (towards
  galactic north), z towards the galactic core and x towards galactic longitude
  270 - a left-handed frame, so the viewer mirrors z to draw it the right way
  round.
* `Status.json` - rewritten whole whenever the ship's state changes: `Flags`,
  and near a planet `Latitude`, `Longitude` (degrees), `Altitude` (metres),
  `Heading`, `BodyName` and `PlanetRadius` (metres).

Surface positions are put into a local tangent plane per body, in metres: the
first fix on a body is its origin, x east, z north, y up (altitude), the same
handedness as the galaxy frame, and the rows carry the body's name as their
`session` so a drive or a walk on a planet draws as a route of its own. Galaxy
rows have session "galaxy". The two scales live in one table; pick the session
in the viewer.

Format: Frontier's "Elite Dangerous Player Journal" manual (published with each
update on the official forums, https://forums.frontier.co.uk/ - "Journal Manual",
v4.x), events and fields as documented there; Status.json flags per the same
manual's "Status File" chapter.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..session import Axes, Session
from ._logfollow import Follower, game_started, newest, read_shared
from .base import Adapter, Context

#: Status.json Flags bit 21: the game is giving latitude and longitude.
HAS_LAT_LONG = 1 << 21

#: Keyword names of Session.event that a journal field must not shadow.
_RESERVED = {"t", "ms", "x", "y", "z", "player", "team", "session", "event", "time"}

#: Journal events that pay credits, and how much: the earnings stat.
_EARNINGS: dict[str, Callable[[dict], float]] = {
    "MissionCompleted": lambda e: e.get("Reward", 0),
    "MarketSell": lambda e: e.get("TotalSale", 0),
    "SellExplorationData": lambda e: e.get("TotalEarnings", e.get("BaseValue", 0) + e.get("Bonus", 0)),
    "MultiSellExplorationData": lambda e: e.get("TotalEarnings", e.get("BaseValue", 0) + e.get("Bonus", 0)),
    "SellOrganicData": lambda e: sum(b.get("Value", 0) + b.get("Bonus", 0) for b in e.get("BioData", ())),
    # Bounties and combat bonds are paid when the voucher is redeemed, not when earned.
    "RedeemVoucher": lambda e: e.get("Amount", 0),
    "SearchAndRescue": lambda e: e.get("Reward", 0),
    "DatalinkVoucher": lambda e: e.get("Reward", 0),
    "CommunityGoalReward": lambda e: e.get("Reward", 0),
    "SellDrones": lambda e: e.get("TotalSale", 0),
    "SellMicroResources": lambda e: e.get("Price", 0),
}

#: Journal events that are a kill by the player.
_KILLS = ("Bounty", "FactionKillBond", "PVPKill", "CapShipBond")


def journal_dir() -> Path:
    return Path(os.environ.get("USERPROFILE") or Path.home()) / "Saved Games" / "Frontier Developments" / "Elite Dangerous"


def parse_time(text: Any) -> float | None:
    try:
        return datetime.strptime(str(text), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def line_time(line: str) -> float | None:
    """A journal line's timestamp without parsing all of it, for the follower."""
    i = line.find('"timestamp":"')
    return parse_time(line[i + 13 : i + 33]) if i >= 0 else None


def attributes(entry: dict) -> dict[str, Any]:
    """The plain fields of a journal entry, the localised text in place of the symbol where there is one."""
    out: dict[str, Any] = {}
    for k, v in entry.items():
        if k in ("timestamp", "event") or k.endswith("_Localised"):
            continue
        if isinstance(v, (dict, list)):
            continue
        loc = entry.get(f"{k}_Localised")
        if isinstance(loc, str):
            v = loc
        if isinstance(v, str) and len(v) > 300:
            v = v[:300]
        out[f"j_{k}" if k in _RESERVED else k] = v
    return out


def tangent(lat: float, lon: float, alt: float, origin: tuple[float, float], radius: float) -> tuple[float, float, float]:
    """(lat, lon, alt) on a sphere of `radius` as metres east, up, north of the surface point `origin`."""
    la, lo = math.radians(lat), math.radians(lon)
    la0, lo0 = math.radians(origin[0]), math.radians(origin[1])
    r = radius + alt
    px, py, pz = r * math.cos(la) * math.cos(lo), r * math.cos(la) * math.sin(lo), r * math.sin(la)
    ox, oy, oz = radius * math.cos(la0) * math.cos(lo0), radius * math.cos(la0) * math.sin(lo0), radius * math.sin(la0)
    dx, dy, dz = px - ox, py - oy, pz - oz
    east = -math.sin(lo0) * dx + math.cos(lo0) * dy
    north = -math.sin(la0) * math.cos(lo0) * dx - math.sin(la0) * math.sin(lo0) * dy + math.cos(la0) * dz
    up = math.cos(la0) * math.cos(lo0) * dx + math.cos(la0) * math.sin(lo0) * dy + math.sin(la0) * dz
    return east, up, north


class Journal:
    """Journal entries and Status.json snapshots into a session: every event, the positions, the stats."""

    def __init__(self, session_for: Callable[[float], Session]):
        #: The session for a moment in time; opened on the first thing worth recording.
        self.session_for = session_for
        self.commander: str | None = None
        self.star_pos: tuple[float, float, float] | None = None
        self.system: str | None = None
        self.pending_spawn = False
        self.radii: dict[str, float] = {}
        self.origins: dict[str, tuple[float, float]] = {}
        self.scanned: set[str] = set()
        self.systems: set[int] = set()
        self.on_surface: str | None = None
        self.last_fix: tuple | None = None
        self.last_status: str | None = None
        self.started: float | None = None

    @property
    def me(self) -> str:
        return self.commander or "me"

    # ------------------------------------------------------------------ input

    def line(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        try:
            entry = json.loads(text)
            if not isinstance(entry, dict) or "event" not in entry:
                raise ValueError("not a journal entry")
        except ValueError:
            if self.started is not None:
                self.session_for(self.started).stat("bad_lines", 1, "add")
            return
        self.entry(entry)

    def entry(self, e: dict) -> None:
        ts = parse_time(e.get("timestamp"))
        s = self.session_for(ts or 0.0)
        if self.started is None:
            self.started = s.started
        ev = str(e["event"])
        t, ms = self._when(s, ts)
        s.event(ev, t=t, ms=ms, player=self.me, **attributes(e))
        try:
            self._meaning(s, ev, e, t, ms)
        except (TypeError, ValueError, KeyError):
            s.stat("bad_lines", 1, "add")

    def status(self, doc: dict) -> None:
        """One Status.json snapshot: a surface fix when the game gives latitude and longitude."""
        ts = parse_time(doc.get("timestamp"))
        flags = int(doc.get("Flags") or 0)
        body = doc.get("BodyName")
        has = bool(flags & HAS_LAT_LONG) and "Latitude" in doc and "Longitude" in doc
        if not has:
            if self.on_surface and self.last_fix is not None:
                s = self.session_for(ts or 0.0)
                t, ms = self._when(s, ts)
                self._surface_row(s, "surface_end", self.on_surface, self.last_fix, t, ms)
            self.on_surface = None
            return
        body = str(body or self.on_surface or "body")
        radius = doc.get("PlanetRadius")
        if isinstance(radius, (int, float)) and radius > 0:
            self.radii[body] = float(radius)
        fix = (float(doc["Latitude"]), float(doc["Longitude"]), float(doc.get("Altitude") or 0.0), doc.get("Heading"))
        if self.on_surface == body and self.last_fix is not None and fix[:3] == self.last_fix[:3]:
            return
        s = self.session_for(ts or 0.0)
        t, ms = self._when(s, ts)
        if self.on_surface != body:
            self.on_surface = body
            self.last_fix = fix
            self._surface_row(s, "surface_start", body, fix, t, ms)
        self.last_fix = fix
        self._surface_row(s, "surface", body, fix, t, ms)
        s.stat("surface_fixes", 1, "add")

    # ---------------------------------------------------------------- meaning

    @staticmethod
    def _when(s: Session, ts: float | None) -> tuple[float | None, float | None]:
        if ts is None:
            return None, None
        return ts - s.started, ts * 1000.0

    def _galaxy(self, s: Session, event: str, journal: str, t, ms, **attrs: Any) -> None:
        x, y, z = self.star_pos if self.star_pos else (None, None, None)
        s.event(event, t=t, ms=ms, x=x, y=y, z=z, player=self.me, session="galaxy", journal=journal,
                system=self.system, **attrs)

    def _surface_row(self, s: Session, event: str, body: str, fix: tuple, t, ms, **attrs: Any) -> None:
        lat, lon, alt, heading = fix
        radius = self.radii.get(body)
        x = y = z = None
        if radius:
            origin = self.origins.setdefault(body, (lat, lon))
            x, y, z = tangent(lat, lon, alt, origin, radius)
        s.event(event, t=t, ms=ms, x=x, y=y, z=z, player=self.me, session=body, lat=lat, lon=lon, alt=alt,
                heading=heading, body=body, **attrs)

    def _meaning(self, s: Session, ev: str, e: dict, t, ms) -> None:
        if ev == "LoadGame" and e.get("Commander"):
            self.commander = str(e["Commander"])
        elif ev == "Commander" and e.get("Name"):
            self.commander = str(e["Name"])
        if ev == "LoadGame":
            self.pending_spawn = True
            if isinstance(e.get("Credits"), (int, float)):
                s.stat("credits", e["Credits"])
            s.stat("ship", str(e.get("Ship_Localised") or e.get("Ship") or ""))
            return
        if ev == "Scan" and e.get("BodyName"):
            name = str(e["BodyName"])
            if isinstance(e.get("Radius"), (int, float)) and e["Radius"] > 0:
                self.radii[name] = float(e["Radius"])
            if name not in self.scanned:
                self.scanned.add(name)
                s.stat("bodies_scanned", len(self.scanned))
        if ev in ("FSDJump", "Location", "CarrierJump") and isinstance(e.get("StarPos"), list):
            self.star_pos = tuple(float(v) for v in e["StarPos"][:3])  # type: ignore[assignment]
            self.system = e.get("StarSystem") or self.system
            if isinstance(e.get("SystemAddress"), int):
                self.systems.add(e["SystemAddress"])
                s.stat("systems_visited", len(self.systems))
            if self.pending_spawn:
                self.pending_spawn = False
                self._galaxy(s, "spawn", "LoadGame", t, ms)
            if ev == "FSDJump":
                s.stat("jumps", 1, "add")
                if isinstance(e.get("JumpDist"), (int, float)):
                    s.stat("distance_ly", float(e["JumpDist"]), "add")
                self._galaxy(s, "jump", ev, t, ms, jump_ly=e.get("JumpDist"))
            elif ev == "CarrierJump":
                s.stat("carrier_jumps", 1, "add")
                self._galaxy(s, "jump", ev, t, ms)
            else:
                self._galaxy(s, "location", ev, t, ms, body=e.get("Body"), docked=e.get("Docked"))
            return
        if ev == "Resurrect":
            self._galaxy(s, "spawn", ev, t, ms, option=e.get("Option"), cost=e.get("Cost"))
        elif ev == "Died":
            s.stat("deaths", 1, "add")
            killer = e.get("KillerName_Localised") or e.get("KillerName")
            if not killer and isinstance(e.get("Killers"), list) and e["Killers"]:
                killer = ", ".join(str(k.get("Name", "")) for k in e["Killers"] if isinstance(k, dict))
            self._galaxy(s, "death", ev, t, ms, killer=killer, killer_ship=e.get("KillerShip"))
        elif ev in ("Docked", "Undocked"):
            if ev == "Docked":
                s.stat("dockings", 1, "add")
            self._galaxy(s, ev.lower(), ev, t, ms, station=e.get("StationName"))
        elif ev in ("Touchdown", "Liftoff"):
            if ev == "Touchdown":
                s.stat("landings", 1, "add")
            body = e.get("Body")
            self._galaxy(s, ev.lower(), ev, t, ms, body=body)
            if body and "Latitude" in e and "Longitude" in e:
                alt = self.last_fix[2] if self.last_fix and self.on_surface == body else 0.0
                fix = (float(e["Latitude"]), float(e["Longitude"]), alt, None)
                self._surface_row(s, ev.lower(), str(body), fix, t, ms)
        elif ev in ("Disembark", "Embark", "LaunchSRV", "DockSRV") and self.on_surface and self.last_fix:
            self._surface_row(s, ev.lower(), self.on_surface, self.last_fix, t, ms)
        elif ev == "Shutdown":
            self._galaxy(s, "session_end", ev, t, ms)
        if ev in _KILLS:
            s.stat("kills", 1, "add")
        if ev == "MissionCompleted":
            s.stat("missions_completed", 1, "add")
        pay = _EARNINGS.get(ev)
        if pay is not None:
            amount = pay(e)
            if isinstance(amount, (int, float)) and amount > 0:
                s.stat("credits_earned", amount, "add")


@dataclass
class EliteAdapter(Adapter):
    #: The journal folder; empty: the game's default under Saved Games.
    folder: Path | None = None
    _journal: Journal | None = field(default=None, repr=False)
    _follow: Follower | None = field(default=None, repr=False)
    _session: Session | None = field(default=None, repr=False)
    _status_stamp: int = field(default=0, repr=False)

    def where(self, ctx: Context | None = None) -> Path:
        if self.folder:
            return Path(self.folder)
        if ctx is not None:
            extra = ctx.settings.replay_folders.get(self.id)
            if extra:
                return Path(extra[0])
        return journal_dir()

    def start(self, ctx: Context) -> None:
        folder = self.where(ctx)
        self._session = None
        self._status_stamp = 0

        def session_for(ts: float) -> Session:
            if self._session is None:
                when = ts if ts and abs(time.time() - ts) < 86400 else None
                self._session = ctx.open(self, when=when)
            return self._session

        self._journal = Journal(session_for)
        self._follow = Follower(lambda: newest(folder.glob("Journal.*.log")), game_started(ctx.pid), line_time)

    def poll(self, ctx: Context) -> None:
        if self._follow is None or self._journal is None:
            return
        for line in self._follow.lines():
            self._journal.line(line)
        status = self.where(ctx) / "Status.json"
        try:
            stamp = status.stat().st_mtime_ns
        except OSError:
            return
        if stamp == self._status_stamp:
            return
        self._status_stamp = stamp
        try:
            doc = json.loads(read_shared(status).decode("utf-8", errors="replace") or "{}")
        except (OSError, ValueError):
            self._status_stamp = 0  # caught mid-write: read it again next time
            return
        if isinstance(doc, dict) and doc.get("event") == "Status":
            try:
                self._journal.status(doc)
            except (TypeError, ValueError):
                if self._session is not None:
                    self._session.stat("bad_lines", 1, "add")

    def stop(self, ctx: Context) -> None:
        if self._follow is not None and self._journal is not None:
            for line in self._follow.lines():
                self._journal.line(line)
        ctx.close(self._session)
        self._session = None
        self._journal = None
        self._follow = None


ADAPTERS = [
    EliteAdapter(
        id="elite",
        game="Elite Dangerous",
        kind="log",
        exes=("EliteDangerous64.exe",),
        multiplayer=True,
        source=(
            r"Player Journal %USERPROFILE%\Saved Games\Frontier Developments\Elite Dangerous\Journal.*.log "
            r"(newest file, followed as it is written) and Status.json in the same folder"
        ),
        records=(
            "every journal event, under its journal name, with its plain fields",
            "galaxy position (StarPos, light years) on jumps, locations, spawns, deaths, docking, landing",
            "spawn on LoadGame/Resurrect, death on Died, docked/undocked, touchdown/liftoff, jump, session_end",
            "planet surface route from Status.json: latitude, longitude, altitude, heading, and metres in a "
            "tangent plane per body (session = body name)",
            "jumps, distance jumped, deaths, kills, credits earned, missions completed, bodies scanned, "
            "systems visited, dockings, landings",
        ),
        enable="",
        interval=0.25,
        axes=Axes(
            up="y",
            flip_z=True,
            units="light years (galaxy rows, session 'galaxy'); metres east/up/north of the first fix on a body "
            "(surface rows, session = body name)",
        ),
        verified="parser replayed over 43 real journal files on this machine (event counts match the stats); "
        "Status.json surface fixes tested on synthetic snapshots only",
    )
]
