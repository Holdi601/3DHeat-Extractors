"""
War Thunder: the game's own local web map on http://127.0.0.1:8111.

The game serves the page a second screen or a phone can show (the "web map"),
and the JSON behind it, while it runs - in every mode, multiplayer included:

    /map_info.json   {"map_min": [x, y], "map_max": [x, y], "grid_steps", "grid_zero", "map_generation",
                      "hud_type", "valid"}: the map's extent in metres; valid false in the hangar
    /map_obj.json    [{"type": aircraft|ground_model|airfield|capture_zone|bombing_point|defending_point|
                      respawn_base_*|..., "icon": Player|Fighter|Bomber|Mediumtank|SPAA|..., "color": "#rrggbb",
                      "x", "y" (0..1 of the map picture, y down), "dx", "dy" (direction),
                      airfields "sx", "sy", "ex", "ey"}]: what the minimap shows, the player's own
                      vehicle with icon "Player"; no ids, no heights
    /state           {"valid", "H, m", "TAS, km/h", "IAS, km/h", "M", "Vy, m/s", ...}: flight values,
                      valid only in an aircraft
    /indicators      {"valid", "army": air|tank|..., "type": the vehicle}
    /hudmsg?lastEvt=N&lastDmg=M   {"events": [...], "damage": [...]} newer than the ids given, each
                      {"id", "msg", "sender", "enemy", "mode", "time"}; damage lines read like
                      "Name (Vehicle) shot down Name (Vehicle)"

A map position becomes metres with the extent: u = min_x + x * (max_x - min_x)
east, v = max_y - y * (max_y - min_y) north (the picture's y runs down). Rows are
x = u, y = height (the own aircraft's "H, m"; 0 for everything else), z = v, so
Y is up; z is mirrored for the viewer so that north is not drawn south.

Colours tell the sides as the player sees them: own yellow (#faC81E by default),
squad green, allies blue (#174DFF), enemies red (#f40C00); colour-blind palettes
change them, so the hue decides, not the exact value. The game names nobody on
the map and gives objects no ids: other objects get ids here by following them
from poll to poll (nearest of the same type and icon), which may swap two
aircraft of one type that pass close by. The player's own name is not in the
API either: give it (`player_name`) and kill-feed lines about the player become
kills and deaths; without it they are recorded as the kill feed.

Sources: the game's own page at http://127.0.0.1:8111 and its JSON; field facts
as documented by the JoyPro 2 project (docs/telemetry-recording.md; GPL, used for
facts only).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from ..session import Axes, Session
from .base import Adapter, Context

log = logging.getLogger("heat3d_watch")

#: Objects that stay where they are: recorded once, and again when they change side.
PLACES = ("airfield", "capture_zone", "bombing_point", "defending_point", "waypoint")
#: Top speed (m/s) and margin (m) for following an object between polls.
GATES = {"aircraft": (700.0, 150.0), "ground_model": (40.0, 60.0)}
#: A track not seen for this long is forgotten.
GRACE = 3.0

NAME = r"(?P<{0}>.+?) \((?P<{0}_vehicle>[^()]*(?:\([^()]*\)[^()]*)*)\)"
FEED = re.compile(
    "^" + NAME.format("actor") + r" (?P<verb>shot down|destroyed|set afire|severely damaged|critically damaged|"
    r"damaged|has destroyed) " + NAME.format("target") + r"\.?$"
)
ALONE = re.compile("^" + NAME.format("actor") + r" (?P<verb>has crashed|has been wrecked|has been destroyed)\.?$")
KILL_VERBS = ("shot down", "destroyed", "has destroyed")


def side(color: str | None, own: str | None) -> str:
    """friendly / enemy / neutral from a map colour, by hue (palettes differ)."""
    c = (color or "").strip()
    if own and c.lower() == own.lower():
        return "friendly"
    if len(c) != 7 or not c.startswith("#"):
        return "neutral"
    try:
        rgb = int(c[1:], 16)
    except ValueError:
        return "neutral"
    r, g, b = (rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255
    if r - max(g, b) > 80:
        return "enemy"
    if max(g, b) - r > 40:
        return "friendly"
    return "neutral"


def heading(dx: Any, dy: Any) -> float | None:
    """The picture's direction (x right, y down) as degrees clockwise from north."""
    try:
        dx, dy = float(dx), float(dy)
    except (TypeError, ValueError):
        return None
    if dx == 0 and dy == 0:
        return None
    return round(math.degrees(math.atan2(dx, -dy)) % 360, 1)


@dataclass
class MapInfo:
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    grid: tuple[float, ...] = ()

    @classmethod
    def parse(cls, doc: Any) -> "MapInfo | None":
        if not isinstance(doc, dict) or doc.get("valid") is False:
            return None
        try:
            (x0, y0), (x1, y1) = doc["map_min"], doc["map_max"]
            info = cls(float(x0), float(y0), float(x1), float(y1),
                       tuple(float(v) for v in [*(doc.get("grid_steps") or []), *(doc.get("grid_zero") or [])]))
        except (KeyError, TypeError, ValueError):
            return None
        return info if info.max_x > info.min_x and info.max_y > info.min_y else None

    def metres(self, x: float, y: float) -> tuple[float, float]:
        return self.min_x + x * (self.max_x - self.min_x), self.max_y - y * (self.max_y - self.min_y)

    @property
    def id(self) -> str:
        """The game names no map here: an id from its extent and grid."""
        key = json.dumps([self.min_x, self.min_y, self.max_x, self.max_y, *self.grid])
        w, h = round(self.max_x - self.min_x), round(self.max_y - self.min_y)
        return f"wt_{w}x{h}_{hashlib.sha1(key.encode()).hexdigest()[:8]}"


@dataclass
class _Track:
    id: str
    kind: str
    u: float
    v: float
    seen: float
    vu: float = 0.0
    vv: float = 0.0


class Tracker:
    """Gives map objects - which have no ids in the game - ids that last from poll to poll."""

    def __init__(self) -> None:
        self.tracks: dict[str, _Track] = {}
        self.counter = 0

    def assign(self, objects: list[tuple[str, float, float]], now: float) -> list[str]:
        """(kind, u, v) per object -> an id per object, nearest prediction first."""
        ids: list[str | None] = [None] * len(objects)
        pairs = []
        for i, (kind, u, v) in enumerate(objects):
            speed, margin = GATES.get(kind.split("|")[0], (0.0, 60.0))
            for t in self.tracks.values():
                if t.kind != kind:
                    continue
                dt = max(0.0, now - t.seen)
                pu, pv = t.u + t.vu * dt, t.v + t.vv * dt
                d = math.hypot(u - pu, v - pv)
                if d <= speed * max(0.5, dt) + margin:
                    pairs.append((d, i, t.id))
        used: set[str] = set()
        for d, i, tid in sorted(pairs):
            if ids[i] is not None or tid in used:
                continue
            ids[i] = tid
            used.add(tid)
        for i, (kind, u, v) in enumerate(objects):
            tid = ids[i]
            if tid is None:
                self.counter += 1
                tid = ids[i] = str(self.counter)
                self.tracks[tid] = _Track(tid, kind, u, v, now)
                continue
            t = self.tracks[tid]
            dt = now - t.seen
            if dt < 0.2:
                # Polls this close together say more about timing than about speed: keep the last estimate.
                t.u, t.v, t.seen = u, v, now
                continue
            speed = GATES.get(kind.split("|")[0], (0.0, 0.0))[0]
            vu, vv = (u - t.u) / dt, (v - t.v) / dt
            k = min(1.0, speed / math.hypot(vu, vv)) if (vu or vv) else 1.0
            t.vu, t.vv = vu * k, vv * k
            t.u, t.v, t.seen = u, v, now
        for tid in [t.id for t in self.tracks.values() if now - t.seen > GRACE]:
            del self.tracks[tid]
        return [str(i) for i in ids]


@dataclass
class WarThunder(Adapter):
    #: The game's web map. Only 127.0.0.1 is ever asked.
    url: str = "http://127.0.0.1:8111"
    #: The player's name as the kill feed writes it. Empty: kill-feed lines stay the kill feed.
    player_name: str = ""
    map_every: float = 10.0
    hud_every: float = 1.0
    indicators_every: float = 2.0
    #: Seconds of "no battle" (invalid map, no answer) before a battle counts as over.
    gone_after: float = 10.0
    #: Seconds the own vehicle may be missing from the map before it counts as gone.
    vehicle_grace: float = 2.0
    _session: Session | None = None
    _map: MapInfo | None = None
    _next_map: float = 0.0
    _next_hud: float = 0.0
    _next_ind: float = 0.0
    _no_battle_since: float | None = None
    _last_evt: int | None = None
    _last_dmg: int | None = None
    _army: str | None = None
    _vehicle: str | None = None
    _tracker: Tracker = field(default_factory=Tracker)
    _places: dict[str, str] = field(default_factory=dict)
    _own: tuple[float, float, float] | None = None
    _own_seen: float = 0.0
    _alive: bool = False
    _sortie: int = 0
    _distance: float = 0.0
    _bad: int = 0

    def missing(self) -> str | None:
        if (urlparse(self.url).hostname or "") not in ("127.0.0.1", "localhost"):
            return f"the web map is local: {self.url} is not 127.0.0.1"
        return None

    def get(self, path: str) -> Any:
        if self.missing():
            return None
        try:
            with urllib.request.urlopen(self.url.rstrip("/") + path, timeout=0.3) as r:
                body = r.read()
        except (urllib.error.URLError, OSError, ValueError):
            return None
        try:
            return json.loads(body.decode("utf-8", errors="replace"))
        except ValueError:
            self._bad += 1
            return None

    # -- lifecycle ---------------------------------------------------------------

    def start(self, ctx: Context) -> None:
        self._session = None
        self._map = None
        self._next_map = self._next_hud = self._next_ind = 0.0
        self._no_battle_since = None
        self._last_evt = self._last_dmg = None
        self._army = self._vehicle = None
        self._bad = 0
        self._new_battle()

    def _new_battle(self) -> None:
        self._tracker = Tracker()
        self._places = {}
        self._own = None
        self._alive = False
        self._sortie = 0
        self._distance = 0.0

    def stop(self, ctx: Context) -> None:
        self._end(ctx)

    def _end(self, ctx: Context) -> None:
        s = self._session
        if s is None:
            return
        if self._alive and self._own is not None:
            self._left(s, time.time())
        if self._bad:
            s.stat("bad_packets", self._bad)
        ctx.close(s)
        self._session = None
        self._new_battle()

    def poll(self, ctx: Context) -> None:
        now = time.monotonic()
        wall = time.time()
        if now >= self._next_map or self._map is None:
            self._next_map = now + self.map_every
            info = MapInfo.parse(self.get("/map_info.json"))
            if info is None:
                self._no_battle_since = self._no_battle_since or now
                if self._session is not None and now - self._no_battle_since >= self.gone_after:
                    self._end(ctx)
                    self._map = None
            else:
                self._no_battle_since = None
                if self._map is not None and info != self._map:
                    self._end(ctx)
                self._map = info
        if self._map is not None and now >= self._next_ind:
            self._next_ind = now + self.indicators_every
            ind = self.get("/indicators")
            if isinstance(ind, dict) and ind.get("valid"):
                self._army = str(ind.get("army") or "") or None
                self._vehicle = str(ind.get("type") or "") or None
        if self._map is not None:
            objs = self.get("/map_obj.json")
            if isinstance(objs, list):
                if self._session is None:
                    m = self._map
                    self._session = ctx.open(self, meta={"map_id": m.id, "map_min": [m.min_x, m.min_y],
                                                         "map_max": [m.max_x, m.max_y]})
                    self._no_battle_since = None
                has_own = any(isinstance(o, dict) and o.get("icon") == "Player" for o in objs)
                state = self.get("/state") if has_own else None
                self._objects(self._session, objs, state if isinstance(state, dict) else {}, now, wall)
        if now >= self._next_hud:
            self._next_hud = now + self.hud_every
            self._hud(self._session, wall)
        if self._session is not None and self._bad:
            self._session.stat("bad_packets", self._bad)

    # -- the map -------------------------------------------------------------------

    def _objects(self, s: Session, objs: list[Any], state: dict[str, Any], now: float, wall: float) -> None:
        info = self._map
        assert info is not None
        ms = int(wall * 1000)
        own_color = next((str(o.get("color") or "") for o in objs
                          if isinstance(o, dict) and o.get("icon") == "Player"), None)
        movers: list[tuple[str, float, float]] = []
        rows: list[dict[str, Any]] = []
        player_seen = False
        for o in objs:
            if not isinstance(o, dict):
                continue
            kind, icon = str(o.get("type") or ""), str(o.get("icon") or "")
            try:
                if "sx" in o and "ex" in o:
                    x, y = (float(o["sx"]) + float(o["ex"])) / 2, (float(o["sy"]) + float(o["ey"])) / 2
                    dx, dy = float(o["ex"]) - float(o["sx"]), float(o["ey"]) - float(o["sy"])
                else:
                    x, y = float(o["x"]), float(o["y"])
                    dx, dy = o.get("dx"), o.get("dy")
            except (KeyError, TypeError, ValueError):
                self._bad += 1
                continue
            u, v = info.metres(x, y)
            if icon == "Player":
                player_seen = True
                self._own_position(s, u, v, state, heading(dx, dy), ms)
                continue
            team = side(o.get("color"), own_color)
            if kind in PLACES or kind.startswith("respawn_base"):
                key = f"{kind}|{icon}|{round(u)}|{round(v)}"
                if key not in self._places:
                    s.event("map_object", ms=ms, x=u, y=0.0, z=v, player=f"{kind}", team=team, kind=kind, icon=icon)
                elif self._places[key] != team:
                    s.event("zone_changed", ms=ms, x=u, y=0.0, z=v, player=f"{kind}", team=team, kind=kind,
                            icon=icon, was=self._places[key])
                self._places[key] = team
                continue
            movers.append((f"{kind}|{icon}", u, v))
            rows.append({"team": team, "kind": kind, "icon": icon, "heading": heading(dx, dy)})
        ids = self._tracker.assign(movers, now)
        for (key, u, v), tid, r in zip(movers, ids, rows):
            s.position(u, 0.0, v, ms=ms, player=f"{r['kind']}-{tid}", team=r["team"], icon=r["icon"],
                       heading=r["heading"])
        if not player_seen and self._alive and now - self._own_seen > self.vehicle_grace:
            self._left(s, wall)

    def _own_position(self, s: Session, u: float, v: float, state: dict[str, Any], hdg: float | None, ms: int) -> None:
        alt = None
        if state.get("valid"):
            try:
                alt = float(state.get("H, m"))
            except (TypeError, ValueError):
                alt = None
        y = alt if alt is not None else 0.0
        tas = state.get("TAS, km/h") if state.get("valid") else None
        self._own_seen = time.monotonic()
        if not self._alive:
            self._sortie += 1
            self._alive = True
            s.event("spawn", ms=ms, x=u, y=y, z=v, player="me", team="friendly", session=f"sortie {self._sortie}",
                    vehicle=self._vehicle, army=self._army)
            s.stat("spawns", 1, "add")
            if self._vehicle:
                s.stat("vehicle", self._vehicle)
        elif self._own is not None:
            self._distance += math.dist(self._own, (u, y, v))
            s.stat("distance_m", round(self._distance, 1))
        self._own = (u, y, v)
        s.position(u, y, v, ms=ms, player="me", team="friendly", session=f"sortie {self._sortie}", heading=hdg,
                   tas_kmh=tas, vehicle=self._vehicle)
        if alt is not None:
            s.stat("max_altitude_m", round(alt, 1), "max")
        if isinstance(tas, (int, float)):
            s.stat("max_tas_kmh", round(float(tas), 1), "max")

    def _left(self, s: Session, wall: float) -> None:
        """The own vehicle is gone from the map: destroyed, crashed, landed and left, or back to the hangar."""
        x, y, z = self._own or (None, None, None)
        s.event("vehicle_left", ms=int(wall * 1000), x=x, y=y, z=z, player="me", team="friendly",
                session=f"sortie {self._sortie}", vehicle=self._vehicle)
        self._alive = False

    # -- the HUD messages -----------------------------------------------------------

    def _hud(self, s: Session | None, wall: float) -> None:
        first = self._last_evt is None
        doc = self.get(f"/hudmsg?lastEvt={self._last_evt or 0}&lastDmg={self._last_dmg or 0}")
        if not isinstance(doc, dict):
            return
        for key, attr in (("events", "_last_evt"), ("damage", "_last_dmg")):
            entries = [e for e in doc.get(key) or [] if isinstance(e, dict)]
            last = getattr(self, attr) or 0
            for e in sorted(entries, key=lambda e: e.get("id", 0) if isinstance(e.get("id"), (int, float)) else 0):
                try:
                    eid = int(e.get("id"))
                except (TypeError, ValueError):
                    self._bad += 1
                    continue
                # Ids grow for as long as the game runs; the game answers only newer ones, but say it twice anyway.
                if eid <= last:
                    continue
                last = eid
                # Lines from before the watcher started, or said in the hangar, belong to no battle here.
                if not first and s is not None:
                    self._message(s, key, e, wall)
            setattr(self, attr, last)

    def is_me(self, name: str) -> bool:
        p = self.player_name.strip()
        return bool(p) and (name == p or name.endswith(" " + p))

    def _message(self, s: Session, key: str, e: dict[str, Any], wall: float) -> None:
        text = str(e.get("msg") or "").strip()
        if not text:
            return
        ms = int(wall * 1000)
        extra = {"text": text, "game_time": e.get("time"), "message_id": e.get("id")}
        if key == "events":
            s.event("hud_event", ms=ms, **extra)
            return
        where = {}
        if self._own is not None:
            where = dict(x=self._own[0], y=self._own[1], z=self._own[2])
        m = FEED.match(text)
        if m:
            actor, target, verb = m.group("actor"), m.group("target"), m.group("verb")
            kill = verb in KILL_VERBS
            attrs = dict(actor=actor, actor_vehicle=m.group("actor_vehicle"), target=target,
                         target_vehicle=m.group("target_vehicle"), verb=verb, **extra)
            if self.is_me(actor):
                s.event("kill" if kill else "damage_dealt", ms=ms, player="me", session=f"sortie {self._sortie}",
                        **where, **attrs)
                s.stat("kills" if kill else "hits_dealt", 1, "add")
            elif self.is_me(target):
                s.event("death" if kill else "damage_taken", ms=ms, player="me", session=f"sortie {self._sortie}",
                        **where, **attrs)
                s.stat("deaths" if kill else "hits_taken", 1, "add")
            else:
                s.event("kill_feed" if kill else "damage_feed", ms=ms, player=actor, **attrs)
            return
        m = ALONE.match(text)
        if m:
            actor = m.group("actor")
            attrs = dict(actor=actor, actor_vehicle=m.group("actor_vehicle"), verb=m.group("verb"), **extra)
            if self.is_me(actor):
                s.event("death", ms=ms, player="me", session=f"sortie {self._sortie}", cause="crash", **where, **attrs)
                s.stat("deaths", 1, "add")
            else:
                s.event("crash_feed", ms=ms, player=actor, **attrs)
            return
        s.event("hud_message", ms=ms, enemy=e.get("enemy"), **extra)


ADAPTERS = [
    WarThunder(
        id="warthunder-webmap",
        game="War Thunder",
        kind="gsi",
        exes=("aces.exe",),
        interval=0.5,
        source="the game's local web map, http://127.0.0.1:8111 (/map_info.json, /map_obj.json, /state, "
        "/indicators, /hudmsg)",
        records=(
            "own position (map east/north in metres; height from /state in aircraft, else 0) twice a second, "
            "heading, speed",
            "spawns and the own vehicle leaving the map (destroyed, landed, back to the hangar), one sortie each",
            "other aircraft and ground units on the minimap as positions with tracked ids and side (no height)",
            "airfields, capture zones and bases once, and zones changing side",
            "kill and damage feed lines (who, vehicle, verb); kills and deaths when `player_name` is set",
            "spawns, distance, top altitude, top true airspeed, vehicle; kills and deaths with `player_name`",
        ),
        enable="",
        axes=Axes(up="y", flip_z=True, units="metres (map east/north; height only for the own aircraft)"),
        verified="tested against a fake local server serving answers in the format of the game's web map; "
        "War Thunder is installed here but was not run",
    ),
]
