"""
Valve Game State Integration: Counter-Strike 2 and Dota 2 tell a local web server what happens.

Both games read every `gamestate_integration_*.cfg` in their cfg folder when they
start, and from then on POST a JSON document to each file's `uri` whenever the
state changes (at most every `throttle` seconds, and every `heartbeat` seconds
when nothing does). The watcher is that web server: it listens on 127.0.0.1 only,
answers every POST with 200 at once and does the work in its own poll thread.

    Counter-Strike 2  <CS2>\\game\\csgo\\cfg\\gamestate_integration_heat3d.cfg      port 3311
    Dota 2            <Dota>\\game\\dota\\cfg\\gamestate_integration\\gamestate_integration_heat3d.cfg   port 3312

What a player is sent - and so what can be recorded - is what the game decides
a player may see:

- CS2: `provider` (the own steamid), `map` (name, mode, phase warmup/live/
  intermission/gameover, round, team_ct/team_t score), `round` (phase
  freezetime/live/over, bomb planted/exploded/defused, win_team), `player` (the
  player being *watched*: oneself while alive, a teammate after death) with
  `state` (health, armor, money, round_kills, round_killhs, round_totaldmg),
  `match_stats` (kills, assists, deaths, mvps, score), `position` and `forward`
  as "x, y, z" strings in world units (inches, Z up), `weapons`; `bomb` only for
  spectators. Other players' positions (`allplayers_*`) are sent only to
  spectators and GOTV, so they are not asked for.
- Dota 2: `map` (matchid, game_state, game_time, clock_time, win_team), `player`
  (kills, deaths, assists, last_hits, denies, gold, gpm, xpm, team_name), `hero`
  (xpos/ypos in world units, level, alive, respawn_seconds), `buildings`
  (health per tower/barracks/ancient), `roshan`, `events` (roshan_killed,
  aegis_picked_up, ...). There is no height: z is 0.

Sources: https://developer.valvesoftware.com/wiki/Counter-Strike:_Global_Offensive_Game_State_Integration
(the format CS2 kept), https://github.com/antonpup/CounterStrike2GSI and
https://github.com/antonpup/Dota2GSI (field lists; MIT, read for facts only).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..session import Axes, Session
from .base import Adapter, Context

log = logging.getLogger("heat3d_watch")

#: A GSI document is a few kilobytes; anything far larger is not one.
MAX_BODY = 1 << 20

#: `"path"  "D:\\SteamLibrary"` lines of Valve's key-value files.
LIBRARY_PATH = re.compile(r'"path"\s*"([^"]+)"')
TOKEN = re.compile(r'"token"\s*"([^"]*)"')

STEAM_DEFAULTS = (r"C:\Program Files (x86)\Steam", r"C:\Program Files\Steam")


def steam_libraries() -> list[Path]:
    """
    Every Steam library this machine knows about.

    The gamefiles package has the same lookup; it is used when it is importable,
    and Steam's own `steamapps\\libraryfolders.vdf` (and the older copy in
    `config\\`) is read either way, so the watcher does not depend on it.
    """
    roots: list[Path] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        key = str(p).lower().rstrip("\\/")
        if key not in seen:
            seen.add(key)
            roots.append(p)

    try:
        from heat3d_gamefiles.forzainstall import steam_libraries as gamefiles_libraries  # type: ignore

        for p in gamefiles_libraries():
            add(Path(p))
    except Exception:  # noqa: BLE001 - not installed, or broken: the own lookup below suffices
        pass
    bases = list(STEAM_DEFAULTS)
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as k:
            bases.insert(0, str(winreg.QueryValueEx(k, "SteamPath")[0]).replace("/", "\\"))
    except (ImportError, OSError):
        pass
    for base in bases:
        for vdf in (Path(base) / "steamapps" / "libraryfolders.vdf", Path(base) / "config" / "libraryfolders.vdf"):
            try:
                text = vdf.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            add(Path(base))
            for found in LIBRARY_PATH.finditer(text):
                add(Path(found.group(1).replace("\\\\", "\\")))
    return roots


def steam_game(folder: str) -> Path | None:
    """`<library>\\steamapps\\common\\<folder>` in the first library that has it."""
    for lib in steam_libraries():
        p = lib / "steamapps" / "common" / folder
        if p.is_dir():
            return p
    return None


def backup_once(path: Path) -> None:
    """`<file>.heat3d-backup`, made the first time a file is changed and never overwritten."""
    backup = path.with_name(path.name + ".heat3d-backup")
    if path.exists() and not backup.exists():
        backup.write_bytes(path.read_bytes())


def vec(text: Any) -> tuple[float, float, float] | None:
    """CS2's "x, y, z" strings; None when it is not three numbers."""
    if not isinstance(text, str):
        return None
    try:
        parts = [float(v) for v in text.split(",")]
    except ValueError:
        return None
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 else None


def num(value: Any, default: float = 0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class _Handler(BaseHTTPRequestHandler):
    server: "_Server"

    def do_POST(self) -> None:  # noqa: N802 - the http.server name
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        body = self.rfile.read(n) if 0 < n <= MAX_BODY else b""
        try:
            self.server.inbox.put_nowait((time.time(), body))
        except queue.Full:
            self.server.dropped += 1
        # The game waits for the answer before it sends the next state: answer first, think later.
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - quiet, the game posts ten times a second
        pass


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port: int):
        super().__init__(("127.0.0.1", port), _Handler)
        self.inbox: queue.Queue[tuple[float, bytes]] = queue.Queue(maxsize=10000)
        self.dropped = 0


@dataclass
class GsiAdapter(Adapter):
    """The web server and the cfg file; what a document means is the game's subclass."""

    kind: str = "gsi"
    interval: float = 0.05
    port: int = 0
    #: The token the cfg file names. Empty: read from the cfg file the game uses.
    token: str = ""
    #: The game's install folder. Empty: next to the running exe, else the Steam libraries.
    install: str = ""
    #: `steamapps\common\<folder>` of the game.
    steam_folder: str = ""
    #: The cfg file, relative to the install folder.
    cfg_file: str = ""
    #: The `data` section of the cfg file: what the game is asked to send.
    providers: tuple[str, ...] = ()
    _server: _Server | None = None
    _thread: threading.Thread | None = None
    _session: Session | None = None
    _accept: str | None = None
    _bad: int = 0
    _bad_auth: int = 0

    # -- where the game is -----------------------------------------------------

    def find_install(self, ctx: Context | None = None) -> Path | None:
        if self.install:
            return Path(self.install)
        if ctx is not None and ctx.exe is not None:
            # <install>\game\bin\win64\<game>.exe
            p = Path(ctx.exe).parents[3] if len(Path(ctx.exe).parents) > 3 else None
            if p is not None and (p / "game").is_dir():
                return p
        return steam_game(self.steam_folder)

    def cfg_text(self, token: str) -> str:
        lines = [
            '"3DHeat Watcher"',
            "{",
            f'\t"uri"\t\t"http://127.0.0.1:{self.port}/"',
            '\t"timeout"\t"5.0"',
            '\t"buffer"\t"0.1"',
            '\t"throttle"\t"0.1"',
            '\t"heartbeat"\t"10.0"',
            '\t"auth"',
            "\t{",
            f'\t\t"token"\t"{token}"',
            "\t}",
            '\t"data"',
            "\t{",
            *[f'\t\t"{p}"\t"1"' for p in self.providers],
            "\t}",
            "}",
            "",
        ]
        return "\r\n".join(lines)

    def setup(self, ctx: Context, *, apply: bool) -> list[str]:
        root = self.find_install(ctx)
        if root is None:
            return []
        path = root / self.cfg_file
        old = ""
        try:
            old = path.read_bytes().decode("utf-8", errors="replace")  # as written: CRLF stays CRLF
        except OSError:
            pass
        found = TOKEN.search(old)
        token = found.group(1) if found and found.group(1) else secrets.token_hex(16)
        text = self.cfg_text(token)
        if old == text:
            return []
        if apply:
            path.parent.mkdir(parents=True, exist_ok=True)
            backup_once(path)
            path.write_text(text, encoding="utf-8", newline="")
        what = "update" if old else "write"
        return [f"{what} {path}: send the game state to http://127.0.0.1:{self.port}/ (read when the game starts)"]

    # -- lifecycle -------------------------------------------------------------

    def start(self, ctx: Context) -> None:
        self._accept = self.token or None
        if self._accept is None:
            root = self.find_install(ctx)
            try:
                text = (root / self.cfg_file).read_text(encoding="utf-8", errors="replace") if root else ""
            except OSError:
                text = ""
            found = TOKEN.search(text)
            self._accept = found.group(1) if found and found.group(1) else None
        if self._accept is None:
            log.info("%s: no cfg file with a token found; accepting any local post (`setup %s --apply`)",
                     self.id, self.id)
        self._bad = self._bad_auth = 0
        self._server = _Server(self.port)
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.2},
                                        name=f"gsi-{self.id}", daemon=True)
        self._thread.start()

    def poll(self, ctx: Context) -> None:
        srv = self._server
        if srv is None:
            return
        deadline = time.monotonic() + 0.1
        while time.monotonic() < deadline:
            try:
                when, body = srv.inbox.get_nowait()
            except queue.Empty:
                break
            try:
                doc = json.loads(body.decode("utf-8"))
                if not isinstance(doc, dict):
                    raise ValueError("not an object")
            except (ValueError, UnicodeDecodeError):
                self._bad += 1
                self._count()
                continue
            if self._accept is not None and str((doc.get("auth") or {}).get("token") or "") != self._accept:
                self._bad_auth += 1
                self._count()
                continue
            try:
                self.handle(ctx, doc, when)
            except (TypeError, ValueError, AttributeError, KeyError):
                # A field of an unexpected shape: one document lost, never the watcher.
                self._bad += 1
                self._count()

    def stop(self, ctx: Context) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self.end(ctx)

    # -- sessions ----------------------------------------------------------------

    def begin(self, ctx: Context, when: float, **meta: Any) -> Session:
        self._session = ctx.open(self, when=when, meta={k: v for k, v in meta.items() if v not in (None, "")})
        self._count()
        return self._session

    def end(self, ctx: Context) -> None:
        if self._session is not None:
            self._count()
            ctx.close(self._session)
            self._session = None

    def _count(self) -> None:
        s = self._session
        if s is None:
            return
        if self._bad:
            s.stat("bad_packets", self._bad)
        if self._bad_auth:
            s.stat("bad_auth", self._bad_auth)
        if self._server is not None and self._server.dropped:
            s.stat("dropped_packets", self._server.dropped)

    def handle(self, ctx: Context, doc: dict[str, Any], when: float) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------- Counter-Strike 2


@dataclass
class Cs2Gsi(GsiAdapter):
    _map: str | None = None
    _phase: str | None = None
    _round: int | None = None
    _round_phase: str | None = None
    _bomb: str | None = None
    _team: str | None = None
    _health: int | None = None
    _pos: tuple[float, float, float] | None = None
    _need_spawn: bool = True
    _kills: int | None = None
    #: Deaths recorded so far; None until the first match statistics give the count to start from.
    _deaths: int | None = None
    _round_hs: int = 0
    _round_dmg: int = 0
    _totals: dict[str, int] = field(default_factory=dict)

    def _reset(self) -> None:
        self._map = self._phase = self._round_phase = self._bomb = self._team = None
        self._round = self._health = self._kills = self._pos = self._deaths = None
        self._need_spawn = True
        self._round_hs = self._round_dmg = 0
        self._totals = {"rounds_won": 0, "rounds_lost": 0, "headshot_kills": 0, "damage": 0}

    def start(self, ctx: Context) -> None:
        self._reset()
        super().start(ctx)

    def _close_round(self) -> None:
        """A round's damage and headshots count once, at the round's end or the next round's start."""
        s = self._session
        self._totals["headshot_kills"] += self._round_hs
        self._totals["damage"] += self._round_dmg
        self._round_hs = self._round_dmg = 0
        if s is not None:
            for k, v in self._totals.items():
                s.stat(k, v)

    def handle(self, ctx: Context, doc: dict[str, Any], when: float) -> None:
        me = str((doc.get("provider") or {}).get("steamid") or "")
        m = doc.get("map")
        if not isinstance(m, dict) or not m.get("name"):
            # The main menu: no map, the match is over for this player.
            if self._session is not None:
                self._close_round()
                self.end(ctx)
                self._reset()
            return
        name, phase = str(m["name"]), str(m.get("phase") or "")
        rnd = int(num(m.get("round"), 0))
        new_match = (
            self._session is None
            or name != self._map
            or (self._phase == "gameover" and phase != "gameover")
            or (self._round is not None and rnd < self._round and phase != "gameover")
        )
        if new_match:
            if self._session is not None:
                self._close_round()
                self.end(ctx)
            self._reset()
            self.begin(ctx, when, map=name, mode=m.get("mode"))
            self._map = name
        s = self._session
        assert s is not None
        ms = int(when * 1000)
        label = f"round {rnd + 1}"
        if self._round is not None and rnd != self._round:
            self._close_round()
            self._need_spawn = True
        self._round = rnd
        if phase != self._phase:
            s.event("match_phase", ms=ms, phase=phase, session=label)
            if phase == "live" and self._phase == "warmup":
                # Warm-up kills are reset to 0 when the match goes live.
                self._kills, self._deaths = None, None
                self._need_spawn = True

        p = doc.get("player")
        own = isinstance(p, dict) and (not me or str(p.get("steamid") or "") == me)
        if own:
            self._team = str(p.get("team") or "") or self._team

        r = doc.get("round") if isinstance(doc.get("round"), dict) else {}
        rphase = str(r.get("phase") or "")
        if rphase != self._round_phase:
            if rphase == "freezetime":
                self._need_spawn = True
                s.event("round_start", ms=ms, session=label, round=rnd + 1)
            elif rphase == "over":
                winner = str(r.get("win_team") or "")
                won = bool(self._team) and winner == self._team
                if winner and self._team:
                    self._totals["rounds_won" if won else "rounds_lost"] += 1
                s.event("round_end", ms=ms, session=label, round=rnd + 1, winner=winner,
                        won=won if self._team else None, team=self._team)
            self._round_phase = rphase
        bomb = str(r.get("bomb") or "")
        if bomb != self._bomb:
            if bomb in ("planted", "exploded", "defused"):
                b = doc.get("bomb") if isinstance(doc.get("bomb"), dict) else {}
                at = vec(b.get("position"))
                kw = dict(x=at[0], y=at[1], z=at[2]) if at else {}
                s.event(f"bomb_{bomb}", ms=ms, session=label, **kw)
            self._bomb = bomb

        if own:
            self._own(s, p, ms, label)

        if phase != self._phase and phase == "gameover":
            ct = int(num((m.get("team_ct") or {}).get("score")))
            t = int(num((m.get("team_t") or {}).get("score")))
            mine, theirs = (ct, t) if self._team == "CT" else (t, ct) if self._team == "T" else (None, None)
            result = None if mine is None else "win" if mine > theirs else "loss" if mine < theirs else "tie"
            self._close_round()
            s.event("match_end", ms=ms, score_ct=ct, score_t=t, result=result, team=self._team)
            if result:
                s.stat("result", result)
                s.stat("score_own", mine)
                s.stat("score_other", theirs)
        self._phase = phase
        self._count()

    def _own(self, s: Session, p: dict[str, Any], ms: int, label: str) -> None:
        st = p.get("state") if isinstance(p.get("state"), dict) else {}
        mstats = p.get("match_stats") if isinstance(p.get("match_stats"), dict) else {}
        pos = vec(p.get("position")) or self._pos
        health = int(num(st.get("health"), self._health if self._health is not None else 0))
        team = self._team
        where = dict(x=pos[0], y=pos[1], z=pos[2]) if pos else {}
        # 1. A death seen as health going to 0.
        if self._health is not None and self._health > 0 and health == 0:
            s.event("death", ms=ms, player="me", team=team, session=label, **where)
            if self._deaths is not None:
                self._deaths += 1
        # 2. Deaths the game counted while it showed a teammate: at the last place seen.
        if "deaths" in mstats:
            counted = int(num(mstats.get("deaths")))
            if self._deaths is None or counted < self._deaths:
                self._deaths = counted
            last = dict(x=self._pos[0], y=self._pos[1], z=self._pos[2]) if self._pos else {}
            while self._deaths < counted:
                s.event("death", ms=ms, player="me", team=team, session=label, late=True, **last)
                self._deaths += 1
        # 3. Alive: a spawn at a round's first sighting, or after a death (respawn modes), then the position.
        if pos and health > 0:
            if self._need_spawn or (self._health == 0):
                s.event("spawn", ms=ms, player="me", team=team, session=label, **where)
                self._need_spawn = False
            if pos != self._pos or self._health != health:
                fwd = vec(p.get("forward"))
                s.position(*pos, ms=ms, player="me", team=team, session=label, health=health,
                           armor=int(num(st.get("armor"))), money=int(num(st.get("money"))),
                           fx=fwd[0] if fwd else None, fy=fwd[1] if fwd else None, fz=fwd[2] if fwd else None)
            self._pos = pos
        self._health = health
        # 4. Kills, from the match statistics growing.
        if "kills" in mstats:
            kills = int(num(mstats.get("kills")))
            if self._kills is not None and kills > self._kills:
                weapon = active_weapon(p)
                for _ in range(kills - self._kills):
                    s.event("kill", ms=ms, player="me", team=team, session=label, weapon=weapon, **where)
            self._kills = kills
            for key in ("kills", "assists", "deaths", "mvps", "score"):
                if key in mstats:
                    s.stat(key, int(num(mstats.get(key))))
        if "money" in st:
            s.stat("money", int(num(st.get("money"))))
        self._round_hs = max(self._round_hs, int(num(st.get("round_killhs"))))
        self._round_dmg = max(self._round_dmg, int(num(st.get("round_totaldmg"))))


def active_weapon(player: dict[str, Any]) -> str | None:
    weapons = player.get("weapons")
    if not isinstance(weapons, dict):
        return None
    for w in weapons.values():
        if isinstance(w, dict) and w.get("state") == "active":
            return str(w.get("name") or "") or None
    return None


# ---------------------------------------------------------------------- Dota 2


@dataclass
class DotaGsi(GsiAdapter):
    _match: str | None = None
    _state: str | None = None
    _alive: bool | None = None
    _pos: tuple[float, float, float] | None = None
    _kills: int | None = None
    _deaths: int = 0
    _buildings: dict[str, float] = field(default_factory=dict)
    _roshan: bool | None = None
    _seen_events: set[str] = field(default_factory=set)

    def _reset(self) -> None:
        self._match = self._state = self._alive = self._pos = self._kills = self._roshan = None
        self._deaths = 0
        self._buildings = {}
        self._seen_events = set()

    def start(self, ctx: Context) -> None:
        self._reset()
        super().start(ctx)

    def handle(self, ctx: Context, doc: dict[str, Any], when: float) -> None:
        m = doc.get("map")
        if not isinstance(m, dict) or not (m.get("matchid") or m.get("name")):
            if self._session is not None:
                self.end(ctx)
                self._reset()
            return
        match = str(m.get("matchid") or m.get("name"))
        if self._session is None or match != self._match:
            self.end(ctx)
            self._reset()
            self.begin(ctx, when, map=m.get("name"), custom_game=m.get("customgamename"))
            self._match = match
        s = self._session
        assert s is not None
        ms = int(when * 1000)
        clock = num(m.get("clock_time"), 0)
        p = doc.get("player") if isinstance(doc.get("player"), dict) else {}
        # Spectating, `player` is teams of players without a steamid of its own: not a player's own game.
        own = "steamid" in p or "kills" in p
        team = str(p.get("team_name") or "") or None if own else None
        state = str(m.get("game_state") or "")
        if state != self._state:
            s.event("game_state", ms=ms, state=state, clock=clock, session=match)
            self._state = state

        h = doc.get("hero") if isinstance(doc.get("hero"), dict) else {}
        if own and "xpos" in h and "ypos" in h:
            pos = (num(h.get("xpos")), num(h.get("ypos")), 0.0)
            alive = bool(h.get("alive", True))
            where = dict(x=pos[0], y=pos[1], z=pos[2])
            hero = h.get("name")
            if self._alive is None and alive:
                s.event("spawn", ms=ms, player="me", team=team, session=match, hero=hero, clock=clock, **where)
            elif self._alive and not alive:
                last = self._pos or pos
                s.event("death", ms=ms, player="me", team=team, session=match, hero=hero, clock=clock,
                        respawn_seconds=num(h.get("respawn_seconds"), 0), x=last[0], y=last[1], z=last[2])
                self._deaths += 1
            elif self._alive is False and alive:
                s.event("respawn", ms=ms, player="me", team=team, session=match, hero=hero, clock=clock, **where)
            if alive and pos != self._pos:
                s.position(*pos, ms=ms, player="me", team=team, session=match, clock=clock,
                           level=int(num(h.get("level"))), health=int(num(h.get("health"))))
                self._pos = pos
            self._alive = alive
            if "level" in h:
                s.stat("level", int(num(h.get("level"))), "max")
            if hero:
                s.stat("hero", str(hero))
        if own:
            if "kills" in p:
                kills = int(num(p.get("kills")))
                if self._kills is not None and kills > self._kills:
                    where = dict(x=self._pos[0], y=self._pos[1], z=self._pos[2]) if self._pos else {}
                    for _ in range(kills - self._kills):
                        s.event("kill", ms=ms, player="me", team=team, session=match, clock=clock, **where)
                self._kills = kills
            for key in ("kills", "deaths", "assists", "last_hits", "denies", "gold", "gpm", "xpm", "net_worth"):
                if key in p:
                    s.stat(key, int(num(p.get(key))))
        self._buildings_seen(s, doc.get("buildings"), ms, match, clock)
        self._roshan_seen(s, doc.get("roshan"), ms, match, clock)
        self._events_seen(s, doc.get("events"), ms, match)
        win = str(m.get("win_team") or "none")
        if win not in ("none", "") and s.stats.get("result") is None:
            result = None if not team else "win" if win == team else "loss"
            s.event("match_end", ms=ms, session=match, winner=win, result=result, clock=clock,
                    radiant_score=int(num(m.get("radiant_score"))), dire_score=int(num(m.get("dire_score"))))
            s.stat("result", result or win)
        self._count()

    def _buildings_seen(self, s: Session, b: Any, ms: int, match: str, clock: float) -> None:
        if not isinstance(b, dict):
            return
        now: dict[str, float] = {}
        for side, items in b.items():
            if not isinstance(items, dict):
                continue
            for name, v in items.items():
                if isinstance(v, dict):
                    now[f"{side}/{name}"] = num(v.get("health"), 0)
        for key, health in self._buildings.items():
            if health > 0 and now.get(key, 0) <= 0:
                side, _, name = key.partition("/")
                s.event("building_destroyed", ms=ms, session=match, team=side, building=name, clock=clock)
        self._buildings = now

    def _roshan_seen(self, s: Session, r: Any, ms: int, match: str, clock: float) -> None:
        if not isinstance(r, dict) or "alive" not in r:
            return
        alive = bool(r.get("alive"))
        if self._roshan is not None and alive != self._roshan:
            s.event("roshan_alive" if alive else "roshan_down", ms=ms, session=match, clock=clock)
        self._roshan = alive

    def _events_seen(self, s: Session, events: Any, ms: int, match: str) -> None:
        """The `events` list repeats what happened lately: each one is recorded once."""
        if not isinstance(events, list):
            return
        for e in events:
            if not isinstance(e, dict) or not e.get("event_type"):
                continue
            key = json.dumps(e, sort_keys=True)
            if key in self._seen_events:
                continue
            self._seen_events.add(key)
            kind = re.sub(r"[^a-z0-9_]+", "_", str(e["event_type"]).lower())
            extra = {k: v for k, v in e.items() if k != "event_type" and isinstance(v, (str, int, float, bool))}
            s.event(kind, ms=ms, session=match, **{("game_" + k if k in ("time", "ms", "event") else k): v
                                                  for k, v in extra.items()})
            if kind == "roshan_killed":
                s.stat("roshan_kills_seen", 1, "add")


ADAPTERS = [
    Cs2Gsi(
        id="cs2-gsi",
        game="Counter-Strike 2",
        exes=("cs2.exe",),
        port=int(os.environ.get("HEAT3D_CS2_GSI_PORT", "3311")),
        steam_folder="Counter-Strike Global Offensive",
        cfg_file=r"game\csgo\cfg\gamestate_integration_heat3d.cfg",
        providers=(
            "provider", "map", "round", "player_id", "player_state", "player_match_stats", "player_position",
            "player_weapons", "bomb", "phase_countdowns", "map_round_wins",
        ),
        source=r"Game State Integration: HTTP POSTs to 127.0.0.1:3311 configured by "
        r"<CS2>\game\csgo\cfg\gamestate_integration_heat3d.cfg",
        records=(
            "own position (world XYZ, inches) and view direction at up to 10 Hz while alive",
            "spawns (each round's first sighting, respawns) and deaths (health to 0, or the death count)",
            "kills with the active weapon, from the match statistics",
            "rounds (start, end, winner), bomb planted/defused/exploded, match phases and the result",
            "kills, assists, deaths, MVPs, score, money, damage, headshot kills, rounds won and lost",
        ),
        enable=r"`python -m heat3d_watch setup cs2-gsi --apply`, then start CS2 (it reads the file at start)",
        axes=Axes(up="z", flip_y=True, units="inches (Hammer units)"),
        verified="tested on documents in the published GSI format posted to the running adapter; "
        "the install and cfg folder were found on this machine; untested against the running game",
    ),
    DotaGsi(
        id="dota2-gsi",
        game="Dota 2",
        exes=("dota2.exe",),
        port=int(os.environ.get("HEAT3D_DOTA_GSI_PORT", "3312")),
        steam_folder="dota 2 beta",
        cfg_file=r"game\dota\cfg\gamestate_integration\gamestate_integration_heat3d.cfg",
        providers=("provider", "map", "player", "hero", "buildings", "roshan", "events"),
        source=r"Game State Integration: HTTP POSTs to 127.0.0.1:3312 configured by "
        r"<Dota>\game\dota\cfg\gamestate_integration\gamestate_integration_heat3d.cfg",
        records=(
            "own hero position (world X/Y, Dota units; no height) while alive",
            "spawn, deaths (with respawn time) and respawns of the own hero",
            "kills, from the kill count growing",
            "buildings destroyed, Roshan, game events (roshan_killed, aegis_picked_up, ...), game states, result",
            "kills, deaths, assists, last hits, denies, gold, GPM, XPM, level, hero",
        ),
        enable=r"`python -m heat3d_watch setup dota2-gsi --apply`, then start Dota 2 (it reads the file at start)",
        axes=Axes(up="z", flip_y=True, units="Dota world units (z always 0)"),
        verified="tested on documents in the published GSI format posted to the running adapter; "
        "untested against the running game (not installed here)",
    ),
]
