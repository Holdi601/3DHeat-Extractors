"""
Rocket League: the official Stats API, a local socket the game writes match data to.

Switched on in `<RL>\\TAGame\\Config\\TAStatsAPI.ini` - or `DefaultStatsAPI.ini`
when that does not exist, which the game reads only then - under
`[TAGame.MatchStatsExporter_TA]`:

    PacketSendRate   UpdateState messages per second; 0 (the default) switches the API off, at most 120
    Port             the TCP port the game listens on, default 49123 (WebPort 49124 is a WebSocket copy)

The game is the server: a client connects to 127.0.0.1:49123 and reads a stream
of JSON objects `{"Event": name, "Data": {...}}` (some builds send Data as a
JSON string), one after another without separators. Events are sent on the
tick they happen, independent of the rate:

    UpdateState   MatchGuid, Players[] (Name, PrimaryId, Shortcut, TeamNum 0 blue / 1 orange,
                  Score, Goals, Shots, Assists, Saves, Touches, CarTouches, Demos; car state
                  only for spectators), Game (Teams[], TimeSeconds, bOvertime, Ball {Speed,
                  TeamNum}, bReplay, bHasWinner, Winner, Arena, bHasTarget, Target)
    BallHit       Players[] (Name, Shortcut, TeamNum), Ball {PreHitSpeed, PostHitSpeed, Location {X,Y,Z}}
    GoalScored    GoalSpeed, GoalTime, ImpactLocation {X,Y,Z}, Scorer, Assister, BallLastTouch
    CrossbarHit   BallLocation {X,Y,Z}, BallSpeed, ImpactForce, BallLastTouch
    StatfeedEvent EventName, Type, MainTarget, SecondaryTarget
    MatchEnded    WinnerTeamNum
    MatchCreated, MatchInitialized, CountdownBegin, RoundStarted, GoalReplayStart/WillEnd/End,
    MatchPaused/Unpaused, PodiumStart, ReplayCreated, MatchDestroyed, ClockUpdatedSeconds

Locations are Unreal units (centimetres), Z up; the field is about 8200 x 10240
with the goals at Y = +-5120. Cars' positions are not part of the API: what has
a place is the ball where it was hit, scored or hit the crossbar.

Sources: https://www.rocketleague.com/en/developer/stats-api (the official page);
field casing and the TCP framing confirmed against the rlstatsapi client
(https://docs.rs/crate/rlstatsapi, MIT; read for facts only).
"""

from __future__ import annotations

import codecs
import glob
import json
import logging
import os
import re
import socket
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..session import Axes, Session
from .base import Adapter, Context
from .gsi import backup_once, steam_game

log = logging.getLogger("heat3d_watch")

SECTION = "[TAGame.MatchStatsExporter_TA]"
DEFAULT_PORT = 49123
TEAMS = {0: "blue", 1: "orange"}
#: A message is a few kilobytes; a buffer far beyond that is not a stream of them.
MAX_BUFFER = 4 << 20
STATS = (("Goals", "goals"), ("Assists", "assists"), ("Saves", "saves"), ("Shots", "shots"),
         ("Demos", "demos"), ("Touches", "touches"), ("Score", "score"))


def snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def split_objects(buf: str) -> tuple[list[str], str, int]:
    """
    Complete top-level JSON objects at the start of `buf`, what is left, and junk skipped.

    The stream has no separators, so an object ends where its braces close -
    counted outside strings. A message cut in half stays in the rest until the
    other half arrives.
    """
    out: list[str] = []
    junk = 0
    i, n = 0, len(buf)
    while i < n:
        c = buf[i]
        if c != "{":
            if not c.isspace() and c != "\x00":
                junk += 1
            i += 1
            continue
        depth, j, in_str, esc = 0, i, False, False
        end = -1
        while j < n:
            ch = buf[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
            j += 1
        if end < 0:
            return out, buf[i:], junk
        out.append(buf[i:end + 1])
        i = end + 1
    return out, "", junk


def vec(v: Any) -> tuple[float, float, float] | None:
    if not isinstance(v, dict):
        return None
    try:
        return float(v["X"]), float(v["Y"]), float(v["Z"])
    except (KeyError, TypeError, ValueError):
        return None


def epic_install(launch: str = "RocketLeague.exe") -> Path | None:
    """The Epic Games launcher's manifests say where each game is installed."""
    base = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
    for item in glob.glob(os.path.join(base, "Epic", "EpicGamesLauncher", "Data", "Manifests", "*.item")):
        try:
            doc = json.loads(Path(item).read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            continue
        if str(doc.get("LaunchExecutable") or "").replace("\\", "/").lower().endswith(launch.lower()):
            p = Path(str(doc.get("InstallLocation") or ""))
            if p.is_dir():
                return p
    return None


def read_ini(path: Path) -> dict[str, str]:
    """Keys of the Stats API section."""
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    inside = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("["):
            inside = s.lower() == SECTION.lower()
        elif inside and "=" in s and not s.startswith(";"):
            k, _, v = s.partition("=")
            out[k.strip()] = v.strip()
    return out


@dataclass
class RocketLeagueStats(Adapter):
    host: str = "127.0.0.1"
    #: The game's Stats API port. 0: the `Port` of the ini file, else 49123.
    port: int = 0
    #: The game's install folder. Empty: next to the running exe, Steam, or the Epic launcher.
    install: str = ""
    #: The rate `setup` sets when the API is off.
    packet_rate: int = 30
    #: The player's own name in matches. Empty: the player the game's camera follows outside replays.
    player_name: str = ""
    #: Seconds between connection attempts while the game does not answer.
    retry: float = 2.0
    _sock: socket.socket | None = None
    _next_try: float = 0.0
    _decoder: Any = None
    _buf: str = ""
    _session: Session | None = None
    _guid: str | None = None
    _targets: Counter = field(default_factory=Counter)
    _players: dict[str, dict[str, Any]] = field(default_factory=dict)
    _teams: dict[int, int] = field(default_factory=dict)
    _ended: bool = False
    _bad: int = 0

    # -- where the game is -----------------------------------------------------

    def find_install(self, ctx: Context | None = None) -> Path | None:
        if self.install:
            return Path(self.install)
        if ctx is not None and ctx.exe is not None:
            parents = Path(ctx.exe).parents  # <RL>\Binaries\Win64\RocketLeague.exe
            if len(parents) > 2 and (parents[2] / "TAGame").is_dir():
                return parents[2]
        return steam_game("rocketleague") or epic_install()

    def ini_path(self, root: Path) -> Path:
        config = root / "TAGame" / "Config"
        ta = config / "TAStatsAPI.ini"
        return ta if ta.exists() else config / "DefaultStatsAPI.ini"

    def setup(self, ctx: Context, *, apply: bool) -> list[str]:
        root = self.find_install(ctx)
        if root is None or not (root / "TAGame" / "Config").is_dir():
            return []
        path = self.ini_path(root)
        try:
            raw = path.read_bytes().decode("utf-8", errors="replace")
        except OSError:
            raw = ""
        nl = "\r\n" if "\r\n" in raw or not raw else "\n"
        lines = raw.splitlines()
        have = read_ini(path) if raw else {}
        want: dict[str, str] = {}
        try:
            rate = float(have.get("PacketSendRate", "0") or 0)
        except ValueError:
            rate = 0
        if rate <= 0:
            want["PacketSendRate"] = str(self.packet_rate)
        if not have.get("Port"):
            want["Port"] = str(self.port or DEFAULT_PORT)
        if not want:
            return []
        start = next((i for i, l in enumerate(lines) if l.strip().lower() == SECTION.lower()), None)
        if start is None:
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(SECTION)
            start = len(lines) - 1
        end = next((i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("[")), len(lines))
        todo = dict(want)
        for i in range(start + 1, end):
            k = lines[i].partition("=")[0].strip()
            if k in todo and not lines[i].strip().startswith(";"):
                lines[i] = f"{k}={todo.pop(k)}"
        insert = [f"{k}={v}" for k, v in todo.items()]
        while end > start + 1 and not lines[end - 1].strip():
            end -= 1
        lines[end:end] = insert
        if apply:
            backup_once(path)
            path.write_text(nl.join(lines) + nl, encoding="utf-8", newline="")
        said = ", ".join(f"{k}={v}" for k, v in want.items())
        return [f"set {said} in {SECTION} of {path} (read when Rocket League starts)"]

    # -- the connection ----------------------------------------------------------

    def start(self, ctx: Context) -> None:
        if not self.port:
            root = self.find_install(ctx)
            try:
                self.port = int(read_ini(self.ini_path(root)).get("Port") or DEFAULT_PORT) if root else DEFAULT_PORT
            except ValueError:
                self.port = DEFAULT_PORT
        self._next_try = 0.0
        self._reset_match()

    def _connect(self) -> None:
        now = time.monotonic()
        if now < self._next_try:
            return
        self._next_try = now + self.retry
        try:
            s = socket.create_connection((self.host, self.port), timeout=0.2)
        except OSError:
            return
        s.settimeout(0.05)
        self._sock = s
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._buf = ""

    def _drop(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None

    def poll(self, ctx: Context) -> None:
        if self._sock is None:
            self._connect()
            if self._sock is None:
                return
        deadline = time.monotonic() + 0.1
        while time.monotonic() < deadline:
            try:
                chunk = self._sock.recv(65536)
            except (socket.timeout, BlockingIOError):
                break
            except OSError:
                self._drop()
                break
            if not chunk:
                # The game closed the socket: it is quitting, or the API was switched off.
                self._drop()
                break
            self._buf += self._decoder.decode(chunk)
        objects, self._buf, junk = split_objects(self._buf)
        if len(self._buf) > MAX_BUFFER:
            self._buf, junk = "", junk + 1
        if junk:
            self._bad += 1
        for text in objects:
            try:
                msg = json.loads(text)
                name = str(msg.get("Event") or msg.get("event") or "")
                data = msg.get("Data", msg.get("data"))
                if isinstance(data, str):
                    data = json.loads(data) if data.strip() else {}
                if not name or not isinstance(data, (dict, type(None))):
                    raise ValueError("not a Stats API message")
            except (ValueError, AttributeError):
                self._bad += 1
                continue
            try:
                self.handle(ctx, name, data or {})
            except (TypeError, ValueError, AttributeError, KeyError):
                self._bad += 1
        if self._session is not None and self._bad:
            self._session.stat("bad_packets", self._bad)

    def stop(self, ctx: Context) -> None:
        self._drop()
        self._end(ctx)

    # -- matches -------------------------------------------------------------------

    def _reset_match(self) -> None:
        self._guid = None
        self._targets = Counter()
        self._players = {}
        self._teams = {}
        self._ended = False

    def _open(self, ctx: Context, guid: str | None) -> Session:
        self._end(ctx)
        self._reset_match()
        self._guid = guid or None
        self._session = ctx.open(self, meta={"match_guid": guid} if guid else {})
        return self._session

    def _end(self, ctx: Context) -> None:
        s = self._session
        if s is None:
            return
        for name, p in self._players.items():
            s.event("player_result", player=self._who(name), team=TEAMS.get(p.get("TeamNum")),
                    session=self._guid, **{stat: p.get(key) for key, stat in STATS})
        if self._bad:
            s.stat("bad_packets", self._bad)
        ctx.close(s)
        self._session = None

    def me(self) -> str | None:
        if self.player_name:
            return self.player_name
        if self._targets:
            return self._targets.most_common(1)[0][0]
        return None

    def _who(self, name: Any) -> str:
        name = str(name or "")
        return "me" if name and name == self.me() else name

    def _team(self, ref: Any) -> str | None:
        return TEAMS.get(ref.get("TeamNum")) if isinstance(ref, dict) else None

    def handle(self, ctx: Context, name: str, data: dict[str, Any]) -> None:
        guid = str(data.get("MatchGuid") or "") or None
        if name in ("MatchCreated", "MatchInitialized"):
            if self._session is None or (guid and guid != self._guid) or self._ended:
                self._open(ctx, guid)
            self._session.event(snake(name), session=self._guid)
            return
        if name == "MatchDestroyed":
            if self._session is not None:
                self._session.event("match_destroyed", session=self._guid)
            self._end(ctx)
            self._reset_match()
            return
        if self._session is None or (guid and self._guid and guid != self._guid):
            if name not in ("UpdateState", "RoundStarted", "CountdownBegin", "BallHit", "GoalScored"):
                return  # between matches: nothing to file it under
            self._open(ctx, guid)
        if guid and not self._guid:
            self._guid = guid
            self._session.meta["match_guid"] = guid
        s = self._session
        sid = self._guid
        if name == "UpdateState":
            self._update(s, data)
        elif name == "BallHit":
            ball = data.get("Ball") if isinstance(data.get("Ball"), dict) else {}
            at = vec(ball.get("Location"))
            for p in data.get("Players") or [{}]:
                p = p if isinstance(p, dict) else {}
                s.event("ball_hit", player=self._who(p.get("Name")) or None, team=self._team(p), session=sid,
                        pre_speed=ball.get("PreHitSpeed"), post_speed=ball.get("PostHitSpeed"),
                        **({"x": at[0], "y": at[1], "z": at[2]} if at else {}))
        elif name == "GoalScored":
            at = vec(data.get("ImpactLocation"))
            scorer = data.get("Scorer") if isinstance(data.get("Scorer"), dict) else {}
            assister = data.get("Assister") if isinstance(data.get("Assister"), dict) else {}
            s.event("goal", player=self._who(scorer.get("Name")) or None, team=self._team(scorer), session=sid,
                    assister=self._who(assister.get("Name")) or None, speed=data.get("GoalSpeed"),
                    goal_time=data.get("GoalTime"), **({"x": at[0], "y": at[1], "z": at[2]} if at else {}))
        elif name == "CrossbarHit":
            at = vec(data.get("BallLocation"))
            last = data.get("BallLastTouch") if isinstance(data.get("BallLastTouch"), dict) else {}
            p = last.get("Player") if isinstance(last.get("Player"), dict) else {}
            s.event("crossbar_hit", player=self._who(p.get("Name")) or None, team=self._team(p), session=sid,
                    speed=data.get("BallSpeed"), force=data.get("ImpactForce"),
                    **({"x": at[0], "y": at[1], "z": at[2]} if at else {}))
        elif name == "StatfeedEvent":
            main = data.get("MainTarget") if isinstance(data.get("MainTarget"), dict) else {}
            other = data.get("SecondaryTarget") if isinstance(data.get("SecondaryTarget"), dict) else {}
            s.event("feed_" + snake(str(data.get("EventName") or "event")), player=self._who(main.get("Name")) or None,
                    team=self._team(main), session=sid, label=data.get("Type"),
                    target=self._who(other.get("Name")) or None)
            if str(data.get("EventName")) == "Demolish" and self.me():
                if self._who(other.get("Name")) == "me":
                    s.stat("demolished", 1, "add")
        elif name == "MatchEnded":
            winner = data.get("WinnerTeamNum")
            mine = self._players.get(self.me() or "", {}).get("TeamNum")
            result = None if mine is None or winner is None else "win" if int(winner) == int(mine) else "loss"
            s.event("match_end", session=sid, winner=TEAMS.get(winner, winner), result=result,
                    blue=self._teams.get(0), orange=self._teams.get(1))
            if result:
                s.stat("result", result)
            self._ended = True
        elif name == "ClockUpdatedSeconds":
            if data.get("bOvertime"):
                s.stat("overtime", 1, "max")
        else:
            s.event({"RoundStarted": "kickoff"}.get(name, snake(name)), session=sid)

    def _update(self, s: Session, data: dict[str, Any]) -> None:
        game = data.get("Game") if isinstance(data.get("Game"), dict) else {}
        target = game.get("Target") if isinstance(game.get("Target"), dict) else {}
        if game.get("bHasTarget") and not game.get("bReplay") and target.get("Name"):
            self._targets[str(target["Name"])] += 1
        for p in data.get("Players") or []:
            if isinstance(p, dict) and p.get("Name"):
                self._players[str(p["Name"])] = p
        for t in game.get("Teams") or []:
            if isinstance(t, dict) and t.get("TeamNum") is not None:
                self._teams[int(t["TeamNum"])] = int(t.get("Score") or 0)
        if game.get("Arena"):
            if s.meta.get("arena") != game["Arena"]:
                s.meta["arena"] = str(game["Arena"])
        me = self._players.get(self.me() or "")
        if me is not None:
            for key, stat in STATS:
                if key in me:
                    s.stat(stat, int(me[key]))
            team = me.get("TeamNum")
            if team in self._teams and (1 - team) in self._teams:
                s.stat("team_goals", self._teams[team])
                s.stat("other_goals", self._teams[1 - team])


ADAPTERS = [
    RocketLeagueStats(
        id="rocketleague-stats",
        game="Rocket League",
        kind="gsi",
        exes=("RocketLeague.exe",),
        interval=0.05,
        source=r"Stats API: TCP JSON stream from 127.0.0.1:49123 (Port in "
        r"<RL>\TAGame\Config\TAStatsAPI.ini or DefaultStatsAPI.ini)",
        records=(
            "ball hits with the ball's place (XYZ, Unreal units), who hit it and the speeds",
            "goals with the impact place, scorer, assister and speed; crossbar hits with the place",
            "stat feed (demolitions, saves, shots, epic saves, ...), kickoffs, replays, match start and end, winner",
            "own goals, assists, saves, shots, demos, touches, score and the team scores, from UpdateState",
            "every player's final line at match end",
            "no car positions: the API has none",
        ),
        enable=r"`python -m heat3d_watch setup rocketleague-stats --apply` (sets PacketSendRate=30), "
        "then start Rocket League",
        axes=Axes(up="z", units="Unreal units (centimetres)"),
        verified="tested on messages in the documented format sent by a fake local server; "
        "untested against the game (not installed here)",
    ),
]
