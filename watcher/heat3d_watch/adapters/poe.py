"""
Path of Exile and Path of Exile 2: the client log, `<install>\\logs\\Client.txt`.

Both games append to one Client.txt in their install folder for as long as
they are installed (it grows across launches). Lines look like

    2024/01/01 12:00:00 123456789 cff94598 [INFO Client 1234] : You have entered Lioneye's Watch.
    2024/01/01 12:00:00 123456789 cff94598 [DEBUG Client 1234] Generating level 2 area "1_1_2" with seed 1
    2024/01/01 12:00:00 123456789 cff94598 [INFO Client 1234] : Name (Witch) is now level 2
    2024/01/01 12:00:00 123456789 cff94598 [INFO Client 1234] : Name has been slain.
    2024/01/01 12:00:00 123456789 cff94598 [INFO Client 1234] @From Name: Hi, I would like to buy your ...

- local time, a millisecond counter, a hash, the level and thread, then what the
  chat window shows. The messages are the English client's; another language's
  client writes them translated and is not understood here. Nothing gives a
  position. The format is not documented by Grinding Gear Games; it is the
  game's plain-text log, as community tools (trade helpers, level guides) have
  read it for years - e.g. https://github.com/klayveR/poe-log-monitor (MIT) for
  the event list.

The install folder is the running executable's; without one the Steam
libraries and the standalone client's default folder are searched.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..session import Axes, Session
from ._logfollow import Follower, game_started, newest, steam_libraries
from .base import Adapter, Context

_LINE = re.compile(r"^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) \d+ \S+ \[(\w+) [^\]]*\] ?(.*)$")
_ENTERED = re.compile(r"^: You have entered (.+?)\.?$")
_GENERATING = re.compile(r'^Generating level (\d+) area "([^"]+)"')
_LEVEL = re.compile(r"^: (\S+) \(([^)]+)\) is now level (\d+)")
_SLAIN = re.compile(r"^: (\S+) has been slain\.?$")
_JOINED = re.compile(r"^: (\S+) has (joined|left) the area\.?$")
_WHISPER = re.compile(r"^@(From|To) (?:<[^>]*> )?([^:]+): (.*)$")
_TRADE = re.compile(r"^: Trade (accepted|cancelled)\.?$")

#: Words that make a whisper a trade request, as the trade site writes them.
_BUY = re.compile(r"\b(would like to buy|wtb|i'd like to buy)\b", re.IGNORECASE)

FOLDERS = ("Path of Exile", "Path of Exile 2")


def line_time(line: str) -> float | None:
    try:
        return time.mktime(time.strptime(line[:19], "%Y/%m/%d %H:%M:%S"))
    except ValueError:
        return None


def candidates(exe: Path | None) -> list[Path]:
    """Where Client.txt can be: next to the running game, else the usual install folders."""
    if exe is not None:
        return [exe.parent / "logs" / "Client.txt"]
    out: list[Path] = []
    for lib in steam_libraries():
        out += [lib / "steamapps" / "common" / f / "logs" / "Client.txt" for f in FOLDERS]
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
        if base:
            out += [Path(base) / "Grinding Gear Games" / f / "logs" / "Client.txt" for f in FOLDERS]
    return out


def game_name(path: Path) -> str:
    return "Path of Exile 2" if any("path of exile 2" in p.lower() for p in path.parts) else "Path of Exile"


class ClientLog:
    """Client.txt lines into a session: zones, levels, deaths, whispers and trades."""

    def __init__(self, session_for):
        self.session_for = session_for
        self.me: str | None = None
        self.area_level: int | None = None
        self.area: str | None = None
        self.zones: set[str] = set()

    def line(self, text: str) -> None:
        m = _LINE.match(text)
        if not m:
            return
        stamp, level, msg = m.groups()
        ts = line_time(stamp)
        msg = msg.strip()
        if (g := _GENERATING.match(msg)) is not None:
            self.area_level, self.area = int(g.group(1)), g.group(2)
            return
        s: Session | None = None

        def ev(name: str, **kw) -> Session:
            nonlocal s
            s = self.session_for(ts)
            t = ts - s.started if ts else None
            s.event(name, t=t, ms=ts * 1000 if ts else None, **kw)
            return s

        if (g := _ENTERED.match(msg)) is not None:
            zone = g.group(1)
            ev("zone", player=self.me or "me", zone=zone, area_level=self.area_level, area=self.area)
            self.zones.add(zone)
            s.stat("zones_entered", 1, "add")
            s.stat("zones_distinct", len(self.zones))
            self.area_level = self.area = None
        elif (g := _LEVEL.match(msg)) is not None:
            name, cls, lvl = g.group(1), g.group(2), int(g.group(3))
            self.me = self.me or name
            ev("level_up", player=name, character_class=cls, level=lvl)
            if name == self.me:
                s.stat("levels_gained", 1, "add")
                s.stat("level", lvl, "max")
                s.stat("character_class", cls)
        elif (g := _SLAIN.match(msg)) is not None:
            name = g.group(1)
            ev("death", player=name)
            if self.me is None or name == self.me:
                s.stat("deaths", 1, "add")
        elif (g := _JOINED.match(msg)) is not None:
            ev("player_" + g.group(2), player=g.group(1))
        elif (g := _WHISPER.match(msg)) is not None:
            way, who, said = g.groups()
            trade = bool(_BUY.search(said))
            ev("whisper_in" if way == "From" else "whisper_out", player=who.strip(), trade=trade)
            s.stat("whispers", 1, "add")
            if trade:
                s.stat("trade_requests_received" if way == "From" else "trade_requests_sent", 1, "add")
        elif (g := _TRADE.match(msg)) is not None:
            ev("trade_" + g.group(1), player=self.me or "me")
            if g.group(1) == "accepted":
                s.stat("trades", 1, "add")


@dataclass
class PoEAdapter(Adapter):
    #: Client.txt to follow; empty: found from the running game or the usual installs.
    path: Path | None = None
    _session: Session | None = field(default=None, repr=False)
    _follow: Follower | None = field(default=None, repr=False)
    _log: ClientLog | None = field(default=None, repr=False)

    def start(self, ctx: Context) -> None:
        self._session = None
        paths = [Path(self.path)] if self.path else candidates(ctx.exe)

        def find() -> Path | None:
            return newest(p for p in paths if p.exists())

        def session_for(ts: float | None) -> Session:
            if self._session is None:
                current = find()
                game = game_name(current) if current else self.game
                self._session = ctx.open(self, game=game, when=ts if ts and abs(time.time() - ts) < 86400 else None)
            return self._session

        self._log = ClientLog(session_for)
        self._follow = Follower(find, game_started(ctx.pid), line_time)

    def poll(self, ctx: Context) -> None:
        if self._follow is None or self._log is None:
            return
        for line in self._follow.lines():
            try:
                self._log.line(line)
            except (ValueError, IndexError, OverflowError):
                if self._session is not None:
                    self._session.stat("bad_lines", 1, "add")

    def stop(self, ctx: Context) -> None:
        self.poll(ctx)
        ctx.close(self._session)
        self._session = None
        self._follow = None
        self._log = None


ADAPTERS = [
    PoEAdapter(
        id="poe",
        game="Path of Exile",
        kind="log",
        exes=(
            "PathOfExile.exe",
            "PathOfExile_x64.exe",
            "PathOfExileSteam.exe",
            "PathOfExile_x64Steam.exe",
            "PathOfExileEGS.exe",
            "PathOfExile_x64EGS.exe",
            "PathOfExile_KG.exe",
            "PathOfExile_x64_KG.exe",
        ),
        multiplayer=True,
        source=r"<install>\logs\Client.txt of Path of Exile or Path of Exile 2 (followed; the install folder is "
        r"the running executable's)",
        records=(
            "zones entered with area level",
            "level-ups with class and level",
            "deaths (\"has been slain\")",
            "players joining and leaving the area, whispers in and out, trade requests, trades accepted",
            "deaths, levels gained, highest level, zones entered, distinct zones, whispers, trade requests, trades",
        ),
        enable="",
        interval=0.5,
        axes=Axes(units="no positions recorded"),
        verified="tested on synthetic Client.txt lines in the game's format; untested against the game",
    )
]
