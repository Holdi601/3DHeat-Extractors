"""
EVE Online: the client's game logs, `Documents\\EVE\\logs\\Gamelogs\\*.txt`.

The client writes one file per character per login,
`<yyyymmdd>_<hhmmss>[_<character id>].txt`; logging to file is on by default. Several clients running side by
side each write their own, so every file written since the game started is
followed. Header and lines (EVE time is UTC):

    ------------------------------------------------------------
      Gamelog
      Listener: Character Name
      Session Started: 2024.07.02 19:41:40
    ------------------------------------------------------------
    [ 2024.07.02 20:31:28 ] (combat) <color=0xff00ffff><b>200</b> <color=0x77ffffff><font size=10>to</font>
        <b><color=0xffffffff>Pilot[TICK](Ship)</b><font size=10><color=0x77ffffff> - Weapon - Penetrates
    [ 2024.06.25 15:20:01 ] (combat) <color=0xffcc0000><b>375</b> ... from ... - Weapon - Hits
    [ 2024.06.25 15:21:00 ] (bounty) <font size=12><b><color=0xff00aa00>15,000 ISK</b> added to next bounty payout
    [ 2024.06.25 15:30:00 ] (None) Jumping from System A to System B

(the combat line is one line in the file). Formatting tags are stripped before
matching. The file has been UTF-8 and UTF-16 over the years; both are read.
The format is not documented by CCP; it is the client's plain-text log, read as
community combat-log tools read it (e.g. https://docs.rs/crate/eve-log-parser).
No positions.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..session import Axes, Session
from ._logfollow import Follower, SLACK, game_started
from .base import Adapter, Context

_TAGS = re.compile(r"<[^>]*>")
_LINE = re.compile(r"^\[ (\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}) \] \((\w+)\) ?(.*)$")
_LISTENER = re.compile(r"^\s*Listener: (.+?)\s*$")
_HIT = re.compile(r"^(\d+) (to|from) (.+?) - (.+?)(?: - ([A-Za-z][A-Za-z ]*))?\s*$")
_MISS_OUT = re.compile(r"^Your (?:group of )?(.+?) misses (.+?) completely(?: - (.+))?$")
_MISS_IN = re.compile(r"^(.+?) misses you completely(?: - (.+))?$")
_BOUNTY = re.compile(r"^([\d,.]+) ISK added to next bounty payout")
_JUMP = re.compile(r"Jumping from (.+?) to (.+?)\.?\s*$")
_WHO = re.compile(r"^(?P<name>[^\[(]+?)\s*(?:\[(?P<corp>[^\]]*)\])?\s*(?:\((?P<ship>[^)]*)\))?$")


def line_time(line: str) -> float | None:
    m = _LINE.match(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y.%m.%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def gamelog_dirs() -> list[Path]:
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    out = [home / "Documents" / "EVE" / "logs" / "Gamelogs"]
    for env in ("OneDrive", "OneDriveConsumer"):
        if os.environ.get(env):
            out.append(Path(os.environ[env]) / "Documents" / "EVE" / "logs" / "Gamelogs")
    return out


class Gamelog:
    """Lines of one character's gamelog into the session: damage, misses, bounties, jumps."""

    def __init__(self, session_for):
        self.session_for = session_for
        self.me: str | None = None

    def line(self, text: str) -> None:
        if self.me is None and (m := _LISTENER.match(text)) is not None:
            self.me = m.group(1)
            return
        m = _LINE.match(text)
        if not m:
            return
        ts = line_time(text) or time.time()
        kind, body = m.group(2), _TAGS.sub("", m.group(3)).strip()
        body = re.sub(r"\s+", " ", body)
        me = self.me or "me"

        def ev(name: str, **kw) -> Session:
            s = self.session_for(ts)
            s.event(name, t=ts - s.started, ms=ts * 1000, **kw)
            return s

        if kind == "combat":
            if (h := _HIT.match(body)) is not None:
                amount, way, other, weapon, quality = h.groups()
                who = _WHO.match(other.strip())
                name = who.group("name").strip() if who else other
                ship = who.group("ship") if who else None
                if way == "to":
                    s = ev("damage_dealt", player=me, target=name, target_ship=ship, damage=int(amount), weapon=weapon,
                           quality=quality)
                    s.stat("damage_dealt", int(amount), "add")
                    s.stat("hits_dealt", 1, "add")
                else:
                    s = ev("damage_received", player=me, attacker=name, attacker_ship=ship, damage=int(amount),
                           weapon=weapon, quality=quality)
                    s.stat("damage_received", int(amount), "add")
                    s.stat("hits_received", 1, "add")
            elif (h := _MISS_IN.match(body)) is not None:
                s = ev("miss_received", player=me, attacker=h.group(1))
                s.stat("misses_received", 1, "add")
            elif (h := _MISS_OUT.match(body)) is not None:
                s = ev("miss_dealt", player=me, weapon=h.group(1), target=h.group(2))
                s.stat("misses_dealt", 1, "add")
        elif kind == "bounty" and (b := _BOUNTY.match(body)) is not None:
            isk = float(b.group(1).replace(",", ""))
            s = ev("bounty", player=me, isk=isk)
            s.stat("bounty_kills", 1, "add")
            s.stat("bounty_isk", isk, "add")
        elif (j := _JUMP.search(body)) is not None:
            s = ev("jump", player=me, origin=j.group(1), destination=j.group(2))
            s.stat("jumps", 1, "add")


@dataclass
class EveAdapter(Adapter):
    #: Gamelogs folder; empty: Documents\EVE\logs\Gamelogs.
    folder: Path | None = None
    _session: Session | None = field(default=None, repr=False)
    _files: dict[Path, tuple[Follower, Gamelog]] = field(default_factory=dict, repr=False)
    _since: float = field(default=0.0, repr=False)
    _next_scan: float = field(default=0.0, repr=False)

    def folders(self) -> list[Path]:
        return [Path(self.folder)] if self.folder else gamelog_dirs()

    def start(self, ctx: Context) -> None:
        self._session = None
        self._files = {}
        self._next_scan = 0.0
        self._since = game_started(ctx.pid)

    def _session_for(self, ctx: Context):
        def session_for(ts: float | None) -> Session:
            if self._session is None:
                self._session = ctx.open(self, when=ts if ts and abs(time.time() - ts) < 86400 else None)
            return self._session

        return session_for

    def _scan(self, ctx: Context) -> None:
        for folder in self.folders():
            for path in folder.glob("*.txt"):
                if path in self._files:
                    continue
                try:
                    if path.stat().st_mtime < self._since - SLACK:
                        continue
                except OSError:
                    continue
                follower = Follower(lambda p=path: p, self._since, line_time)
                self._files[path] = (follower, Gamelog(self._session_for(ctx)))

    def poll(self, ctx: Context) -> None:
        if time.monotonic() >= self._next_scan:
            self._next_scan = time.monotonic() + 2.0
            self._scan(ctx)
        for follower, log in self._files.values():
            for line in follower.lines():
                try:
                    log.line(line)
                except (ValueError, IndexError):
                    if self._session is not None:
                        self._session.stat("bad_lines", 1, "add")

    def stop(self, ctx: Context) -> None:
        self.poll(ctx)
        ctx.close(self._session)
        self._session = None
        self._files = {}


ADAPTERS = [
    EveAdapter(
        id="eve",
        game="EVE Online",
        kind="log",
        exes=("exefile.exe",),
        multiplayer=True,
        source=r"%USERPROFILE%\Documents\EVE\logs\Gamelogs\*.txt (every file written since the client started)",
        records=(
            "damage dealt and received with the other ship, pilot, weapon and hit quality",
            "misses dealt and received",
            "bounties paid (NPC kills) with ISK",
            "jumps between systems",
            "damage dealt and received, hits, misses, bounty kills and ISK, jumps",
        ),
        enable="",
        interval=0.5,
        axes=Axes(units="no positions recorded"),
        verified="tested on synthetic gamelog lines in the format published by community parsers; untested "
        "against the game",
    )
]
