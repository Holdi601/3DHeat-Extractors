"""
Hearthstone: the game's Power.log, which it writes once `log.config` asks for it.

Hearthstone reads `%LOCALAPPDATA%\\Blizzard\\Hearthstone\\log.config` at launch;
a `[Power]` section with `LogLevel=1` and `FilePrinting=true` makes it write
every game's state changes to `<install>\\Logs\\Power.log` (newer clients put
each launch's logs in a folder of their own, `Logs\\Hearthstone_<date>\\`). This
is the documented debug logging deck trackers have switched on since 2014
(Hearthstone Deck Tracker's setup writes the same section). `setup()` adds the
section, with a backup of the file as it was; the game needs a restart to read it.

The lines used, all from `GameState.DebugPrintPower()` (the `PowerTaskList`
lines repeat them and are skipped):

    D 20:15:01.1234567 GameState.DebugPrintPower() - CREATE_GAME
    D 20:15:01.1234567 GameState.DebugPrintPower() -     Player EntityID=2 PlayerID=1 GameAccountId=[hi=1 lo=2]
    D 20:15:01.1234567 GameState.DebugPrintGame() - PlayerID=1, PlayerName=Name#1234
    D 20:15:02.1234567 GameState.DebugPrintPower() -     FULL_ENTITY - Updating [entityName=... id=4 zone=HAND
                       zonePos=0 cardId=CS2_029 player=1] CardID=CS2_029
    D 20:16:00.1234567 GameState.DebugPrintPower() - TAG_CHANGE Entity=GameEntity tag=TURN value=3
    D 20:25:00.1234567 GameState.DebugPrintPower() - TAG_CHANGE Entity=Name#1234 tag=PLAYSTATE value=WON
    D 20:25:00.1234567 GameState.DebugPrintPower() - TAG_CHANGE Entity=GameEntity tag=STATE value=COMPLETE

The player's own side is the one whose starting hand has its cards named
(the opponent's are hidden), else the side that is not the AI (account 0) or
"UNKNOWN HUMAN PLAYER". Lines carry only the time of day; the date is today's.
No positions.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from ..session import Axes, Session
from ._logfollow import Follower, game_started, newest
from .base import Adapter, Context

_LINE = re.compile(r"^[DIWE] (\d{1,2}:\d{2}:\d{2}(?:\.\d+)?) GameState\.DebugPrint(Power|Game)\(\) - (.*)$")
_PLAYER = re.compile(r"Player EntityID=(\d+) PlayerID=(\d+) GameAccountId=\[hi=(\d+) lo=(\d+)\]")
_NAME = re.compile(r"PlayerID=(\d+), PlayerName=(.+)$")
_HAND = re.compile(r"\[entityName=.*? id=\d+ zone=HAND zonePos=\d+ cardId=(\S*) player=(\d+)\]")
_CREATING = re.compile(r"FULL_ENTITY - Creating ID=\d+ CardID=(\S*)")
_TAG = re.compile(r"^\s*tag=(\w+) value=(\S+)")
_CHANGE = re.compile(r"TAG_CHANGE Entity=(.+?) tag=(\w+) value=(\S+)")
_ENTITY_ID = re.compile(r"\bid=(\d+)\b")

#: Sections of log.config this adapter needs, as Hearthstone's own logger reads them.
CONFIG = {"Power": {"LogLevel": "1", "FilePrinting": "true", "ConsolePrinting": "false", "ScreenPrinting": "false",
                    "Verbose": "true"}}


def config_path() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "Blizzard" / "Hearthstone" / "log.config"


def clock_time(clock: str, now: float | None = None) -> float:
    """Today's date with the line's time of day; yesterday's when that would be in the future."""
    now = time.time() if now is None else now
    hms, _, frac = clock.partition(".")
    h, m, s = (int(v) for v in hms.split(":"))
    day = datetime.fromtimestamp(now).replace(hour=h, minute=m, second=s, microsecond=0)
    ts = day.timestamp() + (float("0." + frac) if frac else 0.0)
    if ts > now + 60:
        ts = (day - timedelta(days=1)).timestamp() + (float("0." + frac) if frac else 0.0)
    return ts


def merge_config(text: str) -> tuple[str, list[str]]:
    """`text` with CONFIG's settings in it; and what changed. Other sections and keys stay as they are."""
    lines = text.splitlines()
    changes: list[str] = []
    for section, keys in CONFIG.items():
        start = next((i for i, ln in enumerate(lines) if ln.strip().lower() == f"[{section.lower()}]"), None)
        if start is None:
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(f"[{section}]")
            lines += [f"{k}={v}" for k, v in keys.items()]
            changes.append(f"add section [{section}] (" + ", ".join(f"{k}={v}" for k, v in keys.items()) + ")")
            continue
        end = next((i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("[")), len(lines))
        for k, v in keys.items():
            idx = next((i for i in range(start + 1, end) if lines[i].split("=", 1)[0].strip().lower() == k.lower()), None)
            if idx is None:
                lines.insert(end, f"{k}={v}")
                end += 1
                changes.append(f"[{section}] add {k}={v}")
            elif lines[idx].split("=", 1)[1].strip().lower() != v:
                changes.append(f"[{section}] {lines[idx].strip()} -> {k}={v}")
                lines[idx] = f"{k}={v}"
    return "\n".join(lines) + "\n", changes


class PowerLog:
    """Power.log lines into a session: one game after another, who won, how many turns."""

    def __init__(self, session_for, clock=clock_time):
        self.session_for = session_for
        self.clock = clock
        self.games = 0
        self.reset()

    def reset(self) -> None:
        self.entity_player: dict[int, int] = {}
        self.ai: set[int] = set()
        self.names: dict[int, str] = {}
        self.friendly: int | None = None
        self.turn = 0
        self.results: dict[int, str] = {}
        self.pending_card = False
        self.pending_controller: int | None = None
        self.in_game = False

    def line(self, text: str) -> None:
        m = _LINE.match(text)
        if not m:
            return
        clock, kind, msg = m.groups()
        ts = self.clock(clock)
        if kind == "Game":
            n = _NAME.search(msg)
            if n:
                self.names[int(n.group(1))] = n.group(2).strip()
            return
        body = msg.strip()
        if body == "CREATE_GAME":
            if self.in_game:
                self._end(ts, complete=False)
            self.reset()
            self.in_game = True
            self.games += 1
            self._event(ts, "game_start")
            return
        if (p := _PLAYER.search(body)) is not None:
            ent, pid, hi, lo = (int(v) for v in p.groups())
            self.entity_player[ent] = pid
            if hi == 0 and lo == 0:
                self.ai.add(pid)
            return
        if (h := _HAND.search(body)) is not None and body.startswith(("FULL_ENTITY", "SHOW_ENTITY")):
            if h.group(1) and self.friendly is None and self.turn <= 1:
                self.friendly = int(h.group(2))
            return
        if (c := _CREATING.search(body)) is not None:
            self.pending_card, self.pending_controller = bool(c.group(1)), None
            return
        if (t := _TAG.match(msg)) is not None:
            tag, value = t.groups()
            if tag == "CONTROLLER" and value.isdigit():
                self.pending_controller = int(value)
            elif tag == "ZONE" and value == "HAND" and self.pending_card and self.pending_controller:
                if self.friendly is None and self.turn <= 1:
                    self.friendly = self.pending_controller
            return
        if (ch := _CHANGE.search(body)) is not None:
            self.pending_card = False
            entity, tag, value = ch.groups()
            if tag == "TURN" and entity == "GameEntity" and value.isdigit():
                self.turn = int(value)
                self._event(ts, "turn", turn=self.turn)
            elif tag == "PLAYSTATE" and value in ("WON", "LOST", "TIED"):
                pid = self._player_of(entity)
                if pid is not None:
                    self.results[pid] = value
            elif tag == "STATE" and entity == "GameEntity" and value == "COMPLETE":
                self._end(ts, complete=True)

    # ----------------------------------------------------------------------

    def _player_of(self, entity: str) -> int | None:
        for pid, name in self.names.items():
            if name == entity:
                return pid
        m = _ENTITY_ID.search(entity)
        if m:
            return self.entity_player.get(int(m.group(1)))
        if entity.isdigit():
            return self.entity_player.get(int(entity))
        return None

    def me(self) -> int | None:
        if self.friendly is not None:
            return self.friendly
        players = set(self.entity_player.values()) | set(self.names)
        humans = [p for p in players if p not in self.ai and self.names.get(p) != "UNKNOWN HUMAN PLAYER"]
        return humans[0] if len(humans) == 1 else None

    def _event(self, ts: float, name: str, **kw) -> Session:
        s = self.session_for(ts)
        s.event(name, t=ts - s.started, ms=ts * 1000, session=f"game-{self.games}", **kw)
        return s

    def _end(self, ts: float, *, complete: bool) -> None:
        me = self.me()
        result = self.results.get(me) if me is not None else None
        if result is None and me is not None and self.results:
            # Only the other side's line was seen: the opposite is ours.
            other = next(iter(self.results.values()))
            result = {"WON": "LOST", "LOST": "WON", "TIED": "TIED"}.get(other)
        opponent = next((n for p, n in self.names.items() if p != me), None) if me is not None else None
        s = self._event(
            ts,
            "game_end",
            player=self.names.get(me, "me") if me is not None else "me",
            result=(result or "unknown").lower(),
            turns=self.turn,
            opponent=opponent,
            complete=complete,
        )
        self.in_game = False
        s.stat("games", 1, "add")
        if result == "WON":
            s.stat("wins", 1, "add")
        elif result == "LOST":
            s.stat("losses", 1, "add")
        elif result == "TIED":
            s.stat("ties", 1, "add")
        s.stat("turns", self.turn, "add")
        s.stat("longest_game_turns", self.turn, "max")


def install_candidates(exe: Path | None) -> list[Path]:
    if exe is not None:
        return [exe.parent]
    out = []
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
        if base:
            out.append(Path(base) / "Hearthstone")
    return out


@dataclass
class HearthstoneAdapter(Adapter):
    #: The install folder; empty: the running game's, else the default.
    install: Path | None = None
    #: log.config; empty: %LOCALAPPDATA%\Blizzard\Hearthstone\log.config.
    config: Path | None = None
    _session: Session | None = field(default=None, repr=False)
    _follow: Follower | None = field(default=None, repr=False)
    _log: PowerLog | None = field(default=None, repr=False)

    def config_file(self) -> Path:
        return Path(self.config) if self.config else config_path()

    def setup(self, ctx: Context, *, apply: bool) -> list[str]:
        path = self.config_file()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            text = ""
        except OSError as e:
            return [f"{path}: cannot read ({e})"]
        new, changes = merge_config(text)
        if not changes:
            return []
        out = [f"{path}: {c}" for c in changes]
        if apply:
            path.parent.mkdir(parents=True, exist_ok=True)
            backup = path.with_name(path.name + ".heat3d-backup")
            if text and not backup.exists():
                backup.write_text(text, encoding="utf-8")
                out.append(f"{path}: backup kept as {backup.name}")
            path.write_text(new, encoding="utf-8")
            out.append("restart Hearthstone for it to read log.config")
        return out

    def start(self, ctx: Context) -> None:
        self._session = None
        roots = [Path(self.install)] if self.install else install_candidates(ctx.exe)

        def find() -> Path | None:
            files: list[Path] = []
            for r in roots:
                logs = r / "Logs"
                files += [p for p in (logs / "Power.log",) if p.exists()]
                files += list(logs.glob("*/Power.log"))
            return newest(files)

        def session_for(ts: float | None) -> Session:
            if self._session is None:
                self._session = ctx.open(self, when=ts if ts and abs(time.time() - ts) < 86400 else None)
            return self._session

        self._log = PowerLog(session_for)
        # The log has no dates to tell this sitting's lines by; it is new at each launch, read whole then.
        self._follow = Follower(find, game_started(ctx.pid))

    def poll(self, ctx: Context) -> None:
        if self._follow is None or self._log is None:
            return
        for line in self._follow.lines():
            try:
                self._log.line(line)
            except (ValueError, IndexError):
                if self._session is not None:
                    self._session.stat("bad_lines", 1, "add")

    def stop(self, ctx: Context) -> None:
        self.poll(ctx)
        ctx.close(self._session)
        self._session = None
        self._follow = None
        self._log = None


ADAPTERS = [
    HearthstoneAdapter(
        id="hearthstone",
        game="Hearthstone",
        kind="log",
        exes=("Hearthstone.exe",),
        multiplayer=True,
        source=r"<install>\Logs\Power.log (or Logs\Hearthstone_*\Power.log), written once "
        r"%LOCALAPPDATA%\Blizzard\Hearthstone\log.config has a [Power] section",
        records=(
            "game start and end, with the result (won, lost, tied), turns and the opponent's name",
            "each turn",
            "games, wins, losses, ties, turns, longest game",
        ),
        enable=r"`python -m heat3d_watch setup hearthstone --apply` adds [Power] logging to "
        r"%LOCALAPPDATA%\Blizzard\Hearthstone\log.config (backup kept); restart the game",
        interval=0.5,
        axes=Axes(units="no positions recorded"),
        verified="tested on synthetic Power.log lines in the game's format; untested against the game",
    )
]
