"""
Minecraft (Java Edition): the log, statistics and player files the game writes.

Where (`%APPDATA%\\.minecraft\\`):

* `logs\\latest.log` - the game's log, new at each launch (the old one is
  packed into a dated .log.gz). Lines `[hh:mm:ss] [thread/LEVEL]: message`
  (mod loaders write `[ddMMMyyyy hh:mm:ss.mmm] [thread/LEVEL] [logger/]: ...`).
  What players see in chat is in it: joins and leaves, death messages,
  advancements, chat - the integrated server of a single-player world writes
  the same lines once more, so a line repeated within the same second counts
  once. "Setting user: <name>" names the player; the server's
  "<name>[...] logged in with entity id N at (x, y, z)" gives where they
  appeared.
* `saves\\<world>\\stats\\<uuid>.json` - the statistics screen, written when
  the world saves (autosave every few minutes, and on leaving): `stats` ->
  `minecraft:custom` (play_time in ticks, mob_kills, *_one_cm distances, jump,
  ...), `minecraft:mined`, `minecraft:crafted`, ... Totals for the world; the
  session's figures are the difference to what the file said when the game
  started. Format: https://minecraft.wiki/w/Statistics
* `saves\\<world>\\playerdata\\<uuid>.dat` (and `level.dat` -> Data -> Player
  for a single-player host) - gzip-compressed NBT, written on the same saves:
  `Pos` [x, y, z] doubles, `Rotation` [yaw, pitch], `Dimension`
  ("minecraft:overworld"; an int 0/-1/1 before 1.16). NBT per
  https://minecraft.wiki/w/NBT_format - read here by a small reader of our own.

Coordinates are blocks, one block a metre, y up; x east, z south - a right-
handed frame, drawn as it is. Positions come only when the world saves, so a
route is a few points per autosave, not a trail; the dimension is the
`session`, so the Nether is not drawn over the Overworld.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import struct
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..session import Axes, Session
from ._logfollow import Follower, game_started, read_shared
from .base import Adapter, Context

# ------------------------------------------------------------------------ NBT

_END, _BYTE, _SHORT, _INT, _LONG, _FLOAT, _DOUBLE, _BYTES, _STRING, _LIST, _COMPOUND, _INTS, _LONGS = range(13)


class NBTError(ValueError):
    pass


def read_nbt(data: bytes) -> dict[str, Any]:
    """An NBT file's root compound (gzip, zlib or uncompressed), as dicts, lists and numbers."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    elif data[:1] == b"\x78":
        data = zlib.decompress(data)
    if not data or data[0] != _COMPOUND:
        raise NBTError("not an NBT compound")
    reader = _NBT(data)
    reader.pos = 1
    reader.string()  # the root's name, usually empty
    return reader.payload(_COMPOUND)


class _NBT:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.depth = 0

    def take(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        if self.pos + size > len(self.data):
            raise NBTError("cut short")
        v = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += size
        return v[0] if len(v) == 1 else v

    def string(self) -> str:
        n = self.take(">H")
        raw = self.data[self.pos : self.pos + n]
        if len(raw) < n:
            raise NBTError("cut short")
        self.pos += n
        return raw.decode("utf-8", errors="replace")  # Java's modified UTF-8: the same for all but NUL/astral

    def array(self, fmt: str) -> list:
        n = self.take(">i")
        if n < 0 or self.pos + n * struct.calcsize(fmt) > len(self.data):
            raise NBTError("bad array length")
        out = list(struct.unpack_from(f">{n}{fmt}", self.data, self.pos))
        self.pos += n * struct.calcsize(fmt)
        return out

    def payload(self, tag: int) -> Any:
        if tag == _BYTE:
            return self.take(">b")
        if tag == _SHORT:
            return self.take(">h")
        if tag == _INT:
            return self.take(">i")
        if tag == _LONG:
            return self.take(">q")
        if tag == _FLOAT:
            return self.take(">f")
        if tag == _DOUBLE:
            return self.take(">d")
        if tag == _BYTES:
            return self.array("b")
        if tag == _STRING:
            return self.string()
        if tag == _INTS:
            return self.array("i")
        if tag == _LONGS:
            return self.array("q")
        self.depth += 1
        if self.depth > 512:
            raise NBTError("nested too deep")
        try:
            if tag == _LIST:
                kind, n = self.take(">b"), self.take(">i")
                if n < 0 or n > len(self.data):
                    raise NBTError("bad list length")
                return [self.payload(kind) for _ in range(n)] if kind != _END else []
            if tag == _COMPOUND:
                out: dict[str, Any] = {}
                while True:
                    kind = self.take(">b")
                    if kind == _END:
                        return out
                    name = self.string()
                    out[name] = self.payload(kind)
            raise NBTError(f"unknown tag {tag}")
        finally:
            self.depth -= 1


# ------------------------------------------------------------------------ log

_LINE = re.compile(r"^\[(?:[^\]]*?)(\d{1,2}:\d{2}:\d{2})(?:\.\d+)?\] \[([^\]/]+)/(\w+)\](?: \[[^\]]*\])?: (.*)$")
_CHAT_PREFIX = re.compile(r"^(?:\[System\] )?\[CHAT\] ")
_NAME = r"([A-Za-z0-9_]{1,16})"
_USER = re.compile(r"^Setting user: (\S+)")
_LOGIN = re.compile(_NAME + r"\[[^\]]*\] logged in with entity id \d+ at \(([-\d.,\sE]+)\)")
_JOIN = re.compile(r"^" + _NAME + r" joined the game$")
_LEAVE = re.compile(r"^" + _NAME + r" left the game$")
_ADVANCEMENT = re.compile(r"^" + _NAME + r" has (made the advancement|completed the challenge|reached the goal) \[(.+)\]$")
_CHAT = re.compile(r"^<" + _NAME + r"> (.*)$")
#: The death messages' verbs (https://minecraft.wiki/w/Death_messages); the line starts with the name.
_DEATH = re.compile(
    r"^" + _NAME + r" (was (?:slain|shot|killed|blown up|pummeled|pricked|squashed|squished|impaled|skewered|"
    r"fireballed|stung|poked|struck by lightning|burnt|frozen|obliterated|doomed|roasted|stabbed)|"
    r"walked into|drowned|experienced kinetic energy|blew up|hit the ground too hard|fell |went up in flames|"
    r"went off with a bang|burned to death|tried to swim in lava|discovered the floor was lava|froze to death|"
    r"starved to death|suffocated in a wall|left the confines of this world|didn't want to live|withered away|"
    r"died)(.*)$"
)


def minecraft_dir() -> Path:
    return Path(os.environ.get("APPDATA") or Path.home()) / ".minecraft"


# ---------------------------------------------------------------------- stats

def stat_totals(doc: dict) -> dict[str, float]:
    """The numbers worth a session's statistics from one stats/<uuid>.json (1.13+ layout)."""
    stats = doc.get("stats") if isinstance(doc, dict) else None
    if not isinstance(stats, dict):
        return {}
    custom = stats.get("minecraft:custom") or {}

    def total(group: str) -> float:
        g = stats.get(group) or {}
        return float(sum(v for v in g.values() if isinstance(v, (int, float))))

    ticks = custom.get("minecraft:play_time", custom.get("minecraft:play_one_minute", 0))
    return {
        "play_seconds": float(ticks) / 20.0,
        "distance_m": sum(float(v) for k, v in custom.items() if k.endswith("_one_cm") and isinstance(v, (int, float)))
        / 100.0,
        "mob_kills": float(custom.get("minecraft:mob_kills", 0)),
        "player_kills": float(custom.get("minecraft:player_kills", 0)),
        "jumps": float(custom.get("minecraft:jump", 0)),
        "damage_dealt": float(custom.get("minecraft:damage_dealt", 0)) / 10.0,  # tenths of a heart -> hearts
        "damage_taken": float(custom.get("minecraft:damage_taken", 0)) / 10.0,
        "blocks_mined": total("minecraft:mined"),
        "items_crafted": total("minecraft:crafted"),
    }


def player_fix(root: dict) -> tuple[tuple[float, float, float], str, list | None] | None:
    """(Pos, dimension, Rotation) from a player NBT compound, or None."""
    pos = root.get("Pos")
    if not (isinstance(pos, list) and len(pos) == 3 and all(isinstance(v, (int, float)) for v in pos)):
        return None
    dim = root.get("Dimension", "minecraft:overworld")
    if isinstance(dim, int):
        dim = {0: "minecraft:overworld", -1: "minecraft:the_nether", 1: "minecraft:the_end"}.get(dim, str(dim))
    rot = root.get("Rotation") if isinstance(root.get("Rotation"), list) else None
    return (float(pos[0]), float(pos[1]), float(pos[2])), str(dim), rot


@dataclass
class MinecraftAdapter(Adapter):
    #: The game folder; empty: %APPDATA%\.minecraft.
    folder: Path | None = None
    #: Seconds between looks at the world folders (the log is read every poll).
    world_every: float = 3.0
    _session: Session | None = field(default=None, repr=False)
    _follow: Follower | None = field(default=None, repr=False)
    _me: str | None = field(default=None, repr=False)
    _since: float = field(default=0.0, repr=False)
    _seen: dict[Path, int] = field(default_factory=dict, repr=False)
    _baseline: dict[Path, dict[str, float]] = field(default_factory=dict, repr=False)
    _recent: list[tuple[str, str]] = field(default_factory=list, repr=False)
    _next_world: float = field(default=0.0, repr=False)

    def root(self) -> Path:
        return Path(self.folder) if self.folder else minecraft_dir()

    def missing(self) -> str | None:
        return None if self.root().is_dir() else f"no Minecraft folder at {self.root()}"

    # ---------------------------------------------------------------- lifecycle

    def start(self, ctx: Context) -> None:
        self._session = None
        self._me = None
        self._recent = []
        self._seen = {}
        self._baseline = {}
        self._next_world = 0.0
        self._since = game_started(ctx.pid)
        log = self.root() / "logs" / "latest.log"
        self._follow = Follower(lambda: log if log.exists() else None, self._since)
        # What the world files say now: the session's numbers are what changes from here.
        for path in self._world_files():
            try:
                self._seen[path] = path.stat().st_mtime_ns
            except OSError:
                continue
            if path.suffix == ".json":
                try:
                    self._baseline[path] = stat_totals(json.loads(read_shared(path)))
                except (OSError, ValueError):
                    self._baseline[path] = {}

    def poll(self, ctx: Context) -> None:
        if self._follow is not None:
            for line in self._follow.lines():
                try:
                    self.line(ctx, line)
                except (ValueError, IndexError):
                    self._open(ctx).stat("bad_lines", 1, "add")
        if time.monotonic() >= self._next_world:
            self._next_world = time.monotonic() + self.world_every
            self.worlds(ctx)

    def stop(self, ctx: Context) -> None:
        if self._follow is not None:
            for line in self._follow.lines():
                try:
                    self.line(ctx, line)
                except (ValueError, IndexError):
                    pass
        self.worlds(ctx)  # the save on leaving the world
        ctx.close(self._session)
        self._session = None
        self._follow = None

    def _open(self, ctx: Context) -> Session:
        if self._session is None:
            self._session = ctx.open(self)
        return self._session

    # --------------------------------------------------------------------- log

    def line(self, ctx: Context, text: str) -> None:
        m = _LINE.match(text)
        if not m:
            return
        clock, _thread, _level, msg = m.groups()
        msg = _CHAT_PREFIX.sub("", msg).strip()
        u = _USER.match(msg)
        if u:
            self._me = u.group(1)
            return
        # Single-player: the integrated server and the client both log what is said in chat.
        key = (clock, msg)
        if key in self._recent:
            return
        self._recent = (self._recent + [key])[-64:]
        me = self._me
        if (g := _LOGIN.search(msg)) is not None:
            parts = [float(v) for v in g.group(2).split(",")]
            if len(parts) == 3:
                name = g.group(1)
                s = self._open(ctx)
                s.event("spawn", x=parts[0], y=parts[1], z=parts[2], player=self._who(name), clock=clock, how="login")
            return
        if (g := _JOIN.match(msg)) is not None:
            self._open(ctx).event("join", player=self._who(g.group(1)), clock=clock)
            self._open(ctx).stat("joins", 1, "add")
        elif (g := _LEAVE.match(msg)) is not None:
            self._open(ctx).event("leave", player=self._who(g.group(1)), clock=clock)
        elif (g := _ADVANCEMENT.match(msg)) is not None:
            s = self._open(ctx)
            s.event("advancement", player=self._who(g.group(1)), kind=g.group(2).split()[-1], name=g.group(3), clock=clock)
            if me is None or g.group(1) == me:
                s.stat("advancements", 1, "add")
        elif (g := _CHAT.match(msg)) is not None:
            s = self._open(ctx)
            s.event("chat", player=self._who(g.group(1)), text=g.group(2)[:300], clock=clock)
            s.stat("chat_messages", 1, "add")
        elif (g := _DEATH.match(msg)) is not None:
            s = self._open(ctx)
            s.event("death", player=self._who(g.group(1)), message=msg[:300], clock=clock)
            if me is None or g.group(1) == me:
                s.stat("deaths", 1, "add")
            else:
                s.stat("others_deaths", 1, "add")
        elif msg.startswith("Stopping!") or msg == "Stopping server":
            self._open(ctx).event("session_end", player="me", clock=clock)

    def _who(self, name: str) -> str:
        return "me" if self._me and name == self._me else name

    # ------------------------------------------------------------------ worlds

    def _world_files(self) -> list[Path]:
        saves = self.root() / "saves"
        out: list[Path] = []
        try:
            worlds = [w for w in saves.iterdir() if w.is_dir()]
        except OSError:
            return out
        for w in worlds:
            out += list((w / "stats").glob("*.json"))
            out += list((w / "playerdata").glob("*.dat"))
            if (w / "level.dat").exists():
                out.append(w / "level.dat")
        return out

    def worlds(self, ctx: Context) -> None:
        """Stats and player files that were saved since the last look."""
        for path in self._world_files():
            try:
                stamp = path.stat().st_mtime_ns
            except OSError:
                continue
            if self._seen.get(path) == stamp:
                continue
            first = path not in self._seen
            self._seen[path] = stamp
            # A world not touched in this sitting. With a little slack: a file's time
            # and the clock are not read from the same source, and a save just after
            # the start can carry a stamp a few milliseconds before it.
            if first and stamp / 1e9 < self._since - 2.0:
                continue
            try:
                data = read_shared(path)
            except OSError:
                self._seen.pop(path, None)
                continue
            try:
                if path.suffix == ".json":
                    self._stats(ctx, path, json.loads(data))
                else:
                    self._player(ctx, path, read_nbt(data))
            except (ValueError, OSError, EOFError, struct.error, zlib.error):
                self._seen.pop(path, None)  # caught mid-save: next look
                self._open(ctx).stat("bad_files", 1, "add")

    def _stats(self, ctx: Context, path: Path, doc: dict) -> None:
        now = stat_totals(doc)
        base = self._baseline.setdefault(path, {})
        s = self._open(ctx)
        for k, v in now.items():
            s.stat(k, round(v - base.get(k, 0.0), 3))
        s.stat("world", path.parent.parent.name)

    def _player(self, ctx: Context, path: Path, root: dict) -> None:
        world = path.parent.parent if path.parent.name == "playerdata" else path.parent
        if path.name == "level.dat":
            # The single-player host is in level.dat as well; use it only where no playerdata exists.
            if any((world / "playerdata").glob("*.dat")):
                return
            root = (root.get("Data") or {}).get("Player") or {}
            who = "me"
        else:
            others = list((world / "playerdata").glob("*.dat"))
            who = "me" if len(others) <= 1 else path.stem
        fix = player_fix(root)
        if fix is None:
            return
        (x, y, z), dim, rot = fix
        s = self._open(ctx)
        yaw = rot[0] if rot else None
        pitch = rot[1] if rot and len(rot) > 1 else None
        s.position(x, y, z, player=who, session=dim, world=world.name, yaw=yaw, pitch=pitch,
                   health=root.get("Health"), level=root.get("XpLevel"))


ADAPTERS = [
    MinecraftAdapter(
        id="minecraft",
        game="Minecraft",
        kind="log",
        exes=("javaw.exe", "java.exe"),
        multiplayer=True,
        source=(
            r"%APPDATA%\.minecraft\logs\latest.log (followed), saves\<world>\stats\<uuid>.json and "
            r"saves\<world>\playerdata\<uuid>.dat / level.dat (gzip NBT), read when the game saves them"
        ),
        records=(
            "own and other players' position (blocks = m, y up) each time the world saves, dimension as session",
            "spawn with position when a player logs in to the (integrated) server",
            "joins, leaves, deaths with the death message, advancements, chat lines",
            "deaths, advancements, chat messages; from the stats file: play time, distance, mob and player kills, "
            "jumps, damage, blocks mined, items crafted (this session's share)",
        ),
        enable="",
        interval=0.5,
        axes=Axes(up="y", units="blocks (1 block = 1 m)"),
        verified="tested on synthetic log lines, stats JSON and NBT files built from the published formats; "
        "untested against the game",
    )
]
