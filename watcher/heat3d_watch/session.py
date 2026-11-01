"""
One recording: what a game said during one sitting, written as it is said.

A session is a folder. `events.jsonl` gets one line per event the moment it
happens, so a crash, a power cut or a killed process loses nothing that was
already said. When the session ends, the lines with a position become
`heat.parquet` - the table the 3DHeat viewer opens, with the reading
instructions in its footer (which column is the player, the time, the event,
which way is up) - and a summary goes into `meta.json` and the statistics
database.

Rows carry the same columns in every game:

    time     seconds since the session began (or the game's own clock)
    ms       wall-clock time, Unix milliseconds
    event    what happened: "position", "death", "kill", "spawn", ...
    x y z    where, in the game's own units and axes (see `axes` in meta.json)
    player   who: a name, an id, "me" for the person playing
    team     their side, when the game says
    session  the match, round or life it belongs to, when the game says

plus whatever else a game reports, by name. Rows without a position - a kill
feed line, a level-up, a stat - stay in `events.jsonl` and count in the
statistics; the table holds the rows that can be drawn.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

#: Columns every table has, in this order.
BASE_COLUMNS = ("time", "ms", "event", "x", "y", "z", "player", "team", "session")


@dataclass
class Axes:
    """Which way the game's world is, for the viewer: the up axis, mirrored axes, units."""

    up: str = "y"
    flip_x: bool = False
    flip_y: bool = False
    flip_z: bool = False
    #: World units per metre, so the viewer can size things; 1 for metres.
    scale: float = 1.0
    #: A short note on the units, said in meta.json ("metres", "inches", "light years").
    units: str = "metres"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:60] or "game"


@dataclass
class Session:
    game: str
    adapter: str
    folder: Path
    axes: Axes = field(default_factory=Axes)
    #: Where the numbers came from, said in meta.json: "UDP telemetry", "Journal files", ...
    source: str = ""
    multiplayer: bool = False
    started: float = field(default_factory=time.time)
    ended: float | None = None
    stats: dict[str, float | str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    positions: int = 0
    _mono: float = field(default_factory=time.monotonic)
    _file: Any = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    #: When meta.json was last written; it is kept current while the session runs.
    _flushed: float = 0.0

    @classmethod
    def open(cls, root: Path, game: str, adapter: str, *, when: float | None = None, **kw: Any) -> "Session":
        stamp = datetime.fromtimestamp(when or time.time()).strftime("%Y-%m-%d_%H-%M-%S")
        folder = root / "sessions" / _slug(game) / f"{stamp}_{_slug(adapter)}"
        n = 1
        while folder.exists():
            n += 1
            folder = folder.with_name(f"{stamp}_{_slug(adapter)}_{n}")
        folder.mkdir(parents=True)
        s = cls(game=game, adapter=adapter, folder=folder, **kw)
        if when:
            s.started = when
        s._file = (folder / "events.jsonl").open("a", encoding="utf-8", buffering=1)
        s.write_meta()
        return s

    @property
    def id(self) -> str:
        """Unique across games: the game's folder and the session's, as on disk."""
        return f"{self.folder.parent.name}/{self.folder.name}"

    def now(self) -> float:
        """Seconds since the session began."""
        return time.monotonic() - self._mono

    # ------------------------------------------------------------------ record

    def event(
        self,
        event: str,
        *,
        t: float | None = None,
        ms: float | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        player: str | None = None,
        team: str | None = None,
        session: str | None = None,
        **attrs: Any,
    ) -> None:
        """Record one thing that happened. Coordinates are optional; so is everything but its name."""
        row: dict[str, Any] = {
            "time": round(self.now() if t is None else float(t), 4),
            "ms": int(ms if ms is not None else time.time() * 1000),
            "event": event,
        }
        has_position = x is not None and y is not None and z is not None
        if has_position:
            if not all(math.isfinite(float(v)) for v in (x, y, z)):
                has_position = False
            else:
                row["x"], row["y"], row["z"] = float(x), float(y), float(z)
        if player is not None:
            row["player"] = str(player)
        if team is not None:
            row["team"] = str(team)
        if session is not None:
            row["session"] = str(session)
        for k, v in attrs.items():
            if v is None:
                continue
            if isinstance(v, float) and not math.isfinite(v):
                continue
            row[k] = v
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            if self._file is None:
                return
            self._file.write(line + "\n")
            self.counts[event] = self.counts.get(event, 0) + 1
            if has_position:
                self.positions += 1
        self._flush()

    def position(self, x: float, y: float, z: float, **kw: Any) -> None:
        """Where someone is now: an event named "position"."""
        self.event("position", x=x, y=y, z=z, **kw)

    def stat(self, key: str, value: float | str, how: str = "set") -> None:
        """A number for the statistics: `set` it, `add` to it, keep the `max` or `min`."""
        with self._lock:
            old = self.stats.get(key)
            if how == "add" and isinstance(old, (int, float)) and isinstance(value, (int, float)):
                value = old + value
            elif how == "max" and isinstance(old, (int, float)) and isinstance(value, (int, float)):
                value = max(old, value)
            elif how == "min" and isinstance(old, (int, float)) and isinstance(value, (int, float)):
                value = min(old, value)
            self.stats[key] = value
        self._flush()

    def _flush(self, every: float = 30.0) -> None:
        """
        meta.json brought up to date now and then, not only at the end.

        The lines are safe the moment they are written; the statistics live in
        memory until the session closes. A watcher killed mid-game - a crash, a
        shutdown, a task manager - would lose them, so they are written out every
        half minute and `recover()` finishes such a session the next time.
        """
        now = time.monotonic()
        if now - self._flushed < every:
            return
        self._flushed = now
        try:
            self.write_meta()
        except OSError:
            pass

    # ------------------------------------------------------------------- close

    def write_meta(self) -> None:
        from . import __version__

        doc = {
            "format": "heat3d-watch-session",
            "version": 1,
            "watcher": __version__,
            "game": self.game,
            "adapter": self.adapter,
            "source": self.source,
            "multiplayer": self.multiplayer,
            "started": datetime.fromtimestamp(self.started).isoformat(timespec="seconds"),
            "ended": datetime.fromtimestamp(self.ended).isoformat(timespec="seconds") if self.ended else None,
            "seconds": round((self.ended or time.time()) - self.started, 1),
            "axes": self.axes.__dict__,
            "events": dict(sorted(self.counts.items(), key=lambda kv: -kv[1])),
            "positions": self.positions,
            "stats": self.stats,
            **self.meta,
        }
        (self.folder / "meta.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    def close(self) -> Path | None:
        """End the session: the table, the summary. Returns the table, or None when nothing was said."""
        with self._lock:
            if self._file is None:
                return None
            self._file.close()
            self._file = None
        self.ended = time.time()
        self.write_meta()
        if not self.counts:
            return None
        try:
            return write_table(self.folder, self.axes, self.game)
        except ImportError:
            # pyarrow missing: the lines are the record, the table can be made later.
            return None


def read_rows(folder: Path) -> list[dict[str, Any]]:
    rows = []
    path = folder / "events.jsonl"
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                # A line cut short by a crash: everything before it stands.
                continue
    return rows


def write_table(folder: Path, axes: Axes, game: str) -> Path | None:
    """`events.jsonl` as `heat.parquet`, with the viewer's reading instructions in the footer."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [r for r in read_rows(folder) if "x" in r]
    if not rows:
        return None
    extra: list[str] = []
    for r in rows:
        for k in r:
            if k not in BASE_COLUMNS and k not in extra:
                extra.append(k)
    columns: dict[str, pa.Array] = {}
    for name in (*BASE_COLUMNS, *extra):
        values = [r.get(name) for r in rows]
        numeric = all(v is None or (isinstance(v, (int, float)) and not isinstance(v, bool)) for v in values)
        if name in ("x", "y", "z", "time"):
            columns[name] = pa.array([float(v) if v is not None else None for v in values], type=pa.float64())
        elif name == "ms":
            columns[name] = pa.array(values, type=pa.int64())
        elif numeric and any(v is not None for v in values):
            columns[name] = pa.array([float(v) if v is not None else None for v in values], type=pa.float64())
        else:
            text = ["" if v is None else (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)) for v in values]
            columns[name] = pa.array(text, type=pa.string()).dictionary_encode()
    table = pa.table(columns)
    hints = {
        "version": 1,
        "kind": "heat3d-watch",
        "game": game,
        "axes": {"up": axes.up, "flipX": axes.flip_x, "flipY": axes.flip_y, "flipZ": axes.flip_z, "scale": axes.scale},
        "columns": {
            "x": "x",
            "y": "y",
            "z": "z",
            "time": "time",
            "timeUnit": "seconds",
            "personId": "player",
            "teamId": "team",
            "sessionId": "session",
            "extra": ["event", *[c for c in extra]],
        },
    }
    meta = dict(table.schema.metadata or {})
    meta[b"heat3d"] = json.dumps(hints).encode("utf-8")
    table = table.replace_schema_metadata(meta)
    out = folder / "heat.parquet"
    pq.write_table(table, out)
    return out


def recover(data: Path) -> list[Session]:
    """
    Sessions a killed watcher left open, finished now: ended at their last event,
    their table written, their statistics as last saved. Returns them, for the
    statistics database.
    """
    out: list[Session] = []
    for meta_path in sorted((data / "sessions").glob("*/*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if meta.get("format") != "heat3d-watch-session" or meta.get("ended"):
            continue
        folder = meta_path.parent
        rows = read_rows(folder)
        if not rows:
            continue
        a = meta.get("axes") or {}
        axes = Axes(**{k: v for k, v in a.items() if k in Axes.__dataclass_fields__})
        try:
            started = datetime.fromisoformat(meta["started"]).timestamp()
        except (KeyError, ValueError):
            started = folder.stat().st_mtime
        known = {"format", "version", "watcher", "game", "adapter", "source", "multiplayer", "started", "ended",
                 "seconds", "axes", "events", "positions", "stats"}
        s = Session(
            game=meta.get("game", folder.parent.name),
            adapter=meta.get("adapter", ""),
            folder=folder,
            axes=axes,
            source=meta.get("source", ""),
            multiplayer=bool(meta.get("multiplayer")),
            started=started,
            stats=dict(meta.get("stats") or {}),
            meta={k: v for k, v in meta.items() if k not in known},
        )
        for r in rows:
            e = str(r.get("event", ""))
            s.counts[e] = s.counts.get(e, 0) + 1
            if "x" in r:
                s.positions += 1
        last_ms = max((r.get("ms") or 0) for r in rows)
        s.ended = last_ms / 1000.0 if last_ms else folder.stat().st_mtime
        s.meta["recovered"] = True
        s.write_meta()
        try:
            write_table(folder, axes, s.game)
        except ImportError:
            pass
        out.append(s)
    return out
