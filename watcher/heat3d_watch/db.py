"""
The statistics: one SQLite file the player can open with anything.

Three tables, deliberately plain so a spreadsheet, a notebook or a SQL prompt
reads them without this program:

    sessions  one row per sitting: game, adapter, start, end, seconds, folder,
              how many events and positions were recorded, multiplayer or not
    events    per session, how many times each event happened
    stats     per session, every number a game reported or the watcher counted
              (kills, deaths, distance, best lap, ...), as a value or a text

`python -m heat3d_watch stats` sums them per game; `--csv` writes them out.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from .session import Session

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  game TEXT NOT NULL,
  adapter TEXT NOT NULL,
  started REAL NOT NULL,
  ended REAL,
  seconds REAL,
  folder TEXT NOT NULL,
  events INTEGER NOT NULL DEFAULT 0,
  positions INTEGER NOT NULL DEFAULT 0,
  multiplayer INTEGER NOT NULL DEFAULT 0,
  source TEXT
);
CREATE TABLE IF NOT EXISTS events (
  session TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  event TEXT NOT NULL,
  count INTEGER NOT NULL,
  PRIMARY KEY (session, event)
);
CREATE TABLE IF NOT EXISTS stats (
  session TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  key TEXT NOT NULL,
  value REAL,
  text TEXT,
  PRIMARY KEY (session, key)
);
CREATE INDEX IF NOT EXISTS sessions_game ON sessions(game);
"""


def connect(data: Path) -> sqlite3.Connection:
    data.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(data / "watcher.sqlite", timeout=10)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


def record(con: sqlite3.Connection, s: Session) -> None:
    """A finished session into the database; recording it again replaces it."""
    seconds = (s.ended or s.started) - s.started
    with con:
        con.execute(
            "INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                s.id,
                s.game,
                s.adapter,
                s.started,
                s.ended,
                seconds,
                str(s.folder),
                sum(s.counts.values()),
                s.positions,
                1 if s.multiplayer else 0,
                s.source,
            ),
        )
        con.execute("DELETE FROM events WHERE session = ?", (s.id,))
        con.executemany(
            "INSERT INTO events VALUES (?,?,?)", [(s.id, e, n) for e, n in s.counts.items()]
        )
        con.execute("DELETE FROM stats WHERE session = ?", (s.id,))
        con.executemany(
            "INSERT INTO stats VALUES (?,?,?,?)",
            [
                (s.id, k, float(v), None) if isinstance(v, (int, float)) and not isinstance(v, bool) else (s.id, k, None, str(v))
                for k, v in s.stats.items()
            ],
        )


def per_game(con: sqlite3.Connection) -> list[tuple]:
    """Sessions, hours and events per game, most played first."""
    return con.execute(
        """SELECT game, COUNT(*), ROUND(SUM(COALESCE(seconds, 0)) / 3600.0, 2), SUM(events), SUM(positions),
                  datetime(MAX(started), 'unixepoch', 'localtime')
           FROM sessions GROUP BY game ORDER BY SUM(COALESCE(seconds, 0)) DESC"""
    ).fetchall()


def event_totals(con: sqlite3.Connection, game: str) -> list[tuple]:
    return con.execute(
        """SELECT e.event, SUM(e.count) FROM events e JOIN sessions s ON s.id = e.session
           WHERE s.game = ? GROUP BY e.event ORDER BY SUM(e.count) DESC""",
        (game,),
    ).fetchall()


def stat_totals(con: sqlite3.Connection, game: str) -> list[tuple]:
    """Each numeric stat of a game: summed, best, average over the sessions that have it."""
    return con.execute(
        """SELECT t.key, COUNT(*), SUM(t.value), MAX(t.value), AVG(t.value) FROM stats t
           JOIN sessions s ON s.id = t.session
           WHERE s.game = ? AND t.value IS NOT NULL GROUP BY t.key ORDER BY t.key""",
        (game,),
    ).fetchall()


def rows(con: sqlite3.Connection, table: str) -> tuple[list[str], Iterable[tuple]]:
    if table not in ("sessions", "events", "stats"):
        raise ValueError(table)
    cur = con.execute(f"SELECT * FROM {table}")
    return [d[0] for d in cur.description], cur.fetchall()
