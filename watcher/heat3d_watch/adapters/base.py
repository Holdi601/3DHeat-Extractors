"""
What a game adapter is: how the watcher learns from one game.

An adapter says which game it is for, how the game shows it is running (its
process names), where its numbers come from, and what it records - and then is
started, polled and stopped while the game runs. `records` and `source` are
not decoration: `python -m heat3d_watch games` and the documentation are
written from them, so they say exactly what is recorded and from where.

Kinds, which decide what an adapter may do:

    telemetry  a port or shared memory the game publishes for dashboards and rigs
    gsi        a game-state integration or local API the game serves
    log        files the game writes while it runs: journals, logs, stats
    replay     demos and replays the game saves, converted after the match
    platform   what the launcher or driver keeps: Steam stats, highlights
    memory     single-player games only: memory read, never written, behind the
               checks in `safety.py`

Only `memory` touches another process, and only through `memory.read`, which
cannot write. Every other kind reads what the game hands out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..session import Axes, Session

if TYPE_CHECKING:
    from ..config import Settings

KINDS = ("telemetry", "gsi", "log", "replay", "platform", "memory")


@dataclass
class Context:
    """What an adapter is given: the settings, the data folder, and how to open sessions."""

    settings: "Settings"
    data: Path
    #: The running game's process id, when the adapter was started for one.
    pid: int | None = None
    #: The executable's full path, when known: the install folder is next to it.
    exe: Path | None = None

    def open(self, adapter: "Adapter", *, game: str | None = None, when: float | None = None, **kw) -> Session:
        return Session.open(
            self.data,
            game or adapter.game,
            adapter.id,
            when=when,
            axes=kw.pop("axes", adapter.axes),
            source=kw.pop("source", adapter.source),
            multiplayer=kw.pop("multiplayer", adapter.multiplayer),
            **kw,
        )

    def close(self, session: Session | None) -> Path | None:
        """End a session: its table, its summary, and its line in the statistics."""
        if session is None:
            return None
        table = session.close()
        if not session.counts:
            # The game ran and said nothing - a menu, a crash at start. An empty
            # folder per launch would bury the sittings worth keeping.
            import shutil

            shutil.rmtree(session.folder, ignore_errors=True)
            return None
        if session.counts:
            from .. import db

            con = db.connect(self.data)
            try:
                db.record(con, session)
            finally:
                con.close()
        return table


@dataclass
class Adapter:
    #: Short, stable: the session folder and settings refer to it.
    id: str = ""
    #: The game's name, as people write it.
    game: str = ""
    kind: str = "telemetry"
    #: Executable names (any case) that mean the game is running. Empty: always on.
    exes: tuple[str, ...] = ()
    #: Played online: never read from memory, whatever else is allowed.
    multiplayer: bool = True
    #: Where the numbers come from, said once: "UDP telemetry on port 20777, switched on in ...".
    source: str = ""
    #: What is recorded, one item per kind of thing, for the list and the documentation.
    records: tuple[str, ...] = ()
    #: How to turn the source on, when the game needs telling; empty when it needs nothing.
    enable: str = ""
    #: Seconds between polls while running.
    interval: float = 0.1
    axes: Axes = field(default_factory=Axes)
    #: What has been checked, honestly: "tested against real journals", "packets built from the spec".
    verified: str = ""

    # -- lifecycle; adapters override what they need --------------------------------

    def missing(self) -> str | None:
        """Why this adapter cannot run here (a package not installed, ...), or None."""
        return None

    def setup(self, ctx: Context, *, apply: bool) -> list[str]:
        """Switch the game's own output on. Returns what was (or would be) changed."""
        return []

    def start(self, ctx: Context) -> None:
        """The game was seen running."""

    def poll(self, ctx: Context) -> None:
        """Take whatever the game has said since the last poll."""

    def stop(self, ctx: Context) -> None:
        """The game is gone, or the watcher is stopping. Close sessions."""

    @property
    def always(self) -> bool:
        return not self.exes
