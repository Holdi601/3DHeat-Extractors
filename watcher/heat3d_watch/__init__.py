"""
3DHeat Watcher: records what games publish about a player's play, in the
background, into files on the player's own machine.

Every source it reads is one the game offers for the purpose - a telemetry
port, a game-state integration, a journal or log file, a replay the game saves,
the platform's own statistics files, the highlights the graphics driver keeps -
or, for single-player games only, the game's memory read and never written. See
`docs/watcher.md` for the rules, and `python -m heat3d_watch games` for what
each game gives.
"""

from __future__ import annotations

from pathlib import Path

#: The tools' version: the repository's VERSION file, which the updater compares with releases.
try:
    __version__ = (Path(__file__).resolve().parents[2] / "VERSION").read_text(encoding="utf-8").strip()
except OSError:
    __version__ = "0"
