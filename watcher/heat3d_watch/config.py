"""
Where the watcher keeps what it records, and the few settings it has.

Everything lives under one folder - `%LOCALAPPDATA%\\3DHeat\\Watcher` unless
`HEAT3D_WATCH_HOME` or the settings say otherwise - so a player can see, copy
or delete all of it in one place. Nothing is sent anywhere.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def home() -> Path:
    env = os.environ.get("HEAT3D_WATCH_HOME")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / "3DHeat" / "Watcher"


@dataclass
class Settings:
    #: Where sessions and the statistics database go. Empty: `home()`.
    data: str = ""
    #: Adapters switched off by id. Everything else that finds its game runs.
    disabled: list[str] = field(default_factory=list)
    #: Update the tools from GitHub releases while no game is running.
    auto_update: bool = True
    #: Seconds between looks at which games are running.
    scan_seconds: float = 2.0
    #: Read single-player games' memory where a profile exists for the exact build.
    #: Off by default: it is the one source that is not the game's own output.
    memory: bool = False
    #: Watch the folders games save replays and demos to, and convert new ones.
    replays: bool = True
    #: Extra folders to look for replays in, by adapter id.
    replay_folders: dict[str, list[str]] = field(default_factory=dict)

    @property
    def data_dir(self) -> Path:
        return Path(self.data) if self.data else home()


def settings_path() -> Path:
    return home() / "settings.json"


def load() -> Settings:
    try:
        raw = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Settings()
    known = {f for f in Settings.__dataclass_fields__}
    return Settings(**{k: v for k, v in raw.items() if k in known})


def save(s: Settings) -> Path:
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(s), indent=2), encoding="utf-8")
    return p
