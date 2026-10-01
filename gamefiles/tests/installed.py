"""
Where games are installed on this machine, for the tests that check a reader
against shipped files.

Every Steam library Steam knows about is searched, so the tests find a game
wherever it was installed and skip when it is not there.
"""

from __future__ import annotations

import glob
from pathlib import Path

from heat3d_gamefiles.forzainstall import STEAM_DEFAULTS, steam_libraries


def steam_common() -> list[Path]:
    """Every Steam library's `steamapps/common`, Steam's own folder included."""
    found: list[Path] = []
    seen: set[str] = set()
    for library in [Path(p) for p in STEAM_DEFAULTS] + steam_libraries():
        common = library / "steamapps" / "common"
        key = str(common).lower()
        if key not in seen and common.is_dir():
            seen.add(key)
            found.append(common)
    return found


def installed(*patterns: str) -> list[Path]:
    """Paths matching any of `patterns` (globs under `steamapps/common`) in any library."""
    found: list[Path] = []
    for common in steam_common():
        for pattern in patterns:
            found += [Path(p) for p in sorted(glob.glob(str(common / pattern), recursive="**" in pattern))]
    return found
