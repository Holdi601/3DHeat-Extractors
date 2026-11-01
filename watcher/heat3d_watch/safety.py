"""
When reading a game's memory is allowed: single-player, no anti-cheat, read only.

Memory is the one source the watcher reads that the game did not hand out, so
it is fenced in more than the others, and every fence is checked *before* the
game's memory is opened - opening a protected game for reading is itself what
an anti-cheat sees, and refusing afterwards would be too late.

In order, all of which must pass:

1. **The game has a profile** for this exact build (`memory.py`): unknown games
   and unknown versions are never read. A profile declares whether the game
   can be played online at all.
2. **No anti-cheat anywhere near it.** Its install folder holds none of the
   known anti-cheat files, and no anti-cheat driver or service is running on
   the machine at all - a kernel anti-cheat for another game still watches
   every process.
3. **Not online.** A game a profile says can be played online must have no
   connection to anything but this machine; one whose mode cannot be told is
   refused.
4. Only then is the process opened, for reading and nothing else
   (`PROCESS_VM_READ | PROCESS_QUERY_LIMITED_INFORMATION`), and its loaded
   modules are checked once more for an anti-cheat client.

There is no elevation, no debug privilege, and no function in this package that
writes to another process; `test_safety.py` fails the build if one appears.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .adapters.base import Adapter

#: Files and folders an anti-cheat leaves in a game's install, lower case.
ANTICHEAT_FILES = (
    "easyanticheat",  # folder, and EasyAntiCheat_EOS_Setup.exe inside it
    "easyanticheat_eos_setup.exe",
    "start_protected_game.exe",  # EAC's launcher, e.g. Elden Ring
    "battleye",  # folder
    "beclient_x64.dll",
    "beservice_x64.exe",
    "eaanticheat.gameservicelauncher.exe",  # EA Javelin
    "anticheatexpert",  # Tencent ACE folder
    "sguard64.exe",
    "gameguard",  # nProtect folder
    "gamemon.des",
    "xigncode",
    "xhunter1.sys",
    "vanguard",
    "faceit",
)
#: Drivers and services of anti-cheats, lower case. Any of these running refuses all reading.
ANTICHEAT_SERVICES = (
    "easyanticheat",
    "easyanticheat_eos",
    "beservice",
    "bedaisy",
    "vgk",
    "vgc",
    "ace-base",
    "ace-game",
    "xhunter1",
    "faceit",
    "faceitservice",
    "eaanticheatservice",
    "denuvo anti-cheat",
    "denuvoanticheat",
    "npggsvc",
    "ricochet",
)
#: Modules that mean an anti-cheat client lives in the game, checked after opening.
ANTICHEAT_MODULES = ("beclient_x64.dll", "beclient.dll", "easyanticheat_x64.dll", "easyanticheat.dll", "eosanticheat")


def install_markers(exe: Path | None, depth: int = 3) -> list[str]:
    """Anti-cheat files beside the game: in its folder and up to `depth` folders above it."""
    if exe is None:
        return ["the game's install folder is unknown"]
    found: list[str] = []
    folder = exe.parent
    for _ in range(depth + 1):
        try:
            names = [p.name.lower() for p in folder.iterdir()]
        except OSError:
            names = []
        for marker in ANTICHEAT_FILES:
            if any(n == marker or n.startswith(marker) for n in names):
                found.append(f"{marker} in {folder}")
        if folder.parent == folder or (folder / "steamapps").exists():
            break
        # Stop at a library's `common` folder: above it are other games.
        if folder.name.lower() == "common":
            break
        folder = folder.parent
    return found


def running_anticheat() -> list[str]:
    """Anti-cheat drivers and services running on this machine right now."""
    if os.name != "nt":
        return []
    names: list[str] = []
    for kind in ("driver", "service"):
        try:
            out = subprocess.run(
                ["sc", "query", "type=", kind, "state=", "active"],
                capture_output=True,
                text=True,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return ["the running services could not be listed"]
        for line in out.splitlines():
            line = line.strip()
            for prefix in ("SERVICE_NAME:", "DISPLAY_NAME:"):
                if line.upper().startswith(prefix):
                    names.append(line[len(prefix) :].strip().lower())
    return sorted({n for n in names if any(n == s or n.startswith(s) for s in ANTICHEAT_SERVICES)})


def remote_connections(pid: int) -> list[str]:
    """Connections the game has to anything but this machine."""
    import psutil

    try:
        conns = psutil.Process(pid).net_connections(kind="inet")
    except (psutil.Error, OSError):
        return ["its connections could not be read"]
    out = []
    for c in conns:
        if not c.raddr:
            continue
        host = c.raddr.ip
        if host.startswith("127.") or host in ("::1", "0.0.0.0", "::"):
            continue
        out.append(f"{host}:{c.raddr.port}")
    return out


def memory_allowed(adapter: "Adapter", pid: int, exe: Path | None) -> str | None:
    """Why the game's memory must not be read, or None when every check passed."""
    if adapter.kind != "memory":
        return "not a memory adapter"
    if adapter.multiplayer and not getattr(adapter, "offline_check", False):
        return "the game is played online"
    markers = install_markers(exe)
    if markers:
        return "anti-cheat: " + "; ".join(markers)
    running = running_anticheat()
    if running:
        return "an anti-cheat is running on this machine: " + ", ".join(running)
    if getattr(adapter, "offline_check", False):
        remote = remote_connections(pid)
        if remote:
            return "the game is connected to " + ", ".join(remote[:3])
    return None


def loaded_anticheat(pid: int) -> list[str]:
    """Anti-cheat modules inside the game. Asked only after every check before opening passed."""
    from .memory import ReadError, modules

    try:
        loaded = modules(pid)
    except ReadError as e:
        return [str(e)]
    return [str(m.path) for m in loaded if any(m.name.lower().startswith(a) for a in ANTICHEAT_MODULES)]
