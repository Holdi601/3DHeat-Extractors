"""
Following the files a game writes while it runs: journals, logs, gamelogs.

Shared by the log adapters (Elite Dangerous, Minecraft, Path of Exile,
Hearthstone, Star Citizen, EVE Online). The rules every one of them needs:

* **Read shared, hold nothing.** The file is opened for each read and closed
  again, with every share flag Windows has (read, write, delete), so the game can
  keep writing, rename its log at the next launch (Minecraft) or replace it
  (Star Citizen, Hearthstone) without ever meeting a locked file.
* **Only this sitting.** When the game starts, a file it created before then is
  history: following starts at its end. Lines written between the game's start
  and the watcher noticing it are still taken - by their own timestamps where the
  game writes them (`stamp`), so nothing of this session is lost and nothing of
  the last one is recorded twice.
* **New files.** A newer file (the next journal, a fresh log after a restart) is
  followed from its first line once the old one has been read to the end; a file
  replaced or cut short in place is read again from the start.
* **Whole lines only.** A line the game is still writing waits for its newline.

Text is UTF-8 unless the file starts with a UTF-16 byte order mark (EVE's logs
have been both over the years).
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import IO, Callable, Iterable

#: A file created this many seconds before the game's process counts as made by it.
SLACK = 10.0


def open_shared(path: Path) -> IO[bytes]:
    """Open for reading without keeping the game from writing, renaming or deleting the file."""
    if sys.platform != "win32":
        return open(path, "rb")
    import ctypes
    import msvcrt
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    GENERIC_READ, OPEN_EXISTING = 0x80000000, 3
    SHARE_ALL = 0x1 | 0x2 | 0x4  # read, write, delete
    handle = k32.CreateFileW(str(path), GENERIC_READ, SHARE_ALL, None, OPEN_EXISTING, 0x80, None)
    if handle in (None, wintypes.HANDLE(-1).value):
        raise OSError(ctypes.get_last_error(), "cannot open", str(path))
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except OSError:
        k32.CloseHandle(handle)
        raise
    return os.fdopen(fd, "rb")


def read_shared(path: Path) -> bytes:
    with open_shared(path) as f:
        return f.read()


def created(st: os.stat_result) -> float:
    """When a file was made: the birth time where the system keeps one (Windows does)."""
    birth = getattr(st, "st_birthtime", None)
    if birth:
        return float(birth)
    return float(st.st_ctime if sys.platform == "win32" else st.st_mtime)


def game_started(pid: int | None) -> float:
    """When the game's process began (from the process list, not its memory); now when unknown."""
    if pid:
        try:
            import psutil

            return float(psutil.Process(pid).create_time())
        except Exception:  # noqa: BLE001 - gone, not allowed, psutil missing: start now
            pass
    return time.time()


def newest(paths: Iterable[Path]) -> Path | None:
    best, best_t = None, -1.0
    for p in paths:
        try:
            t = p.stat().st_mtime
        except OSError:
            continue
        if t > best_t:
            best, best_t = p, t
    return best


class Follower:
    """The lines a game appends to its newest log file, as they are written."""

    def __init__(
        self,
        find: Callable[[], Path | None],
        since: float,
        stamp: Callable[[str], float | None] | None = None,
        backfill: int = 2 << 20,
    ):
        #: Which file is the one being written now.
        self.find = find
        #: The game's start: older files are history, newer ones are read whole.
        self.since = since
        #: A line's own time, for the lines written before the watcher saw the game.
        self.stamp = stamp
        self.backfill = backfill
        self.path: Path | None = None
        self.pos = 0
        self.ident: tuple | None = None
        self.utf16 = False

    # ------------------------------------------------------------------ public

    def lines(self) -> list[str]:
        """Every complete line written since the last call."""
        out: list[str] = []
        try:
            current = self.find()
        except OSError:
            current = None
        if current is not None and current != self.path:
            if self.path is not None:
                out += self._read()  # the rest of the old file first
            out += self._begin(current)
        if self.path is not None:
            out += self._read()
        return out

    # ----------------------------------------------------------------- reading

    def _begin(self, path: Path) -> list[str]:
        self.path, self.pos, self.ident, self.utf16 = path, 0, None, False
        try:
            st = path.stat()
        except OSError:
            return []
        self.ident = self._identity(st)
        if created(st) >= self.since - SLACK:
            return []  # made by this run of the game: _read takes it from the top
        if st.st_mtime < self.since - SLACK or self.stamp is None:
            self.pos = st.st_size  # history, or no way to tell this sitting's lines: the end
            self._sniff()
            return []
        # Written to since the game started: take the lines whose time says so.
        start = max(0, st.st_size - self.backfill)
        self._sniff()
        if self.utf16 and start % 2:
            start += 1
        self.pos = max(start, 2 if self.utf16 else 0)
        lines = self._read(partial_first=start > 0)
        keep: list[str] = []
        taking = False
        for line in lines:
            t = self.stamp(line)
            if t is not None:
                taking = t >= self.since - 1.0
            if taking:
                keep.append(line)
        return keep

    def _sniff(self) -> None:
        try:
            with open_shared(self.path) as f:
                head = f.read(3)
        except OSError:
            return
        if head[:2] == b"\xff\xfe":
            self.utf16 = True
            self.pos = max(self.pos, 2)
        elif head == b"\xef\xbb\xbf":
            self.pos = max(self.pos, 3)

    @staticmethod
    def _identity(st: os.stat_result) -> tuple:
        return (st.st_ino, st.st_dev, round(created(st), 3))

    def _read(self, partial_first: bool = False) -> list[str]:
        path = self.path
        if path is None:
            return []
        try:
            st = path.stat()
        except OSError:
            return []
        ident = self._identity(st)
        if ident != self.ident or st.st_size < self.pos:
            # Replaced, or cut short in place: a new log under the old name.
            self.ident, self.pos, self.utf16 = ident, 0, False
        if self.pos == 0:
            self._sniff()
        if st.st_size <= self.pos:
            return []
        try:
            with open_shared(path) as f:
                f.seek(self.pos)
                data = f.read(st.st_size - self.pos + (1 << 16))
        except OSError:
            return []
        nl = b"\n\x00" if self.utf16 else b"\n"
        end = self._last_newline(data, nl)
        if end < 0:
            return []
        chunk = data[: end + len(nl)]
        self.pos += len(chunk)
        text = chunk.decode("utf-16-le" if self.utf16 else "utf-8", errors="replace")
        lines = text.split("\n")[:-1]
        if partial_first and lines:
            lines = lines[1:]
        return [ln.rstrip("\r") for ln in lines]

    def _last_newline(self, data: bytes, nl: bytes) -> int:
        if not self.utf16:
            return data.rfind(nl)
        end = len(data)
        while True:
            i = data.rfind(nl, 0, end)
            if i < 0 or i % 2 == 0:
                return i
            # An odd hit is the high byte of one character and the low byte of the next: look further back.
            end = i + 1


# -------------------------------------------------------------------- installs


def steam_libraries() -> list[Path]:
    """Every Steam library folder (`steamapps\\common` parents), from Steam's own libraryfolders.vdf."""
    roots: list[Path] = []
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
        if base:
            roots.append(Path(base) / "Steam")
    out: list[Path] = []
    for steam in roots:
        vdf = steam / "steamapps" / "libraryfolders.vdf"
        if steam.is_dir() and steam not in out:
            out.append(steam)
        try:
            text = vdf.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in re.finditer(r'"path"\s+"([^"]+)"', text):
            p = Path(m.group(1).replace("\\\\", "\\"))
            if p not in out:
                out.append(p)
    return out
