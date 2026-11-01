"""
Replays and demos: what a game saves of a match, converted once the match is over.

A replay holds everything a live interface keeps back in multiplayer games -
every player's position, every kill where it happened - and it is a file the
game writes for its players, so reading it touches nothing in the game. The
watcher looks at the folders games save them to, waits until a new file has
stopped growing (the game is still writing it otherwise), and converts it in a
worker thread, so a long conversion never holds up the rest of the watcher.

The first time a folder is seen, the replays already in it are only noted, not
converted: a player with years of demos would otherwise find the watcher busy
for a day. `python -m heat3d_watch replays <adapter> [folder]` converts the
existing ones on request.

Counter-Strike 2
----------------
CS2 saves the demos a player downloads from the match history (watch tab) to
`<CS2>/game/csgo/replays/` and ones recorded with the `record` console command
to `<CS2>/game/csgo/`. Each is converted with this repository's own CS2
converter (`cs2/heat3d_cs2`: every player's position, look direction, health and
money every eighth tick; kills, deaths, damage, grenades, plants and defuses
where they happened; rounds, score, economy) into the session folder, as the
`<map>/<match>.parquet` table and `<match>.match.json` summary the viewer's heat
and match tabs read. The match's scoreboard becomes the session's statistics:
for the player whose Steam account is signed in on this machine (from Steam's
`loginusers.vdf`) under its plain names - kills, deaths, assists, ADR, KAST,
headshot percentage, opening kills, clutches - and the score and rounds.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..session import Axes
from .base import Adapter, Context

log = logging.getLogger("heat3d_watch")

#: How long a new file must keep its size before it counts as written.
SETTLE_SECONDS = 10.0
#: The repository root: the converters of other games live beside the watcher.
ROOT = Path(__file__).resolve().parents[3]


def steam_root() -> Path | None:
    """Steam's own folder, from the registry or the usual place."""
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as k:
                path = Path(winreg.QueryValueEx(k, "SteamPath")[0])
                if path.is_dir():
                    return path
        except OSError:
            pass
    for p in (Path(r"C:\Program Files (x86)\Steam"), Path.home() / ".steam" / "steam"):
        if p.is_dir():
            return p
    return None


def steam_libraries() -> list[Path]:
    """Every Steam library's `steamapps/common`, from `libraryfolders.vdf`."""
    root = steam_root()
    if root is None:
        return []
    found = [root]
    vdf = root / "steamapps" / "libraryfolders.vdf"
    try:
        text = vdf.read_text(encoding="utf-8", errors="replace")
        found += [Path(m.replace("\\\\", "\\")) for m in re.findall(r'"path"\s+"([^"]+)"', text)]
    except OSError:
        pass
    out, seen = [], set()
    for lib in found:
        common = lib / "steamapps" / "common"
        key = str(common).lower()
        if key not in seen and common.is_dir():
            seen.add(key)
            out.append(common)
    return out


def signed_in_steam_ids() -> set[str]:
    """The 64-bit ids of the Steam accounts signed in on this machine (`loginusers.vdf`)."""
    root = steam_root()
    if root is None:
        return set()
    try:
        text = (root / "config" / "loginusers.vdf").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    return set(re.findall(r'"(7656\d{13})"', text))


@dataclass
class ReplayAdapter(Adapter):
    """A folder of replays, watched; each new one converted into a session."""

    kind: str = "replay"
    interval: float = 5.0
    #: File name patterns of this game's replays.
    patterns: tuple[str, ...] = ()
    _seen: dict[str, list[float]] = field(default_factory=dict)
    _pending: dict[str, tuple[float, float]] = field(default_factory=dict)
    _queue: "queue.Queue[Path]" = field(default_factory=queue.Queue)
    _worker: threading.Thread | None = None
    _halt: threading.Event = field(default_factory=threading.Event)

    def folders(self, ctx: Context) -> list[Path]:
        """Where this game saves replays, plus the folders the settings add."""
        return [Path(p) for p in ctx.settings.replay_folders.get(self.id, [])]

    def convert(self, path: Path, ctx: Context) -> None:
        """Turn one replay into a session. Runs in the worker thread."""
        raise NotImplementedError

    # -- the folder watch, shared by every game ---------------------------------

    def _state_path(self, ctx: Context) -> Path:
        return ctx.data / "state" / f"replays-{self.id}.json"

    def _files(self, ctx: Context) -> list[Path]:
        out: list[Path] = []
        for folder in self.folders(ctx):
            if not folder.is_dir():
                continue
            for pattern in self.patterns:
                out += [p for p in folder.glob(pattern) if p.is_file()]
        return out

    def start(self, ctx: Context) -> None:
        path = self._state_path(ctx)
        first = not path.exists()
        try:
            self._seen = json.loads(path.read_text(encoding="utf-8")) if not first else {}
        except (OSError, ValueError):
            self._seen = {}
        if first:
            # Only note what is already there; see the module docstring.
            for f in self._files(ctx):
                st = f.stat()
                self._seen[str(f)] = [st.st_size, st.st_mtime]
            self._save(ctx)
        self._halt.clear()
        self._worker = threading.Thread(target=self._work, args=(ctx,), name=f"convert-{self.id}", daemon=True)
        self._worker.start()

    def poll(self, ctx: Context) -> None:
        now = time.time()
        for f in self._files(ctx):
            key = str(f)
            try:
                st = f.stat()
            except OSError:
                continue
            if self._seen.get(key) == [st.st_size, st.st_mtime]:
                continue
            last = self._pending.get(key)
            if last is None or last[0] != st.st_size:
                self._pending[key] = (st.st_size, now)
                continue
            if now - last[1] < SETTLE_SECONDS:
                continue
            del self._pending[key]
            self._seen[key] = [st.st_size, st.st_mtime]
            self._save(ctx)
            self._queue.put(f)

    def _work(self, ctx: Context) -> None:
        while not self._halt.is_set():
            try:
                f = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                log.info("%s: converting %s", self.id, f.name)
                self.convert(f, ctx)
            except Exception:  # noqa: BLE001 - one bad replay does not stop the next
                log.exception("%s: could not convert %s", self.id, f.name)

    def _save(self, ctx: Context) -> None:
        p = self._state_path(ctx)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._seen), encoding="utf-8")

    def stop(self, ctx: Context) -> None:
        self._halt.set()
        if self._worker is not None:
            self._worker.join(5)

    def convert_existing(self, ctx: Context, folder: Path | None = None) -> int:
        """Convert every replay in the folders (or `folder`) now, in this thread. Returns how many."""
        files = (
            [p for pat in self.patterns for p in folder.glob(pat)] if folder is not None else self._files(ctx)
        )
        for f in sorted(files):
            self.convert(f, ctx)
        return len(files)


@dataclass
class Cs2Demos(ReplayAdapter):
    def folders(self, ctx: Context) -> list[Path]:
        out = []
        for common in steam_libraries():
            csgo = common / "Counter-Strike Global Offensive" / "game" / "csgo"
            if csgo.is_dir():
                out += [csgo / "replays", csgo]
        return out + super().folders(ctx)

    def missing(self) -> str | None:
        try:
            import demoparser2  # noqa: F401
        except ImportError:
            return "pip install -r ../cs2/requirements.txt"
        if not (ROOT / "cs2" / "heat3d_cs2").is_dir():
            return "the cs2 converter is not next to the watcher"
        return None

    def convert(self, path: Path, ctx: Context) -> None:
        when = path.stat().st_mtime
        session = ctx.open(self, when=when, meta={"replay": str(path)})
        try:
            out = subprocess.run(
                [sys.executable, "-m", "heat3d_cs2", "demos", str(path), "-o", str(session.folder), "--jobs", "1"],
                cwd=ROOT / "cs2",
                capture_output=True,
                text=True,
                timeout=3600,
                env={**os.environ, "HEAT3D_NO_UPDATE": "1"},
            )
            session.meta["converter"] = (out.stdout + out.stderr).strip()[-2000:]
            summaries = sorted(session.folder.glob("*/*.match.json"))
            if out.returncode != 0 or not summaries:
                session.event("replay_failed", file=path.name)
                return
            record_match(session, json.loads(summaries[0].read_text(encoding="utf-8")), signed_in_steam_ids())
            tables = sorted(session.folder.glob("*/*.parquet"))
            if tables:
                count_events(session, tables[0])
        finally:
            ctx.close(session)


def record_match(session, match: dict, me: set[str]) -> None:
    """A CS2 match summary into the session: what happened, the score, and the signed-in player's line."""
    session.meta["match"] = {k: match.get(k) for k in ("match", "map", "teams", "score")}
    session.stat("map", str(match.get("map") or ""))
    session.stat("rounds", len(match.get("rounds") or []))
    for team, score in (match.get("score") or {}).items():
        session.stat(f"score.{team}", score)
    for p in match.get("players") or []:
        if str(p.get("steamid")) not in me:
            continue
        session.meta["player"] = p.get("name")
        for key, name in (
            ("kills", "kills"),
            ("deaths", "deaths"),
            ("assists", "assists"),
            ("adr", "adr"),
            ("kast", "kast"),
            ("headshotPct", "headshot_pct"),
            ("openingKills", "opening_kills"),
            ("openingDeaths", "opening_deaths"),
            ("tradeKills", "trade_kills"),
            ("clutchesWon", "clutches_won"),
            ("utilityDamage", "utility_damage"),
            ("enemiesFlashed", "enemies_flashed"),
            ("plants", "plants"),
            ("defuses", "defuses"),
            ("roundsPlayed", "rounds_played"),
        ):
            value = p.get(key)
            if isinstance(value, (int, float)):
                session.stat(name, value)
        won = None
        for team, score in (match.get("score") or {}).items():
            if team == p.get("team"):
                others = [s for t, s in (match.get("score") or {}).items() if t != team]
                won = bool(others) and score > max(others)
        if won is not None:
            session.stat("won", 1 if won else 0)


def count_events(session, table: Path) -> None:
    """The converter's table's events, counted for the statistics without copying its rows."""
    import pyarrow.parquet as pq

    events = pq.read_table(table, columns=["event"]).column("event").to_pylist()
    counts: dict[str, int] = {}
    for e in events:
        counts[str(e)] = counts.get(str(e), 0) + 1
    with session._lock:
        for e, n in counts.items():
            session.counts[e] = session.counts.get(e, 0) + n
        session.positions += len(events)


#: Seconds from 0001-01-01 (the start of .NET's ticks) to the Unix epoch.
_TICKS_TO_UNIX = 62135596800


def _osr_string(buf: bytes, at: int) -> tuple[str, int]:
    """An .osr string: 0x00 for none, or 0x0b, a ULEB128 length and that many UTF-8 bytes."""
    kind = buf[at]
    at += 1
    if kind == 0x00:
        return "", at
    if kind != 0x0B:
        raise ValueError(f"not an osu! string at {at - 1}")
    length = shift = 0
    while True:
        b = buf[at]
        at += 1
        length |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            break
    return buf[at : at + length].decode("utf-8", errors="replace"), at + length


def parse_osr(buf: bytes) -> dict:
    """
    An osu! replay (.osr), per the file format on the osu! wiki
    (https://osu.ppy.sh/wiki/en/Client/File_formats/osr_(file_format)):
    the header, and the cursor frames - milliseconds since the previous frame,
    x 0..512, y 0..384 on the playfield, and the keys held.
    """
    import lzma
    import struct as st

    at = 0
    mode, version = st.unpack_from("<Bi", buf, at)
    at += 5
    beatmap, at = _osr_string(buf, at)
    player, at = _osr_string(buf, at)
    _replay_hash, at = _osr_string(buf, at)
    n300, n100, n50, geki, katu, miss = st.unpack_from("<6h", buf, at)
    at += 12
    score, combo, perfect, mods = st.unpack_from("<ihBi", buf, at)
    at += 11
    _life, at = _osr_string(buf, at)
    ticks, length = st.unpack_from("<qi", buf, at)
    at += 12
    data = lzma.decompress(buf[at : at + length]) if length > 0 else b""
    frames = []
    t = 0
    for part in data.decode("ascii", errors="replace").split(","):
        fields = part.split("|")
        if len(fields) != 4:
            continue
        w, x, y, z = fields
        if w == "-12345":
            continue  # the RNG seed, not a frame
        t += int(w)
        frames.append((t, float(x), float(y), int(float(z))))
    hits = n300 + n100 + n50 + miss
    return {
        "mode": mode,
        "version": version,
        "beatmap": beatmap,
        "player": player,
        "count_300": n300,
        "count_100": n100,
        "count_50": n50,
        "count_geki": geki,
        "count_katu": katu,
        "misses": miss,
        "score": score,
        "max_combo": combo,
        "perfect": perfect,
        "mods": mods,
        "played_at": ticks / 1e7 - _TICKS_TO_UNIX if ticks > 0 else None,
        # Standard mode's accuracy; the other modes weigh their judgements differently.
        "accuracy": (300 * n300 + 100 * n100 + 50 * n50) / (300 * hits) if mode == 0 and hits else None,
        "frames": frames,
    }


@dataclass
class OsuReplays(ReplayAdapter):
    def folders(self, ctx: Context) -> list[Path]:
        base = Path(os.environ.get("LOCALAPPDATA", "")) / "osu!"
        return [base / "Data" / "r", base / "Replays"] + super().folders(ctx)

    def convert(self, path: Path, ctx: Context) -> None:
        r = parse_osr(path.read_bytes())
        session = ctx.open(self, when=r["played_at"] or path.stat().st_mtime, meta={"replay": str(path)})
        try:
            play = r["beatmap"] or path.stem
            frames = r["frames"]
            # The playfield drawn on the ground: x across, its y away from the viewer.
            for i, (t, x, y, keys) in enumerate(frames):
                event = "session_start" if i == 0 else "finish" if i == len(frames) - 1 else "position"
                session.event(event, t=t / 1000.0, x=x, y=0.0, z=y, player=r["player"] or "me", session=play, keys=keys)
            for key in ("score", "max_combo", "count_300", "count_100", "count_50", "misses", "mode", "mods"):
                session.stat(key, r[key])
            if r["accuracy"] is not None:
                session.stat("accuracy", round(r["accuracy"], 5))
            session.stat("full_combo", 1 if r["perfect"] else 0)
        finally:
            ctx.close(session)


ADAPTERS = [
    Cs2Demos(
        id="cs2-demos",
        game="Counter-Strike 2",
        patterns=("*.dem",),
        source="demos CS2 saves: <CS2>/game/csgo/replays/*.dem (downloaded from the match history) and <CS2>/game/csgo/*.dem (recorded)",
        records=(
            "every player's position, look direction, health and money (every 8th tick)",
            "kills, deaths, damage, grenades, plants and defuses where they happened",
            "rounds, score, economy; the signed-in player's scoreboard line as statistics",
        ),
        enable="download a match's demo in CS2 (Watch > Your Matches), or record one with the console's `record`",
        axes=Axes(up="z", flip_y=True, units="inches (Source units)", scale=39.37),
        verified="the converter is tested on 165 real tournament demos; the folder watch and statistics by tests",
        multiplayer=True,
    ),
    OsuReplays(
        id="osu-replays",
        game="osu!",
        patterns=("*.osr",),
        source="replays osu! (stable) saves: %LOCALAPPDATA%/osu!/Data/r/*.osr (every play) and osu!/Replays/*.osr (exported)",
        records=(
            "the cursor's path over the playfield (x 0..512, y 0..384) and the keys held, every frame",
            "score, combo, 300s/100s/50s/misses, accuracy, mods, full combo",
        ),
        axes=Axes(up="y", flip_z=True, units="osu! pixels (playfield 512 x 384)"),
        verified="tested on replays built to the published .osr format; osu!lazer keeps replays in its own store and is not read",
        multiplayer=True,
    ),
]
