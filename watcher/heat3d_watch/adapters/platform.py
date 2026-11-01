"""
What the launcher and the graphics driver keep about play: Steam's statistics,
achievements, playtime and timelines, and NVIDIA's highlights and clips.

None of these sources needs a game to cooperate or to be running, so the three
adapters here are always on and look at a few files every few seconds.

Steam statistics (`steam-stats`)
    `<Steam>\\appcache\\stats\\UserGameStats_<account>_<appid>.bin`, rewritten by
    the Steam client whenever a game stores its stats, decoded with the schema
    `UserGameStatsSchema_<appid>.bin` next to it (binary KeyValues, see `_steam.py`).
    Stats are integers or floats as the game defines them; achievements are bits
    with an unlock time in Unix seconds. Playtime is in
    `<Steam>\\userdata\\<account>\\config\\localconfig.vdf` under apps/<appid>:
    Playtime and Playtime2wks in minutes, LastPlayed in Unix seconds.
    Steam is found through HKCU\\Software\\Valve\\Steam SteamPath.
    Stats API: https://partner.steamgames.com/doc/features/achievements

Steam Timeline (`steam-timeline`)
    `<Steam>\\userdata\\<account>\\gamerecordings\\timelines\\timeline_<appid><yyyymmdd>_<hhmmss>.json`,
    written while Steam Game Recording is on. The file is JSON with string
    numbers: "daterecorded" (Unix seconds), "starttime"/"endtime" (ms) and
    "entries", each {"id", "time" (ms since daterecorded), "type", ...}. Types
    seen on real files:
        achievement        achievement_name - Steam adds these for every game
        gamemode           mode, an ETimelineGameMode: 1 playing, 2 staging, 3 menus,
                           4 loading screen
        error              description - Steam's own ("#GameRecording_RecordingFailed")
        event              a game's AddInstantaneousTimelineEvent/AddRangeTimelineEvent:
                           title, description, icon, priority, duration (ms),
                           possible_clip - "You killed <name>" / "with the <weapon>"
        state_description  title - SetTimelineTooltip ("Round 3")
        phase              a StartGamePhase..EndGamePhase span: duration (ms), tags
                           [{name, icon, group, priority}] (mode, map ...)
    The file layout is Steam's and undocumented; the fields are the API's
    parameters. Any other type is recorded under its own name with its fields.
    https://partner.steamgames.com/doc/api/ISteamTimeline
    https://partner.steamgames.com/doc/features/timeline

NVIDIA highlights and clips (`nvidia-highlights`)
    What NVIDIA App / GeForce Experience keeps, all under
    `%LOCALAPPDATA%\\NVIDIA Corporation\\NVIDIA Overlay`:
      UGCTracker.json      the gallery index: {"UGC": [{id, path, type (video,
                           image), subType (InstantReplay, Normal, ...), date
                           (UTC, when the file was written), profileName, drsName
                           (the exe), shortName, msDuration, ...}], "apps", "info"}
      GallerySettings.json settings.currentDirectoryV2 (where clips are saved),
                           settings.tempDirectory (highlights wait in
                           <temp>\\Highlights\\<game> until saved)
      Highlights\\<game>\\config.json  the highlight kinds a game registered:
                           {"highlights": {id: {significance, highlightTags,
                           name: {locale: text}}}, profileName, drsName, shortName}
    and the clip files themselves, `<capture folder>\\<Game>\\<Game> yyyy.mm.dd -
    hh.mm.ss.ff.DVR.mp4` (screenshots: `<Game> Screenshot yyyy.mm.dd - ...png`), the
    name's time being the local time the clip was saved. GfeSDK games also log
    their highlight calls in `%LOCALAPPDATA%\\NVIDIA Corporation\\GfeSDK\\<exe>_<pid>.log`.

    What this can and cannot deliver: a game that supports NVIDIA Highlights
    (GfeSDK) registers highlight kinds once - an id like "kill", a
    significance, tags, a display name - and later says only "highlight <id>,
    from <ms> before now to <ms> after". There are no positions, no match or
    round, no opponent, no custom data: the SDK has no field for any of it. A
    saved clip adds a file, a game name, a length and a time. So what is
    recorded is "this kind of moment, in this game, at this time" - and a place
    only by looking the time up in another adapter's recording of the same game
    (`position_at`), which this adapter does for every clip it records.
    https://developer.nvidia.com/highlights

Each adapter keeps what it has already seen in `<data>\\state\\<id>.json`, so a
restart records nothing twice.
"""

from __future__ import annotations

import json
import logging
import os
import re
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..session import Axes, Session, _slug
from . import _steam
from .base import Adapter, Context

log = logging.getLogger("heat3d_watch")


# ------------------------------------------------------------------------------ state


def _state_path(ctx: Context, adapter_id: str) -> Path:
    return ctx.data / "state" / f"{adapter_id}.json"


def load_state(ctx: Context, adapter_id: str) -> dict[str, Any] | None:
    """What an adapter saw before, or None the first time it runs."""
    try:
        doc = json.loads(_state_path(ctx, adapter_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def save_state(ctx: Context, adapter_id: str, doc: dict[str, Any]) -> None:
    """Written beside and swapped in, so a crash mid-write keeps the old state, not half a new one."""
    path = _state_path(ctx, adapter_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def _unique_when(ctx: Context, adapter_id: str, when: float) -> float:
    """A start time whose second no other session of this adapter has, in any game.

    A session's id in the statistics is its folder name, `<start>_<adapter>`, which
    does not name the game: two games' stats files written in the same second
    (Steam syncs them together at login) would share one id and one would replace
    the other. Moving the later one by a second keeps both.
    """
    root = ctx.data / "sessions"
    slug = _slug(adapter_id)
    for _ in range(3600):
        stamp = datetime.fromtimestamp(when).strftime("%Y-%m-%d_%H-%M-%S")
        if not any(root.glob(f"*/{stamp}_{slug}")):
            return when
        when += 1.0
    return when


def _close_at(ctx: Context, session: Session, ended: float) -> None:
    """Close a session that describes the past: its end is when the source says, not now.

    Session.close stamps the current time as the end; a snapshot of last year's
    stats or a timeline written hours ago would otherwise claim hours of play.
    Closing first and fixing the end before `ctx.close` puts the right length
    into meta.json and the statistics.
    """
    session.close()
    session.ended = max(ended, session.started)
    session.write_meta()
    ctx.close(session)


# ------------------------------------------------------------------------ steam-stats


_STATS_FILE = re.compile(r"UserGameStats_(\d+)_(\d+)\.bin$", re.I)


@dataclass
class SteamStats(Adapter):
    """Steam's own copy of every game's stats and achievements, recorded when it changes."""

    #: Steam's folder; empty: from the registry.
    root: str = ""

    def __post_init__(self) -> None:
        self._steam: _steam.Steam | None = None
        self._state: dict[str, Any] | None = None

    def steam(self) -> _steam.Steam | None:
        if self._steam is None:
            root = Path(self.root) if self.root else _steam.steam_root()
            if root is None:
                return None
            self._steam = _steam.Steam(root)
        return self._steam

    def missing(self) -> str | None:
        st = self.steam()
        if st is None or not st.stats_dir.is_dir():
            return "Steam not found (no appcache\\stats folder)"
        return None

    def start(self, ctx: Context) -> None:
        self._state = load_state(ctx, self.id)
        self.poll(ctx)

    def poll(self, ctx: Context) -> None:
        st = self.steam()
        if st is None:
            return
        first = self._state is None
        state = self._state or {"version": 1, "games": {}, "playtime": {}, "localconfig": {}}
        games: dict[str, Any] = state.setdefault("games", {})
        play_seen: dict[str, Any] = state.setdefault("playtime", {})
        lc_seen: dict[str, float] = state.setdefault("localconfig", {})

        changed: dict[str, dict[str, Any]] = {}  # "<account>_<appid>" -> what to record
        dirty = first

        # Playtime first: a game's stats file and its playtime usually change together,
        # when the game closes, and belong in the same session.
        play_now: dict[str, dict[str, int]] = {}
        for account in _steam.accounts(st.root):
            path = st.root / "userdata" / account / "config" / "localconfig.vdf"
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if lc_seen.get(account) == mtime:
                continue
            lc_seen[account] = mtime
            dirty = True
            for appid, row in _steam.playtimes(st.root, account).items():
                key = f"{account}_{appid}"
                play_now[key] = row
                if play_seen.get(key) != row:
                    old = play_seen.get(key)
                    play_seen[key] = row
                    # The first look is a baseline for games without stats; the stats
                    # snapshot below brings the playtime of the games that have them.
                    if not first and old is not None and old.get("Playtime") != row.get("Playtime"):
                        changed.setdefault(key, {})["playtime"] = (old, row)
                    elif not first and old is None and row.get("Playtime"):
                        changed.setdefault(key, {})["playtime"] = ({}, row)

        try:
            files = list(st.stats_dir.iterdir())
        except OSError:
            files = []
        for f in files:
            m = _STATS_FILE.match(f.name)
            if not m:
                continue
            account, appid = m.groups()
            key = f"{account}_{appid}"
            try:
                mtime = f.stat().st_mtime
            except OSError:
                continue
            seen = games.get(key)
            if seen and seen.get("mtime") == mtime:
                continue
            dirty = True
            schema = st.schema(appid)
            try:
                stats, unlocked = _steam.user_stats(_steam.parse_binary_kv(f.read_bytes()), schema)
            except (OSError, ValueError) as e:
                # Mid-write or damaged: remember the time so it is not retried every poll;
                # the next write gives it a new one.
                log.warning("%s: could not read %s: %s", self.id, f.name, e)
                games.setdefault(key, {})["mtime"] = mtime
                state["bad_files"] = int(state.get("bad_files", 0)) + 1
                continue
            old_stats = (seen or {}).get("stats", {})
            old_ach = (seen or {}).get("achievements", {})
            games[key] = {"mtime": mtime, "stats": stats, "achievements": unlocked}
            if seen is not None and "stats" not in seen:
                # Was unreadable before: everything counts as new now, but without
                # pretending it all changed - take it as this game's first snapshot.
                old_stats, old_ach, seen = {}, {}, None
            new_ach = {a: t for a, t in unlocked.items() if a not in old_ach}
            diff = {k: (old_stats.get(k), v) for k, v in stats.items() if old_stats.get(k) != v}
            if seen is None or diff or new_ach:
                c = changed.setdefault(key, {})
                c.update(stats=stats, unlocked=unlocked, new_ach=new_ach, diff=diff if seen else {},
                         mtime=mtime, snapshot=seen is None)
                if key in play_now or key in play_seen:
                    c.setdefault("playtime", ({}, play_now.get(key) or play_seen.get(key)))

        for key, c in sorted(changed.items()):
            try:
                self._record(ctx, st, key, c)
            except Exception:  # noqa: BLE001 - one game's odd file must not lose the others
                log.exception("%s: could not record %s", self.id, key)
        self._state = state
        if dirty:
            save_state(ctx, self.id, state)

    def _record(self, ctx: Context, st: _steam.Steam, key: str, c: dict[str, Any]) -> None:
        account, appid = key.split("_", 1)
        schema = st.schema(appid)
        when = float(c.get("mtime") or time.time())
        s = ctx.open(self, game=st.app_name(appid), when=_unique_when(ctx, self.id, when))
        s.meta.update({"steam_appid": int(appid)})
        stats: dict[str, float] = c.get("stats") or {}
        unlocked: dict[str, int] = c.get("unlocked") or {}
        new_ach: dict[str, int] = c.get("new_ach") or {}
        if "stats" in c:
            s.event(
                "stats_snapshot" if c.get("snapshot") else "stats_update",
                t=0.0,
                ms=when * 1000,
                player="me",
                stats=len(stats),
                achievements=len(unlocked),
                changed=len(c.get("diff") or {}) + len(new_ach),
            )
        for name, (old, new) in (c.get("diff") or {}).items():
            delta = new - old if isinstance(old, (int, float)) and isinstance(new, (int, float)) else None
            s.event("stat_change", t=0.0, ms=when * 1000, player="me", stat=name, old=old, new=new, delta=delta)
        for api, unlock in sorted(new_ach.items(), key=lambda kv: kv[1]):
            ach = schema.by_api(api) if schema else None
            s.event(
                "achievement",
                t=0.0,
                ms=(unlock or when) * 1000,
                player="me",
                achievement=api,
                name=(ach.name if ach else "") or api,
                description=ach.desc if ach and ach.desc else None,
                unlocked=datetime.fromtimestamp(unlock, timezone.utc).isoformat(timespec="seconds") if unlock else None,
            )
        for name, value in stats.items():
            s.stat(_steam.snake(name), value)
        if "stats" in c:
            s.stat("achievements_unlocked", len(unlocked))
            if schema and schema.achievements:
                s.stat("achievements_total", len(schema.achievements))
        play = c.get("playtime")
        if play:
            old, row = play
            if "stats" not in c:
                s.event("playtime_update", t=0.0, ms=when * 1000, player="me",
                        minutes=row.get("Playtime"), before=old.get("Playtime"))
            if "Playtime" in row:
                s.stat("playtime_minutes", row["Playtime"])
            if "Playtime2wks" in row:
                s.stat("playtime_2wks_minutes", row["Playtime2wks"])
            if "LastPlayed" in row:
                s.stat("last_played", row["LastPlayed"])
        _close_at(ctx, s, when)


# --------------------------------------------------------------------- steam-timeline


_TIMELINE_FILE = re.compile(r"timeline_(\d+)(\d{8})_(\d{6})\.json$", re.I)
GAME_MODES = {0: "invalid", 1: "playing", 2: "staging", 3: "menus", 4: "loading_screen"}
#: Attributes a timeline entry may not take as they are: the table's own columns.
_RESERVED = {"time", "ms", "event", "x", "y", "z", "player", "team", "session", "id", "type", "t"}
#: Fields Steam writes as strings that are numbers (ms, priorities, ETimelineEventClipPriority).
_NUMERIC = {"duration", "priority", "possible_clip", "mode", "starttime", "endtime"}


def _num(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_timeline(text: str) -> dict[str, Any] | None:
    """A timeline file, also while Steam is still writing it (the closing brackets missing)."""
    try:
        doc = json.loads(text)
        return doc if isinstance(doc, dict) else None
    except ValueError:
        pass
    body = text.rstrip().rstrip(",")
    for tail in ("}", "]}", "}]}", "\n]\n}"):
        try:
            doc = json.loads(body + tail)
            return doc if isinstance(doc, dict) else None
        except ValueError:
            continue
    # Cut inside an entry: drop the unfinished one.
    cut = body.rfind("}")
    while cut > 0:
        try:
            doc = json.loads(body[: cut + 1] + "]}")
            return doc if isinstance(doc, dict) else None
        except ValueError:
            cut = body.rfind("}", 0, cut)
    return None


def _entry_key(e: dict[str, Any]) -> str:
    """An entry's identity across reads: its id, or its time and type where it has none."""
    return str(e.get("id") or f"{e.get('time')}/{e.get('type')}")


def timeline_event(entry: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """An entry as an event name and its attributes."""
    kind = str(entry.get("type") or "entry")
    attrs: dict[str, Any] = {}
    for k, v in entry.items():
        if k in ("id", "time", "type") or isinstance(v, (dict, list)):
            continue
        if k in _NUMERIC and isinstance(v, str):
            v = _num(v, 0.0)
            v = int(v) if v == int(v) else v
        attrs[f"tl_{k}" if k in _RESERVED else k] = v
    # A phase's tags (AddGamePhaseTag: name, icon, group, priority): one attribute per group.
    for tag in entry.get("tags") or []:
        if isinstance(tag, dict) and tag.get("name"):
            attrs[f"tag_{_steam.snake(str(tag.get('group') or 'tag'))}"] = str(tag["name"])
    if kind == "gamemode":
        mode = int(_num(entry.get("mode"), 0))
        attrs["mode"] = mode
        attrs["mode_name"] = GAME_MODES.get(mode, str(mode))
        return "game_mode", attrs
    if kind == "event":
        # A game's own AddTimelineEvent. Its title is free text ("You killed <name> with
        # <weapon>"), its icon the closest thing to a kind: Steam's ("steam_death") or the
        # game's ("<game>_gun_kill"). Icons that only count ("<game>_12") say nothing.
        icon = str(entry.get("icon") or "")
        name = icon[6:] if icon.startswith("steam_") else icon
        if not name or re.fullmatch(r"(.*_)?\d+", name):
            return "timeline_event", attrs
        return _steam.snake(name), attrs
    return _steam.snake(kind), attrs


@dataclass
class SteamTimeline(Adapter):
    """Steam Game Recording's timelines: one session per timeline file."""

    root: str = ""
    #: A file not written to for this long, still without an end, is taken as ended (a crash).
    stale_seconds: float = 2 * 3600

    def __post_init__(self) -> None:
        self._steam: _steam.Steam | None = None
        self._state: dict[str, Any] = {}
        self._open: dict[str, Session] = {}

    def steam(self) -> _steam.Steam | None:
        if self._steam is None:
            root = Path(self.root) if self.root else _steam.steam_root()
            if root is None:
                return None
            self._steam = _steam.Steam(root)
        return self._steam

    def folders(self) -> list[Path]:
        st = self.steam()
        if st is None:
            return []
        return [st.root / "userdata" / a / "gamerecordings" / "timelines" for a in _steam.accounts(st.root)]

    def missing(self) -> str | None:
        if self.steam() is None:
            return "Steam not found"
        return None

    def start(self, ctx: Context) -> None:
        self._state = load_state(ctx, self.id) or {"version": 1, "files": {}}
        self.poll(ctx)

    def poll(self, ctx: Context) -> None:
        files: dict[str, Any] = self._state.setdefault("files", {})
        dirty = False
        for folder in self.folders():
            try:
                paths = list(folder.glob("timeline_*.json"))
            except OSError:
                continue
            for path in paths:
                m = _TIMELINE_FILE.match(path.name)
                if not m:
                    continue
                key = f"{folder.parent.parent.name}/{path.name}"
                try:
                    stt = path.stat()
                except OSError:
                    continue
                seen = files.get(key) or {}
                if seen.get("ended") or (seen.get("mtime") == stt.st_mtime and seen.get("size") == stt.st_size):
                    if not seen.get("ended") and key in self._open and time.time() - stt.st_mtime > self.stale_seconds:
                        self._finish(ctx, key, stt.st_mtime)
                        seen["ended"] = True
                        dirty = True
                    continue
                try:
                    self._read(ctx, key, path, m.group(1), stt.st_mtime, files)
                except Exception:  # noqa: BLE001
                    log.exception("%s: could not read %s", self.id, path.name)
                files.setdefault(key, {}).update(mtime=stt.st_mtime, size=stt.st_size)
                dirty = True
        if dirty:
            save_state(ctx, self.id, self._state)

    def _read(self, ctx: Context, key: str, path: Path, appid: str, mtime: float, files: dict[str, Any]) -> None:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                doc = load_timeline(f.read())
        except OSError:
            return
        info = files.setdefault(key, {"ids": []})
        if doc is None:
            info["bad"] = int(info.get("bad", 0)) + 1
            return
        recorded = _num(doc.get("daterecorded"), mtime)
        seen_ids = set(info.setdefault("ids", []))
        entries = [e for e in (doc.get("entries") or []) if isinstance(e, dict)]
        new = [e for e in entries if _entry_key(e) not in seen_ids]
        ended = "endtime" in doc
        s = self._open.get(key)
        if s is None and (new or ended or not info.get("started")):
            st = self.steam()
            game = st.app_name(appid) if st else f"Steam app {appid}"
            s = ctx.open(self, game=game, when=_unique_when(ctx, self.id, recorded))
            s.meta.update({"steam_appid": int(appid), "timeline": path.name})
            self._open[key] = s
            if not info.get("started"):
                s.event("session_start", t=_num(doc.get("starttime")) / 1000, ms=recorded * 1000, player="me")
                info["started"] = True
        if s is not None:
            schema = self.steam().schema(appid) if self.steam() else None
            # Game phases (a match, a mission) bound the events in them: their id goes in
            # the session column, so the viewer can split a sitting into its matches.
            phases = sorted(
                (_num(e.get("time")), _num(e.get("time")) + _num(e.get("duration")), f"phase-{e.get('id')}")
                for e in entries if e.get("type") == "phase"
            )
            for e in sorted(new, key=lambda e: _num(e.get("time"))):
                t_ms = _num(e.get("time"))
                name, attrs = timeline_event(e)
                phase = next((pid for a, b, pid in reversed(phases) if a <= t_ms <= b), None)
                if phase:
                    attrs["session"] = phase
                if name == "achievement" and schema:
                    ach = schema.by_api(str(e.get("achievement_name") or ""))
                    if ach and ach.name:
                        attrs["name"] = ach.name
                    s.stat("achievements", 1, "add")
                elif name == "game_mode":
                    s.stat("game_mode_changes", 1, "add")
                elif name == "phase":
                    s.stat("phases", 1, "add")
                elif e.get("type") == "event":
                    s.stat("timeline_events", 1, "add")
                s.event(name, t=t_ms / 1000, ms=recorded * 1000 + t_ms, player="me", **attrs)
                info["ids"].append(_entry_key(e))
            if ended:
                end_ms = _num(doc.get("endtime"))
                s.event("session_end", t=end_ms / 1000, ms=recorded * 1000 + end_ms, player="me")
                s.stat("recorded_seconds", round(end_ms / 1000, 1))
                self._finish(ctx, key, recorded + end_ms / 1000)
                info["ended"] = True
            elif time.time() - mtime > self.stale_seconds:
                self._finish(ctx, key, mtime)
                info["ended"] = True

    def _finish(self, ctx: Context, key: str, ended: float) -> None:
        s = self._open.pop(key, None)
        if s is not None:
            _close_at(ctx, s, ended)

    def stop(self, ctx: Context) -> None:
        for key in list(self._open):
            s = self._open.pop(key)
            ctx.close(s)
        save_state(ctx, self.id, self._state)


# -------------------------------------------------------------------- nvidia-highlights


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _same_game(a: str, b: str) -> bool:
    """NVIDIA's profile names and the watcher's game names differ in spelling, rarely in words."""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    short, long = sorted((na, nb), key=len)
    return len(short) >= 5 and short in long


def _meta_time(v: Any) -> float | None:
    if not isinstance(v, str):
        return None
    try:
        return datetime.fromisoformat(v).timestamp()
    except ValueError:
        return None


def position_at(data_dir: Path, game: str, ms: float, *, within_ms: float = 5000,
                skip_adapter: str = "nvidia-highlights") -> dict[str, Any] | None:
    """Where "me" was in `game` at wall-clock `ms`: the last recorded position at or before it.

    Looks through `<data>/sessions/<game>/*/events.jsonl` of every session whose
    time span covers `ms` (any adapter but `skip_adapter`), and returns the
    latest own row with a position no more than `within_ms` before `ms`:
    {"x", "y", "z", "ms", "session" (the row's match/round, if any), "folder",
    "adapter", "axes"} - or None when no recording was there.
    """
    root = data_dir / "sessions"
    try:
        game_dirs = [d for d in root.iterdir() if d.is_dir() and _same_game(d.name, game)]
    except OSError:
        return None
    best: dict[str, Any] | None = None
    for gdir in game_dirs:
        try:
            folders = [f for f in gdir.iterdir() if f.is_dir()]
        except OSError:
            continue
        for folder in folders:
            try:
                meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if meta.get("adapter") == skip_adapter or not meta.get("positions") and meta.get("ended"):
                continue
            started = _meta_time(meta.get("started"))
            ended = _meta_time(meta.get("ended")) or time.time()
            # meta.json keeps whole seconds; allow for that at both ends.
            if started is None or ms < started * 1000 - 1000 or ms > ended * 1000 + 1000 + within_ms:
                continue
            row = _last_own_position(folder / "events.jsonl", ms, within_ms)
            if row and (best is None or row["ms"] > best["ms"]):
                best = {
                    "x": row["x"], "y": row["y"], "z": row["z"], "ms": row["ms"],
                    "session": row.get("session"), "folder": folder.name,
                    "adapter": meta.get("adapter"), "axes": meta.get("axes"),
                }
    return best


def _last_own_position(path: Path, ms: float, within_ms: float) -> dict[str, Any] | None:
    best = None
    try:
        f = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return None
    with f:
        for line in f:
            if '"x"' not in line or '"me"' not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("player") != "me" or "x" not in r:
                continue
            t = r.get("ms")
            if not isinstance(t, (int, float)) or t > ms or t < ms - within_ms:
                continue
            if best is None or t >= best["ms"]:
                best = r
    return best


_CLIP_NAME = re.compile(
    r"^(?P<game>.*?)\s+(?P<shot>Screenshot\s+)?(?P<date>\d{4}\.\d{2}\.\d{2})\s+-\s+"
    r"(?P<time>\d{2}\.\d{2}\.\d{2})(?:\.(?P<frac>\d{1,3}))?(?P<rest>(?:\.[^.]+)*?)\.(?P<ext>[A-Za-z0-9]+)$"
)
_MEDIA = {"mp4": "video", "mkv": "video", "mov": "video", "png": "image", "jpg": "image", "jpeg": "image"}


def parse_clip_name(name: str) -> dict[str, Any] | None:
    """`<Game> 2025.10.20 - 00.49.45.115.DVR.mp4` -> game, local save time (Unix ms), kind, media."""
    m = _CLIP_NAME.match(name)
    if not m:
        return None
    try:
        dt = datetime.strptime(f"{m['date']} {m['time']}", "%Y.%m.%d %H.%M.%S")
    except ValueError:
        return None
    frac = m["frac"] or ""
    ms = dt.timestamp() * 1000 + (int(frac.ljust(3, "0")) if frac else 0)
    rest = (m["rest"] or "").strip(".")
    kind = "InstantReplay" if rest.upper() == "DVR" else ("Screenshot" if m["shot"] else (rest or "Clip"))
    return {"game": m["game"].strip(), "ms": ms, "kind": kind, "media": _MEDIA.get(m["ext"].lower(), "file")}


def mp4_duration(path: Path) -> float | None:
    """Seconds, from the movie header (moov/mvhd) - the box is found by seeking, the video is not read."""
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            end = f.tell()
            pos = 0
            while pos + 8 <= end:
                f.seek(pos)
                size, kind = struct.unpack(">I4s", f.read(8))
                header = 8
                if size == 1:
                    size = struct.unpack(">Q", f.read(8))[0]
                    header = 16
                elif size == 0:
                    size = end - pos
                if size < header:
                    return None
                if kind == b"moov":
                    inner, stop = pos + header, pos + size
                    while inner + 8 <= stop:
                        f.seek(inner)
                        isize, ikind = struct.unpack(">I4s", f.read(8))
                        if ikind == b"mvhd":
                            version = f.read(4)[0]
                            if version == 1:
                                _, _, scale, dur = struct.unpack(">QQIQ", f.read(28))
                            else:
                                _, _, scale, dur = struct.unpack(">IIII", f.read(16))
                            return dur / scale if scale else None
                        if isize < 8:
                            return None
                        inner += isize
                    return None
                pos += size
    except (OSError, struct.error, IndexError):
        return None
    return None


def _hms(text: Any) -> float | None:
    if not isinstance(text, str):
        return None
    parts = text.split(":")
    try:
        secs = 0.0
        for p in parts:
            secs = secs * 60 + float(p)
        return secs
    except ValueError:
        return None


@dataclass
class NvidiaHighlights(Adapter):
    """NVIDIA's saved clips, screenshots and highlights, each placed where a recording says the player was."""

    #: The NVIDIA Overlay folder; empty: under %LOCALAPPDATA%.
    overlay: str = ""
    #: The capture folder; empty: from GallerySettings.json, else %USERPROFILE%\\Videos\\NVIDIA.
    captures: str = ""
    #: The GfeSDK log folder; empty: under %LOCALAPPDATA%.
    gfesdk: str = ""
    #: The first run takes what is there as already seen; False records it too.
    baseline: bool = True
    #: A game's session closes after this long without a new clip.
    idle_seconds: float = 1800.0
    #: How far back a position may be from the moment and still count as where it happened.
    within_ms: float = 5000.0

    def __post_init__(self) -> None:
        self._state: dict[str, Any] = {}
        self._first = False
        #: game -> (session, wall time of its last item, Unix ms of its last moment)
        self._open: dict[str, tuple[Session, float, float]] = {}
        self._seen: set[str] = set()
        self._pending: dict[str, tuple[int, float]] = {}
        self._defs: dict[str, dict[str, Any]] = {}

    # -- where things are ---------------------------------------------------------------

    def overlay_dir(self) -> Path | None:
        if self.overlay:
            return Path(self.overlay)
        return _steam.environ_path("LOCALAPPDATA", "NVIDIA Corporation", "NVIDIA Overlay")

    def gfesdk_dir(self) -> Path | None:
        if self.gfesdk:
            return Path(self.gfesdk)
        return _steam.environ_path("LOCALAPPDATA", "NVIDIA Corporation", "GfeSDK")

    def gallery_settings(self) -> dict[str, Any]:
        o = self.overlay_dir()
        try:
            doc = json.loads((o / "GallerySettings.json").read_text(encoding="utf-8-sig")) if o else {}
        except (OSError, ValueError):
            return {}
        s = doc.get("settings") if isinstance(doc, dict) else None
        return s if isinstance(s, dict) else {}

    def capture_dir(self) -> Path | None:
        if self.captures:
            return Path(self.captures)
        cur = self.gallery_settings().get("currentDirectoryV2")
        if isinstance(cur, str) and cur:
            return Path(cur)
        return _steam.environ_path("USERPROFILE", "Videos", "NVIDIA")

    def temp_highlights_dir(self) -> Path | None:
        tmp = self.gallery_settings().get("tempDirectory")
        return Path(tmp) / "Highlights" if isinstance(tmp, str) and tmp else None

    def missing(self) -> str | None:
        o, c = self.overlay_dir(), self.capture_dir()
        if (o and o.is_dir()) or (c and c.is_dir()):
            return None
        return "NVIDIA App / GeForce Experience not found (no NVIDIA Overlay or capture folder)"

    # -- what the games registered ------------------------------------------------------

    def definitions(self) -> dict[str, dict[str, Any]]:
        """shortName -> {"profile", "exe", "highlights": {id: display name}} from Highlights\\*\\config.json."""
        out: dict[str, dict[str, Any]] = {}
        o = self.overlay_dir()
        if not o:
            return out
        try:
            configs = list((o / "Highlights").glob("*/config.json"))
        except OSError:
            return out
        for cfg in configs:
            try:
                doc = json.loads(cfg.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                continue
            if not isinstance(doc, dict):
                continue
            hl = doc.get("highlights") if isinstance(doc.get("highlights"), dict) else {}
            names = {}
            for hid, d in hl.items():
                n = d.get("name") if isinstance(d, dict) else None
                names[hid] = _steam._english(n) or hid  # some games register ids only
            out[str(doc.get("shortName") or cfg.parent.name)] = {
                "profile": str(doc.get("profileName") or ""),
                "exe": str(doc.get("drsName") or ""),
                "highlights": names,
            }
        return out

    def _definition_for(self, *names: str) -> tuple[str, dict[str, Any]] | None:
        for short, d in self._defs.items():
            for n in names:
                if n and (n == short or _same_game(n, d["profile"]) or n.lower() == d["exe"].lower()):
                    return short, d
        return None

    # -- what is there ------------------------------------------------------------------

    def items(self) -> dict[str, dict[str, Any]]:
        """Every clip and screenshot known, by normalised path: the gallery index and the folders."""
        found: dict[str, dict[str, Any]] = {}
        o = self.overlay_dir()
        if o:
            try:
                doc = json.loads((o / "UGCTracker.json").read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                doc = {}
            for u in (doc.get("UGC") or []) if isinstance(doc, dict) else []:
                if isinstance(u, dict) and isinstance(u.get("path"), str):
                    found[os.path.normcase(u["path"])] = {"path": u["path"], "ugc": u}
        roots: list[tuple[Path, bool]] = []
        c = self.capture_dir()
        if c:
            roots.append((c, False))
        t = self.temp_highlights_dir()
        if t:
            roots.append((t, True))
        for root, temp in roots:
            try:
                game_dirs = [d for d in root.iterdir() if d.is_dir()]
            except OSError:
                continue
            for d in game_dirs:
                try:
                    files = [f for f in d.iterdir() if f.suffix.lower().lstrip(".") in _MEDIA]
                except OSError:
                    continue
                for f in files:
                    key = os.path.normcase(str(f))
                    found.setdefault(key, {"path": str(f)})
                    found[key].update(folder_game=d.name, temp=temp, file=f)
        return found

    # -- lifecycle ----------------------------------------------------------------------

    def start(self, ctx: Context) -> None:
        state = load_state(ctx, self.id)
        self._first = state is None
        self._state = state or {"version": 1, "seen": [], "logs": {}}
        self._seen = set(self._state.get("seen") or [])
        self.poll(ctx)

    def poll(self, ctx: Context) -> None:
        self._defs = self.definitions()
        now = time.time()
        items = self.items()
        fresh: list[dict[str, Any]] = []
        for key, it in items.items():
            if key in self._seen:
                continue
            f: Path | None = it.get("file")
            if f is not None and not self._first:
                # A clip is written for a while after the key press: wait until it holds still.
                try:
                    st = f.stat()
                except OSError:
                    continue
                last = self._pending.get(key)
                self._pending[key] = (st.st_size, st.st_mtime)
                if last != (st.st_size, st.st_mtime) or now - st.st_mtime < 5:
                    continue
            self._pending.pop(key, None)
            self._seen.add(key)
            if not (self._first and self.baseline):
                fresh.append(it)
        fresh.sort(key=lambda it: self._moment(it)[0])
        for it in fresh:
            try:
                self._record(ctx, it)
            except Exception:  # noqa: BLE001
                log.exception("%s: could not record %s", self.id, it.get("path"))
        logs_changed = self._read_logs(ctx)
        for game, (s, last, last_ms) in list(self._open.items()):
            if now - last > self.idle_seconds:
                _close_at(ctx, s, last_ms / 1000)
                del self._open[game]
        if fresh or self._first or logs_changed:
            self._state["seen"] = sorted(self._seen)
            save_state(ctx, self.id, self._state)
        self._first = False

    def stop(self, ctx: Context) -> None:
        for game, (s, _, last_ms) in list(self._open.items()):
            # Ends at its last moment: a session per game holds moments, not hours of play.
            _close_at(ctx, s, last_ms / 1000)
        self._open.clear()
        self._state["seen"] = sorted(self._seen)
        save_state(ctx, self.id, self._state)

    # -- recording ----------------------------------------------------------------------

    def _moment(self, it: dict[str, Any]) -> tuple[float, dict[str, Any] | None]:
        """When the moment was (Unix ms) - the save time in the file name, else the index, else the file."""
        parsed = parse_clip_name(Path(it["path"]).name)
        if parsed:
            return parsed["ms"], parsed
        u = it.get("ugc") or {}
        d = u.get("date")
        if isinstance(d, str):
            try:
                return datetime.fromisoformat(d.replace("Z", "+00:00")).timestamp() * 1000, None
            except ValueError:
                pass
        try:
            return Path(it["path"]).stat().st_mtime * 1000, None
        except OSError:
            return time.time() * 1000, None

    def _record(self, ctx: Context, it: dict[str, Any]) -> None:
        u: dict[str, Any] = it.get("ugc") or {}
        path = Path(it["path"])
        ms, parsed = self._moment(it)
        game = str(u.get("profileName") or it.get("folder_game") or (parsed or {}).get("game") or "Unknown game")
        short = str(u.get("shortName") or "")
        hit = self._definition_for(short, str(it.get("folder_game") or ""), game, str(u.get("drsName") or ""))
        if hit and it.get("temp") and hit[1]["profile"]:
            game = hit[1]["profile"]  # temp highlight folders are named by shortName
        media = str(u.get("type") or (parsed or {}).get("media") or _MEDIA.get(path.suffix.lower().lstrip("."), "file"))
        kind = str(u.get("subType") or (parsed or {}).get("kind") or "")
        if it.get("temp"):
            kind = kind if kind and kind != "Clip" else "Highlight"
        duration = None
        if isinstance(u.get("msDuration"), (int, float)):
            duration = u["msDuration"] / 1000
        elif u.get("duration"):
            duration = _hms(u.get("duration"))
        elif media == "video" and path.exists():
            duration = mp4_duration(path)
        # A highlight's own id, when anything names it: a field of the index entry, or
        # the id appearing in the file name. Neither is documented; both are cheap to try.
        hid = next((str(v) for k, v in u.items() if "highlight" in k.lower() and isinstance(v, (str, int))), None)
        if hid is None and hit and it.get("temp"):
            # Only in the temp highlight folders: a saved clip's name is the game's name,
            # which may well contain a highlight id ("kill") by chance.
            stem = _norm(path.stem)
            hid = next((h for h in sorted(hit[1]["highlights"], key=len, reverse=True) if _norm(h) and _norm(h) in stem
                        and len(_norm(h)) >= 4), None)
        hname = hit[1]["highlights"].get(hid) if hit and hid else None
        pos = position_at(ctx.data, game, ms, within_ms=self.within_ms, skip_adapter=self.id)
        s = self._session(ctx, game, ms, pos)
        event = "screenshot" if media == "image" else "highlight"
        extra: dict[str, Any] = {}
        if pos:
            extra.update(x=pos["x"], y=pos["y"], z=pos["z"], session=pos.get("session") or pos["folder"],
                         position_from=pos["folder"], position_lag_ms=int(ms - pos["ms"]))
        s.event(
            event,
            t=max(0.0, ms / 1000 - s.started),
            ms=ms,
            player="me",
            game=game,
            clip=str(path),
            media=media,
            kind=kind or None,
            duration_s=round(duration, 3) if duration else None,
            clip_start_ms=int(ms - duration * 1000) if duration and media == "video" else None,
            highlight_id=hid,
            highlight_name=hname,
            exe=u.get("drsName") or None,
            **extra,
        )
        s.stat("clips" if media == "video" else "screenshots", 1, "add")
        if duration and media == "video":
            s.stat("clip_seconds", round(duration, 3), "add")
        if pos:
            s.stat("placed", 1, "add")

    def _session(self, ctx: Context, game: str, ms: float, pos: dict[str, Any] | None) -> Session:
        key = _slug(game).lower()
        hit = self._open.get(key)
        if hit and ms - hit[2] > self.idle_seconds * 1000:
            # Moments far apart are separate sittings (an older clip found late, a backfill).
            _close_at(ctx, hit[0], hit[2] / 1000)
            del self._open[key]
            hit = None
        if hit:
            self._open[key] = (hit[0], time.time(), max(hit[2], ms))
            return hit[0]
        axes = self.axes
        if pos and isinstance(pos.get("axes"), dict):
            # The positions are another adapter's: draw them with its axes.
            known = {k: v for k, v in pos["axes"].items() if k in Axes.__dataclass_fields__}
            try:
                axes = Axes(**known)
            except TypeError:
                pass
        s = ctx.open(self, game=game, when=_unique_when(ctx, self.id, min(ms / 1000, time.time())), axes=axes)
        self._open[key] = (s, time.time(), ms)
        return s

    # -- GfeSDK logs: the game asking for a highlight, even when no clip is kept ----------

    _LOG_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)\[\w\].*?Posted Request (HL\w+)\[")
    #: Requests that configure rather than mark a moment.
    _LOG_SKIP = {"HLConfigure", "HLGetConfig", "HLGetUserSettings", "HLSetDefaultLocale"}

    def _read_logs(self, ctx: Context) -> bool:
        d = self.gfesdk_dir()
        offsets: dict[str, int] = self._state.setdefault("logs", {})
        if not d:
            return False
        try:
            logs = list(d.glob("*.log"))
        except OSError:
            return False
        changed = False
        for p in logs:
            try:
                size = p.stat().st_size
            except OSError:
                continue
            key = p.name
            if key not in offsets:
                offsets[key] = size if (self._first and self.baseline) else 0
                changed = True
            off = offsets[key]
            if size < off:
                off = 0  # rewritten
            if size == off:
                continue
            try:
                with p.open("rb") as f:
                    f.seek(off)
                    chunk = f.read(size - off)
            except OSError:
                continue
            cut = chunk.rfind(b"\n") + 1  # keep a line still being written for next time
            offsets[key] = off + cut
            changed = True
            exe = key.rsplit("_", 1)[0]
            for raw in chunk[:cut].decode("utf-8", "replace").splitlines():
                m = self._LOG_LINE.match(raw)
                if not m or m.group(2) in self._LOG_SKIP:
                    continue
                try:
                    ms = datetime.fromisoformat(m.group(1)).timestamp() * 1000
                except ValueError:
                    continue
                hit = self._definition_for(exe, exe + ".exe")
                game = hit[1]["profile"] if hit and hit[1]["profile"] else exe
                pos = position_at(ctx.data, game, ms, within_ms=self.within_ms, skip_adapter=self.id)
                s = self._session(ctx, game, ms, pos)
                extra = {}
                if pos:
                    extra = dict(x=pos["x"], y=pos["y"], z=pos["z"], session=pos.get("session") or pos["folder"],
                                 position_from=pos["folder"])
                s.event("highlight_request", t=max(0.0, ms / 1000 - s.started), ms=ms, player="me",
                        game=game, request=m.group(2), **extra)
        return changed


# ---------------------------------------------------------------------------- adapters

ADAPTERS = [
    SteamStats(
        id="steam-stats",
        game="Steam statistics",
        kind="platform",
        multiplayer=False,
        interval=30.0,
        source=r"Steam's local stats cache %STEAM%\appcache\stats\UserGameStats_<account>_<appid>.bin with "
        r"UserGameStatsSchema_<appid>.bin; playtime from %STEAM%\userdata\<account>\config\localconfig.vdf "
        r"(%STEAM% from HKCU\Software\Valve\Steam SteamPath)",
        records=(
            "every stat a game stores in Steam (its own names and values), per game, when it changes",
            "achievement unlocks with display name, description and unlock time",
            "stat changes with old value, new value and difference",
            "playtime (minutes, last two weeks) and last played time",
        ),
        enable="",
        verified="decoded all real stats files and schemas on this machine (local test); "
        "change detection tested on files built in the binary KeyValues format",
    ),
    SteamTimeline(
        id="steam-timeline",
        game="Steam Timeline",
        kind="platform",
        multiplayer=False,
        interval=10.0,
        source=r"%STEAM%\userdata\<account>\gamerecordings\timelines\timeline_<appid><date>_<time>.json",
        records=(
            "one session per timeline file: its start and end (session_start, session_end)",
            "achievements unlocked during play, with time",
            "game mode changes (playing, staging, menus, loading screen)",
            "events games add through ISteamTimeline, with title and description, under their icon's name",
            "game phases (a match, a mission) with their tags; events inside one carry its id as session",
            "state descriptions the game shows on the timeline (\"Round 3\")",
            "Steam's own recording errors",
        ),
        enable="Steam > Settings > Game Recording: record in background or on demand (Steam writes timelines then)",
        verified="tested on the real timeline files on this machine (local test): achievement, gamemode, error, "
        "and one game's own event, state_description and phase entries; the file format itself is undocumented",
    ),
    NvidiaHighlights(
        id="nvidia-highlights",
        game="NVIDIA highlights",
        kind="platform",
        multiplayer=False,
        interval=10.0,
        source=r"%LOCALAPPDATA%\NVIDIA Corporation\NVIDIA Overlay\UGCTracker.json, GallerySettings.json, "
        r"Highlights\<game>\config.json; clips in the capture folder (%USERPROFILE%\Videos\NVIDIA\<Game>\*.mp4) "
        r"and <temp>\Highlights\<game>; %LOCALAPPDATA%\NVIDIA Corporation\GfeSDK\<exe>_<pid>.log",
        records=(
            "each new clip, highlight or screenshot: game, file, kind, length and the moment (wall-clock ms)",
            "the highlight's id and name when the game registered it with NVIDIA",
            "where the player was at that moment, when another adapter recorded the same game then",
            "highlight requests GfeSDK games log, even when no clip is kept",
        ),
        enable="NVIDIA App: Instant Replay or Highlights switched on, clips saved by hotkey or by the game",
        verified="tested on the real gallery index, settings, highlight definitions and clips on this machine "
        "(local test); GfeSDK highlight request lines and highlight ids in clips untested - none occur here",
    ),
]
