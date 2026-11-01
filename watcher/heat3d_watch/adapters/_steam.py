"""
What the Steam client keeps on disk, read without the client's help.

Three formats, all written by Steam itself for its own use and readable by
anyone who can read the files:

    binary KeyValues  `appcache\\stats\\UserGameStats*_<appid>.bin` and the stats
                      schemas next to them. A node is a type byte, a
                      NUL-terminated name, and a value; type 0 opens a child
                      node, 8 (or 11) closes it. Value types: 1 string, 2 int32,
                      3 float32, 4 pointer (int32), 5 wide string, 6 colour,
                      7 uint64, 10 int64 - all little-endian.
                      https://developer.valvesoftware.com/wiki/Binary_VDF
    text KeyValues    `localconfig.vdf`, `libraryfolders.vdf`, `appmanifest_*.acf`:
                      quoted keys and values, braces for children, // comments.
                      https://developer.valvesoftware.com/wiki/KeyValues
    the registry      HKCU\\Software\\Valve\\Steam SteamPath: where Steam is.

Nothing here talks to Steam or the network. The module starts with an
underscore so the adapter loader passes it by: it has no adapters of its own.
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(r"C:\Program Files (x86)\Steam")


def steam_root() -> Path | None:
    """Where Steam is installed: the registry first, then the usual folder."""
    candidates: list[Path] = []
    try:
        import winreg

        for hive, key, value in (
            (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam", "InstallPath"),
        ):
            try:
                with winreg.OpenKey(hive, key) as k:
                    candidates.append(Path(str(winreg.QueryValueEx(k, value)[0])))
            except OSError:
                continue
    except ImportError:
        pass
    candidates.append(DEFAULT_ROOT)
    home = Path.home()
    candidates += [home / ".steam" / "steam", home / ".local" / "share" / "Steam"]
    for c in candidates:
        if (c / "appcache").is_dir() or (c / "userdata").is_dir():
            return c
    return None


# --------------------------------------------------------------------- binary KeyValues


def parse_binary_kv(data: bytes) -> dict[str, Any]:
    """A binary KeyValues document as nested dicts. Raises ValueError on anything malformed."""
    out, _ = _bkv_node(data, 0, 0)
    return out


def _bkv_cstr(data: bytes, i: int) -> tuple[str, int]:
    j = data.find(b"\x00", i)
    if j < 0:
        raise ValueError(f"unterminated string at {i}")
    raw = data[i:j]
    try:
        return raw.decode("utf-8"), j + 1
    except UnicodeDecodeError:
        # Older schemas were written in the Windows code page: a bare 0x99 is the
        # trademark sign in "Battlefield(TM) 6", not a broken UTF-8 sequence.
        return raw.decode("cp1252", "replace"), j + 1


def _bkv_node(data: bytes, i: int, depth: int) -> tuple[dict[str, Any], int]:
    if depth > 64:
        raise ValueError("nested too deep")
    out: dict[str, Any] = {}
    n = len(data)
    while True:
        if i >= n:
            # The root may simply end; a child node may not.
            if depth == 0:
                return out, i
            raise ValueError("truncated")
        t = data[i]
        i += 1
        if t in (0x08, 0x0B):
            return out, i
        key, i = _bkv_cstr(data, i)
        if t == 0x00:
            value, i = _bkv_node(data, i, depth + 1)
        elif t == 0x01:
            value, i = _bkv_cstr(data, i)
        elif t in (0x02, 0x04, 0x06):
            if i + 4 > n:
                raise ValueError("truncated int32")
            value = struct.unpack_from("<i" if t != 0x06 else "<I", data, i)[0]
            i += 4
        elif t == 0x03:
            if i + 4 > n:
                raise ValueError("truncated float")
            value = struct.unpack_from("<f", data, i)[0]
            i += 4
        elif t == 0x05:
            j = i
            while j + 1 < n and data[j : j + 2] != b"\x00\x00":
                j += 2
            if j + 1 >= n:
                raise ValueError("unterminated wide string")
            value = data[i:j].decode("utf-16-le", "replace")
            i = j + 2
        elif t in (0x07, 0x0A):
            if i + 8 > n:
                raise ValueError("truncated int64")
            value = struct.unpack_from("<Q" if t == 0x07 else "<q", data, i)[0]
            i += 8
        else:
            raise ValueError(f"unknown type {t:#x} at {i - 1}")
        out[key] = value


def bkv_encode(doc: dict[str, Any]) -> bytes:
    """The other way round, for tests: str, int (int32), float (float32) and dict values."""
    out = bytearray()

    def node(d: dict[str, Any]) -> None:
        for k, v in d.items():
            name = k.encode("utf-8") + b"\x00"
            if isinstance(v, dict):
                out.extend(b"\x00" + name)
                node(v)
                out.append(0x08)
            elif isinstance(v, str):
                out.extend(b"\x01" + name + v.encode("utf-8") + b"\x00")
            elif isinstance(v, float):
                out.extend(b"\x03" + name + struct.pack("<f", v))
            elif isinstance(v, int):
                out.extend(b"\x02" + name + struct.pack("<i", v))
            else:
                raise TypeError(type(v))

    node(doc)
    out.append(0x08)
    return bytes(out)


# ----------------------------------------------------------------------- text KeyValues

_TOKEN = re.compile(r'\s*(?://[^\n]*\n\s*)*(?:"((?:[^"\\]|\\.)*)"|([{}])|([^\s"{}]+))', re.S)
_ESCAPES = {"n": "\n", "t": "\t", "\\": "\\", '"': '"'}


def parse_text_kv(text: str) -> dict[str, Any]:
    """Text KeyValues as nested dicts. Lenient: a broken tail ends the document, it does not raise."""
    root: dict[str, Any] = {}
    stack = [root]
    key: str | None = None
    pos = 0
    n = len(text)
    while pos < n:
        m = _TOKEN.match(text, pos)
        if not m:
            break
        pos = m.end()
        quoted, brace, bare = m.groups()
        if brace == "{":
            child: dict[str, Any] = {}
            stack[-1][key if key is not None else ""] = child
            stack.append(child)
            key = None
        elif brace == "}":
            if len(stack) > 1:
                stack.pop()
            key = None
        else:
            tok = quoted if quoted is not None else bare
            if quoted is not None and "\\" in tok:
                tok = re.sub(r"\\(.)", lambda e: _ESCAPES.get(e.group(1), e.group(1)), tok)
            if key is None:
                key = tok
            else:
                stack[-1][key] = tok
                key = None
    return root


def kv_get(d: Any, *path: str) -> Any:
    """Follow keys without minding their case, as Steam does ("apps" and "Apps" both occur)."""
    for p in path:
        if not isinstance(d, dict):
            return None
        if p in d:
            d = d[p]
            continue
        low = p.lower()
        d = next((v for k, v in d.items() if k.lower() == low), None)
    return d


def read_text_kv(path: Path) -> dict[str, Any]:
    try:
        return parse_text_kv(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return {}


# ------------------------------------------------------------------------- what is there


def accounts(root: Path) -> list[str]:
    """The Steam account ids (32-bit) that have a folder under userdata."""
    try:
        return sorted(p.name for p in (root / "userdata").iterdir() if p.is_dir() and p.name.isdigit() and p.name != "0")
    except OSError:
        return []


def playtimes(root: Path, account: str) -> dict[str, dict[str, int]]:
    """appid -> {Playtime, Playtime2wks (minutes), LastPlayed (Unix seconds)} from localconfig.vdf."""
    doc = read_text_kv(root / "userdata" / account / "config" / "localconfig.vdf")
    apps = kv_get(doc, "UserLocalConfigStore", "Software", "Valve", "Steam", "apps")
    out: dict[str, dict[str, int]] = {}
    if not isinstance(apps, dict):
        return out
    for appid, a in apps.items():
        if not isinstance(a, dict) or not appid.isdigit():
            continue
        row = {}
        for k in ("Playtime", "Playtime2wks", "LastPlayed"):
            v = kv_get(a, k)
            if isinstance(v, str) and v.lstrip("-").isdigit():
                row[k] = int(v)
        if row:
            out[appid] = row
    return out


def library_folders(root: Path) -> list[Path]:
    doc = read_text_kv(root / "steamapps" / "libraryfolders.vdf")
    out = [root]
    folders = kv_get(doc, "libraryfolders")
    if isinstance(folders, dict):
        for v in folders.values():
            p = kv_get(v, "path") if isinstance(v, dict) else None
            if isinstance(p, str) and Path(p) not in out:
                out.append(Path(p))
    return out


@dataclass
class Achievement:
    api: str
    stat: str
    bit: int
    name: str = ""
    desc: str = ""


@dataclass
class StatDef:
    id: str
    api: str
    kind: str  # int, float, avgrate, achievements
    name: str = ""


@dataclass
class Schema:
    appid: str
    gamename: str = ""
    stats: dict[str, StatDef] = field(default_factory=dict)
    #: (stat id, bit) -> achievement
    achievements: dict[tuple[str, int], Achievement] = field(default_factory=dict)

    def by_api(self, api: str) -> Achievement | None:
        return next((a for a in self.achievements.values() if a.api == api), None)


_STAT_KINDS = {
    "1": "int", "int": "int",
    "2": "float", "float": "float",
    "3": "avgrate", "avgrate": "avgrate",
    "4": "achievements", "achievements": "achievements",
    "5": "achievements", "groupachievements": "achievements",
}


def _english(v: Any) -> str:
    """A display string: Steam stores one per language, or a plain string in old schemas."""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        low = {k.lower(): x for k, x in v.items()}
        for k in ("english", "en", "en-us", "en-gb"):
            if isinstance(low.get(k), str):
                return low[k]
        return next((x for k, x in v.items() if isinstance(x, str) and k != "token"), "")
    return ""


def parse_schema(appid: str, doc: dict[str, Any]) -> Schema:
    root = doc.get(appid)
    if not isinstance(root, dict):
        root = next((v for v in doc.values() if isinstance(v, dict)), {})
    s = Schema(appid, gamename=str(root.get("gamename") or ""))
    stats = root.get("stats")
    if not isinstance(stats, dict):
        return s
    for sid, st in stats.items():
        if not isinstance(st, dict):
            continue
        kind = _STAT_KINDS.get(str(st.get("type", "")).lower()) or _STAT_KINDS.get(str(st.get("type_int", "")), "int")
        api = str(st.get("name") or sid)
        s.stats[sid] = StatDef(sid, api, kind, _english(kv_get(st, "display", "name")))
        bits = st.get("bits")
        if kind == "achievements" and isinstance(bits, dict):
            for key, b in bits.items():
                if not isinstance(b, dict):
                    continue
                bit = b.get("bit")
                if not isinstance(bit, int):
                    # Newer schemas leave "bit" out: the key is the bit.
                    if not key.isdigit():
                        continue
                    bit = int(key)
                disp = b.get("display") if isinstance(b.get("display"), dict) else {}
                s.achievements[(sid, bit)] = Achievement(
                    str(b.get("name") or f"{sid}_{bit}"), sid, bit, _english(disp.get("name")), _english(disp.get("desc"))
                )
    return s


@dataclass
class Steam:
    """One Steam installation, with what is looked up more than once remembered."""

    root: Path
    _schemas: dict[str, tuple[float, Schema | None]] = field(default_factory=dict)
    _manifest_names: dict[str, str] | None = None

    @property
    def stats_dir(self) -> Path:
        return self.root / "appcache" / "stats"

    def schema(self, appid: str) -> Schema | None:
        path = self.stats_dir / f"UserGameStatsSchema_{appid}.bin"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        hit = self._schemas.get(appid)
        if hit and hit[0] == mtime:
            return hit[1]
        try:
            schema: Schema | None = parse_schema(appid, parse_binary_kv(path.read_bytes()))
        except (OSError, ValueError):
            schema = None
        self._schemas[appid] = (mtime, schema)
        return schema

    def app_name(self, appid: str) -> str:
        """The game's name: its install manifest, then its stats schema, then the app id.

        The manifest comes first because a schema's "gamename" can be a placeholder
        ("ValveTestApp260") where a game took over an older one's app id.
        """
        if self._manifest_names is None:
            names: dict[str, str] = {}
            for lib in library_folders(self.root):
                try:
                    manifests = list((lib / "steamapps").glob("appmanifest_*.acf"))
                except OSError:
                    continue
                for m in manifests:
                    doc = read_text_kv(m)
                    a = kv_get(doc, "AppState")
                    if isinstance(a, dict) and isinstance(kv_get(a, "name"), str):
                        names[str(kv_get(a, "appid") or m.stem.split("_")[-1])] = kv_get(a, "name")
            self._manifest_names = names
        if appid in self._manifest_names:
            return self._manifest_names[appid]
        s = self.schema(appid)
        if s and s.gamename and not re.fullmatch(r"ValveTestApp\d+", s.gamename):
            return s.gamename
        return f"Steam app {appid}"


def user_stats(doc: dict[str, Any], schema: Schema | None) -> tuple[dict[str, float], dict[str, int]]:
    """A UserGameStats file: (stat API name -> value, achievement API name -> unlock time, Unix s).

    Each stat is stored under its schema id as {"data": int32}. Float stats keep
    their IEEE bits in that int32; an achievement stat's int32 is a bit field,
    one bit per achievement, with the unlock times in "AchievementTimes" keyed by
    bit. A stat the player never changed is not stored and keeps its default.
    """
    cache = doc.get("cache") if isinstance(doc.get("cache"), dict) else doc
    stats: dict[str, float] = {}
    unlocked: dict[str, int] = {}
    for sid, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        data = entry.get("data")
        if not isinstance(data, int):
            continue
        sd = schema.stats.get(sid) if schema else None
        kind = sd.kind if sd else ("achievements" if "AchievementTimes" in entry else "int")
        if kind == "achievements":
            times = entry.get("AchievementTimes") if isinstance(entry.get("AchievementTimes"), dict) else {}
            bits = data & 0xFFFFFFFF
            for bit in range(32):
                if not bits >> bit & 1:
                    continue
                ach = schema.achievements.get((sid, bit)) if schema else None
                if schema and ach is None:
                    # Bits the schema no longer defines: achievements the game retired
                    # (a sequel took over the app id). Nameless, and not the player's now.
                    continue
                api = ach.api if ach else f"{sid}_{bit}"
                t = times.get(str(bit))
                unlocked[api] = int(t) if isinstance(t, int) else 0
            continue
        api = sd.api if sd else f"stat_{sid}"
        if kind in ("float", "avgrate"):
            v = struct.unpack("<f", struct.pack("<I", data & 0xFFFFFFFF))[0]
            if v == v and abs(v) != float("inf"):
                stats[api] = round(v, 6)
        else:
            stats[api] = data
    return stats, unlocked


def snake(name: str) -> str:
    """A stat's API name as a statistics key: "TotalKills" -> "total_kills", "SCORE_2" -> "score_2"."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s or "stat"


def environ_path(var: str, *parts: str) -> Path | None:
    base = os.environ.get(var)
    return Path(base, *parts) if base else None
