"""Steam stats, Steam Timeline and NVIDIA highlights: decoded from files built per their formats, then the real ones."""

from __future__ import annotations

import json
import os
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from heat3d_watch import db
from heat3d_watch.adapters import _steam
from heat3d_watch.adapters.base import Context
from heat3d_watch.adapters.platform import (
    NvidiaHighlights,
    SteamStats,
    SteamTimeline,
    load_timeline,
    mp4_duration,
    parse_clip_name,
    position_at,
    timeline_event,
)
from heat3d_watch.config import Settings
from heat3d_watch.session import Session, read_rows

ACCOUNT = "111"
APPID = "480"


def _ctx(tmp_path: Path) -> Context:
    s = Settings(data=str(tmp_path / "data"))
    return Context(s, s.data_dir)


def _sessions(ctx: Context, adapter: str) -> list[Path]:
    out = []
    for meta in sorted((ctx.data / "sessions").glob("*/*/meta.json")):
        if json.loads(meta.read_text(encoding="utf-8"))["adapter"] == adapter:
            out.append(meta.parent)
    return out


def _f32_bits(v: float) -> int:
    return struct.unpack("<i", struct.pack("<f", v))[0]


# ----------------------------------------------------------------------------- KeyValues


def test_binary_keyvalues_all_value_types():
    raw = (
        b"\x00root\x00"
        b"\x01s\x00text\x00"
        b"\x02i\x00" + struct.pack("<i", -5)
        + b"\x03f\x00" + struct.pack("<f", 1.5)
        + b"\x05w\x00" + "wide".encode("utf-16-le") + b"\x00\x00"
        + b"\x07u\x00" + struct.pack("<Q", 2**40)
        + b"\x0aq\x00" + struct.pack("<q", -(2**40))
        + b"\x00child\x00\x02n\x00" + struct.pack("<i", 1) + b"\x08"
        + b"\x08\x08"
    )
    doc = _steam.parse_binary_kv(raw)
    assert doc == {"root": {"s": "text", "i": -5, "f": 1.5, "w": "wide", "u": 2**40, "q": -(2**40), "child": {"n": 1}}}
    assert _steam.parse_binary_kv(_steam.bkv_encode({"a": {"b": "c", "n": 3}})) == {"a": {"b": "c", "n": 3}}
    with pytest.raises(ValueError):
        _steam.parse_binary_kv(raw[:20])
    with pytest.raises(ValueError):
        _steam.parse_binary_kv(b"\x09bad\x00")


def test_text_keyvalues_with_comments_escapes_and_case():
    doc = _steam.parse_text_kv(
        '// a comment\n"UserLocalConfigStore"\n{\n\t"Software" { "Valve" { "Steam" { "Apps" {\n'
        '\t\t"480" { "LastPlayed" "1700000000" "Playtime" "125" "Playtime2wks" "30" }\n'
        '\t\t"10" { "cloud" { "last_sync_state" "synchronized" } }\n'
        '\t} } } }\n\t"path" "C:\\\\Games\\\\a \\"b\\""\n}\n'
    )
    assert _steam.kv_get(doc, "userlocalconfigstore", "software", "valve", "steam", "apps", "480", "Playtime") == "125"
    assert _steam.kv_get(doc, "UserLocalConfigStore", "path") == 'C:\\Games\\a "b"'
    assert _steam.parse_text_kv('"a" { "b" "c"') == {"a": {"b": "c"}}  # a cut-off file still reads


def test_snake_case_stat_keys():
    assert [_steam.snake(n) for n in ("TotalKills", "SCORE_2", "total_distance", "HTTPServer", "ACH_WinMatch")] == [
        "total_kills", "score_2", "total_distance", "http_server", "ach_win_match"]


# ----------------------------------------------------------------------------- steam-stats


def _schema(gamename: str = "Spacewar Test") -> dict:
    return {APPID: {
        "gamename": gamename,
        "version": "3",
        "stats": {
            "1": {"type": "1", "id": "1", "name": "TotalKills", "display": {"name": "Kills"}},
            "2": {"type": "FLOAT", "name": "max_speed", "display": {"name": "Top speed"}},
            "3": {"type": "4", "id": "3", "bits": {
                "0": {"name": "ACH_FIRST", "bit": 0, "display": {"name": {"english": "First!", "german": "Erster!"},
                                                                 "desc": {"english": "Do it once"}}},
                # Newer schemas leave "bit" out; the key is the bit.
                "1": {"name": "ACH_SECOND", "display": {"name": {"token": "#x", "english": "Second"}}},
                "2": {"name": "ACH_THIRD", "display": {"name": {"english": "Third"}}},
            }},
        },
    }}


def _user(kills: int, speed: float, bits: int, times: dict[str, int]) -> dict:
    return {"cache": {
        "crc": 123,
        "PendingChanges": 0,
        "1": {"data": kills},
        "2": {"data": _f32_bits(speed)},
        "3": {"data": bits, "AchievementTimes": times},
    }}


def _steam_root(tmp_path: Path) -> Path:
    root = tmp_path / "Steam"
    (root / "appcache" / "stats").mkdir(parents=True)
    (root / "userdata" / ACCOUNT / "config").mkdir(parents=True)
    (root / "appcache" / "stats" / f"UserGameStatsSchema_{APPID}.bin").write_bytes(_steam.bkv_encode(_schema()))
    return root


def _write_user(root: Path, doc: dict, mtime: float, appid: str = APPID) -> None:
    p = root / "appcache" / "stats" / f"UserGameStats_{ACCOUNT}_{appid}.bin"
    p.write_bytes(_steam.bkv_encode(doc))
    os.utime(p, (mtime, mtime))


def _write_localconfig(root: Path, apps: dict[str, dict[str, int]], mtime: float) -> None:
    body = "".join(
        f'\t\t\t\t\t"{a}"\n\t\t\t\t\t{{\n' + "".join(f'\t\t\t\t\t\t"{k}"\t\t"{v}"\n' for k, v in row.items()) + "\t\t\t\t\t}\n"
        for a, row in apps.items()
    )
    text = ('"UserLocalConfigStore"\n{\n\t"Software"\n\t{\n\t\t"Valve"\n\t\t{\n\t\t\t"Steam"\n\t\t\t{\n'
            f'\t\t\t\t"apps"\n\t\t\t\t{{\n{body}\t\t\t\t}}\n\t\t\t}}\n\t\t}}\n\t}}\n}}\n')
    p = root / "userdata" / ACCOUNT / "config" / "localconfig.vdf"
    p.write_text(text, encoding="utf-8")
    os.utime(p, (mtime, mtime))


def test_user_stats_decoding():
    schema = _steam.parse_schema(APPID, _schema())
    assert schema.gamename == "Spacewar Test"
    assert schema.achievements[("3", 1)].name == "Second"
    assert schema.achievements[("3", 0)].desc == "Do it once"
    # Bit 5 is set but the schema no longer has it: a retired achievement, left out.
    stats, unlocked = _steam.user_stats(_user(42, 205.5, 0b100011, {"0": 1700000000, "1": 1700000100}), schema)
    assert stats == {"TotalKills": 42, "max_speed": 205.5}
    assert unlocked == {"ACH_FIRST": 1700000000, "ACH_SECOND": 1700000100}
    # Without a schema everything is kept under its ids.
    stats, unlocked = _steam.user_stats(_user(1, 0.5, 0b1, {"0": 5}), None)
    assert stats == {"stat_1": 1, "stat_2": _f32_bits(0.5)} and unlocked == {"3_0": 5}


def test_steam_stats_records_a_snapshot_then_only_changes(tmp_path):
    root = _steam_root(tmp_path)
    t0 = time.time() - 3600
    _write_user(root, _user(10, 100.25, 0b1, {"0": 1700000000}), t0)
    _write_localconfig(root, {APPID: {"LastPlayed": 1700000500, "Playtime": 125, "Playtime2wks": 30},
                              "999": {"Playtime": 7}}, t0)
    ctx = _ctx(tmp_path)
    a = SteamStats(id="steam-stats", game="Steam statistics", kind="platform", root=str(root))
    assert a.missing() is None
    a.start(ctx)
    [first] = _sessions(ctx, "steam-stats")
    assert first.parent.name == "Spacewar-Test"
    meta = json.loads((first / "meta.json").read_text(encoding="utf-8"))
    assert meta["stats"]["total_kills"] == 10 and meta["stats"]["max_speed"] == 100.25
    assert meta["stats"]["playtime_minutes"] == 125 and meta["stats"]["playtime_2wks_minutes"] == 30
    assert meta["stats"]["last_played"] == 1700000500
    assert meta["stats"]["achievements_unlocked"] == 1 and meta["stats"]["achievements_total"] == 3
    assert meta["seconds"] == 0  # a snapshot, not an hour of play
    rows = read_rows(first)
    assert [r["event"] for r in rows] == ["stats_snapshot", "achievement"]
    assert rows[1]["name"] == "First!" and rows[1]["ms"] == 1700000000 * 1000 and rows[1]["description"] == "Do it once"
    assert json.loads((ctx.data / "state" / "steam-stats.json").read_text(encoding="utf-8"))["games"]

    a.poll(ctx)  # nothing changed
    assert len(_sessions(ctx, "steam-stats")) == 1

    _write_user(root, _user(13, 100.25, 0b101, {"0": 1700000000, "2": 1700009000}), t0 + 60)
    a.poll(ctx)
    second = [s for s in _sessions(ctx, "steam-stats") if s != first]
    assert len(second) == 1
    rows = read_rows(second[0])
    assert [r["event"] for r in rows] == ["stats_update", "stat_change", "achievement"]
    assert rows[1] == {**rows[1], "stat": "TotalKills", "old": 10, "new": 13, "delta": 3}
    assert rows[2]["achievement"] == "ACH_THIRD" and rows[2]["name"] == "Third"

    # Played without stats changing: the playtime alone makes a session.
    _write_localconfig(root, {APPID: {"LastPlayed": 1700100000, "Playtime": 185, "Playtime2wks": 90},
                              "999": {"Playtime": 7}}, t0 + 120)
    a.poll(ctx)
    third = [s for s in _sessions(ctx, "steam-stats") if s not in (first, second[0])]
    assert len(third) == 1
    rows = read_rows(third[0])
    assert rows[0]["event"] == "playtime_update" and rows[0]["minutes"] == 185 and rows[0]["before"] == 125
    a.stop(ctx)

    # A restart remembers what it saw.
    b = SteamStats(id="steam-stats", game="Steam statistics", kind="platform", root=str(root))
    b.start(ctx)
    b.poll(ctx)
    assert len(_sessions(ctx, "steam-stats")) == 3
    con = db.connect(ctx.data)
    assert con.execute("SELECT COUNT(*) FROM sessions WHERE adapter = 'steam-stats'").fetchone()[0] == 3


def test_steam_stats_written_in_the_same_second_keep_their_own_sessions(tmp_path):
    root = _steam_root(tmp_path)
    (root / "appcache" / "stats" / "UserGameStatsSchema_481.bin").write_bytes(
        _steam.bkv_encode({"481": {**_schema("Other Test")[APPID]}}))
    t0 = float(int(time.time()) - 600)
    _write_user(root, _user(1, 1.0, 0, {}), t0)
    _write_user(root, _user(2, 2.0, 0, {}), t0, appid="481")
    ctx = _ctx(tmp_path)
    a = SteamStats(id="steam-stats", game="Steam statistics", kind="platform", root=str(root))
    a.start(ctx)
    a.stop(ctx)
    con = db.connect(ctx.data)
    assert sorted(r[0] for r in con.execute("SELECT game FROM sessions")) == ["Other Test", "Spacewar Test"]


def test_a_damaged_stats_file_is_skipped_until_it_changes(tmp_path):
    root = _steam_root(tmp_path)
    p = root / "appcache" / "stats" / f"UserGameStats_{ACCOUNT}_{APPID}.bin"
    p.write_bytes(b"\x00cache\x00\x02crc")
    ctx = _ctx(tmp_path)
    a = SteamStats(id="steam-stats", game="Steam statistics", kind="platform", root=str(root))
    a.start(ctx)
    assert _sessions(ctx, "steam-stats") == []
    _write_user(root, _user(5, 1.0, 0, {}), time.time() - 30)
    a.poll(ctx)
    [s] = _sessions(ctx, "steam-stats")
    assert read_rows(s)[0]["event"] == "stats_snapshot"


# -------------------------------------------------------------------------- steam-timeline

# Built in the layout Steam writes (observed in real timeline files): string numbers,
# times in ms after "daterecorded"; the fields of "event", "state_description" and "phase"
# entries are the ISteamTimeline parameters
# (https://partner.steamgames.com/doc/api/ISteamTimeline).


def _timeline_entries() -> list[dict]:
    return [
        {"id": "1", "time": "14698", "type": "gamemode", "mode": 3},
        {"id": "2", "time": "87620", "type": "gamemode", "mode": 1},
        {"id": "3", "time": "100000", "type": "phase", "duration": "60000",
         "tags": [{"name": "Ranked", "icon": "x", "group": "Mode", "priority": 500},
                  {"name": "Test Arena", "icon": "y", "group": "Map", "priority": 500}]},
        {"id": "4", "time": "100000", "type": "state_description", "title": "Round 1"},
        {"id": "5", "time": "105000", "type": "event", "title": "Start of round 1", "description": "",
         "icon": "game_7", "priority": 1000, "duration": "0", "possible_clip": 1},
        {"id": "6", "time": "120000", "type": "event", "title": "You killed Someone", "description": "with a pistol",
         "icon": "game_gun_kill", "priority": 1000, "duration": "0", "possible_clip": 2},
        {"id": "7", "time": "150000", "type": "event", "title": "Died", "icon": "steam_death", "priority": 100},
        {"id": "8", "time": "170000", "type": "achievement", "achievement_name": "ACH_FIRST"},
    ]


def _timeline_text(recorded: int, entries: list[dict], end: int | None) -> str:
    doc = {"daterecorded": str(recorded), "starttime": "0", "entries": entries}
    if end is not None:
        doc["endtime"] = str(end)
    return json.dumps(doc, indent="\t")


def test_timeline_entries_become_events():
    e = _timeline_entries()
    assert timeline_event(e[0]) == ("game_mode", {"mode": 3, "mode_name": "menus"})
    name, attrs = timeline_event(e[2])
    assert name == "phase" and attrs == {"duration": 60000, "tag_mode": "Ranked", "tag_map": "Test Arena"}
    assert timeline_event(e[3]) == ("state_description", {"title": "Round 1"})
    assert timeline_event(e[4])[0] == "timeline_event"  # a counter icon says nothing
    name, attrs = timeline_event(e[5])
    assert name == "game_gun_kill" and attrs["title"] == "You killed Someone" and attrs["possible_clip"] == 2
    assert timeline_event(e[6])[0] == "death"
    assert timeline_event({"id": "9", "time": "1", "type": "error", "description": "#X"}) == ("error", {"description": "#X"})
    assert timeline_event({"type": "other", "session": "s", "player": "p"})[1] == {"tl_session": "s", "tl_player": "p"}


def test_a_timeline_being_written_still_reads():
    text = _timeline_text(1700000000, _timeline_entries(), None)
    assert len(load_timeline(text[:-2])["entries"]) == 8  # closing brackets missing
    cut = text[: text.index('"Died"')]  # cut inside an entry
    assert [e["id"] for e in load_timeline(cut)["entries"]] == ["1", "2", "3", "4", "5", "6"]
    assert load_timeline("not json") is None


def test_steam_timeline_follows_a_file_until_it_ends(tmp_path):
    root = _steam_root(tmp_path)
    folder = root / "userdata" / ACCOUNT / "gamerecordings" / "timelines"
    folder.mkdir(parents=True)
    recorded = int(time.time()) - 600
    name = f"timeline_{APPID}{datetime.fromtimestamp(recorded, timezone.utc):%Y%m%d_%H%M%S}.json"
    path = folder / name
    entries = _timeline_entries()
    path.write_text(_timeline_text(recorded, entries[:4], None), encoding="utf-8")
    # A copy inside a saved clip is the same timeline: not read twice.
    clip = root / "userdata" / ACCOUNT / "gamerecordings" / "clips" / "clip_x" / "timelines"
    clip.mkdir(parents=True)
    (clip / name).write_text(_timeline_text(recorded, entries, 200000), encoding="utf-8")

    ctx = _ctx(tmp_path)
    a = SteamTimeline(id="steam-timeline", game="Steam Timeline", kind="platform", root=str(root))
    a.start(ctx)
    assert len(a._open) == 1  # still being played
    path.write_text(_timeline_text(recorded, entries, 200000), encoding="utf-8")
    os.utime(path, (time.time() + 1, time.time() + 1))
    a.poll(ctx)
    assert a._open == {}
    [s] = _sessions(ctx, "steam-timeline")
    assert s.parent.name == "Spacewar-Test"
    rows = read_rows(s)
    events = [r["event"] for r in rows]
    assert events == ["session_start", "game_mode", "game_mode", "phase", "state_description",
                      "timeline_event", "game_gun_kill", "death", "achievement", "session_end"]
    assert rows[0]["ms"] == recorded * 1000
    kill = rows[6]
    assert kill["ms"] == recorded * 1000 + 120000 and kill["time"] == 120.0 and kill["session"] == "phase-3"
    assert kill["description"] == "with a pistol"
    assert "session" not in rows[8]  # after the phase ended
    assert rows[8]["name"] == "First!"  # the achievement's display name, from the stats schema
    meta = json.loads((s / "meta.json").read_text(encoding="utf-8"))
    assert meta["seconds"] == 200.0 and meta["stats"]["recorded_seconds"] == 200.0
    assert meta["stats"]["achievements"] == 1 and meta["stats"]["phases"] == 1 and meta["stats"]["timeline_events"] == 3
    a.stop(ctx)

    b = SteamTimeline(id="steam-timeline", game="Steam Timeline", kind="platform", root=str(root))
    b.start(ctx)
    os.utime(path, (time.time() + 5, time.time() + 5))
    b.poll(ctx)
    b.stop(ctx)
    assert len(_sessions(ctx, "steam-timeline")) == 1


def test_an_abandoned_timeline_is_closed_at_its_last_write(tmp_path):
    root = _steam_root(tmp_path)
    folder = root / "userdata" / ACCOUNT / "gamerecordings" / "timelines"
    folder.mkdir(parents=True)
    recorded = int(time.time()) - 86400
    path = folder / f"timeline_{APPID}20240101_120000.json"
    path.write_text(_timeline_text(recorded, _timeline_entries()[:2], None), encoding="utf-8")
    os.utime(path, (recorded + 300, recorded + 300))
    ctx = _ctx(tmp_path)
    a = SteamTimeline(id="steam-timeline", game="Steam Timeline", kind="platform", root=str(root))
    a.start(ctx)
    assert a._open == {}
    [s] = _sessions(ctx, "steam-timeline")
    assert json.loads((s / "meta.json").read_text(encoding="utf-8"))["seconds"] == 300.0


# ----------------------------------------------------------------------- nvidia-highlights


def _mp4(seconds: float, scale: int = 1000) -> bytes:
    ftyp = struct.pack(">I4s4sI", 16, b"ftyp", b"isom", 0)
    mdat = struct.pack(">I4s", 16, b"mdat") + b"\x00" * 8
    mvhd_body = struct.pack(">B3xIIII", 0, 0, 0, scale, int(seconds * scale)) + b"\x00" * 80
    mvhd = struct.pack(">I4s", 8 + len(mvhd_body), b"mvhd") + mvhd_body
    moov = struct.pack(">I4s", 8 + len(mvhd), b"moov") + mvhd
    return ftyp + mdat + moov  # moov after the data, as a recorder writes it


def _clip_name(game: str, ms: float, suffix: str = ".DVR.mp4") -> str:
    dt = datetime.fromtimestamp(ms / 1000)
    return f"{game} {dt:%Y.%m.%d - %H.%M.%S}.{int(ms % 1000):03d}{suffix}"


def test_clip_names_and_durations(tmp_path):
    p = parse_clip_name("Some Game 2025.10.20 - 00.49.45.115.DVR.mp4")
    assert p["game"] == "Some Game" and p["kind"] == "InstantReplay" and p["media"] == "video"
    assert p["ms"] == datetime(2025, 10, 20, 0, 49, 45).timestamp() * 1000 + 115
    assert parse_clip_name("Some Game 2025.10.23 - 11.29.04.04.DVR.mp4")["ms"] % 1000 == 40
    s = parse_clip_name("A - B Game Screenshot 2025.05.25 - 15.43.20.94.png")
    assert s["game"] == "A - B Game" and s["kind"] == "Screenshot" and s["media"] == "image"
    assert parse_clip_name("hitreg.mkv") is None
    f = tmp_path / "c.mp4"
    f.write_bytes(_mp4(464.049))
    assert mp4_duration(f) == pytest.approx(464.049)
    (tmp_path / "bad.mp4").write_bytes(b"\x00\x00")
    assert mp4_duration(tmp_path / "bad.mp4") is None


def _game_session(ctx: Context, game: str, start_ms: float) -> Session:
    s = Session.open(ctx.data, game, "some-game", when=start_ms / 1000 - 10)
    for i in range(20):
        s.position(float(i), 2.0, float(-i), player="me", ms=start_ms + i * 1000, session="round-2")
        s.position(99.0, 99.0, 99.0, player="Enemy", ms=start_ms + i * 1000 + 500)
    return s


def test_position_at_finds_the_last_own_position_before_the_moment(tmp_path):
    ctx = _ctx(tmp_path)
    start = (time.time() - 120) * 1000
    s = _game_session(ctx, "Some Game", start)
    hit = position_at(ctx.data, "SOME GAME", start + 5600)
    assert hit and (hit["x"], hit["y"], hit["z"]) == (5.0, 2.0, -5.0) and hit["session"] == "round-2"
    assert hit["adapter"] == "some-game"
    s.close()  # a finished session is found the same way
    assert position_at(ctx.data, "Some Game", start + 5600)["x"] == 5.0
    assert position_at(ctx.data, "Some Game", start + 19000 + 5001) is None  # more than 5 s after the last
    assert position_at(ctx.data, "Some Game", start - 1) is None
    assert position_at(ctx.data, "Another Game", start + 5600) is None


def _nvidia(tmp_path: Path) -> tuple[Path, Path]:
    overlay = tmp_path / "NVIDIA Overlay"
    captures = tmp_path / "Captures"
    (overlay / "Highlights" / "some_game").mkdir(parents=True)
    (captures / "Some Game").mkdir(parents=True)
    (overlay / "GallerySettings.json").write_text(json.dumps({"settings": {
        "currentDirectoryV2": str(captures), "tempDirectory": str(tmp_path / "Temp")}}), encoding="utf-8")
    (overlay / "Highlights" / "some_game" / "config.json").write_text(json.dumps({
        "highlights": {"kill": {"significance": 16, "highlightTags": 4, "userInterest": True,
                                "name": {"en-us": "Enemy destroyed"}}},
        "defaultLocale": "en-US", "drsName": "somegame.exe", "profileName": "Some Game", "shortName": "some_game",
    }), encoding="utf-8")
    return overlay, captures


def _tracker(overlay: Path, entries: list[dict]) -> None:
    (overlay / "UGCTracker.json").write_text(json.dumps({"UGC": entries, "apps": [], "info": {"version": 7}}),
                                             encoding="utf-8")


def test_nvidia_records_new_clips_placed_on_the_map(tmp_path):
    overlay, captures = _nvidia(tmp_path)
    old = captures / "Some Game" / _clip_name("Some Game", (time.time() - 86400) * 1000)
    old.write_bytes(_mp4(30))
    _tracker(overlay, [{"id": 1, "path": str(old), "type": "video", "subType": "InstantReplay",
                        "profileName": "Some Game", "drsName": "somegame.exe", "shortName": "some_game",
                        "msDuration": 30000, "date": "2025-01-01T00:00:00Z"}])
    gfesdk = tmp_path / "GfeSDK"
    gfesdk.mkdir()
    ctx = _ctx(tmp_path)
    a = NvidiaHighlights(id="nvidia-highlights", game="NVIDIA highlights", kind="platform",
                         overlay=str(overlay), gfesdk=str(gfesdk))
    assert a.missing() is None
    assert a.capture_dir() == captures
    a.start(ctx)
    assert _sessions(ctx, "nvidia-highlights") == []  # what was there before is the baseline

    start = (time.time() - 60) * 1000
    game = _game_session(ctx, "Some Game", start)
    moment = start + 7300
    clip = captures / "Some Game" / _clip_name("Some Game", moment)
    clip.write_bytes(_mp4(20))
    a.poll(ctx)  # still being written, as far as anyone can tell
    os.utime(clip, (time.time() - 10, time.time() - 10))
    a.poll(ctx)
    a.poll(ctx)
    shot = captures / "Some Game" / _clip_name("Some Game Screenshot", moment + 2000, ".png")
    shot.write_bytes(b"\x89PNG")
    os.utime(shot, (time.time() - 10, time.time() - 10))
    a.poll(ctx)
    a.poll(ctx)
    # A GfeSDK game's highlight call, logged by the SDK.
    (gfesdk / "somegame_1234.log").write_text(
        f"{datetime.fromtimestamp((start + 9000) / 1000).isoformat(timespec='milliseconds')}[I] GAME"
        "        Controller.cpp:153  Posted Request HLSetVideo[613c0556]\n"
        f"{datetime.fromtimestamp((start + 9000) / 1000).isoformat(timespec='milliseconds')}[I] GAME"
        "        Controller.cpp:153  Posted Request HLConfigure[613c0555]\n", encoding="utf-8")
    a.poll(ctx)
    a.stop(ctx)
    game.close()

    [s] = _sessions(ctx, "nvidia-highlights")
    assert s.parent.name == "Some-Game"
    rows = read_rows(s)
    assert [r["event"] for r in rows] == ["highlight", "screenshot", "highlight_request"]
    h = rows[0]
    assert h["ms"] == int(moment) and h["clip"] == str(clip) and h["kind"] == "InstantReplay"
    assert h["duration_s"] == 20.0 and h["clip_start_ms"] == int(moment - 20000)
    assert (h["x"], h["y"], h["z"]) == (7.0, 2.0, -7.0) and h["session"] == "round-2" and h["player"] == "me"
    assert h["position_lag_ms"] == 300
    assert rows[1]["x"] == 9.0 and rows[1]["media"] == "image"
    assert rows[2]["request"] == "HLSetVideo" and rows[2]["game"] == "Some Game" and rows[2]["x"] == 9.0
    meta = json.loads((s / "meta.json").read_text(encoding="utf-8"))
    assert meta["stats"] == {"clips": 1, "clip_seconds": 20.0, "placed": 2, "screenshots": 1}
    assert meta["positions"] == 3
    assert (s / "heat.parquet").exists()

    b = NvidiaHighlights(id="nvidia-highlights", game="NVIDIA highlights", kind="platform",
                         overlay=str(overlay), gfesdk=str(gfesdk))
    b.start(ctx)
    b.poll(ctx)
    b.stop(ctx)
    assert len(_sessions(ctx, "nvidia-highlights")) == 1  # nothing recorded twice


def test_nvidia_temp_highlights_carry_their_id(tmp_path):
    overlay, captures = _nvidia(tmp_path)
    _tracker(overlay, [])
    temp = tmp_path / "Temp" / "Highlights" / "some_game"
    temp.mkdir(parents=True)
    ctx = _ctx(tmp_path)
    a = NvidiaHighlights(id="nvidia-highlights", game="NVIDIA highlights", kind="platform",
                         overlay=str(overlay), gfesdk=str(tmp_path / "none"))
    a.start(ctx)
    f = temp / "kill_0001.mp4"
    f.write_bytes(_mp4(15))
    os.utime(f, (time.time() - 10, time.time() - 10))
    a.poll(ctx)
    a.poll(ctx)
    a.stop(ctx)
    [s] = _sessions(ctx, "nvidia-highlights")
    [r] = read_rows(s)
    assert r["game"] == "Some Game" and r["kind"] == "Highlight"
    assert r["highlight_id"] == "kill" and r["highlight_name"] == "Enemy destroyed" and "x" not in r


# --------------------------------------------------------------------- real files, local only


def _real_steam() -> Path | None:
    root = _steam.steam_root()
    return root if root and any((root / "appcache" / "stats").glob("UserGameStats_*.bin")) else None


@pytest.mark.local
def test_real_steam_stats(tmp_path):
    root = _real_steam()
    if root is None:
        pytest.skip("no Steam stats on this machine")
    st = _steam.Steam(root)
    files = list(st.stats_dir.glob("UserGameStats_*.bin"))
    for f in files:
        appid = f.stem.split("_")[2]
        schema = st.schema(appid)
        stats, unlocked = _steam.user_stats(_steam.parse_binary_kv(f.read_bytes()), schema)
        if schema:
            assert all(schema.by_api(a) for a in unlocked)
            assert all(isinstance(v, (int, float)) for v in stats.values())
    ctx = _ctx(tmp_path)
    a = SteamStats(id="steam-stats", game="Steam statistics", kind="platform")
    a.start(ctx)
    a.stop(ctx)
    assert len(_sessions(ctx, "steam-stats")) == len(files)
    con = db.connect(ctx.data)
    assert con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == len(files)
    b = SteamStats(id="steam-stats", game="Steam statistics", kind="platform")
    b.start(ctx)
    b.stop(ctx)
    assert len(_sessions(ctx, "steam-stats")) == len(files)


@pytest.mark.local
def test_real_steam_timelines(tmp_path):
    root = _steam.steam_root()
    files = [f for a in (_steam.accounts(root) if root else [])
             for f in (root / "userdata" / a / "gamerecordings" / "timelines").glob("timeline_*.json")]
    if not files:
        pytest.skip("no Steam timelines on this machine")
    for f in files:
        doc = load_timeline(f.read_text(encoding="utf-8"))
        assert doc is not None and "daterecorded" in doc
    ctx = _ctx(tmp_path)
    a = SteamTimeline(id="steam-timeline", game="Steam Timeline", kind="platform")
    a.start(ctx)
    a.stop(ctx)
    sessions = _sessions(ctx, "steam-timeline")
    assert len(sessions) == len(files)
    assert all(read_rows(s)[0]["event"] == "session_start" for s in sessions)


@pytest.mark.local
def test_real_nvidia_gallery(tmp_path):
    a = NvidiaHighlights(id="nvidia-highlights", game="NVIDIA highlights", kind="platform", baseline=False)
    o = a.overlay_dir()
    if not o or not (o / "UGCTracker.json").exists():
        pytest.skip("no NVIDIA gallery on this machine")
    tracker = json.loads((o / "UGCTracker.json").read_text(encoding="utf-8-sig"))
    items = a.items()
    for u in tracker["UGC"]:
        assert os.path.normcase(u["path"]) in items
        p = Path(u["path"])
        assert parse_clip_name(p.name) is not None
        if u.get("type") == "video" and p.exists():
            assert mp4_duration(p) == pytest.approx(u["msDuration"] / 1000, abs=0.002)
    for d in a.definitions().values():
        assert d["highlights"] and all(d["highlights"].values())
    ctx = _ctx(tmp_path)
    a.start(ctx)
    a.stop(ctx)
    rows = [r for s in _sessions(ctx, "nvidia-highlights") for r in read_rows(s)]
    assert len([r for r in rows if r["event"] in ("highlight", "screenshot")]) == len(items)
