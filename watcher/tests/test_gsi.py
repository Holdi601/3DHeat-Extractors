"""Valve GSI: documents in the published format posted to the running adapter, and the cfg files it writes."""

from __future__ import annotations

import dataclasses
import json
import socket
import urllib.request
from pathlib import Path

import pytest

from heat3d_watch.adapters import Context
from heat3d_watch.adapters import gsi
from heat3d_watch.config import Settings
from heat3d_watch.session import read_rows

ME = "76561190000000001"
MATE = "76561190000000002"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(port: int, doc: dict | bytes) -> None:
    body = doc if isinstance(doc, bytes) else json.dumps(doc).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.status == 200


def context(tmp_path: Path) -> Context:
    data = tmp_path / "data"
    return Context(Settings(data=str(data)), data)


def adapter(aid: str, **kw) -> gsi.GsiAdapter:
    a = next(a for a in gsi.ADAPTERS if a.id == aid)
    return dataclasses.replace(a, port=free_port(), **kw)


def drive(a: gsi.GsiAdapter, ctx: Context, docs: list) -> None:
    a.start(ctx)
    try:
        for d in docs:
            post(a.port, d)
            a.poll(ctx)
        a.poll(ctx)
    finally:
        a.stop(ctx)


def only_session(ctx: Context) -> Path:
    folders = [p for p in (ctx.data / "sessions").glob("*/*") if p.is_dir()]
    assert len(folders) == 1, folders
    return folders[0]


# ----------------------------------------------------------------------------- CS2


def cs2(phase="live", rnd=0, rphase="live", player=None, **extra) -> dict:
    doc = {
        "provider": {"name": "Counter-Strike: Global Offensive", "appid": 730, "version": 14000, "steamid": ME,
                     "timestamp": 1700000000},
        "map": {"mode": "competitive", "name": "de_example", "phase": phase, "round": rnd,
                "team_ct": {"score": extra.pop("ct", 0), "consecutive_round_losses": 0, "timeouts_remaining": 1},
                "team_t": {"score": extra.pop("t", 0), "consecutive_round_losses": 0, "timeouts_remaining": 1},
                "num_matches_to_win_series": 0},
        "round": {"phase": rphase, **extra.pop("round", {})},
        "auth": {"token": "secret"},
    }
    if player is not None:
        doc["player"] = player
    doc.update(extra)
    return doc


def me(pos, health=100, kills=0, deaths=0, steamid=ME, team="T", rk=None) -> dict:
    rk = kills if rk is None else rk
    return {
        "steamid": steamid, "name": "someone", "observer_slot": 1, "team": team, "activity": "playing",
        "state": {"health": health, "armor": 100, "helmet": True, "flashed": 0, "smoked": 0, "burning": 0,
                  "money": 800, "round_kills": rk, "round_killhs": rk, "round_totaldmg": 100 * rk,
                  "equip_value": 200},
        "match_stats": {"kills": kills, "assists": 0, "deaths": deaths, "mvps": 0, "score": 2 * kills},
        "position": f"{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}",
        "forward": "0.50, -0.86, 0.00",
        "weapons": {"weapon_0": {"name": "weapon_knife", "type": "Knife", "state": "holstered"},
                    "weapon_1": {"name": "weapon_ak47", "type": "Rifle", "state": "active", "ammo_clip": 30}},
    }


def test_cs2_round_kill_death_and_result(tmp_path):
    ctx = context(tmp_path)
    a = adapter("cs2-gsi", token="secret")
    docs = [
        {"provider": {"name": "Counter-Strike: Global Offensive", "appid": 730, "steamid": ME}, "auth": {"token": "secret"}},
        b"{not json",
        {**cs2(player=me((0, 0, 0))), "auth": {"token": "wrong"}},
        cs2(rphase="freezetime", player=me((100, 200, 10))),
        cs2(rphase="live", player=me((150, 260, 10))),
        cs2(rphase="live", player=me((300, 400, 12), kills=1)),
        cs2(rphase="live", player=me((300, 400, 12), health=0, kills=1, deaths=1)),
        # Dead: the game now shows a teammate. Nothing of theirs is the player's.
        cs2(rphase="live", player=me((-999, -999, 0), steamid=MATE)),
        cs2(rphase="over", player=me((-999, -999, 0), steamid=MATE), round={"win_team": "CT"}, ct=1),
        cs2(rnd=1, rphase="freezetime", player=me((100, 200, 10), kills=1, deaths=1, rk=0), ct=1),
        cs2(phase="gameover", rnd=1, rphase="over", player=me((110, 210, 10), kills=1, deaths=1, rk=0),
            round={"win_team": "CT"}, ct=2),
        {"provider": {"name": "Counter-Strike: Global Offensive", "appid": 730, "steamid": ME}, "auth": {"token": "secret"}},
    ]
    drive(a, ctx, docs)
    folder = only_session(ctx)
    rows = read_rows(folder)
    events = [r["event"] for r in rows]
    assert events.count("spawn") == 2
    assert events.count("kill") == 1 and events.count("death") == 1
    kill = next(r for r in rows if r["event"] == "kill")
    assert (kill["x"], kill["y"], kill["z"], kill["weapon"]) == (300, 400, 12, "weapon_ak47")
    death = next(r for r in rows if r["event"] == "death")
    assert (death["x"], death["y"], death["player"], death["session"]) == (300, 400, "me", "round 1")
    assert not any(r.get("x") == -999 for r in rows), "a teammate's position was recorded"
    spawn2 = [r for r in rows if r["event"] == "spawn"][1]
    assert spawn2["session"] == "round 2"
    end = next(r for r in rows if r["event"] == "round_end")
    assert end["winner"] == "CT" and end["won"] is False
    assert next(r for r in rows if r["event"] == "match_end")["result"] == "loss"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    st = meta["stats"]
    assert st["kills"] == 1 and st["deaths"] == 1 and st["rounds_lost"] == 2 and st["result"] == "loss"
    assert st["bad_packets"] == 1 and st["bad_auth"] == 1
    assert st["headshot_kills"] == 1 and st["damage"] == 100
    assert meta["axes"]["up"] == "z" and meta["axes"]["flip_y"] is True
    assert meta["map"] == "de_example"
    assert (folder / "heat.parquet").exists()


def test_cs2_late_death_from_the_count(tmp_path):
    """Health never seen at 0 (the game switched to a teammate first): the death count still tells."""
    ctx = context(tmp_path)
    a = adapter("cs2-gsi", token="secret")
    drive(a, ctx, [
        cs2(rphase="freezetime", player=me((10, 20, 0), kills=3, deaths=2)),
        cs2(rphase="live", player=me((50, 60, 0), kills=3, deaths=2)),
        cs2(rphase="live", player=me((1, 1, 1), steamid=MATE)),
        cs2(rnd=1, rphase="freezetime", player=me((10, 20, 0), kills=3, deaths=3)),
    ])
    rows = read_rows(only_session(ctx))
    deaths = [r for r in rows if r["event"] == "death"]
    # Mid-match start: the two deaths before the watcher came are not invented.
    assert len(deaths) == 1 and deaths[0]["late"] is True and (deaths[0]["x"], deaths[0]["y"]) == (50, 60)
    assert [r["event"] for r in rows].count("kill") == 0


def test_cs2_setup_writes_cfg_once_with_backup(tmp_path):
    install = tmp_path / "cs2"
    (install / "game" / "csgo" / "cfg").mkdir(parents=True)
    a = adapter("cs2-gsi", install=str(install))
    ctx = context(tmp_path)
    cfg = install / a.cfg_file
    lines = a.setup(ctx, apply=False)
    assert len(lines) == 1 and str(cfg) in lines[0] and not cfg.exists()
    a.setup(ctx, apply=True)
    text = cfg.read_text(encoding="utf-8")
    assert f'"uri"\t\t"http://127.0.0.1:{a.port}/"' in text and '"player_position"\t"1"' in text
    token = gsi.TOKEN.search(text).group(1)
    assert len(token) == 32
    assert a.setup(ctx, apply=True) == []  # idempotent: same token, same text
    assert not cfg.with_name(cfg.name + ".heat3d-backup").exists()
    # A changed port rewrites it, keeping the token and a backup of the first version.
    b = dataclasses.replace(a, port=a.port + 1)
    assert b.setup(ctx, apply=True)
    assert gsi.TOKEN.search(cfg.read_text(encoding="utf-8")).group(1) == token
    assert cfg.with_name(cfg.name + ".heat3d-backup").read_text(encoding="utf-8") == text
    # Started without a token of its own, the adapter takes the cfg file's.
    drive(b, ctx, [
        {**cs2(rphase="live", player=me((1, 2, 3))), "auth": {"token": "wrong"}},
        {**cs2(rphase="live", player=me((1, 2, 3))), "auth": {"token": token}},
    ])
    meta = json.loads((only_session(ctx) / "meta.json").read_text(encoding="utf-8"))
    assert meta["stats"]["bad_auth"] == 1 and meta["positions"] >= 1


def test_vec_and_bad_shapes():
    assert gsi.vec("1.5, -2, 3") == (1.5, -2.0, 3.0)
    assert gsi.vec("1, 2") is None and gsi.vec(None) is None and gsi.vec("a, b, c") is None


def test_steam_libraries_parse(tmp_path, monkeypatch):
    steam = tmp_path / "Steam"
    (steam / "steamapps").mkdir(parents=True)
    lib = tmp_path / "Lib"
    (lib / "steamapps" / "common" / "dota 2 beta").mkdir(parents=True)
    (steam / "steamapps" / "libraryfolders.vdf").write_text(
        '"libraryfolders"\n{\n\t"0"\n\t{\n\t\t"path"\t\t"%s"\n\t}\n\t"1"\n\t{\n\t\t"path"\t\t"%s"\n\t}\n}\n'
        % (str(steam).replace("\\", "\\\\"), str(lib).replace("\\", "\\\\")), encoding="utf-8")
    monkeypatch.setattr(gsi, "STEAM_DEFAULTS", (str(steam),))
    import builtins

    real_import = builtins.__import__

    def no_winreg(name, *args, **kw):
        if name in ("winreg", "heat3d_gamefiles.forzainstall"):
            raise ImportError(name)
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", no_winreg)
    assert lib in gsi.steam_libraries()
    assert gsi.steam_game("dota 2 beta") == lib / "steamapps" / "common" / "dota 2 beta"


@pytest.mark.local
def test_cs2_install_found_on_this_machine(tmp_path):
    a = next(a for a in gsi.ADAPTERS if a.id == "cs2-gsi")
    root = a.find_install(None)
    if root is None:
        pytest.skip("CS2 is not installed here")
    assert (root / "game" / "csgo" / "cfg").is_dir()
    lines = a.setup(context(tmp_path), apply=False)  # never applied by a test
    if (root / a.cfg_file).exists():
        assert all(str(root / a.cfg_file) in line for line in lines)
    else:
        assert len(lines) == 1 and str(root / a.cfg_file) in lines[0]


# ----------------------------------------------------------------------------- Dota 2


def dota(state="DOTA_GAMERULES_STATE_GAME_IN_PROGRESS", clock=60, hero=None, player=None, win="none", **extra):
    doc = {
        "provider": {"name": "Dota 2", "appid": 570, "version": 47, "timestamp": 1700000000},
        "map": {"name": "start", "matchid": "7000000001", "game_time": clock + 90, "clock_time": clock,
                "daytime": True, "nightstalker_night": False, "radiant_score": 1, "dire_score": 0,
                "game_state": state, "paused": False, "win_team": win, "customgamename": ""},
        "auth": {"token": "secret"},
    }
    if player is not None:
        doc["player"] = player
    if hero is not None:
        doc["hero"] = hero
    doc.update(extra)
    return doc


def dplayer(kills=0, deaths=0, assists=0) -> dict:
    return {"steamid": ME, "accountid": "1", "name": "someone", "activity": "playing", "kills": kills,
            "deaths": deaths, "assists": assists, "last_hits": 10, "denies": 2, "kill_streak": 0,
            "team_name": "radiant", "gold": 600, "gold_reliable": 0, "gold_unreliable": 600, "gpm": 300, "xpm": 400}


def dhero(x, y, alive=True, level=3) -> dict:
    return {"xpos": x, "ypos": y, "id": 1, "name": "npc_dota_hero_example", "level": level, "alive": alive,
            "respawn_seconds": 0 if alive else 12, "health": 600 if alive else 0, "max_health": 640}


def towers(top1=1800) -> dict:
    return {"radiant": {"dota_goodguys_tower1_top": {"health": 1800, "max_health": 1800}},
            "dire": {"dota_badguys_tower1_top": {"health": top1, "max_health": 1800}}}


def test_dota_hero_route_kills_buildings_roshan(tmp_path):
    ctx = context(tmp_path)
    a = adapter("dota2-gsi", token="secret")
    ev = [{"game_time": 900, "event_type": "roshan_killed", "killed_by_team": "radiant", "killer_player_id": 2}]
    drive(a, ctx, [
        dota(state="DOTA_GAMERULES_STATE_PRE_GAME", clock=-30, hero=dhero(-6700, -6700), player=dplayer(),
             buildings=towers(), roshan={"alive": True, "health": 6000}),
        dota(clock=60, hero=dhero(-5000, -5200), player=dplayer(), buildings=towers()),
        dota(clock=70, hero=dhero(-4000, -4100), player=dplayer(kills=1), buildings=towers(900)),
        dota(clock=80, hero=dhero(-3000, -3000, alive=False), player=dplayer(kills=1, deaths=1),
             buildings=towers(0), roshan={"alive": False, "health": 0}, events=ev),
        dota(clock=95, hero=dhero(-6700, -6700), player=dplayer(kills=1, deaths=1), events=ev),
        dota(state="DOTA_GAMERULES_STATE_POST_GAME", clock=2000, hero=dhero(-6600, -6600),
             player=dplayer(kills=1, deaths=1, assists=4), win="radiant"),
    ])
    folder = only_session(ctx)
    rows = read_rows(folder)
    events = [r["event"] for r in rows]
    assert events.count("spawn") == 1 and events.count("death") == 1 and events.count("respawn") == 1
    death = next(r for r in rows if r["event"] == "death")
    assert (death["x"], death["y"], death["z"]) == (-4000, -4100, 0) and death["respawn_seconds"] == 12
    kill = next(r for r in rows if r["event"] == "kill")
    assert (kill["x"], kill["y"]) == (-4000, -4100)
    assert not any(r.get("x") == -3000 for r in rows), "a dead hero has no route"
    b = next(r for r in rows if r["event"] == "building_destroyed")
    assert (b["team"], b["building"]) == ("dire", "dota_badguys_tower1_top")
    assert events.count("roshan_killed") == 1 and events.count("roshan_down") == 1
    assert next(r for r in rows if r["event"] == "roshan_killed")["killed_by_team"] == "radiant"
    assert next(r for r in rows if r["event"] == "match_end")["result"] == "win"
    st = json.loads((folder / "meta.json").read_text(encoding="utf-8"))["stats"]
    assert st["kills"] == 1 and st["deaths"] == 1 and st["assists"] == 4 and st["result"] == "win"
    assert st["level"] == 3 and st["last_hits"] == 10
    assert all(r["session"] == "7000000001" for r in rows if "x" in r)
    assert (folder / "heat.parquet").exists()


def test_dota_setup_creates_folder(tmp_path):
    install = tmp_path / "dota"
    (install / "game" / "dota" / "cfg").mkdir(parents=True)
    a = adapter("dota2-gsi", install=str(install))
    assert a.setup(context(tmp_path), apply=True)
    text = (install / a.cfg_file).read_text(encoding="utf-8")
    assert '"hero"\t"1"' in text and '"buildings"\t"1"' in text
    assert a.setup(context(tmp_path), apply=True) == []


def test_spectating_dota_is_not_the_players_own(tmp_path):
    ctx = context(tmp_path)
    a = adapter("dota2-gsi", token="secret")
    drive(a, ctx, [dota(player={"team2": {"player0": {"kills": 3}}}, hero={"team2": {"player0": {"xpos": 1}}})])
    rows = read_rows(only_session(ctx))
    assert not any("x" in r for r in rows)
