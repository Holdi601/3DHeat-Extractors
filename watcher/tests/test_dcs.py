"""DCS: the hook file setup writes, and the hook's line format sent to the adapter as real UDP."""

from __future__ import annotations

import json
import socket
import time
from dataclasses import replace

import pytest

from heat3d_watch.adapters.base import Context
from heat3d_watch.adapters.dcs import ADAPTERS, BACKUP, HOOK_NAME, MARK, hook_lua, parse_line
from heat3d_watch.config import Settings
from heat3d_watch.session import read_rows


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_setup_adds_its_own_hook_and_leaves_others_alone(tmp_path):
    games = tmp_path / "Saved Games"
    (games / "DCS" / "Scripts" / "Hooks").mkdir(parents=True)
    (games / "DCS" / "Scripts" / "Hooks" / "SomeoneElse.lua").write_text("-- theirs", encoding="utf-8")
    (games / "DCS.openbeta" / "Config").mkdir(parents=True)
    a = replace(ADAPTERS[0], saved_games=str(games))
    ctx = Context(Settings(data=str(tmp_path / "data")), tmp_path / "data")
    planned = a.setup(ctx, apply=False)
    assert len(planned) == 2 and all(line.startswith("add ") for line in planned)
    assert not (games / "DCS" / "Scripts" / "Hooks" / HOOK_NAME).exists()  # a dry run changes nothing
    assert len(a.setup(ctx, apply=True)) == 2
    for d in ("DCS", "DCS.openbeta"):
        text = (games / d / "Scripts" / "Hooks" / HOOK_NAME).read_text(encoding="utf-8")
        assert text.startswith(MARK) and "41993" in text
    assert (games / "DCS" / "Scripts" / "Hooks" / "SomeoneElse.lua").read_text(encoding="utf-8") == "-- theirs"
    assert a.setup(ctx, apply=True) == []  # idempotent
    # An older version of our own hook is updated, with a backup kept once.
    hook = games / "DCS" / "Scripts" / "Hooks" / HOOK_NAME
    hook.write_text(MARK + " 0\n-- old", encoding="utf-8")
    assert a.setup(ctx, apply=True)[0].startswith("update ")
    assert hook.with_name(HOOK_NAME + BACKUP).read_text(encoding="utf-8").endswith("-- old")
    # A file of that name someone else wrote is not ours to replace.
    hook.write_text("-- not the watcher's", encoding="utf-8")
    assert a.setup(ctx, apply=True)[0].startswith("leave ")
    assert hook.read_text(encoding="utf-8") == "-- not the watcher's"


def test_lines_parse():
    kind, f = parse_line("p;t=1.5;x=-281000.50;y=1500.25;z=647000.75;type=F-16C_50\n")
    assert kind == "p" and f["x"] == "-281000.50" and f["type"] == "F-16C_50"
    assert parse_line("ev;t=2;name=kill;args=1|A|2|-1|B|1|W")[1]["args"] == "1|A|2|-1|B|1|W"
    assert parse_line("hello;v=1") == ("hello", {"v": "1"})


def _p(t, x, y, z, **kw):
    fields = dict(t=f"{t:.3f}", x=f"{x:.2f}", y=f"{y:.2f}", z=f"{z:.2f}", lat="42.1", lon="41.7", alt=f"{y:.2f}",
                  hdg="1.5", pitch="0.1", bank="0", ias="150", agl="900", g="1.2", type="F-16C_50")
    fields.update(kw)
    return "p;" + ";".join(f"{k}={v}" for k, v in fields.items())


def test_a_mission_over_udp(tmp_path):
    port = _free_port()
    a = replace(ADAPTERS[0], port=port, respawn_gap=0.5)
    ctx = Context(Settings(data=str(tmp_path)), tmp_path)
    a.start(ctx)
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(*lines):
        for line in lines:
            out.sendto(line.encode(), ("127.0.0.1", port))
        time.sleep(0.02)
        a.poll(ctx)

    send("hello;v=1", "start;mission=Training;map=Testland;me=1;name=Pilot;mp=0")
    folder = a._session.folder
    send(*[_p(i * 0.1, -281000 + i * 10, 20, 647000) for i in range(3)])
    send("ev;t=0.3;name=takeoff;args=1|16777217|Some Field")
    send(*[_p(0.3 + i * 0.1, -280970 + i * 20, 30 + i * 10, 647000, g=str(1 + i)) for i in range(3)])
    send("ev;t=0.6;name=kill;args=1|F-16C_50|2|-1|Su-27|1|AIM-120C")
    send("ev;t=0.7;name=kill;args=-1|MiG-29S|1|1|F-16C_50|2|R-73")
    send("ev;t=0.7;name=eject;args=1|16777217", "ev;t=0.8;name=pilot_death;args=1|16777217")
    send(_p(0.9, -280900, 50, 647000))  # the view after death: not a new life
    time.sleep(0.6)
    send(_p(30.0, -100000, 20, 500000))  # respawned, later
    send("this is not a line", "p;t=1;x=;y=;z=")
    send("ev;t=31;name=mission_end;args=2|Red wins")
    send("stop;t=31")
    assert a._session is None
    a.stop(ctx)
    rows = read_rows(folder)
    events = [r["event"] for r in rows if r["event"] != "position"]
    assert events == [
        "mission_start", "spawn", "takeoff", "kill", "killed", "eject", "session_end", "death",
        "spawn", "mission_end", "mission_stop", "session_end",
    ]
    kill = next(r for r in rows if r["event"] == "kill")
    assert kill["victim_type"] == "Su-27" and kill["weapon"] == "AIM-120C" and kill["player"] == "me"
    assert kill["x"] == pytest.approx(-280930)  # where the shooter was
    takeoff = next(r for r in rows if r["event"] == "takeoff")
    assert takeoff["airfield"] == "Some Field" and takeoff["session"] == "life-1"
    pos = [r for r in rows if r["event"] == "position"]
    assert len(pos) == 7 and pos[-1]["session"] == "life-2" and pos[-1]["z"] == 500000
    assert pos[0]["lat"] == 42.1 and pos[0]["aircraft"] == "F-16C_50"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["map"] == "Testland" and meta["pilot"] == "Pilot" and meta["multiplayer"] is False
    assert meta["axes"] == {"up": "y", "flip_x": False, "flip_y": False, "flip_z": False, "scale": 1.0,
                            "units": "metres (DCS world: x north, y up, z east)"}
    st = meta["stats"]
    assert st["kills"] == 1 and st["shot_down"] == 1 and st["deaths"] == 1 and st["ejections"] == 1
    assert st["takeoffs"] == 1 and st["lives"] == 2 and st["max_g"] == 3 and st["bad_packets"] == 2
    assert st["max_altitude_m"] == 50


def test_a_server_that_forbids_export_leaves_a_session_saying_so(tmp_path):
    port = _free_port()
    a = replace(ADAPTERS[0], port=port)
    ctx = Context(Settings(data=str(tmp_path)), tmp_path)
    a.start(ctx)
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for line in ("start;mission=M;map=Testland;me=7;name=P;mp=1", "noexport;t=1", "noexport;t=6",
                 "ev;t=8;name=takeoff;args=3|1|Field", "ev;t=9;name=takeoff;args=7|1|Field"):
        out.sendto(line.encode(), ("127.0.0.1", port))
    time.sleep(0.05)
    a.poll(ctx)
    folder = a._session.folder
    a.stop(ctx)
    rows = read_rows(folder)
    assert [r["event"] for r in rows] == ["mission_start", "no_export", "takeoff", "takeoff"]
    assert [r.get("player") for r in rows[2:]] == ["player-3", "me"]
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["multiplayer"] is True and meta["stats"]["export_blocked"] == 1 and "no_export" in meta


def test_the_hook_runs_under_lua_5_1():
    """With lupa (MIT) installed: the hook loaded in Lua 5.1 with stand-ins for DCS, sends the lines parsed above."""
    lupa = pytest.importorskip("lupa")
    try:
        from lupa.lua51 import LuaRuntime
    except ImportError:
        LuaRuntime = lupa.LuaRuntime
    L = LuaRuntime(unpack_returned_tuples=True)
    sent = []
    L.globals().py_send = lambda text, host, port: sent.append((text, host, port))
    L.execute("""
    package.preload['socket'] = function()
      return { udp = function()
        local u = {}
        function u:settimeout(t) end
        function u:sendto(text, host, port) py_send(text, host, port) end
        return u
      end }
    end
    model = 0
    DCS = { getModelTime = function() return model end, getMissionName = function() return 'M;1' end,
            getCurrentMission = function() return { mission = { theatre = 'Testland' } } end,
            isMultiplayer = function() return false end, setUserCallbacks = function(cb) callbacks = cb end }
    net = { get_my_player_id = function() return 1 end, get_player_info = function() return 'Pilot' end }
    Export = { LoGetSelfData = function() return { Name = 'A-10C', Heading = 1, Pitch = 0, Bank = 0,
      LatLongAlt = { Lat = 42.5, Long = 41.5, Alt = 100 }, Position = { x = 1, y = 100, z = 2 } } end }
    """)
    L.execute(hook_lua())
    L.execute("callbacks.onSimulationStart(); callbacks.onSimulationFrame(); callbacks.onGameEvent('crash', 1, 5)")
    kinds = [parse_line(t)[0] for t, _, _ in sent]
    assert kinds == ["hello", "start", "p", "ev"] and all(p == 41993 for _, _, p in sent)
    assert parse_line(sent[2][0])[1]["type"] == "A-10C" and parse_line(sent[1][0])[1]["mission"] == "M 1"
