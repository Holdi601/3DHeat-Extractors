"""War Thunder web map: a fake local game serving its JSON, a battle played through it."""

from __future__ import annotations

import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from heat3d_watch.adapters import Context
from heat3d_watch.adapters import warthunder as wt
from heat3d_watch.config import Settings
from heat3d_watch.session import read_rows

YELLOW, BLUE, RED, GREY = "#faC81E", "#174DFF", "#f40C00", "#E0E0E0"


class FakeGame:
    def __init__(self) -> None:
        self.map_info: dict = {"valid": False}
        self.map_obj: list = []
        self.state: dict = {"valid": False}
        self.indicators: dict = {"valid": False}
        self.events: list[dict] = []
        self.damage: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                u = urlparse(self.path)
                q = parse_qs(u.query)
                if u.path == "/hudmsg":
                    evt, dmg = int(q.get("lastEvt", ["0"])[0] or 0), int(q.get("lastDmg", ["0"])[0] or 0)
                    doc = {"events": [e for e in fake.events if e["id"] > evt],
                           "damage": [e for e in fake.damage if e["id"] > dmg]}
                else:
                    doc = {"/map_info.json": fake.map_info, "/map_obj.json": fake.map_obj, "/state": fake.state,
                           "/indicators": fake.indicators}.get(u.path)
                body = json.dumps(doc).encode() if doc is not None else b"not found"
                self.send_response(200 if doc is not None else 404)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def obj(kind, icon, color, x, y, dx=0.0, dy=-1.0) -> dict:
    return {"type": kind, "color": color, "color[]": [0, 0, 0], "blink": 0, "icon": icon, "icon_bg": "none",
            "x": x, "y": y, "dx": dx, "dy": dy}


def dmg(i, msg) -> dict:
    return {"id": i, "msg": msg, "sender": "", "enemy": False, "mode": "", "time": 100 + i}


AIRFIELD = {"type": "airfield", "color": BLUE, "color[]": [23, 77, 255], "blink": 0, "icon": "none", "icon_bg": "none",
            "sx": 0.40, "sy": 0.80, "ex": 0.44, "ey": 0.80}


def test_conversion_and_sides():
    info = wt.MapInfo.parse({"grid_steps": [8192, 8192], "grid_zero": [-32768, 32768], "map_generation": 5,
                             "hud_type": 0, "map_max": [32768, 32768], "map_min": [-32768, -32768], "valid": True})
    assert info.metres(0.5, 0.25) == (0.0, 16384.0)
    assert info.metres(0.0, 1.0) == (-32768.0, -32768.0)
    assert info.id.startswith("wt_65536x65536_")
    assert wt.MapInfo.parse({"valid": False}) is None and wt.MapInfo.parse({"map_min": "x"}) is None
    assert wt.side(RED, YELLOW) == "enemy" and wt.side(BLUE, YELLOW) == "friendly"
    assert wt.side(YELLOW, YELLOW) == "friendly" and wt.side(GREY, YELLOW) == "neutral"
    assert wt.side("#10E040", YELLOW) == "friendly"  # squad green
    assert wt.heading(0, -1) == 0.0 and wt.heading(1, 0) == 90.0 and wt.heading(0, 1) == 180.0


def test_feed_patterns():
    m = wt.FEED.match("=TAG= Pilot (Spitfire F Mk IX) shot down Other_1 (Bf 109 F-4)")
    assert m and m.group("actor") == "=TAG= Pilot" and m.group("target_vehicle") == "Bf 109 F-4"
    m = wt.FEED.match("Tanker (T-34 (1942)) destroyed Foe (Pz.IV H)")
    assert m and m.group("actor_vehicle") == "T-34 (1942)" and m.group("verb") == "destroyed"
    assert wt.FEED.match("Pilot (Yak-3) set afire Foe (Ju 87 D)").group("verb") == "set afire"
    assert wt.ALONE.match("Pilot (Yak-3) has crashed.").group("actor") == "Pilot"


def test_tracker_keeps_ids():
    t = wt.Tracker()
    a = t.assign([("aircraft|Fighter", 0, 0), ("aircraft|Fighter", 5000, 0), ("ground_model|SPAA", 0, 0)], 0.0)
    b = t.assign([("aircraft|Fighter", 5100, 0), ("aircraft|Fighter", 90, 10), ("ground_model|SPAA", 2, 0)], 0.5)
    assert len(set(a)) == 3 and b == [a[1], a[0], a[2]]
    c = t.assign([("ground_model|SPAA", 900, 0)], 1.0)  # too far for a ground unit in half a second
    assert c[0] not in a


def test_warthunder_battle(tmp_path):
    game = FakeGame()
    data = tmp_path / "data"
    ctx = Context(Settings(data=str(data)), data)
    a = dataclasses.replace(wt.ADAPTERS[0], url=game.url, player_name="Me", map_every=0, hud_every=0,
                            indicators_every=0, gone_after=0, vehicle_grace=0)
    try:
        a.start(ctx)
        game.damage = [dmg(1, "Old (Yak-3) shot down Older (Bf 109 F-4)")]  # before the watcher: not this battle
        a.poll(ctx)
        assert not (data / "sessions").exists()
        game.map_info = {"grid_steps": [8192, 8192], "grid_zero": [-32768, 32768], "map_generation": 5, "hud_type": 0,
                         "map_max": [32768, 32768], "map_min": [-32768, -32768], "valid": True}
        game.indicators = {"valid": True, "army": "air", "type": "spitfire_example"}
        game.state = {"valid": True, "H, m": 1500, "TAS, km/h": 420, "IAS, km/h": 390, "M": 0.35}
        zone = obj("capture_zone", "capture_zone", GREY, 0.5, 0.5)
        game.map_obj = [obj("aircraft", "Player", YELLOW, 0.5, 0.25), obj("aircraft", "Fighter", RED, 0.6, 0.5),
                        AIRFIELD, zone]
        a.poll(ctx)
        game.map_obj = [obj("aircraft", "Player", YELLOW, 0.501, 0.25, dx=1, dy=0),
                        obj("aircraft", "Fighter", RED, 0.601, 0.5), AIRFIELD, {**zone, "color": RED}]
        game.state = {**game.state, "H, m": 1600, "TAS, km/h": 450}
        game.damage.append(dmg(2, "=TAG= Me (Spitfire F Mk IX) shot down Foe (Bf 109 F-4)"))
        game.damage.append(dmg(3, "Foe2 (Fw 190 A-5) destroyed Mate (Spitfire F Mk IX)"))
        game.events.append({"id": 1, "msg": "Enemy has captured the airfield", "sender": "", "enemy": False,
                            "mode": "", "time": 120})
        a.poll(ctx)
        game.damage.append(dmg(4, "Foe3 (Bf 109 G-2) shot down Me (Spitfire F Mk IX)"))
        game.map_obj = [obj("aircraft", "Fighter", RED, 0.602, 0.5), AIRFIELD]
        game.state = {"valid": False}
        a.poll(ctx)
        a.poll(ctx)
        game.map_info = {"valid": False}
        game.map_obj = []
        a.poll(ctx)
        assert a._session is None
        a.stop(ctx)
    finally:
        game.close()
    folders = list((data / "sessions").glob("*/*"))
    assert len(folders) == 1
    rows = read_rows(folders[0])
    kinds = [r["event"] for r in rows]
    spawn = next(r for r in rows if r["event"] == "spawn")
    assert (spawn["x"], spawn["y"], spawn["z"], spawn["player"]) == (0.0, 1500.0, 16384.0, "me")
    assert spawn["vehicle"] == "spitfire_example" and spawn["session"] == "sortie 1"
    own = [r for r in rows if r["event"] == "position" and r["player"] == "me"]
    assert len(own) == 2 and own[1]["y"] == 1600.0 and own[1]["heading"] == 90.0
    assert own[1]["x"] == pytest.approx(65.536)
    foes = {r["player"] for r in rows if r["event"] == "position" and r["player"] != "me"}
    assert len(foes) == 1 and next(iter(foes)).startswith("aircraft-")
    assert all(r["team"] == "enemy" for r in rows if r["event"] == "position" and r["player"] != "me")
    assert kinds.count("map_object") == 2 and kinds.count("zone_changed") == 1
    field = next(r for r in rows if r["event"] == "map_object" and r["kind"] == "airfield")
    assert field["x"] == pytest.approx(-32768 + 0.42 * 65536) and field["team"] == "friendly"
    kill = next(r for r in rows if r["event"] == "kill")
    assert kill["target"] == "Foe" and kill["target_vehicle"] == "Bf 109 F-4" and kill["player"] == "me"
    assert next(r for r in rows if r["event"] == "kill_feed")["actor"] == "Foe2"
    death = next(r for r in rows if r["event"] == "death")
    assert death["actor"] == "Foe3" and death["session"] == "sortie 1"
    assert "x" in death
    assert kinds.count("vehicle_left") == 1 and kinds.count("hud_event") == 1
    assert not any(r.get("actor") == "Old" for r in rows)
    meta = json.loads((folders[0] / "meta.json").read_text(encoding="utf-8"))
    st = meta["stats"]
    assert st["kills"] == 1 and st["deaths"] == 1 and st["spawns"] == 1
    assert st["max_altitude_m"] == 1600.0 and st["max_tas_kmh"] == 450 and st["distance_m"] > 100
    assert meta["map_id"].startswith("wt_") and meta["axes"]["up"] == "y"
    assert (folders[0] / "heat.parquet").exists()


def test_warthunder_hangar_only(tmp_path):
    game = FakeGame()
    data = tmp_path / "data"
    ctx = Context(Settings(data=str(data)), data)
    a = dataclasses.replace(wt.ADAPTERS[0], url=game.url)
    try:
        a.start(ctx)
        a.poll(ctx)
        a.stop(ctx)
    finally:
        game.close()
    assert not (data / "sessions").exists()
    assert dataclasses.replace(wt.ADAPTERS[0], url="http://192.168.1.2:8111").missing()
