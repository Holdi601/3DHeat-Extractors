"""League of Legends Live Client Data API: a fake local client serving answers in the documented format."""

from __future__ import annotations

import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from heat3d_watch.adapters import Context
from heat3d_watch.adapters import lol
from heat3d_watch.config import Settings
from heat3d_watch.session import read_rows


class FakeClient:
    """`/liveclientdata/allgamedata` and `/eventdata?eventID=N` from a game state the test changes."""

    def __init__(self) -> None:
        self.game: dict | None = None
        self.events: list[dict] = []
        self.asked: list[str] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                u = urlparse(self.path)
                fake.asked.append(self.path)
                if fake.game is None:
                    body, code = b'{"errorCode":"RESOURCE_NOT_FOUND","httpStatus":404}', 404
                elif u.path == "/liveclientdata/allgamedata":
                    body, code = json.dumps({**fake.game, "events": {"Events": fake.events}}).encode(), 200
                elif u.path == "/liveclientdata/eventdata":
                    since = int(parse_qs(u.query).get("eventID", ["0"])[0])
                    body, code = json.dumps({"Events": [e for e in fake.events if e["EventID"] >= since]}).encode(), 200
                else:
                    body, code = b"{}", 404
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
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


def player(name, champ, team, k=0, d=0, a=0, cs=0) -> dict:
    game, _, tag = name.partition("#")
    return {"championName": champ, "isBot": False, "isDead": False, "items": [], "level": 1, "position": "",
            "rawChampionName": f"game_character_displayname_{champ}", "respawnTimer": 0.0,
            "riotId": name, "riotIdGameName": game, "riotIdTagLine": tag, "runes": {},
            "scores": {"assists": a, "creepScore": cs, "deaths": d, "kills": k, "wardScore": 1.5},
            "skinID": 0, "summonerName": name, "summonerSpells": {}, "team": team}


def game(t, k=0, d=0, a=0, cs=0, gold=500.0, level=1) -> dict:
    return {
        "activePlayer": {"abilities": {}, "championStats": {}, "currentGold": gold, "fullRunes": {}, "level": level,
                         "riotId": "Me#TEST", "riotIdGameName": "Me", "riotIdTagLine": "TEST",
                         "summonerName": "Me#TEST", "teamRelativeColors": True},
        "allPlayers": [player("Me#TEST", "Annie", "ORDER", k, d, a, cs), player("Mate#TEST", "Garen", "ORDER"),
                       player("Foe#TEST", "Lux", "CHAOS"), player("Foe2#TEST", "Ashe", "CHAOS")],
        "gameData": {"gameMode": "CLASSIC", "gameTime": t, "mapName": "Map11", "mapNumber": 11,
                     "mapTerrain": "Default"},
    }


def test_lol_match_events_and_stats(tmp_path):
    fake = FakeClient()
    data = tmp_path / "data"
    ctx = Context(Settings(data=str(data)), data)
    a = dataclasses.replace(lol.ADAPTERS[0], url=fake.url, scoreboard_every=0)
    try:
        a.start(ctx)
        a.poll(ctx)  # loading screen: 404, nothing opened
        assert not (data / "sessions").exists()
        fake.game = game(1.0)
        fake.events = [{"EventID": 0, "EventName": "GameStart", "EventTime": 0.05}]
        a.poll(ctx)
        fake.events += [
            {"EventID": 1, "EventName": "MinionsSpawning", "EventTime": 65.0},
            {"EventID": 2, "EventName": "ChampionKill", "EventTime": 300.5, "KillerName": "Me",
             "VictimName": "Foe", "Assisters": ["Mate"]},
            {"EventID": 3, "EventName": "FirstBlood", "EventTime": 300.5, "Recipient": "Me"},
            {"EventID": 4, "EventName": "ChampionKill", "EventTime": 400.0, "KillerName": "Foe2",
             "VictimName": "Me", "Assisters": []},
            {"EventID": 5, "EventName": "ChampionKill", "EventTime": 410.0, "KillerName": "Mate",
             "VictimName": "Foe2", "Assisters": ["Me"]},
            {"EventID": 6, "EventName": "DragonKill", "EventTime": 500.0, "DragonType": "Fire", "Stolen": "False",
             "KillerName": "Mate", "Assisters": ["Me"]},
            {"EventID": 7, "EventName": "TurretKilled", "EventTime": 700.0, "TurretKilled": "Turret_T2_R_03_A",
             "KillerName": "Minion_T1L1S04N0003", "Assisters": []},
            {"EventID": 8, "EventName": "Multikill", "EventTime": 710.0, "KillerName": "Me", "KillStreak": 2},
            {"EventID": 9, "EventName": "Ace", "EventTime": 711.0, "Acer": "Me", "AcingTeam": "ORDER"},
        ]
        fake.game = game(712.0, k=1, d=1, a=2, cs=88, gold=1234.5, level=11)
        a.poll(ctx)
        a.poll(ctx)  # the same events again: each is recorded once
        fake.events.append({"EventID": 10, "EventName": "GameEnd", "EventTime": 1500.0, "Result": "Win"})
        a.poll(ctx)
        a.stop(ctx)
    finally:
        fake.close()
    assert any("eventID=10" in p for p in fake.asked), "events are asked for from the next id on"
    folders = list((data / "sessions").glob("*/*"))
    assert len(folders) == 1
    rows = read_rows(folders[0])
    kinds = [r["event"] for r in rows]
    assert kinds == ["game_start", "minions_spawning", "kill", "first_blood", "death", "assist", "dragon_kill",
                     "turret_killed", "multikill", "ace", "game_end"]
    kill = rows[2]
    assert (kill["player"], kill["victim"], kill["victim_champion"], kill["assisters"]) == ("me", "Foe", "Lux", ["Mate"])
    assert kill["time"] == 300.5
    death = rows[4]
    assert (death["player"], death["killer"], death["killer_champion"]) == ("me", "Foe2", "Ashe")
    turret = rows[7]
    assert turret["building_team"] == "CHAOS" and turret["team"] == "ORDER"
    assert rows[-1]["result"] == "win"
    assert not any("x" in r for r in rows)
    meta = json.loads((folders[0] / "meta.json").read_text(encoding="utf-8"))
    st = meta["stats"]
    assert (st["kills"], st["deaths"], st["assists"], st["creep_score"], st["level"]) == (1, 1, 2, 88, 11)
    assert st["gold"] == 1234.5 and st["champion"] == "Annie" and st["result"] == "win"
    assert st["team_dragons"] == 1 and st["team_turrets"] == 1
    assert meta["champion"] == "Annie" and meta["game_mode"] == "CLASSIC"
    # No positions, so no table: the events are the record.
    assert not (folders[0] / "heat.parquet").exists()


def test_lol_never_asks_anywhere_but_localhost():
    a = dataclasses.replace(lol.ADAPTERS[0], url="https://example.com:2999")
    assert a.missing() and a.get("/liveclientdata/allgamedata") is None
    assert lol.ADAPTERS[0].missing() is None


def test_lol_certificate_context(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    ctx = Context(Settings(data=str(data)), data)
    import ssl

    c = lol.ADAPTERS[0]._context(ctx)
    assert c.verify_mode == ssl.CERT_NONE  # no riotgames.pem here: unverified, for 127.0.0.1 only


def test_structure_owner():
    assert lol.structure_owner("Turret_T1_L_03_A") == "ORDER"
    assert lol.structure_owner("Barracks_T2_R1") == "CHAOS"
    assert lol.structure_owner("Minion_T1L1S04N0003") is None
    assert lol.snake("InhibRespawningSoon") == "inhib_respawning_soon"
