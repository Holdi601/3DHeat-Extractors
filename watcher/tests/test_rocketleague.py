"""Rocket League Stats API: a fake game sending the documented messages over local TCP, and the ini it edits."""

from __future__ import annotations

import dataclasses
import json
import socket
import threading
import time
from pathlib import Path

from heat3d_watch.adapters import Context
from heat3d_watch.adapters import rocketleague as rl
from heat3d_watch.config import Settings
from heat3d_watch.session import read_rows

GUID = "A1B2C3D4E5F60718293A4B5C6D7E8F90"


def ref(name, team, shortcut=1) -> dict:
    return {"Name": name, "Shortcut": shortcut, "TeamNum": team}


def update(goals=0, saves=0, blue=0, orange=0, replay=False) -> dict:
    return {"Event": "UpdateState", "Data": {
        "MatchGuid": GUID,
        "Players": [
            {"Name": "Me", "PrimaryId": "Steam|1|0", "Shortcut": 1, "TeamNum": 0, "Score": 100 * goals + 50 * saves,
             "Goals": goals, "Shots": goals + 1, "Assists": 0, "Saves": saves, "Touches": 5, "CarTouches": 1, "Demos": 1},
            {"Name": "Foe", "PrimaryId": "Epic|2|0", "Shortcut": 2, "TeamNum": 1, "Score": 30, "Goals": orange,
             "Shots": 2, "Assists": 0, "Saves": 0, "Touches": 3, "CarTouches": 0, "Demos": 0},
        ],
        "Game": {"Teams": [{"Name": "Blue", "TeamNum": 0, "Score": blue, "ColorPrimary": "1873FF", "ColorSecondary": "E5E5E5"},
                           {"Name": "Orange", "TeamNum": 1, "Score": orange, "ColorPrimary": "C26418", "ColorSecondary": "E5E5E5"}],
                 "TimeSeconds": 250, "bOvertime": False, "Ball": {"Speed": 1200.5, "TeamNum": 0}, "bReplay": replay,
                 "bHasWinner": False, "Winner": "", "Arena": "Arena_Example", "bHasTarget": True,
                 "Target": ref("Foe" if replay else "Me", 1 if replay else 0)},
    }}


def messages() -> list[dict]:
    return [
        {"Event": "MatchCreated", "Data": {"MatchGuid": GUID}},
        {"Event": "MatchInitialized", "Data": {"MatchGuid": GUID}},
        update(),
        {"Event": "CountdownBegin", "Data": {"MatchGuid": GUID}},
        {"Event": "RoundStarted", "Data": {"MatchGuid": GUID}},
        {"Event": "BallHit", "Data": {"MatchGuid": GUID, "Players": [ref("Me", 0)],
                                      "Ball": {"PreHitSpeed": 0.0, "PostHitSpeed": 1500.25,
                                               "Location": {"X": 0.0, "Y": 0.0, "Z": 93.15}}}},
        # Some builds send Data as a JSON string.
        {"Event": "CrossbarHit", "Data": json.dumps({"MatchGuid": GUID, "BallLocation": {"X": 300.0, "Y": 5120.0, "Z": 640.0},
                                                     "BallSpeed": 2500.0, "ImpactForce": 900.0,
                                                     "BallLastTouch": {"Player": ref("Me", 0), "Speed": 2500.0}})},
        {"Event": "GoalScored", "Data": {"MatchGuid": GUID, "GoalSpeed": 98.4, "GoalTime": 41.2,
                                         "ImpactLocation": {"X": -120.5, "Y": 5120.0, "Z": 300.0},
                                         "Scorer": ref("Me", 0), "BallLastTouch": {"Player": ref("Me", 0), "Speed": 98.4}}},
        {"Event": "StatfeedEvent", "Data": {"MatchGuid": GUID, "EventName": "Goal", "Type": "Goal é",
                                            "MainTarget": ref("Me", 0)}},
        {"Event": "GoalReplayStart", "Data": {"MatchGuid": GUID}},
        update(goals=1, blue=1, replay=True),
        {"Event": "GoalReplayEnd", "Data": {"MatchGuid": GUID}},
        {"Event": "StatfeedEvent", "Data": {"MatchGuid": GUID, "EventName": "Demolish", "Type": "Demolition",
                                            "MainTarget": ref("Foe", 1), "SecondaryTarget": ref("Me", 0)}},
        update(goals=1, saves=2, blue=1),
        {"Event": "MatchEnded", "Data": {"MatchGuid": GUID, "WinnerTeamNum": 0}},
        {"Event": "PodiumStart", "Data": {"MatchGuid": GUID}},
        {"Event": "MatchDestroyed", "Data": {"MatchGuid": GUID}},
    ]


class FakeGame:
    """The game's side: listens, and writes the stream to whoever connects, cut at awkward places."""

    def __init__(self, payload: bytes):
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.done = threading.Event()
        threading.Thread(target=self._serve, args=(payload,), daemon=True).start()

    def _serve(self, payload: bytes) -> None:
        conn, _ = self.srv.accept()
        with conn:
            # 7-byte pieces: messages and multi-byte characters arrive in halves.
            for i in range(0, len(payload), 7):
                conn.sendall(payload[i:i + 7])
                if i % 700 == 0:
                    time.sleep(0.002)
            self.done.set()
            time.sleep(0.3)
        self.srv.close()


def test_split_objects():
    objs, rest, junk = rl.split_objects('{"a":"}{"} \n{"b":{"c":1}}{"d":"\\"}')
    assert objs == ['{"a":"}{"}', '{"b":{"c":1}}'] and rest == '{"d":"\\"}' and junk == 0
    objs, rest, junk = rl.split_objects('xx{"a":1}\x00')
    assert objs == ['{"a":1}'] and rest == "" and junk == 2


def test_rocketleague_match(tmp_path):
    payload = "".join(json.dumps(m, ensure_ascii=False) for m in messages()).encode("utf-8")
    payload = payload.replace(b'{"Event": "PodiumStart"', b'garbage{"Event": "PodiumStart"', 1)
    game = FakeGame(payload)
    data = tmp_path / "data"
    ctx = Context(Settings(data=str(data)), data)
    a = dataclasses.replace(rl.ADAPTERS[0], port=game.port)
    a.start(ctx)
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 10:
            a.poll(ctx)
            if game.done.is_set() and a._session is None and not a._buf:
                break
        a.poll(ctx)
    finally:
        a.stop(ctx)
    folders = list((data / "sessions").glob("*/*"))
    assert len(folders) == 1
    rows = read_rows(folders[0])
    kinds = [r["event"] for r in rows]
    assert kinds[:4] == ["match_created", "match_initialized", "countdown_begin", "kickoff"]
    hit = next(r for r in rows if r["event"] == "ball_hit")
    assert (hit["x"], hit["y"], hit["z"], hit["player"], hit["team"]) == (0.0, 0.0, 93.15, "me", "blue")
    assert hit["post_speed"] == 1500.25 and hit["session"] == GUID
    bar = next(r for r in rows if r["event"] == "crossbar_hit")
    assert (bar["x"], bar["y"], bar["z"], bar["player"]) == (300.0, 5120.0, 640.0, "me")
    goal = next(r for r in rows if r["event"] == "goal")
    assert (goal["x"], goal["y"], goal["z"], goal["speed"]) == (-120.5, 5120.0, 300.0, 98.4)
    feed = next(r for r in rows if r["event"] == "feed_goal")
    assert feed["label"] == "Goal é"  # a character cut in two by the stream arrived whole
    demo = next(r for r in rows if r["event"] == "feed_demolish")
    assert (demo["player"], demo["target"]) == ("Foe", "me")
    end = next(r for r in rows if r["event"] == "match_end")
    assert end["winner"] == "blue" and end["result"] == "win" and end["blue"] == 1
    results = {r["player"]: r for r in rows if r["event"] == "player_result"}
    assert results["me"]["saves"] == 2 and results["Foe"]["team"] == "orange"
    assert "goal_replay_start" in kinds and "goal_replay_end" in kinds and "podium_start" in kinds
    meta = json.loads((folders[0] / "meta.json").read_text(encoding="utf-8"))
    st = meta["stats"]
    assert (st["goals"], st["saves"], st["shots"], st["demos"], st["touches"], st["score"]) == (1, 2, 2, 1, 5, 200)
    assert st["result"] == "win" and st["team_goals"] == 1 and st["other_goals"] == 0 and st["demolished"] == 1
    assert st["bad_packets"] == 1
    assert meta["arena"] == "Arena_Example" and meta["match_guid"] == GUID
    assert meta["axes"]["up"] == "z"
    assert (folders[0] / "heat.parquet").exists()


def test_rocketleague_no_game_is_quiet(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    data = tmp_path / "data"
    ctx = Context(Settings(data=str(data)), data)
    a = dataclasses.replace(rl.ADAPTERS[0], port=port)
    a.start(ctx)
    t = time.monotonic()
    a.poll(ctx)
    a.poll(ctx)  # within the retry pause: no second attempt
    assert time.monotonic() - t < 1.0
    a.stop(ctx)
    assert not (data / "sessions").exists()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def test_setup_switches_the_api_on(tmp_path):
    root = tmp_path / "rocketleague"
    ini = root / "TAGame" / "Config" / "DefaultStatsAPI.ini"
    original = "[TAGame.MatchStatsExporter_TA]\r\n; comment\r\nPacketSendRate=0\r\n\r\n[Other.Section]\r\nKey=1\r\n"
    write(ini, original)
    a = dataclasses.replace(rl.ADAPTERS[0], install=str(root))
    ctx = Context(Settings(data=str(tmp_path / "d")), tmp_path / "d")
    lines = a.setup(ctx, apply=False)
    assert len(lines) == 1 and "PacketSendRate=30" in lines[0] and "Port=49123" in lines[0]
    assert ini.read_bytes().decode() == original
    a.setup(ctx, apply=True)
    text = ini.read_bytes().decode()
    assert text == ("[TAGame.MatchStatsExporter_TA]\r\n; comment\r\nPacketSendRate=30\r\nPort=49123\r\n\r\n"
                    "[Other.Section]\r\nKey=1\r\n")
    assert (ini.parent / "DefaultStatsAPI.ini.heat3d-backup").read_bytes().decode() == original
    assert a.setup(ctx, apply=True) == []
    # The port the adapter connects to is the ini's.
    b = dataclasses.replace(rl.ADAPTERS[0], install=str(root), port=0)
    write(ini, "[TAGame.MatchStatsExporter_TA]\nPacketSendRate=60\nPort=50000\n")
    assert b.setup(ctx, apply=True) == []  # a rate already set is the player's choice
    b.start(ctx)
    assert b.port == 50000
    b.stop(ctx)


def test_setup_prefers_tastatsapi_and_adds_the_section(tmp_path):
    root = tmp_path / "rl"
    write(root / "TAGame" / "Config" / "DefaultStatsAPI.ini", "[TAGame.MatchStatsExporter_TA]\nPacketSendRate=0\n")
    ta = root / "TAGame" / "Config" / "TAStatsAPI.ini"
    write(ta, "[Core.System]\nPaths=x\n")
    a = dataclasses.replace(rl.ADAPTERS[0], install=str(root))
    ctx = Context(Settings(data=str(tmp_path / "d")), tmp_path / "d")
    assert str(ta) in a.setup(ctx, apply=True)[0]
    assert rl.read_ini(ta) == {"PacketSendRate": "30", "Port": "49123"}
    assert ta.read_bytes().decode().startswith("[Core.System]\nPaths=x\n\n[TAGame.MatchStatsExporter_TA]\n")
