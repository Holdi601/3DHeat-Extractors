"""Elite Dangerous: journal lines and Status.json snapshots in the format of Frontier's Journal Manual."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from heat3d_watch.adapters import elite
from heat3d_watch.adapters.base import Context
from heat3d_watch.config import Settings
from heat3d_watch.session import Session, read_rows


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def journal_lines(t0: float) -> list[str]:
    """One short play session, field for field as the Journal Manual writes the events."""
    events = [
        {"event": "Fileheader", "part": 1, "language": "English/UK", "Odyssey": True, "gameversion": "4.0", "build": "r1"},
        {"event": "Commander", "FID": "F1", "Name": "Test Pilot"},
        {"event": "LoadGame", "FID": "F1", "Commander": "Test Pilot", "Ship": "SideWinder", "Ship_Localised": "Sidewinder",
         "Credits": 1000, "GameMode": "Solo"},
        {"event": "Location", "Docked": True, "StarSystem": "Test System A", "SystemAddress": 1, "StarPos": [0.0, 0.0, 0.0],
         "Body": "Test Station", "BodyType": "Station"},
        {"event": "Undocked", "StationName": "Test Station", "StationType": "Coriolis"},
        {"event": "FSDJump", "StarSystem": "Test System B", "SystemAddress": 2, "StarPos": [3.0, 0.5, 4.0],
         "JumpDist": 5.025, "FuelUsed": 0.5, "SystemEconomy": "$economy_Industrial;",
         "SystemEconomy_Localised": "Industrial"},
        {"event": "Scan", "ScanType": "Detailed", "BodyName": "Test System B 1", "Radius": 1000000.0},
        {"event": "Scan", "ScanType": "AutoScan", "BodyName": "Test System B 1", "Radius": 1000000.0},
        {"event": "ApproachBody", "StarSystem": "Test System B", "Body": "Test System B 1", "BodyID": 1},
        {"event": "Touchdown", "PlayerControlled": True, "Body": "Test System B 1", "OnPlanet": True,
         "Latitude": 10.0, "Longitude": 20.0},
        {"event": "Liftoff", "PlayerControlled": True, "Body": "Test System B 1", "OnPlanet": True,
         "Latitude": 10.0, "Longitude": 20.01},
        {"event": "Bounty", "Rewards": [{"Faction": "F", "Reward": 500}], "Target": "viper", "TotalReward": 500},
        {"event": "MarketSell", "Type": "gold", "Count": 2, "SellPrice": 100, "TotalSale": 200},
        {"event": "MissionCompleted", "Name": "Mission_Delivery", "MissionID": 9, "Reward": 3000},
        {"event": "RedeemVoucher", "Type": "bounty", "Amount": 500},
        {"event": "Died", "KillerName": "Pirate", "KillerShip": "viper", "KillerRank": "Deadly"},
        {"event": "Resurrect", "Option": "rebuy", "Cost": 100, "Bankrupt": False},
        {"event": "Docked", "StationName": "Test Station B", "StarSystem": "Test System B", "MarketID": 7},
        {"event": "Shutdown"},
    ]
    out = []
    for i, e in enumerate(events):
        out.append(json.dumps({"timestamp": iso(t0 + i), **e}).replace('{"timestamp"', '{ "timestamp"'))
    out.insert(5, "{ this is not json")
    return out


def status(t: float, lat=None, lon=None, alt=0.0, body="Test System B 1", flags=0) -> str:
    doc = {"timestamp": iso(t), "event": "Status", "Flags": flags}
    if lat is not None:
        doc.update(Flags=flags | elite.HAS_LAT_LONG, Latitude=lat, Longitude=lon, Altitude=alt, Heading=90,
                   BodyName=body, PlanetRadius=1000000.0)
    return json.dumps(doc)


def write_status(folder: Path, text: str, n: int) -> None:
    p = folder / "Status.json"
    p.write_text(text, encoding="utf-8")
    stamp = time.time() + n
    os.utime(p, (stamp, stamp))


def test_the_tangent_plane_is_metres_east_up_north():
    r = 1_000_000.0
    x, y, z = elite.tangent(0.0, 1.0, 0.0, (0.0, 0.0), r)
    assert x == pytest.approx(r * math.radians(1.0), rel=1e-4) and z == pytest.approx(0, abs=1e-6)
    x, y, z = elite.tangent(1.0, 0.0, 50.0, (0.0, 0.0), r)
    assert z == pytest.approx(r * math.radians(1.0), rel=1e-3) and x == pytest.approx(0, abs=1e-6)
    assert elite.tangent(0.0, 0.0, 50.0, (0.0, 0.0), r)[1] == pytest.approx(50.0)


def test_attributes_take_the_localised_text_and_leave_out_lists():
    e = {"timestamp": "x", "event": "FSDJump", "SystemEconomy": "$economy_Industrial;", "SystemEconomy_Localised":
         "Industrial", "StarPos": [1, 2, 3], "x": 5}
    assert elite.attributes(e) == {"SystemEconomy": "Industrial", "j_x": 5}


def test_the_adapter_records_a_sitting(tmp_path):
    folder = tmp_path / "Elite Dangerous"
    folder.mkdir()
    data = tmp_path / "data"
    ctx = Context(Settings(data=str(data)), data)
    a = replace(elite.ADAPTERS[0], folder=folder)
    a.start(ctx)
    t0 = time.time() - 30
    (folder / "Journal.2026-01-01T000000.01.log").write_text("\n".join(journal_lines(t0)) + "\n", encoding="utf-8")
    a.poll(ctx)
    write_status(folder, status(t0 + 20, 10.0, 20.0, 100.0), 1)
    a.poll(ctx)
    write_status(folder, status(t0 + 21, 10.0, 20.01, 120.0), 2)
    a.poll(ctx)
    write_status(folder, status(t0 + 22), 3)  # left the planet
    a.poll(ctx)
    a.stop(ctx)

    [session] = list((data / "sessions").glob("*/*"))
    meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
    st = meta["stats"]
    assert st["jumps"] == 1 and st["distance_ly"] == pytest.approx(5.025)
    assert st["deaths"] == 1 and st["kills"] == 1 and st["missions_completed"] == 1
    assert st["credits_earned"] == 200 + 3000 + 500  # the bounty counts when redeemed
    assert st["bodies_scanned"] == 1 and st["landings"] == 1 and st["bad_lines"] == 1
    assert meta["axes"]["up"] == "y" and meta["axes"]["flip_z"] is True

    rows = read_rows(session)
    ev = [r["event"] for r in rows]
    for name in ("Fileheader", "LoadGame", "FSDJump", "Died", "Shutdown"):
        assert name in ev  # every journal event, under its own name
    jump = next(r for r in rows if r["event"] == "jump")
    assert (jump["x"], jump["y"], jump["z"]) == (3.0, 0.5, 4.0) and jump["session"] == "galaxy"
    assert jump["player"] == "Test Pilot" and jump["system"] == "Test System B"
    spawn = [r for r in rows if r["event"] == "spawn"]
    assert [s["journal"] for s in spawn] == ["LoadGame", "Resurrect"]
    assert (spawn[0]["x"], spawn[0]["z"]) == (0.0, 0.0)  # placed at the Location after LoadGame
    death = next(r for r in rows if r["event"] == "death")
    assert death["killer"] == "Pirate" and death["x"] == 3.0
    raw = next(r for r in rows if r["event"] == "FSDJump")
    assert raw["SystemEconomy"] == "Industrial" and "x" not in raw

    surface = [r for r in rows if r.get("session") == "Test System B 1"]
    assert [r["event"] for r in surface] == ["touchdown", "liftoff", "surface_start", "surface", "surface", "surface_end"]
    first, second = surface[3], surface[4]
    # The touchdown made (10, 20) the body's origin; the second fix is 0.01 degree east.
    assert first["x"] == pytest.approx(0.0, abs=1e-3) and first["y"] == pytest.approx(100.0, abs=0.01)
    assert second["x"] == pytest.approx(1e6 * math.radians(0.01) * math.cos(math.radians(10)), rel=1e-3)
    assert second["lat"] == 10.0 and second["alt"] == 120.0

    t = pq.read_table(session / "heat.parquet")
    assert set(t.column("session").to_pylist()) == {"galaxy", "Test System B 1"}
    assert "jump" in t.column("event").to_pylist()


def test_lines_from_before_the_game_started_are_not_recorded_again(tmp_path, monkeypatch):
    folder = tmp_path / "ed"
    folder.mkdir()
    data = tmp_path / "data"
    old = folder / "Journal.2025-01-01T000000.01.log"
    t_old = time.time() - 7200
    old.write_text("\n".join(journal_lines(t_old)) + "\n", encoding="utf-8")
    started = time.time() + 60  # the file is older than this run of the game
    monkeypatch.setattr(elite, "game_started", lambda pid: started)
    ctx = Context(Settings(data=str(data)), data)
    a = replace(elite.ADAPTERS[0], folder=folder)
    a.start(ctx)
    a.poll(ctx)
    assert a._session is None
    with old.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": iso(started + 1), "event": "Music", "MusicTrack": "NoTrack"}) + "\n")
    a.poll(ctx)
    a.stop(ctx)
    [session] = list((data / "sessions").glob("*/*"))
    assert [r["event"] for r in read_rows(session)] == ["Music"]


# --------------------------------------------------------------------- local


def _replay(path: Path, root: Path) -> Session | None:
    box: dict[str, Session] = {}

    def session_for(ts: float) -> Session:
        if "s" not in box:
            box["s"] = Session.open(root, "Elite Dangerous", "elite", when=ts or None, axes=elite.ADAPTERS[0].axes)
        return box["s"]

    j = elite.Journal(session_for)
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            j.line(line)
    s = box.get("s")
    if s is not None:
        s.close()
    return s


@pytest.mark.local
def test_real_journals_replay_with_plausible_numbers(tmp_path):
    files = sorted(elite.journal_dir().glob("Journal.*.log"))
    if not files:
        pytest.skip("no Elite Dangerous journals on this machine")
    totals: dict[str, float] = {}
    for path in files:
        raw: dict[str, int] = {}
        dist = 0.0
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                raw[e["event"]] = raw.get(e["event"], 0) + 1
                if e["event"] == "FSDJump":
                    dist += e.get("JumpDist", 0.0)
        s = _replay(path, tmp_path)
        if s is None:
            continue
        stats = s.stats
        assert stats.get("jumps", 0) == raw.get("FSDJump", 0), path.name
        assert stats.get("deaths", 0) == raw.get("Died", 0), path.name
        assert stats.get("distance_ly", 0.0) == pytest.approx(dist), path.name
        assert stats.get("bad_lines", 0) == 0, path.name
        # Every journal line became a row under its own name.
        for name, n in raw.items():
            assert s.counts.get(name, 0) == n, (path.name, name)
        assert s.counts.get("jump", 0) == raw.get("FSDJump", 0) + raw.get("CarrierJump", 0)
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                totals[k] = totals.get(k, 0) + v
    assert totals.get("jumps", 0) > 0
    assert 0 < totals["distance_ly"] < totals["jumps"] * 100  # no FSD jumps 100 ly
