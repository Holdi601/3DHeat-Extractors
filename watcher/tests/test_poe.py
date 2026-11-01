"""Path of Exile: synthetic Client.txt lines in the game's format."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

from heat3d_watch.adapters import poe
from heat3d_watch.adapters.base import Context
from heat3d_watch.config import Settings
from heat3d_watch.session import read_rows


def stamp(ts: float) -> str:
    return time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(ts))


def lines(t0: float) -> list[str]:
    msgs = [
        "[INFO Client 4242] : You have entered Test Town.",
        '[DEBUG Client 4242] Generating level 2 area "1_1_2" with seed 12345',
        "[INFO Client 4242] : You have entered The Test Coast.",
        "[INFO Client 4242] : Tester (Witch) is now level 2",
        "[INFO Client 4242] : Tester (Witch) is now level 3",
        "[INFO Client 4242] : Tester has been slain.",
        "[INFO Client 4242] : Friend has joined the area.",
        "[INFO Client 4242] @From <GUILD> Buyer: Hi, I would like to buy your Test Ring listed for 5 chaos",
        "[INFO Client 4242] @To Buyer: sure, inviting",
        "[INFO Client 4242] : Trade accepted.",
        "[INFO Client 4242] : You have entered Test Town.",
        "[WARN Client 4242] something the game warns about",
    ]
    return [f"{stamp(t0 + i)} {123456 + i} cff94598 {m}" for i, m in enumerate(msgs)]


def test_the_adapter_records_zones_levels_deaths_and_trades(tmp_path):
    install = tmp_path / "Path of Exile 2"
    (install / "logs").mkdir(parents=True)
    exe = install / "PathOfExileSteam.exe"
    data = tmp_path / "data"
    ctx = Context(Settings(data=str(data)), data, exe=exe)
    a = replace(poe.ADAPTERS[0])
    a.start(ctx)
    log = install / "logs" / "Client.txt"
    log.write_text("\n".join(lines(time.time())) + "\n", encoding="utf-8")
    a.poll(ctx)
    a.stop(ctx)

    [session] = list((data / "sessions").glob("*/*"))
    assert session.parent.name == "Path-of-Exile-2"
    meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
    st = meta["stats"]
    assert st["zones_entered"] == 3 and st["zones_distinct"] == 2
    assert st["levels_gained"] == 2 and st["level"] == 3 and st["character_class"] == "Witch"
    assert st["deaths"] == 1 and st["whispers"] == 2 and st["trade_requests_received"] == 1 and st["trades"] == 1
    rows = read_rows(session)
    zone = [r for r in rows if r["event"] == "zone"]
    assert zone[1]["zone"] == "The Test Coast" and zone[1]["area_level"] == 2
    assert next(r for r in rows if r["event"] == "death")["player"] == "Tester"
    assert next(r for r in rows if r["event"] == "player_joined")["player"] == "Friend"
    assert next(r for r in rows if r["event"] == "whisper_in")["player"] == "Buyer"
    assert not (session / "heat.parquet").exists()  # nothing with a position


def test_the_log_of_earlier_launches_is_not_recorded_again(tmp_path, monkeypatch):
    install = tmp_path / "Path of Exile"
    (install / "logs").mkdir(parents=True)
    log = install / "logs" / "Client.txt"
    started = time.time() + 60
    old = lines(time.time() - 7200)
    new = lines(started + 1)[:1]
    log.write_text("\n".join(old + new) + "\n", encoding="utf-8")
    os.utime(log, (started + 2, started + 2))  # written to since the game started: by its line times
    monkeypatch.setattr(poe, "game_started", lambda pid: started)
    data = tmp_path / "data"
    ctx = Context(Settings(data=str(data)), data, exe=install / "PathOfExile.exe")
    a = replace(poe.ADAPTERS[0])
    a.start(ctx)
    a.poll(ctx)
    a.stop(ctx)
    [session] = list((data / "sessions").glob("*/*"))
    assert session.parent.name == "Path-of-Exile"
    assert [r["event"] for r in read_rows(session)] == ["zone"]


def test_without_a_running_exe_the_usual_installs_are_searched():
    found = poe.candidates(None)
    assert any(p.parts[-3] == "Path of Exile" for p in found)
    assert poe.candidates(Path(r"D:\Games\Path of Exile\PathOfExile.exe")) == [
        Path(r"D:\Games\Path of Exile\logs\Client.txt")
    ]
