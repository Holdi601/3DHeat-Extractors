"""Hearthstone: log.config set-up and synthetic Power.log lines in the game's format."""

from __future__ import annotations

import json
from dataclasses import replace

from heat3d_watch.adapters import hearthstone as hs
from heat3d_watch.adapters.base import Context
from heat3d_watch.config import Settings
from heat3d_watch.session import read_rows

GS = "GameState.DebugPrintPower() -"
PT = "PowerTaskList.DebugPrintPower() -"


def game(won: bool, clock: str = "20:15") -> list[str]:
    me, them = "Tester#1234", "UNKNOWN HUMAN PLAYER"
    out = [
        f"{GS} CREATE_GAME",
        f"{GS}     GameEntity EntityID=1",
        f"{GS}         tag=TURN value=1",
        f"{GS}     Player EntityID=2 PlayerID=1 GameAccountId=[hi=144115198130930503 lo=11111111]",
        f"{GS}     Player EntityID=3 PlayerID=2 GameAccountId=[hi=144115198130930503 lo=22222222]",
        "GameState.DebugPrintGame() - PlayerID=1, PlayerName=" + me,
        "GameState.DebugPrintGame() - PlayerID=2, PlayerName=" + them,
        f"{GS}     FULL_ENTITY - Updating [entityName=UNKNOWN ENTITY [cardType=INVALID] id=40 zone=HAND zonePos=0 "
        "cardId= player=2] CardID=",
        f"{GS}     FULL_ENTITY - Updating [entityName=Fireball id=4 zone=HAND zonePos=0 cardId=CS2_029 player=1] "
        "CardID=CS2_029",
        f"{PT} TAG_CHANGE Entity=GameEntity tag=TURN value=2 ",  # the repeat of the same change: skipped
        f"{GS} TAG_CHANGE Entity=GameEntity tag=TURN value=2 ",
        f"{GS} TAG_CHANGE Entity=GameEntity tag=TURN value=3 ",
        f"{GS} TAG_CHANGE Entity=GameEntity tag=TURN value=4 ",
        f"{GS} TAG_CHANGE Entity={me} tag=PLAYSTATE value={'WON' if won else 'LOST'} ",
        f"{GS} TAG_CHANGE Entity={them} tag=PLAYSTATE value={'LOST' if won else 'WON'} ",
        f"{GS} TAG_CHANGE Entity=GameEntity tag=STATE value=COMPLETE ",
    ]
    return [f"D {clock}:{i:02d}.1234567 {line}" for i, line in enumerate(out)]


def test_setup_adds_power_logging_and_keeps_the_rest(tmp_path):
    cfg = tmp_path / "log.config"
    cfg.write_text("[Achievements]\nLogLevel=1\n\n[Power]\nLogLevel=2\nFilePrinting=false\n", encoding="utf-8")
    a = replace(hs.ADAPTERS[0], config=cfg)
    ctx = Context(Settings(data=str(tmp_path / "d")), tmp_path / "d")
    plan = a.setup(ctx, apply=False)
    assert plan and cfg.read_text(encoding="utf-8").count("FilePrinting=false") == 1  # nothing written yet
    done = a.setup(ctx, apply=True)
    assert any("backup" in d for d in done)
    text = cfg.read_text(encoding="utf-8")
    assert "[Achievements]\nLogLevel=1" in text
    power = text.split("[Power]")[1]
    assert "LogLevel=1" in power and "FilePrinting=true" in power and "Verbose=true" in power
    backup = cfg.with_name("log.config.heat3d-backup")
    assert "FilePrinting=false" in backup.read_text(encoding="utf-8")
    assert a.setup(ctx, apply=True) == []  # idempotent
    cfg.write_text("", encoding="utf-8")
    a.setup(ctx, apply=True)
    assert "FilePrinting=false" in backup.read_text(encoding="utf-8")  # the first backup is never overwritten


def test_setup_creates_the_file_when_there_is_none(tmp_path):
    cfg = tmp_path / "Blizzard" / "Hearthstone" / "log.config"
    a = replace(hs.ADAPTERS[0], config=cfg)
    ctx = Context(Settings(data=str(tmp_path / "d")), tmp_path / "d")
    a.setup(ctx, apply=True)
    assert cfg.read_text(encoding="utf-8").startswith("[Power]\nLogLevel=1")
    assert not cfg.with_name("log.config.heat3d-backup").exists()


def test_the_adapter_records_games_results_and_turns(tmp_path):
    install = tmp_path / "Hearthstone"
    logs = install / "Logs" / "Hearthstone_2026_01_01_20_00_00"
    logs.mkdir(parents=True)
    data = tmp_path / "data"
    ctx = Context(Settings(data=str(data)), data, exe=install / "Hearthstone.exe")
    a = replace(hs.ADAPTERS[0])
    a.start(ctx)
    power = logs / "Power.log"
    power.write_text("\n".join(game(True, "00:00") + game(False, "00:01")) + "\n", encoding="utf-8")
    a.poll(ctx)
    a.stop(ctx)

    [session] = list((data / "sessions").glob("*/*"))
    st = json.loads((session / "meta.json").read_text(encoding="utf-8"))["stats"]
    assert st["games"] == 2 and st["wins"] == 1 and st["losses"] == 1
    assert st["turns"] == 8 and st["longest_game_turns"] == 4
    rows = read_rows(session)
    ends = [r for r in rows if r["event"] == "game_end"]
    assert [e["result"] for e in ends] == ["won", "lost"]
    assert ends[0]["player"] == "Tester#1234" and ends[0]["session"] == "game-1" and ends[1]["session"] == "game-2"
    assert sum(1 for r in rows if r["event"] == "turn") == 6


def test_the_ai_is_never_the_player():
    log = hs.PowerLog(lambda ts: None)
    log.entity_player = {2: 1, 3: 2}
    log.ai = {2}
    assert log.me() == 1


def test_clock_times_are_today_or_yesterday():
    import time as _t

    now = _t.mktime((2026, 1, 2, 0, 0, 30, 0, 0, -1))
    assert hs.clock_time("00:00:10.5", now) == now - 20 + 0.5
    assert hs.clock_time("23:59:50", now) == now - 40  # just before midnight: yesterday
