"""The watcher's own machinery: sessions, the statistics, the loop, the update check, the memory fences."""

from __future__ import annotations

import json
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from heat3d_watch import db
from heat3d_watch.adapters.base import Adapter, Context
from heat3d_watch.config import Settings
from heat3d_watch.memory import Profile, find, parse_pattern, rip_target
from heat3d_watch.session import Axes, Session, read_rows
from heat3d_watch.watcher import Watcher

PACKAGE = Path(__file__).resolve().parents[1] / "heat3d_watch"


def test_a_session_streams_lines_and_ends_as_a_table(tmp_path):
    s = Session.open(tmp_path, "Some Game", "some-game", axes=Axes(up="z", units="inches"), source="a test")
    s.event("spawn", x=0, y=0, z=0, player="me", session="r1")
    s.position(1, 2, 3, player="me", session="r1", speed=4.5)
    s.event("death", x=5, y=6, z=7, player="me", session="r1", weapon="knife")
    s.event("chat", text="no position here")
    s.stat("kills", 2, "add")
    s.stat("kills", 3, "add")
    s.stat("best_lap", 91.2, "min")
    s.stat("best_lap", 88.0, "min")
    # Written as it happens: readable before the session ends.
    assert [r["event"] for r in read_rows(s.folder)] == ["spawn", "position", "death", "chat"]
    table = s.close()
    assert table is not None and table.name == "heat.parquet"
    t = pq.read_table(table)
    # Only rows that can be drawn; the chat line stays in events.jsonl.
    assert t.column("event").to_pylist() == ["spawn", "position", "death"]
    assert t.column("x").to_pylist() == [0.0, 1.0, 5.0]
    assert t.column("weapon").to_pylist() == ["", "", "knife"]
    hints = json.loads(t.schema.metadata[b"heat3d"])
    assert hints["axes"]["up"] == "z"
    assert hints["columns"]["personId"] == "player"
    assert hints["columns"]["sessionId"] == "session"
    assert "event" in hints["columns"]["extra"]
    meta = json.loads((s.folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["stats"] == {"kills": 5, "best_lap": 88.0}
    assert meta["events"]["position"] == 1 and meta["positions"] == 3
    assert meta["axes"]["units"] == "inches"


def test_a_line_cut_short_by_a_crash_loses_only_itself(tmp_path):
    s = Session.open(tmp_path, "G", "g")
    s.position(1, 1, 1)
    s._file.write('{"time": 1, "event": "posi')  # a write the power cut stopped
    s._file.flush()
    assert len(read_rows(s.folder)) == 1


def test_non_finite_numbers_are_not_positions(tmp_path):
    s = Session.open(tmp_path, "G", "g")
    s.event("position", x=float("nan"), y=0, z=0)
    s.event("position", x=1, y=0, z=0, speed=float("inf"))
    s.close()
    assert s.positions == 1
    assert "speed" not in read_rows(s.folder)[1]


def test_the_statistics_sum_per_game(tmp_path):
    con = db.connect(tmp_path)
    for kills in (3, 5):
        s = Session.open(tmp_path, "Shooter", "shooter")
        s.event("kill", x=0, y=0, z=0)
        s.stat("kills", kills)
        s.close()
        db.record(con, s)
    games = db.per_game(con)
    assert games[0][0] == "Shooter" and games[0][1] == 2
    assert db.event_totals(con, "Shooter") == [("kill", 2)]
    key, n, total, best, avg = db.stat_totals(con, "Shooter")[0]
    assert (key, n, total, best, avg) == ("kills", 2, 8.0, 5.0, 4.0)
    # Recording a session again replaces it rather than counting it twice.
    db.record(con, s)
    assert db.per_game(con)[0][1] == 2


@dataclass
class _Fake(Adapter):
    polls: int = 0

    def start(self, ctx: Context) -> None:
        self.session = ctx.open(self)

    def poll(self, ctx: Context) -> None:
        self.polls += 1
        self.session.position(self.polls, 0, 0, player="me")

    def stop(self, ctx: Context) -> None:
        ctx.close(self.session)


def test_the_loop_starts_a_game_s_adapter_and_counts_its_session(tmp_path, monkeypatch):
    settings = Settings(data=str(tmp_path))
    fake = _Fake(id="fake", game="Fake Game", exes=("fakegame.exe",), interval=0.01)
    w = Watcher(settings)
    w.adapters = {"fake": fake}
    running = {"fakegame.exe": (1234, None)}
    monkeypatch.setattr("heat3d_watch.watcher.running_processes", lambda: running)
    w.step()
    assert "fake" in w.active
    time.sleep(0.2)
    running.clear()
    w.step()
    assert "fake" not in w.active
    con = db.connect(tmp_path)
    (game, n, *_rest) = db.per_game(con)[0]
    assert game == "Fake Game" and n == 1
    assert fake.polls > 2


def test_a_session_with_nothing_in_it_leaves_nothing_behind(tmp_path):
    ctx = Context(Settings(data=str(tmp_path)), tmp_path)
    a = _Fake(id="quiet", game="Quiet")
    s = ctx.open(a)
    ctx.close(s)
    assert not s.folder.exists()


def test_memory_reading_cannot_write():
    """No call that writes to, allocates in, or runs code in another process - anywhere in the package."""
    forbidden = re.compile(
        r"WriteProcessMemory|VirtualAllocEx|VirtualProtectEx|CreateRemoteThread|NtWriteVirtualMemory|"
        r"PROCESS_ALL_ACCESS|PROCESS_VM_WRITE|PROCESS_VM_OPERATION|PROCESS_CREATE_THREAD|SeDebugPrivilege|"
        r"\bpymem\b|LoadLibrary|SetWindowsHookEx|DebugActiveProcess"
    )
    hits = []
    for path in PACKAGE.rglob("*.py"):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if forbidden.search(line):
                hits.append(f"{path.name}:{n}: {line.strip()}")
    assert hits == []
    from heat3d_watch import memory

    assert memory.READ_ACCESS == 0x0010 | 0x1000


def test_patterns_and_rip_relative_addresses():
    image = bytes(16) + bytes([0x48, 0x8B, 0x05]) + struct.pack("<i", 0x100) + bytes(16)
    rx = parse_pattern("48 8B ?? ?")
    assert rx.fullmatch(bytes([0x48, 0x8B, 0x00, 0x0A]))  # any byte, a newline included
    assert not rx.fullmatch(bytes([0x48, 0x8C, 0x00, 0x00]))
    at = find(image, "48 8B 05 ?? ?? ?? ??")
    assert at == 16
    # The address is the end of the 7-byte instruction plus the displacement.
    assert rip_target(image, at, 3, 7) == 16 + 7 + 0x100
    with pytest.raises(Exception):
        find(image + image, "48 8B 05")  # twice is as bad as never


def test_a_profile_must_say_whether_the_game_goes_online(tmp_path):
    base = {
        "schema": "heat3d.memprofile/1",
        "id": "g",
        "game": "G",
        "exe": "g.exe",
        "versions": [{"label": "1.0", "size": 3}],
        "anchors": {},
        "values": {},
    }
    p = tmp_path / "g.json"
    p.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="online"):
        Profile.load(p)
    p.write_text(json.dumps({**base, "online": False}), encoding="utf-8")
    prof = Profile.load(p)
    exe = tmp_path / "g.exe"
    exe.write_bytes(b"abc")
    assert prof.version_of(exe) == "1.0"
    exe.write_bytes(b"abcd")
    assert prof.version_of(exe) is None


def test_memory_is_refused_near_an_anticheat(tmp_path, monkeypatch):
    from heat3d_watch import safety

    game = tmp_path / "Game"
    (game / "EasyAntiCheat").mkdir(parents=True)
    exe = game / "game.exe"
    exe.write_bytes(b"x")
    a = Adapter(id="m", game="G", kind="memory", exes=("game.exe",), multiplayer=False)
    monkeypatch.setattr(safety, "running_anticheat", lambda: [])
    assert "easyanticheat" in (safety.memory_allowed(a, 1, exe) or "")
    (game / "EasyAntiCheat").rmdir()
    assert safety.memory_allowed(a, 1, exe) is None
    monkeypatch.setattr(safety, "running_anticheat", lambda: ["vgk"])
    assert "vgk" in (safety.memory_allowed(a, 1, exe) or "")
    online = Adapter(id="o", game="O", kind="memory", exes=("o.exe",), multiplayer=True)
    monkeypatch.setattr(safety, "running_anticheat", lambda: [])
    assert "online" in (safety.memory_allowed(online, 1, exe) or "")


def test_versions_compare_as_numbers():
    import heat3d_update as up

    assert up.is_newer("v1.10.0", "1.9.9")
    assert not up.is_newer("1.1.0", "1.1.0")
    assert up.is_newer("1.1", "1.0.9")
    assert not up.is_newer("1.0.0-beta", "1.0.0")
    assert up.parse("v2.0.1") == (2, 0, 1)


def test_two_games_in_the_same_second_are_two_sessions(tmp_path):
    con = db.connect(tmp_path)
    when = 1_700_000_000.0
    for game in ("Game A", "Game B"):
        s = Session.open(tmp_path, game, "steam-stats", when=when)
        s.stat("x", 1)
        s.event("snapshot")
        s.close()
        db.record(con, s)
    assert sorted(g[0] for g in db.per_game(con)) == ["Game A", "Game B"]


def test_a_killed_watcher_s_session_is_finished_on_the_next_start(tmp_path):
    from heat3d_watch.session import recover

    s = Session.open(tmp_path, "Some Game", "some-game")
    s.position(1, 2, 3, player="me")
    s.stat("kills", 4)
    s._flush(every=0)  # what the half-minute flush would have saved
    s._file.close()  # killed: no close()
    s._file = None
    finished = recover(tmp_path)
    assert [f.id for f in finished] == [s.id]
    meta = json.loads((s.folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["ended"] and meta["recovered"] and meta["stats"]["kills"] == 4
    assert (s.folder / "heat.parquet").exists()
    assert recover(tmp_path) == []  # once only
