"""Replays: the folder watch, and CS2 demos through the repository's own converter."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pytest

from heat3d_watch.adapters.base import Context
from heat3d_watch.adapters.replays import SETTLE_SECONDS, ReplayAdapter, record_match
from heat3d_watch.config import Settings
from heat3d_watch.session import Session

#: A folder of real CS2 demos to check the conversion on, where one is at hand.
DEMOS = Path(os.environ.get("HEAT3D_CS2_DEMOS", "")) if os.environ.get("HEAT3D_CS2_DEMOS") else None


class _Counting(ReplayAdapter):
    def __init__(self, folder: Path):
        super().__init__(id="test-replays", game="Test", patterns=("*.rep",))
        self.folder = folder
        self.converted: list[str] = []

    def folders(self, ctx):
        return [self.folder]

    def convert(self, path, ctx):
        self.converted.append(path.name)


def test_old_replays_are_noted_and_new_ones_converted_once_written(tmp_path, monkeypatch):
    folder = tmp_path / "replays"
    folder.mkdir()
    (folder / "old.rep").write_bytes(b"x")
    a = _Counting(folder)
    ctx = Context(Settings(data=str(tmp_path / "data")), tmp_path / "data")
    clock = [1000.0]
    monkeypatch.setattr("heat3d_watch.adapters.replays.time.time", lambda: clock[0])
    a.start(ctx)
    try:
        new = folder / "new.rep"
        new.write_bytes(b"ab")
        a.poll(ctx)  # first sight: noted
        new.write_bytes(b"abc")  # still being written
        clock[0] += SETTLE_SECONDS + 1
        a.poll(ctx)  # the size changed: the clock starts again
        clock[0] += SETTLE_SECONDS + 1
        a.poll(ctx)  # unchanged long enough: queued
        deadline = time.time() + 5
        while not a.converted and time.time() < deadline:
            time.sleep(0.05)
        assert a.converted == ["new.rep"]
        a.poll(ctx)
        clock[0] += SETTLE_SECONDS + 1
        a.poll(ctx)
        time.sleep(0.3)
        assert a.converted == ["new.rep"]  # neither again nor the old one
    finally:
        a.stop(ctx)
    # A restart remembers what was converted.
    b = _Counting(folder)
    b.start(ctx)
    b.poll(ctx)
    b.stop(ctx)
    assert b.converted == []


def test_a_match_summary_gives_the_signed_in_player_s_line(tmp_path):
    s = Session.open(tmp_path, "Counter-Strike 2", "cs2-demos")
    match = {
        "match": "m1",
        "map": "a map",
        "teams": ["Blue", "Red"],
        "score": {"Blue": 13, "Red": 9},
        "rounds": [{}] * 22,
        "players": [
            {"name": "me", "steamid": "76561190000000001", "team": "Blue", "kills": 21, "deaths": 14, "adr": 88.5},
            {"name": "other", "steamid": "76561190000000002", "team": "Red", "kills": 30, "deaths": 10},
        ],
    }
    record_match(s, match, {"76561190000000001"})
    assert s.stats["kills"] == 21 and s.stats["deaths"] == 14 and s.stats["adr"] == 88.5
    assert s.stats["won"] == 1 and s.stats["rounds"] == 22 and s.stats["score.Red"] == 9
    s.close()


@pytest.mark.local
def test_a_real_demo_converts_into_a_session(tmp_path):
    demos = sorted(DEMOS.glob("*.dem")) if DEMOS and DEMOS.is_dir() else []
    if not demos:
        pytest.skip("set HEAT3D_CS2_DEMOS to a folder of CS2 demos")
    from heat3d_watch.adapters.replays import ADAPTERS

    a = next(x for x in ADAPTERS if x.id == "cs2-demos")
    if a.missing():
        pytest.skip(a.missing())
    # A whole match: parts of one ("-p2.dem") can hold no complete round of their own.
    whole = [d for d in demos if not re.search(r"[-_]p\d+$", d.stem)]
    demo = min(whole or demos, key=lambda p: p.stat().st_size)
    ctx = Context(Settings(data=str(tmp_path)), tmp_path)
    a.convert(demo, ctx)
    sessions = list((tmp_path / "sessions").rglob("meta.json"))
    assert len(sessions) == 1
    meta = json.loads(sessions[0].read_text(encoding="utf-8"))
    assert meta["stats"]["rounds"] > 0
    assert meta["events"].get("kill", 0) > 0 and meta["positions"] > 1000
    assert list(sessions[0].parent.glob("*/*.parquet"))


def _osr_string(text: str) -> bytes:
    if not text:
        return b"\x00"
    raw = text.encode("utf-8")
    n, out = len(raw), bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            break
    return b"\x0b" + bytes(out) + raw


def _osr(frames: str, player: str = "someone") -> bytes:
    """An .osr built field by field to the layout on the osu! wiki."""
    import lzma
    import struct

    data = lzma.compress(frames.encode("ascii"), format=lzma.FORMAT_ALONE)
    ticks = int((1_700_000_000 + 62135596800) * 1e7)
    return (
        struct.pack("<Bi", 0, 20240101)
        + _osr_string("b" * 32)
        + _osr_string(player)
        + _osr_string("r" * 32)
        + struct.pack("<6h", 90, 8, 2, 30, 4, 1)
        + struct.pack("<ihBi", 1234567, 250, 0, 72)
        + _osr_string("0/1,1000/0.9")
        + struct.pack("<qi", ticks, len(data))
        + data
        + struct.pack("<q", 0)
    )


def test_an_osu_replay_is_read_to_the_published_format(tmp_path):
    from heat3d_watch.adapters.replays import parse_osr

    r = parse_osr(_osr("0|256|192|0,16|260|190|1,17|270.5|180.25|5,-12345|0|0|99"))
    assert r["player"] == "someone" and r["score"] == 1234567 and r["max_combo"] == 250
    assert (r["count_300"], r["count_100"], r["count_50"], r["misses"]) == (90, 8, 2, 1)
    assert r["mods"] == 72 and r["played_at"] == pytest.approx(1_700_000_000)
    # Times accumulate; the seed frame is not a frame.
    assert r["frames"] == [(0, 256.0, 192.0, 0), (16, 260.0, 190.0, 1), (33, 270.5, 180.25, 5)]
    assert r["accuracy"] == pytest.approx((300 * 90 + 100 * 8 + 50 * 2) / (300 * 101))


def test_an_osu_replay_becomes_a_session_with_its_path(tmp_path):
    from heat3d_watch.adapters.replays import ADAPTERS

    a = next(x for x in ADAPTERS if x.id == "osu-replays")
    replay = tmp_path / "play.osr"
    replay.write_bytes(_osr("0|10|20|0,16|30|40|1,16|50|60|0"))
    ctx = Context(Settings(data=str(tmp_path / "data")), tmp_path / "data")
    a.convert(replay, ctx)
    meta = json.loads(next((tmp_path / "data" / "sessions").rglob("meta.json")).read_text(encoding="utf-8"))
    assert meta["events"] == {"session_start": 1, "position": 1, "finish": 1}
    assert meta["stats"]["score"] == 1234567 and meta["stats"]["full_combo"] == 0
    import pyarrow.parquet as pq

    t = pq.read_table(next((tmp_path / "data" / "sessions").rglob("heat.parquet")))
    assert t.column("x").to_pylist() == [10.0, 30.0, 50.0]
    assert t.column("z").to_pylist() == [20.0, 40.0, 60.0]
    assert t.column("time").to_pylist() == [0.0, 0.016, 0.032]
