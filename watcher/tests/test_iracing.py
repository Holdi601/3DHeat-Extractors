"""
iRacing's live telemetry: a shared-memory block built here the way the sim
lays it out (the irsdk header, variable headers, the session info YAML and
round-robin data lines), updated line by line while the adapter polls it.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import sys
import time

import pytest

from heat3d_watch.adapters import iracing
from heat3d_watch.adapters.base import Context
from heat3d_watch.config import Settings
from heat3d_watch.session import read_rows

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="named shared memory is Windows only")

#: irsdk_VarType codes and sizes.
CODES = {0: ("c", 1), 1: ("?", 1), 2: ("i", 4), 3: ("I", 4), 4: ("f", 4), 5: ("d", 8)}
VARS = [
    ("SessionTime", 5, 1, "s"), ("SessionNum", 2, 1, ""), ("SessionState", 2, 1, ""), ("SessionUniqueID", 2, 1, ""),
    ("IsOnTrack", 1, 1, ""), ("IsReplayPlaying", 1, 1, ""), ("Lap", 2, 1, ""), ("LapCompleted", 2, 1, ""),
    ("LapDistPct", 4, 1, "%"), ("LapDist", 4, 1, "m"), ("LapLastLapTime", 4, 1, "s"), ("LapBestLapTime", 4, 1, "s"),
    ("CarIdxLapDistPct", 4, 64, "%"),  # an array the adapter does not read, between the others
    ("Speed", 4, 1, "m/s"), ("Gear", 2, 1, ""), ("Throttle", 4, 1, "%"), ("Brake", 4, 1, "%"), ("RPM", 4, 1, "revs/min"),
    ("OnPitRoad", 1, 1, ""), ("PlayerTrackSurface", 2, 1, ""), ("PlayerCarPosition", 2, 1, ""),
    ("PlayerCarClassPosition", 2, 1, ""), ("PlayerCarMyIncidentCount", 2, 1, ""),
]
YAML = """---
WeekendInfo:
 TrackDisplayName: Test Raceway
 TrackConfigName: Full
SessionInfo:
 Sessions:
 - SessionNum: 0
   SessionType: Race
DriverInfo:
 DriverCarIdx: 1
 Drivers:
 - CarIdx: 0
   UserName: Someone Else
   CarScreenName: Car A
 - CarIdx: 1
   UserName: Test Driver
   CarScreenName: Car B
...
"""


class FakeSim:
    """The memory block, as iRacing writes it: three lines, the newest with the highest tick."""

    BUFS = 3

    def __init__(self, tag: str):
        self.offsets = {}
        at = 0
        for name, t, n, _ in VARS:
            size = CODES[t][1]
            at = (at + size - 1) // size * size
            self.offsets[name] = at
            at += size * n
        self.buf_len = (at + 15) // 16 * 16
        self.var_header_offset = 144
        self.info_offset = self.var_header_offset + 144 * len(VARS)
        info = YAML.encode("cp1252")
        self.info_len = len(info) + 1
        self.line0 = (self.info_offset + self.info_len + 15) // 16 * 16
        self.size = self.line0 + self.BUFS * self.buf_len
        self.mem = mmap.mmap(-1, self.size, tagname=tag)
        for i, (name, t, n, unit) in enumerate(VARS):
            struct.pack_into("<iii?3x32s64s32s", self.mem, self.var_header_offset + 144 * i, t, self.offsets[name], n,
                             False, name.encode(), b"", unit.encode())
        self.mem[self.info_offset : self.info_offset + len(info)] = info
        self.tick = 0
        self.header(status=0)

    def header(self, *, status=1, update=1):
        struct.pack_into("<10i", self.mem, 0, 2, status, 60, update, self.info_len, self.info_offset, len(VARS),
                         self.var_header_offset, self.BUFS, self.buf_len)
        for b in range(self.BUFS):
            struct.pack_into("<ii", self.mem, 48 + 16 * b, 0, self.line0 + b * self.buf_len)

    def write(self, values: dict):
        self.tick += 1
        b = self.tick % self.BUFS
        line = bytearray(self.buf_len)
        for name, t, n, _ in VARS:
            if name in values:
                struct.pack_into("<" + CODES[t][0], line, self.offsets[name], values[name])
        at = self.line0 + b * self.buf_len
        self.mem[at : at + self.buf_len] = bytes(line)
        struct.pack_into("<ii", self.mem, 48 + 16 * b, self.tick, at)
        struct.pack_into("<i", self.mem, 4, 1)

    def close(self):
        self.mem.close()


def test_a_race_its_laps_incidents_pit_stop_and_finish(tmp_path):
    tag = f"Local\\heat3d_test_irsdk_{os.getpid()}"
    sim = FakeSim(tag)
    folder = tmp_path / "telemetry"
    folder.mkdir()
    (folder / "older.ibt").write_bytes(b"old")
    data = tmp_path / "data"
    ctx = Context(Settings(data=str(data)), data)
    a = iracing.IRacingAdapter(id="iracing-test", game="iRacing", exes=("x.exe",), tagname=tag,
                               telemetry_folder=str(folder), hz=20.0)
    try:
        a.start(ctx)
        a.poll(ctx)  # not connected yet: nothing
        sim.header(status=1)
        for k in range(60):
            on = k >= 2
            v = {
                "SessionTime": 100.0 + k * 0.1, "SessionNum": 0, "SessionUniqueID": 77,
                "SessionState": 5 if k >= 40 else 4, "IsOnTrack": on, "IsReplayPlaying": False,
                "Lap": 1 + (k >= 20) + (k >= 45), "LapCompleted": (k >= 20) + (k >= 45),
                "LapDistPct": (k % 20) / 20, "LapDist": (k % 20) * 100.0,
                "LapLastLapTime": 94.0 if k >= 45 else (95.125 if k >= 20 else 0.0),
                "LapBestLapTime": 94.0 if k >= 45 else (95.125 if k >= 20 else -1.0),
                "Speed": 50.0 if on else 0.0, "Gear": 4, "Throttle": 1.0, "Brake": 0.0, "RPM": 7000.0,
                "OnPitRoad": 30 <= k < 35, "PlayerTrackSurface": 3, "PlayerCarPosition": 3,
                "PlayerCarClassPosition": 2, "PlayerCarMyIncidentCount": 2 if k >= 25 else 0,
            }
            sim.write(v)
            a.poll(ctx)
            a.poll(ctx)  # the same tick again: nothing new
            time.sleep(0.01)
        (folder / "a.ibt").write_bytes(b"IBT" * 10)
        a.stop(ctx)
    finally:
        sim.close()

    (session,) = [p for p in (data / "sessions").glob("*/*")]
    rows = read_rows(session)
    by = lambda name: [r for r in rows if r["event"] == name]  # noqa: E731
    (start,) = by("session_start")
    assert start["track"] == "Test Raceway" and start["car"] == "Car B" and start["session_type"] == "Race"
    assert start["player"] == "Test Driver"
    assert [r["lap_time"] for r in by("lap")] == [95.125, 94.0]
    (incident,) = by("incident")
    assert incident["points"] == 2 and incident["incidents"] == 2
    assert len(by("pit_in")) == 1 and len(by("pit_out")) == 1
    assert by("chequered_flag") and by("finish")[0]["race_position"] == 3
    assert len(by("spawn")) == 1
    driving = by("driving")
    assert driving and all(r["speed_kmh"] == 180.0 and r["gear"] == 4 for r in driving)
    assert "x" not in driving[0]  # iRacing gives no position live
    files = by("telemetry_file")
    assert [os.path.basename(r["path"]) for r in files] == ["a.ibt"]

    meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
    s = meta["stats"]
    assert s["laps"] == 2 and s["best_lap"] == 94.0 and s["incidents"] == 2 and s["race_position"] == 3
    assert s["pit_stops"] == 1 and s["top_speed_kmh"] == 180.0
    assert s["distance_km"] == pytest.approx(57 * 5.0 / 1000, rel=0.05)
    assert not (session / "heat.parquet").exists()


def test_the_session_info_yaml_is_read_without_a_yaml_library():
    info = iracing.session_yaml(YAML)
    assert info["car_idx"] == 1 and info["track"] == "Test Raceway" and info["track_config"] == "Full"
    assert info["drivers"][1]["UserName"] == "Test Driver" and info["sessions"] == {0: "Race"}


def test_no_sim_no_mapping_made(tmp_path):
    tag = f"Local\\heat3d_absent_irsdk_{os.getpid()}"
    data = tmp_path / "data"
    ctx = Context(Settings(data=str(data)), data)
    a = iracing.IRacingAdapter(id="iracing-test", game="iRacing", exes=("x.exe",), tagname=tag,
                               telemetry_folder=str(tmp_path / "none"))
    a.start(ctx)
    a.poll(ctx)
    a.stop(ctx)
    from heat3d_watch.adapters.simracing_shm import SharedView

    assert SharedView.open(tag) is None
    assert not (data / "sessions").exists() or not list((data / "sessions").glob("*/*"))
