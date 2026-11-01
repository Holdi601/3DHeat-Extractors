"""
RaceRoom ($R3E) and Automobilista 2 / Project CARS 2 ($pcars2$): pages built
here at the offsets their headers give (r3e.h 3.6, packed; SharedMemory.h,
pack 4), written the way the game writes them, read by the adapters.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import sys
import time

import pyarrow.parquet as pq
import pytest

from heat3d_watch.adapters import simracing_shm as shm
from heat3d_watch.adapters.base import Context
from heat3d_watch.config import Settings
from heat3d_watch.session import read_rows

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="named shared memory is Windows only")


def context(tmp_path):
    data = tmp_path / "data"
    return Context(Settings(data=str(data)), data)


def session_of(ctx):
    (folder,) = [p for p in (ctx.data / "sessions").glob("*/*")]
    return folder, read_rows(folder), json.loads((folder / "meta.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ RaceRoom, r3e.h 3.6

R3E_SIZE = 44024
DRIVERS = 2008  # num_cars; all_drivers_data_1 follows
DRIVER = 328


def r3e_page(*, cars, own_slot=7, menus=0, best=-1.0, previous=-1.0, place=2, incidents=0):
    p = bytearray(R3E_SIZE)
    struct.pack_into("<4i", p, 0, 3, 6, DRIVERS, DRIVER)
    struct.pack_into("<i", p, 24, menus)
    p[600:610] = b"Test Track"
    p[664:666] = b"GP"
    struct.pack_into("<i", p, 780, 2)  # race
    struct.pack_into("<i", p, 988, place)
    struct.pack_into("<f", p, 1068, best)
    struct.pack_into("<f", p, 1084, previous)
    struct.pack_into("<i", p, 1172, incidents)
    struct.pack_into("<i", p, 1196 + 92, own_slot)  # vehicle_info.slot_id
    struct.pack_into("<i", p, 1408, 4)  # gear
    struct.pack_into("<f", p, 1500, 0.75)  # throttle
    struct.pack_into("<i", p, DRIVERS, len(cars))
    for i, c in enumerate(cars):
        at = DRIVERS + 4 + i * DRIVER
        p[at : at + len(c["name"])] = c["name"].encode()
        struct.pack_into("<i", p, at + 92, c["slot"])
        struct.pack_into("<i", p, at + 128, c.get("finish", 0))
        struct.pack_into("<i", p, at + 132, c.get("place", i + 1))
        struct.pack_into("<3f", p, at + 148, *c["pos"])
        struct.pack_into("<i", p, at + 164, c["laps"])
        struct.pack_into("<f", p, at + 172, c.get("clock", 0.0))
        struct.pack_into("<i", p, at + 224, c.get("pit", 0))
        struct.pack_into("<f", p, at + 252, c.get("speed", 40.0))
    return bytes(p)


def test_raceroom_every_car_laps_finish_pit_and_incidents(tmp_path):
    tag = f"Local\\heat3d_test_r3e_{os.getpid()}"
    mem = mmap.mmap(-1, R3E_SIZE, tagname=tag)
    ctx = context(tmp_path)
    a = shm.RaceRoomAdapter(id="r3e-test", game="RaceRoom", exes=("x.exe",), tagname=tag)
    try:
        a.start(ctx)
        for k in range(40):
            own_laps = 1 if k >= 20 else 0
            other_laps = 1 if k >= 22 else 0
            cars = [
                {"name": "Me Myself", "slot": 7, "pos": (k * 4.0, 2.0, 0.0), "laps": own_laps, "place": 2,
                 "pit": 1 if 10 <= k < 12 else 0, "finish": 1 if k >= 35 else 0, "speed": 41.0},
                {"name": "Other Driver", "slot": 9, "pos": (0.0, 2.0, k * 4.0), "laps": other_laps, "place": 1,
                 "clock": (k - 22) * 0.1 + 0.1 if k >= 22 else 60.0 + k},
            ]
            mem[:] = r3e_page(cars=cars, best=80.5 if k >= 20 else -1.0, previous=80.5 if k >= 20 else -1.0,
                              incidents=4 if k >= 30 else 0, menus=1 if k >= 38 else 0)
            a.poll(ctx)
            time.sleep(0.01)
        a.stop(ctx)
    finally:
        mem.close()
    folder, rows, meta = session_of(ctx)
    laps = {r["player"]: r["lap_time"] for r in rows if r["event"] == "lap"}
    assert laps == {"Me Myself": 80.5, "Other Driver": 81.0}  # the other's last running lap time seen
    (finish,) = [r for r in rows if r["event"] == "finish"]
    assert finish["player"] == "Me Myself" and finish["own_car"] == 1
    assert [r["event"] for r in rows if r["event"].startswith("pit_")] == ["pit_in", "pit_out"]
    (start,) = [r for r in rows if r["event"] == "session_start"]
    assert start["track"] == "Test Track GP"
    s = meta["stats"]
    assert s["laps"] == 1 and s["best_lap"] == 80.5 and s["incidents"] == 4 and s["race_position"] == 2
    assert s["pit_stops"] == 1 and s["top_speed_kmh"] == pytest.approx(41.0 * 3.6)
    table = pq.read_table(folder / "heat.parquet").to_pylist()
    positions = [r for r in table if r["event"] == "position"]
    assert {r["player"] for r in positions} == {"Me Myself", "Other Driver"}
    assert max(r["x"] for r in positions if r["player"] == "Me Myself") <= 37 * 4.0  # nothing from the menus


# ------------------------------------------------------------------ AMS2 / PCARS2, SharedMemory.h

PC2_SIZE = 20700


def pc2_page(*, cars, viewed=0, game_state=2, sequence=10, last=0.0, best=-1.0):
    p = bytearray(PC2_SIZE)
    struct.pack_into("<4I", p, 0, 14, 1, game_state, 5)
    struct.pack_into("<2i", p, 20, viewed, len(cars))
    for i, c in enumerate(cars):
        at = 28 + 100 * i
        p[at] = 1
        p[at + 1 : at + 1 + len(c["name"])] = c["name"].encode()
        struct.pack_into("<3f", p, at + 68, *c["pos"])
        struct.pack_into("<3I", p, at + 84, c.get("place", i + 1), c["laps"], c["laps"] + 1)
        struct.pack_into("<f", p, 9200 + 4 * i, c.get("last", 0.0))
        struct.pack_into("<I", p, 9520 + 4 * i, c.get("state", 2))
        struct.pack_into("<f", p, 10800 + 4 * i, 30.0)
    p[6576:6585] = b"Test Ring"
    struct.pack_into("<f", p, 6716, best)
    struct.pack_into("<f", p, 6720, last)
    struct.pack_into("<f", p, 6848, 33.0)
    struct.pack_into("<f", p, 6864, 1.0)
    struct.pack_into("<i", p, 6876, 5)
    struct.pack_into("<I", p, 7320, sequence)
    return bytes(p)


def test_automobilista_reads_every_participant_and_waits_out_a_write(tmp_path):
    tag = f"Local\\heat3d_test_pc2_{os.getpid()}"
    mem = mmap.mmap(-1, PC2_SIZE, tagname=tag)
    ctx = context(tmp_path)
    a = shm.PCars2Adapter(id="ams2-test", game="Automobilista 2", exes=("x.exe",), tagname=tag)
    try:
        a.start(ctx)
        for k in range(40):
            done = 1 if k >= 15 else 0
            cars = [
                {"name": "Player", "pos": (k * 3.0, 1.0, 0.0), "laps": done, "last": 70.25 if done else 0.0},
                {"name": "Rival", "pos": (0.0, 1.0, k * 3.0), "laps": done, "last": 71.0 if done else 0.0,
                 "state": 3 if k >= 30 else 2},
            ]
            # Step 20 is caught mid-write (an odd sequence number): it must not be read.
            seq = 41 if k == 20 else 40 + 2 * k
            if k == 20:
                cars[0]["pos"] = (99999.0, 1.0, 0.0)
            mem[:] = pc2_page(cars=cars, sequence=seq, last=70.25 if done else 0.0, best=70.25 if done else -1.0,
                              game_state=6 if k >= 38 else 2)
            a.poll(ctx)
            time.sleep(0.01)
        a.stop(ctx)
    finally:
        mem.close()
    folder, rows, meta = session_of(ctx)
    laps = {r["player"]: r["lap_time"] for r in rows if r["event"] == "lap"}
    assert laps == {"Player": 70.25, "Rival": 71.0}
    (finish,) = [r for r in rows if r["event"] == "finish"]
    assert finish["player"] == "Rival" and finish["own_car"] == 0
    assert not [r for r in rows if r["event"] == "teleport"]  # the torn copy was never used
    s = meta["stats"]
    assert s["laps"] == 1 and s["best_lap"] == 70.25 and s["top_speed_kmh"] == pytest.approx(33.0 * 3.6)
    own = [r for r in read_rows(folder) if r["event"] == "position" and r.get("own_car") == 1]
    assert own and all(r["player"] == "Player" for r in own) and {r.get("gear") for r in own} == {5}
