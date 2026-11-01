"""
F1 23/24/25 telemetry: packets built from EA's published UDP specification for
each format year, sent over UDP to the adapter, and the session read back.

The layouts below are written from the specification's struct listings, not
from the adapter, and their sizes are checked against the packet sizes the
specification states - which is what tells the 2023, 2024 and 2025 formats apart.
"""

from __future__ import annotations

import json
import socket
import struct
import time

import pyarrow.parquet as pq
import pytest

from heat3d_watch.adapters import f1
from heat3d_watch.adapters.base import Context
from heat3d_watch.config import Settings
from heat3d_watch.session import read_rows

#: The packet sizes the specifications state.
SPEC_SIZES = {
    2023: {"motion": 1349, "lap": 1131, "event": 45, "participants": 1306, "telemetry": 1352, "session": 644},
    2024: {"motion": 1349, "lap": 1285, "event": 45, "participants": 1350, "telemetry": 1352, "session": 753},
    2025: {"motion": 1349, "lap": 1285, "event": 45, "participants": 1284, "telemetry": 1352, "session": 753},
}
#: LapData per car, in the specification's field order.
LAP_FIELDS = {
    2023: [("last", "I"), ("current", "I"), ("s1ms", "H"), ("s1min", "B"), ("s2ms", "H"), ("s2min", "B"),
           ("front", "H"), ("leader", "H"), ("lap_distance", "f"), ("total_distance", "f"), ("sc", "f"),
           ("position", "B"), ("lap", "B"), ("pit_status", "B"), ("pit_stops", "B"), ("sector", "B"),
           ("invalid", "B"), ("penalties", "B"), ("warnings", "B"), ("cc", "B"), ("dt", "B"), ("sg", "B"),
           ("grid", "B"), ("driver_status", "B"), ("result", "B"), ("pit_timer", "B"), ("pit_lane_ms", "H"),
           ("pit_stop_ms", "H"), ("serve", "B")],
    2024: [("last", "I"), ("current", "I"), ("s1ms", "H"), ("s1min", "B"), ("s2ms", "H"), ("s2min", "B"),
           ("front", "H"), ("front_min", "B"), ("leader", "H"), ("leader_min", "B"), ("lap_distance", "f"),
           ("total_distance", "f"), ("sc", "f"), ("position", "B"), ("lap", "B"), ("pit_status", "B"),
           ("pit_stops", "B"), ("sector", "B"), ("invalid", "B"), ("penalties", "B"), ("warnings", "B"),
           ("cc", "B"), ("dt", "B"), ("sg", "B"), ("grid", "B"), ("driver_status", "B"), ("result", "B"),
           ("pit_timer", "B"), ("pit_lane_ms", "H"), ("pit_stop_ms", "H"), ("serve", "B"), ("trap", "f"),
           ("trap_lap", "B")],
}
LAP_FIELDS[2025] = LAP_FIELDS[2024]
#: ParticipantData: bytes per car and name length.
PARTICIPANT = {2023: (58, 48), 2024: (60, 48), 2025: (57, 32)}


def head(fmt: int, pid: int, t: float, frame: int, uid: int = 0xABCDEF0123, player: int = 0) -> bytes:
    return struct.pack("<HBBBBBQfIIBB", fmt, fmt % 100, 1, 2, 1, pid, uid, t, frame, frame, player, 255)


def motion(fmt, t, frame, cars) -> bytes:
    body = b""
    for i in range(22):
        x, y, z, v = cars.get(i, (0.0, 0.0, 0.0, (0.0, 0.0, 0.0)))
        body += struct.pack("<6f6h6f", x, y, z, *v, 0, 0, 32767, 32767, 0, 0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    return head(fmt, 0, t, frame) + body


def lap_data(fmt, t, frame, cars) -> bytes:
    fields = LAP_FIELDS[fmt]
    layout = "<" + "".join(c for _, c in fields)
    body = b""
    for i in range(22):
        values = {"result": 0, **cars.get(i, {})}
        body += struct.pack(layout, *(values.get(n, 0) for n, _ in fields))
    return head(fmt, 2, t, frame) + body + bytes([255, 255])


def participants(fmt, people) -> bytes:
    size, name_len = PARTICIPANT[fmt]
    body = bytes([len(people)])
    for i in range(22):
        name, team, number = people[i] if i < len(people) else ("", 255, 0)
        car = bytes([0 if i == 0 else 1, 255, i, team, 0, number, 10]) + name.encode("utf-8").ljust(name_len, b"\0")
        body += car.ljust(size, b"\0")
    return head(fmt, 4, 0.0, 0) + body


def telemetry(fmt, t, frame, cars) -> bytes:
    body = b""
    for i in range(22):
        speed, gear = cars.get(i, (0, 0))
        body += struct.pack("<HfffBbHBBH4H4B4BH4f4B", speed, 0.9, 0.0, 0.1, 0, gear, 11000, 0, 50, 0,
                            *[300] * 4, *[90] * 4, *[95] * 4, 110, *[23.0] * 4, *[0] * 4)
    return head(fmt, 6, t, frame) + body + bytes([255, 255, 0])


def event(fmt, t, code: str, details: bytes = b"") -> bytes:
    return (head(fmt, 3, t, 0) + code.encode("ascii") + details).ljust(45, b"\0")


def session(fmt, track=7, kind=15) -> bytes:
    p = bytearray(SPEC_SIZES[fmt]["session"])
    p[:29] = head(fmt, 1, 0.0, 0)
    struct.pack_into("<BbbBHBbB", p, 29, 0, 30, 22, 52, 5891, kind, track, 0)
    return bytes(p)


@pytest.mark.parametrize("fmt", [2023, 2024, 2025])
def test_packets_are_the_sizes_the_specification_states(fmt):
    sizes = SPEC_SIZES[fmt]
    assert len(motion(fmt, 0, 0, {})) == sizes["motion"]
    assert len(lap_data(fmt, 0, 0, {})) == sizes["lap"]
    assert len(event(fmt, 0, "SSTA")) == sizes["event"]
    assert len(participants(fmt, [])) == sizes["participants"]
    assert len(telemetry(fmt, 0, 0, {})) == sizes["telemetry"]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("fmt", [2023, 2024, 2025])
def test_a_race_every_car_its_laps_and_its_events(tmp_path, fmt):
    port = free_port()
    data = tmp_path / "data"
    ctx = Context(Settings(data=str(data)), data)
    a = f1.F1Adapter(id="f1-test", game="EA SPORTS F1 25", exes=("x.exe",), port=port)
    a.start(ctx)
    for _ in range(50):
        a.poll(ctx)
        if a._udp.ready:
            break
        time.sleep(0.02)
    assert a._udp.ready

    packets = [
        event(fmt, 0.0, "SSTA"),
        participants(fmt, [("Player One", 8, 4), ("Rival", 6, 44)]),
        session(fmt),
        head(2022, 0, 0.0, 0) + bytes(1300),  # an older format: not read
        motion(fmt, 0.0, 0, {})[:200],  # cut short
    ]
    for k in range(80):
        t = k * 0.05
        lap0 = 1 if k < 40 else 2
        lap1 = 1 if k < 45 else 2
        status0 = 3 if k >= 75 else 2
        cars = {0: (k * 2.5, 1.0, 0.0, (50.0, 0.0, 0.0)), 1: (0.0, 1.0, k * 2.4, (0.0, 0.0, 48.0))}
        packets.append(telemetry(fmt, t, k, {0: (180, 6), 1: (172, 6)}))
        packets.append(motion(fmt, t, k, cars))
        packets.append(lap_data(fmt, t, k, {
            0: {"lap": lap0, "last": 90_500 if lap0 == 2 else 0, "position": 2 if k < 50 else 1, "result": status0, "grid": 3, "pit_stops": 0},
            1: {"lap": lap1, "last": 91_000 if lap1 == 2 else 0, "position": 1 if k < 50 else 2, "result": 2, "grid": 1},
        }))
        if k == 50:
            packets.append(event(fmt, t, "OVTK", bytes([0, 1])))
            packets.append(event(fmt, t, "COLL", bytes([1, 0])))
            packets.append(event(fmt, t, "PENA", bytes([4, 7, 1, 0, 5, 2, 0])))
        if k == 60:
            packets.append(event(fmt, t, "FLBK", struct.pack("<If", 40, 2.0)))
            packets.append(event(fmt, t, "FTLP", bytes([0]) + struct.pack("<f", 90.5)))
            packets.append(event(fmt, t, "BUTN", struct.pack("<I", 1)))
        if k == 74:
            packets.append(event(fmt, t, "CHQF"))
    packets.append(event(fmt, 4.0, "SEND"))

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as out:
        for i, p in enumerate(packets):
            out.sendto(p, ("127.0.0.1", port))
            if i % 3 == 2:
                time.sleep(0.004)
                a.poll(ctx)
    time.sleep(0.05)
    a.poll(ctx)
    a.stop(ctx)

    (folder,) = [p for p in (data / "sessions").glob("*/*")]
    rows = read_rows(folder)
    by = lambda name: [r for r in rows if r["event"] == name]  # noqa: E731
    laps = {r["player"]: r["lap_time"] for r in by("lap")}
    assert laps == {"Player One": 90.5, "Rival": 91.0}
    (overtake,) = by("overtake")
    assert overtake["player"] == "Player One" and overtake["other"] == "Rival" and "x" in overtake
    (penalty,) = by("penalty")
    assert penalty["player"] == "Rival" and penalty["penalty_type"] == 4 and penalty["time"] == 5
    assert by("collision")[0]["player"] == "Rival"
    (finish,) = by("finish")
    assert finish["player"] == "Player One" and finish["race_position"] == 1
    assert by("teleport")[0]["reason"] == "flashback"
    assert by("fastest_lap")[0]["lap_time"] == 90.5
    assert by("chequered_flag") and by("session_start") and by("session_end")
    assert not by("butn") and all(r.get("code") != "BUTN" for r in rows)

    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    stats = meta["stats"]
    assert stats["laps"] == 1 and stats["best_lap"] == 90.5
    assert stats["overtakes"] == 1 and stats["collisions"] == 1 and "penalties" not in stats
    assert stats["race_position"] == 1 and stats["grid_position"] == 3
    assert stats["top_speed_kmh"] == 180.0 and stats["fastest_laps"] == 1 and stats["flashbacks"] == 1
    assert stats["unsupported_packets"] == 1 and stats["bad_packets"] == 1
    (info,) = meta["sessions"].values()
    assert info["track"] == "Silverstone" and info["session_type"] == "race"

    table = pq.read_table(folder / "heat.parquet").to_pylist()
    positions = [r for r in table if r["event"] == "position"]
    assert {r["player"] for r in positions} == {"Player One", "Rival"}
    teams = {r["player"]: r["team"] for r in positions}
    assert teams["Player One"] == "McLaren"
    assert teams["Rival"] == ("Alpha Tauri" if fmt == 2023 else "RB")
    mine = [r for r in positions if r["own_car"] == 1]
    assert mine and all(r["player"] == "Player One" for r in mine)
    assert {r["speed_kmh"] for r in mine if r["speed_kmh"]} == {180.0}
    assert all(r["y"] == 1.0 for r in positions)  # Y is up
    assert meta["axes"]["up"] == "y"


def test_setup_switches_the_udp_output_on(tmp_path, monkeypatch):
    monkeypatch.setattr(f1, "documents", lambda: tmp_path)
    config = tmp_path / "My Games" / "F1 25" / "hardwaresettings" / "hardware_settings_config.xml"
    config.parent.mkdir(parents=True)
    original = ('<hardware_settings_config>\n  <motion>\n    <udp enabled="false" broadcast="false" ip="127.0.0.1" '
                'port="20777" sendRate="20" format="2025" yourTelemetry="restricted" onlineNames="off" />\n'
                '  </motion>\n</hardware_settings_config>\n')
    config.write_text(original, encoding="utf-8")
    a = next(x for x in f1.ADAPTERS if x.id == "f1-25")
    ctx = Context(Settings(data=str(tmp_path / "d")), tmp_path / "d")
    assert a.setup(ctx, apply=False) and config.read_text(encoding="utf-8") == original
    a.setup(ctx, apply=True)
    assert 'enabled="true" broadcast="false"' in config.read_text(encoding="utf-8")
    assert config.with_name(config.name + ".heat3d-backup").read_text(encoding="utf-8") == original
    assert a.setup(ctx, apply=True) == []
