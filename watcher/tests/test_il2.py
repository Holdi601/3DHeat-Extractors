"""IL-2 Great Battles: telemetry packets per the game's Il2_TeleOutput spec, startup.cfg edits, ACMI recordings."""

from __future__ import annotations

import io
import json
import os
import shutil
import socket
import struct
import time
import zipfile
from dataclasses import replace

import pyarrow.parquet as pq
import pytest

from heat3d_watch.adapters.base import Context
from heat3d_watch.adapters.il2 import (
    ADAPTERS,
    BACKUP,
    PACKET_ID,
    acmi_text_lines,
    decode_packet,
    edit_tacview,
    edit_telemetry,
    game_folders,
    read_acmi,
    split_fields,
)
from heat3d_watch.config import Settings
from heat3d_watch.session import read_rows


def packet(tick: int, indicators=(), events=()) -> bytes:
    """Packet_ID, Size, Tick, IC, SIndicator[IC], EC, SEvent[EC] - LSB first, as the spec lays it out."""
    body = struct.pack("<B", len(indicators))
    for iid, values in indicators:
        body += struct.pack("<HB", iid, len(values)) + struct.pack(f"<{len(values)}f", *values)
    body += struct.pack("<B", len(events))
    for eid, data in events:
        body += struct.pack("<HB", eid, len(data)) + data
    return struct.pack("<IHI", PACKET_ID, 10 + len(body), tick) + body


def string(text: str) -> bytes:
    raw = text.encode("ascii")
    return bytes([len(raw)]) + raw


def drop(mass: float, flags: int = 0) -> bytes:
    return struct.pack("<4fH", 0.5, -1.0, 2.0, mass, flags)  # STEDropData, 2-byte packing


def hit() -> bytes:
    return struct.pack("<6f", 1.0, 0.0, -3.0, 300.0, 400.0, 0.0)  # STEHit


def test_packets_decode_per_the_spec():
    raw = packet(
        1234,
        indicators=[(0, [2400.0, 2350.0]), (6, [100.0]), (8, [0.0, 19.6133, 0.0]), (10, [500.0])],
        events=[(0, string("Bf 109 F-4")), (4, drop(250.0)), (6, hit()), (8, struct.pack("<4f", 1, 2, 3, 7.5)),
                (9, b"\x01"), (13, struct.pack("<ii32s", 5, 9, b"Pilot")), (14, struct.pack("<ih", -1, 2))],
    )
    p = decode_packet(raw)
    assert p.tick == 1234
    assert p.indicators["eng_rpm"] == (2400.0, 2350.0) and p.indicators["eas"] == (100.0,)
    names = [n for n, _ in p.events]
    assert names == ["set_focus", "drop_bomb", "hit", "explosion", "gun_fire", "client_dat", "ctrl_dat"]
    assert p.events[0][1]["text"] == "Bf 109 F-4"
    assert p.events[1][1]["mass"] == 250.0 and p.events[1][1]["local"] == (0.5, -1.0, 2.0)
    assert p.events[2][1]["force"] == (300.0, 400.0, 0.0)
    assert p.events[3][1]["radius"] == 7.5
    assert p.events[5][1]["name"] == "Pilot" and p.events[6][1]["coalition"] == 2
    for bad in (b"", struct.pack("<IHIB", 0x494C0100, 11, 0, 0), raw[:-3]):
        with pytest.raises((ValueError, struct.error)):
            decode_packet(bad)


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_sorties_over_udp(tmp_path):
    port = _free_port()
    a = replace(ADAPTERS[0], port=port)
    ctx = Context(Settings(data=str(tmp_path)), tmp_path)
    a.start(ctx)
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(*packets):
        for p in packets:
            out.sendto(p, ("127.0.0.1", port))
        time.sleep(0.03)
        a.poll(ctx)

    fly = [(6, [80.0]), (10, [300.0]), (8, [0.0, 29.42, 0.0]), (14, [5.0])]
    send(packet(100, fly, [(0, string("Yak-1 ser.69"))]))
    folder = a._session.folder
    send(*[packet(100 + i * 10, fly) for i in range(1, 12)])  # 2.2 s of game time
    send(packet(230, fly, [(4, drop(100.0)), (4, drop(50.0)), (5, drop(20.0, 1)), (9, b"\x00"), (9, b"\x01"), (9, b"\x00")]))
    send(packet(240, fly, [(6, hit()), (7, hit()), (8, struct.pack("<4f", 0, 0, 0, 3.0)), (11, string("A server"))]))
    send(b"\x01\x01\x00\x54 junk", packet(250, fly)[:-2])
    send(packet(10, fly, [(0, string("Il-2 mod.1941"))]))  # the clock went back: a new mission, a new sortie
    a.stop(ctx)
    rows = read_rows(folder)
    events = [r["event"] for r in rows if r["event"] != "state"]
    assert events == [
        "spawn", "drop_bomb", "drop_bomb", "rocket_launch", "gun_fire", "hit", "damage", "explosion",
        "session_end", "spawn", "session_end",
    ]
    assert rows[0]["aircraft"] == "Yak-1 ser.69" and rows[0]["session"] == "sortie-1"
    assert next(r for r in rows if r["event"] == "gun_fire")["rounds"] == 3
    assert next(r for r in rows if r["event"] == "hit")["force"] == 500.0
    states = [r for r in rows if r["event"] == "state"]
    assert len(states) == 4 and states[0]["eas"] == 80.0 and states[0]["g"] == pytest.approx(3.0, abs=0.01)
    assert all("x" not in r for r in rows)  # the telemetry has no world position
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    st = meta["stats"]
    assert st["sorties"] == 2 and st["bombs_dropped"] == 2 and st["rockets_fired"] == 1 and st["rounds_fired"] == 3
    assert st["hits_taken"] == 1 and st["damage_taken"] == 1 and st["explosions"] == 1 and st["bad_packets"] == 2
    assert st["max_eas_kmh"] == 288.0 and st["max_g"] == pytest.approx(3.0, abs=0.01) and st["ordnance_kg"] == 170
    assert st["flight_time"] == pytest.approx(2.0)  # state rows at ticks 100, 150, 200: two seconds between them
    assert meta["server_title"] == "A server" and meta["multiplayer"] is True


CFG = (
    "[KEY = graphics]\r\n\tfps_limit = 60\r\n[END]\r\n\r\n"
    "[KEY = telemetrydevice]\r\n\taddr = \"127.0.0.1:41994\"\r\n\tdecimation = 1\r\n\tenable = false\r\n\tport = 4322\r\n[END]\r\n\r\n"
    "[KEY = track_record]\r\n\tfx_sound = 2\r\n\ttacviewrecord = 0\r\n[END]\r\n"
)


def test_startup_cfg_edits_are_minimal_and_idempotent():
    new, changes = edit_telemetry(CFG, 41995)
    assert len(changes) == 2
    assert '\taddr = "127.0.0.1:41994"\r\n\taddr1 = "127.0.0.1:41995"\r\n\tdecimation = 1\r\n\tenable = true\r\n' in new
    assert new.replace('\taddr1 = "127.0.0.1:41995"\r\n', "").replace("enable = true", "enable = false") == CFG
    assert edit_telemetry(new, 41995) == (new, [])
    new2, changes = edit_tacview(new)
    assert changes and "\ttacviewrecord = 1\r\n" in new2 and new2.replace("tacviewrecord = 1", "tacviewrecord = 0") == new
    assert edit_tacview(new2) == (new2, [])
    # Sections that are not there are added, the rest left as it was.
    bare = "[KEY = graphics]\n\tfps_limit = 60\n[END]\n"
    t, c = edit_telemetry(bare)
    assert t.startswith(bare) and "[KEY = telemetrydevice]\n\taddr = \"127.0.0.1:41995\"\n\tdecimation = 5\n\tenable = true" in t
    t, c = edit_tacview(bare)
    assert t.startswith(bare) and "[KEY = track_record]\n\ttacviewrecord = 1\n[END]\n" in t
    t, c = edit_tacview("[KEY = track_record]\n\tfx_sound = 2\n[END]\n")
    assert t == "[KEY = track_record]\n\tfx_sound = 2\n\ttacviewrecord = 1\n[END]\n"


def test_setup_keeps_one_backup(tmp_path):
    game = tmp_path / "game"
    (game / "data").mkdir(parents=True)
    cfg = game / "data" / "startup.cfg"
    cfg.write_bytes(CFG.encode())
    ctx = Context(Settings(data=str(tmp_path / "d")), tmp_path / "d")
    # find=False: never the installed game's own file.
    tele = replace(ADAPTERS[0], folders=(str(game),), find=False)
    tac = replace(ADAPTERS[1], folders=(str(game),), find=False)
    assert len(tele.setup(ctx, apply=False)) == 2 and cfg.read_bytes() == CFG.encode()
    assert len(tele.setup(ctx, apply=True)) == 2
    assert len(tac.setup(ctx, apply=True)) == 1
    assert tele.setup(ctx, apply=True) == [] and tac.setup(ctx, apply=True) == []
    backup = cfg.with_name("startup.cfg" + BACKUP)
    assert backup.read_bytes() == CFG.encode()  # the original, not the first edit
    text = cfg.read_text(encoding="utf-8")
    assert "addr1 = \"127.0.0.1:41995\"" in text and "tacviewrecord = 1" in text


ACMI = "\n".join([
    "﻿FileType=text/acmi/tacview",
    "FileVersion=2.2",
    "0,ReferenceTime=1942-11-19T06:00:00Z",
    "0,RecordingTime=2026-10-01T18:30:00Z",
    "0,ReferenceLongitude=44",
    "0,ReferenceLatitude=48",
    "0,DataSource=Synthetic,Title=A test mission\\, with a comma",
    "0,Comments=two \\",
    "lines",
    "#0",
    "101,T=0.01|0.02|300,Type=Air+FixedWing,Name=Yak-1,Pilot=Alpha,Coalition=Allies",
    "102,T=0.03|0.02|1000|0|5|90,Type=Air+FixedWing,Name=Bf 109,Coalition=Axis",
    "103,T=0.05|0.05|150,Type=Ground+Heavy+Armor+Vehicle+Tank,Name=T-34,Coalition=Allies",
    "201,T=0.03|0.02|1000,Type=Weapon+Missile,Name=bullet",
    "#0.5",
    "101,T=0.0101||310",
    "#1.0",
    "101,T=0.0102||320",
    "102,T=0.031|0.021|1000|1|4|91",
    "-201",
    "0,Event=TakenOff|101|Alpha took off",
    "#2.0",
    "101,T=0.0103|0.0201|400",
    "0,Event=Destroyed|102|",
    "-102",
    "#3.0",
    "0,Event=Message|101|hello",
    "103,T=0.0501|0.05|150",
    "-103",
    "#4.0",
    "101,T=0.0104|0.0202|420",
])


def test_acmi_reads_as_rows():
    assert split_fields("0,Title=a\\,b,X=1") == ["0", "Title=a,b", "X=1"]
    glob, rows = read_acmi(acmi_text_lines(io.BytesIO(ACMI.encode("utf-8"))))
    assert glob["ReferenceLongitude"] == "44" and glob["Title"] == "A test mission, with a comma"
    assert glob["Comments"] == "two \nlines"
    summary = [(r.event, r.attrs.get("player"), r.t) for r in rows]
    assert summary == [
        ("spawn", "Alpha", 0.0),
        ("spawn", "Bf 109 #102", 0.0),
        ("spawn", "T-34 #103", 0.0),
        ("position", "Alpha", 1.0),
        ("position", "Bf 109 #102", 1.0),
        ("takeoff", "Alpha", 1.0),
        ("position", "Alpha", 2.0),
        ("death", "Bf 109 #102", 2.0),
        ("message", None, 3.0),
        ("position", "T-34 #103", 3.0),
        ("left", "T-34 #103", 3.0),
        ("position", "Alpha", 4.0),
        ("end", "Alpha", 4.0),
    ]
    first = rows[0]
    assert (first.lat, first.lon, first.alt) == (48.02, 44.01, 300.0)
    bf = rows[4]
    assert bf.attrs["yaw"] == 91.0 and bf.attrs["team"] == "Axis"
    # Empty components keep the last value: latitude stayed while longitude and altitude moved.
    assert rows[3].lat == pytest.approx(48.02) and rows[3].lon == pytest.approx(44.0102) and rows[3].alt == 320


def test_finished_recordings_become_sessions(tmp_path):
    tracks = tmp_path / "game" / "data" / "Tracks"
    tracks.mkdir(parents=True)
    (tmp_path / "game" / "data" / "startup.cfg").write_text(CFG, encoding="utf-8")
    plain = tracks / "one.txt.acmi"
    plain.write_text(ACMI, encoding="utf-8")
    zipped = tracks / "two.zip.acmi"
    with zipfile.ZipFile(zipped, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("two.txt.acmi", ACMI)
    busy = tracks / "three.acmi"
    busy.write_text(ACMI[:200], encoding="utf-8")
    old = time.time() - 120
    for f in (plain, zipped):
        os.utime(f, (old, old))
    a = replace(ADAPTERS[1], folders=(str(tmp_path / "game"),), find=False, settle=30.0)
    ctx = Context(Settings(data=str(tmp_path / "d")), tmp_path / "d")
    for _ in range(6):
        a.poll(ctx)
    sessions = sorted((tmp_path / "d" / "sessions").rglob("meta.json"))
    assert len(sessions) == 2  # the file still being written waits
    for m in sessions:
        meta = json.loads(m.read_text(encoding="utf-8"))
        assert meta["origin"] == {"lat": 48.0, "lon": 44.0} and meta["started"].startswith("2026-10-01")
        assert meta["stats"]["objects"] == 3 and meta["stats"]["destroyed"] == 1 and meta["stats"]["takeoffs"] == 1
        assert meta["axes"]["flip_z"] is True
        t = pq.read_table(m.parent / "heat.parquet")
        alpha = [r for r in t.to_pylist() if r["player"] == "Alpha"]
        # 0.01 deg of longitude east of the reference at 48 N is about 745 m; 0.02 deg of latitude 2224 m.
        assert alpha[0]["x"] == pytest.approx(745, abs=5) and alpha[0]["z"] == pytest.approx(2224, abs=5)
        assert alpha[0]["y"] == 300 and alpha[0]["team"] == "Allies"
        assert "bullet" not in t.column("name").to_pylist()
    # Done once: nothing more on later looks, and a restart remembers.
    b = replace(ADAPTERS[1], folders=(str(tmp_path / "game"),), find=False, settle=30.0)
    for _ in range(3):
        b.poll(ctx)
    assert len(list((tmp_path / "d" / "sessions").rglob("meta.json"))) == 2


@pytest.mark.local
def test_setup_on_a_copy_of_the_installed_games_startup_cfg(tmp_path):
    games = game_folders()
    if not games:
        pytest.skip("IL-2 Great Battles is not installed through Steam here")
    for n, g in enumerate(games):
        copy = tmp_path / str(n) / "data"
        copy.mkdir(parents=True)
        shutil.copyfile(g / "data" / "startup.cfg", copy / "startup.cfg")  # the real file is only read
        before = (copy / "startup.cfg").read_bytes().decode("utf-8", "surrogateescape")
        text, changes = edit_telemetry(before)
        assert changes
        text2, changes2 = edit_tacview(text)
        assert edit_telemetry(text2) == (text2, []) and edit_tacview(text2) == (text2, [])
        # Every line of the original is still there, in order.
        kept = iter(text2.splitlines())
        assert all(any(line == k for k in kept) for line in before.splitlines() if "tacviewrecord" not in line and "enable" not in line)
