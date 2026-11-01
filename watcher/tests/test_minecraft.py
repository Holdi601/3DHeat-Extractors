"""Minecraft: latest.log lines, stats JSON and player NBT built per the published formats."""

from __future__ import annotations

import gzip
import json
import os
import struct
import time
from dataclasses import replace

import pyarrow.parquet as pq
import pytest

from heat3d_watch.adapters import minecraft as mc
from heat3d_watch.adapters.base import Context
from heat3d_watch.config import Settings
from heat3d_watch.session import read_rows

UUID = "00000000-0000-4000-8000-000000000001"

# ------------------------------------------------------------- NBT, per the spec


def _s(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack(">H", len(raw)) + raw


def _payload(tag: int, value) -> bytes:
    if tag == 1:
        return struct.pack(">b", value)
    if tag == 2:
        return struct.pack(">h", value)
    if tag == 3:
        return struct.pack(">i", value)
    if tag == 4:
        return struct.pack(">q", value)
    if tag == 5:
        return struct.pack(">f", value)
    if tag == 6:
        return struct.pack(">d", value)
    if tag == 8:
        return _s(value)
    if tag == 9:
        kind, items = value
        return struct.pack(">bi", kind, len(items)) + b"".join(_payload(kind, v) for v in items)
    if tag == 10:
        return b"".join(struct.pack(">b", t) + _s(k) + _payload(t, v) for k, (t, v) in value.items()) + b"\x00"
    if tag == 11:
        return struct.pack(">i", len(value)) + struct.pack(f">{len(value)}i", *value)
    raise ValueError(tag)


def nbt_file(root: dict) -> bytes:
    return gzip.compress(b"\x0a" + _s("") + _payload(10, root))


def player(x, y, z, dim="minecraft:overworld"):
    return {
        "Pos": (9, (6, [x, y, z])),
        "Rotation": (9, (5, [90.0, 10.0])),
        "Dimension": (8, dim),
        "Health": (5, 20.0),
        "XpLevel": (3, 7),
        "UUID": (11, [0, 0, 0, 1]),
        "Inventory": (9, (10, [{"id": (8, "minecraft:stone"), "Count": (1, 64)}])),
        "Score": (3, 0),
        "SelectedItemSlot": (3, 0),
        "Motion": (9, (6, [0.0, 0.0, 0.0])),
        "seenCredits": (1, 0),
    }


def test_nbt_reader_reads_the_spec_tags():
    root = mc.read_nbt(nbt_file({**player(1.5, 64.0, -3.25), "big": (4, -(2**40)), "s": (2, -2)}))
    assert root["Pos"] == [1.5, 64.0, -3.25] and root["Dimension"] == "minecraft:overworld"
    assert root["big"] == -(2**40) and root["s"] == -2 and root["UUID"] == [0, 0, 0, 1]
    assert root["Inventory"][0]["id"] == "minecraft:stone"
    assert mc.player_fix(root)[:2] == ((1.5, 64.0, -3.25), "minecraft:overworld")
    # Before 1.16 the dimension was a number.
    assert mc.player_fix({"Pos": [0.0, 0.0, 0.0], "Dimension": -1})[1] == "minecraft:the_nether"
    with pytest.raises(mc.NBTError):
        mc.read_nbt(b"\x0a\x00\x00\x08\x00\x01a\x00\x09")  # a string of 9 bytes that are not there


def stats(play_ticks=0, walk_cm=0, mob_kills=0, mined=0):
    return json.dumps({
        "stats": {
            "minecraft:custom": {"minecraft:play_time": play_ticks, "minecraft:walk_one_cm": walk_cm,
                                 "minecraft:sprint_one_cm": walk_cm, "minecraft:mob_kills": mob_kills,
                                 "minecraft:jump": 3},
            "minecraft:mined": {"minecraft:stone": mined, "minecraft:dirt": 1},
        },
        "DataVersion": 3700,
    })


LOG = """[12:00:00] [main/INFO]: Setting user: Tester
[12:00:01] [Server thread/INFO]: Starting integrated minecraft server version 1.20.4
[12:00:02] [Server thread/INFO]: Tester[local:E:1a2b3c4d] logged in with entity id 42 at (10.5, 64.0, -20.5)
[12:00:02] [Server thread/INFO]: Tester joined the game
[12:00:02] [Render thread/INFO]: [System] [CHAT] Tester joined the game
[12:01:00] [Server thread/INFO]: <Tester> hello
[12:01:00] [Render thread/INFO]: [System] [CHAT] <Tester> hello
[12:02:00] [Server thread/INFO]: Tester has made the advancement [Stone Age]
[12:03:00] [Server thread/INFO]: Tester was slain by Zombie
[12:03:00] [Render thread/INFO]: [System] [CHAT] Tester was slain by Zombie
[12:04:00] [Server thread/INFO]: Friend fell from a high place
[12:05:00] [Server thread/INFO]: Tester left the game
[12:05:01] [Server thread/INFO]: Stopping server
"""


def test_the_adapter_records_log_stats_and_positions(tmp_path):
    root = tmp_path / ".minecraft"
    world = root / "saves" / "Test World"
    (world / "stats").mkdir(parents=True)
    (world / "playerdata").mkdir()
    (root / "logs").mkdir()
    stats_file = world / "stats" / f"{UUID}.json"
    stats_file.write_text(stats(play_ticks=20 * 100, walk_cm=1000, mob_kills=5, mined=10), encoding="utf-8")
    old = time.time() - 3600
    os.utime(stats_file, (old, old))
    data = tmp_path / "data"
    ctx = Context(Settings(data=str(data)), data)
    a = replace(mc.ADAPTERS[0], folder=root, world_every=0.0)
    assert a.missing() is None
    a.start(ctx)
    (root / "logs" / "latest.log").write_text(LOG, encoding="utf-8")
    a.poll(ctx)
    # The world saves: stats and the player file.
    stats_file.write_text(stats(play_ticks=20 * 160, walk_cm=6000, mob_kills=7, mined=30), encoding="utf-8")
    (world / "playerdata" / f"{UUID}.dat").write_bytes(nbt_file(player(100.5, 70.0, -200.25)))
    a.poll(ctx)
    (world / "playerdata" / f"{UUID}.dat").write_bytes(nbt_file(player(12.0, 40.0, 8.0, "minecraft:the_nether")))
    future = time.time() + 5
    os.utime(world / "playerdata" / f"{UUID}.dat", (future, future))
    a.stop(ctx)

    [session] = list((data / "sessions").glob("*/*"))
    meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
    st = meta["stats"]
    assert st["deaths"] == 1 and st["others_deaths"] == 1  # the repeated chat line counts once
    assert st["advancements"] == 1 and st["chat_messages"] == 1 and st["joins"] == 1
    assert st["play_seconds"] == 60 and st["mob_kills"] == 2 and st["blocks_mined"] == 20
    assert st["distance_m"] == pytest.approx(100.0)  # (5000 + 5000) cm walked and sprinted
    assert st["world"] == "Test World"

    rows = read_rows(session)
    spawn = next(r for r in rows if r["event"] == "spawn")
    assert (spawn["x"], spawn["y"], spawn["z"], spawn["player"]) == (10.5, 64.0, -20.5, "me")
    death = [r for r in rows if r["event"] == "death"]
    assert [d["player"] for d in death] == ["me", "Friend"]
    assert death[0]["message"] == "Tester was slain by Zombie"
    pos = [r for r in rows if r["event"] == "position"]
    assert [(p["x"], p["session"]) for p in pos] == [(100.5, "minecraft:overworld"), (12.0, "minecraft:the_nether")]
    assert pos[0]["yaw"] == 90.0 and pos[0]["level"] == 7
    assert "session_end" in [r["event"] for r in rows]

    t = pq.read_table(session / "heat.parquet")
    assert t.column("event").to_pylist() == ["spawn", "position", "position"]
    assert meta["axes"]["up"] == "y"


def test_a_world_not_played_in_this_sitting_is_left_alone(tmp_path):
    root = tmp_path / ".minecraft"
    world = root / "saves" / "Old"
    (world / "playerdata").mkdir(parents=True)
    dat = world / "playerdata" / f"{UUID}.dat"
    dat.write_bytes(nbt_file(player(1, 2, 3)))
    old = time.time() - 86400
    os.utime(dat, (old, old))
    data = tmp_path / "data"
    ctx = Context(Settings(data=str(data)), data)
    a = replace(mc.ADAPTERS[0], folder=root, world_every=0.0)
    a.start(ctx)
    a.poll(ctx)
    a.stop(ctx)
    assert not list((data / "sessions").glob("*/*")) if (data / "sessions").exists() else True


def test_a_broken_player_file_is_counted_not_fatal(tmp_path):
    root = tmp_path / ".minecraft"
    world = root / "saves" / "W"
    (world / "playerdata").mkdir(parents=True)
    data = tmp_path / "data"
    ctx = Context(Settings(data=str(data)), data)
    a = replace(mc.ADAPTERS[0], folder=root, world_every=0.0)
    a.start(ctx)
    (world / "playerdata" / f"{UUID}.dat").write_bytes(gzip.compress(b"\x0a\x00\x00\x09"))
    a.poll(ctx)
    assert a._session.stats["bad_files"] == 1
    a.stop(ctx)
