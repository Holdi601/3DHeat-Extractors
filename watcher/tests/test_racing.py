"""
The racing adapters: Forza, the Assetto Corsa games, the rally games, BeamNG.

Packets and shared-memory pages are built here from the layouts the Race tab's
decoders were written against (`gamecapture/heat3d_capture/telemetry`), sent the
way the game sends them - UDP to localhost, or a named mapping - and the
session folder the adapter leaves behind is read back.
"""

from __future__ import annotations

import json
import math
import mmap
import os
import socket
import struct
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from heat3d_watch.adapters import racing
from heat3d_watch.adapters.base import Context
from heat3d_watch.config import Settings
from heat3d_watch.session import read_rows

from heat3d_capture.telemetry import forza, kunos, rally

windows = pytest.mark.skipif(sys.platform != "win32", reason="named shared memory is Windows only")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def context(tmp_path: Path) -> Context:
    data = tmp_path / "data"
    return Context(Settings(data=str(data)), data)


def only_session(ctx: Context) -> Path:
    folders = [p for p in (ctx.data / "sessions").glob("*/*") if p.is_dir()]
    assert len(folders) == 1, folders
    return folders[0]


def meta(folder: Path) -> dict:
    return json.loads((folder / "meta.json").read_text(encoding="utf-8"))


def events(folder: Path, name: str) -> list[dict]:
    return [r for r in read_rows(folder) if r["event"] == name]


def pump(adapter, ctx, packets, port, *, every: int = 20, pause: float = 0.0) -> None:
    """Send packets to the adapter over UDP and poll it as the watcher would."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as out:
        for i, p in enumerate(packets):
            out.sendto(p, ("127.0.0.1", port))
            if i % every == every - 1:
                adapter.poll(ctx)
                if pause:
                    time.sleep(pause)
        time.sleep(0.05)
        adapter.poll(ctx)


def opened(adapter, ctx) -> None:
    adapter.start(ctx)
    for _ in range(50):
        adapter.poll(ctx)
        if adapter._udp.ready:
            return
        time.sleep(0.02)
    raise AssertionError("the adapter did not take its port")


# ------------------------------------------------------------------ Forza

FH_BASE = forza.FH_DASH_LENGTH - forza.DASH_BLOCK_LENGTH


def forza_packet(*, pos, vel, lap=0, last_lap=0.0, race_position=0, race_on=True, stamp=0, gear=3) -> bytes:
    """A Horizon Dash packet, laid out as `fulllap.SLED`/`DASH` read it."""
    b = bytearray(forza.FH_DASH_LENGTH)
    struct.pack_into("<i", b, 0, 1 if race_on else 0)
    struct.pack_into("<I", b, 4, stamp)
    struct.pack_into("<3f", b, 32, *vel)
    struct.pack_into("<i", b, 212, 1234)  # car ordinal
    struct.pack_into("<i", b, 216, 5)  # class
    struct.pack_into("<i", b, 220, 800)  # PI
    d = FH_BASE
    struct.pack_into("<3f", b, d, *pos)
    struct.pack_into("<f", b, d + 12, math.hypot(*vel))
    struct.pack_into("<f", b, d + 56, last_lap)
    struct.pack_into("<H", b, d + 68, lap)
    struct.pack_into("<B", b, d + 70, race_position)
    struct.pack_into("<B", b, d + 71, 200)
    struct.pack_into("<B", b, d + 75, gear)
    return bytes(b)


def circle(n: int, radius: float, speed: float, *, laps: int = 1):
    """Positions and velocities round a circle, `n` per lap."""
    out = []
    w = speed / radius
    for k in range(n * laps + 1):
        a = 2 * math.pi * k / n
        out.append(((radius * math.cos(a), 10.0, radius * math.sin(a)), (-speed * math.sin(a), 0.0, speed * math.cos(a))))
    return out


def test_forza_records_a_lap_a_race_and_relays_every_packet(tmp_path):
    port, relay = free_port(), free_port()
    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 21)  # FH Companion's side, holding it all
    listener.bind(("127.0.0.1", relay))
    listener.setblocking(False)
    ctx = context(tmp_path)
    a = racing.ForzaAdapter(id="forza-test", game="Forza Horizon 6", exes=("x.exe",), port=port, forward=("127.0.0.1", relay),
                            grace=0.0)
    opened(a, ctx)
    path = circle(200, 100.0, 30.0, laps=2)
    packets = [forza_packet(pos=p, vel=v, race_position=3, lap=0, stamp=i * 16) for i, (p, v) in enumerate(path[:200])]
    packets += [forza_packet(pos=p, vel=v, race_position=2, lap=1, last_lap=21.5, stamp=(200 + i) * 16)
                for i, (p, v) in enumerate(path[200:])]
    packets.append(forza_packet(pos=path[-1][0], vel=(1, 0, 1), race_position=0, lap=1, last_lap=21.5))
    packets.append(forza_packet(pos=path[-1][0], vel=(0, 0, 0), race_on=False))
    packets.append(b"\x00" * 17)  # not Forza at all
    pump(a, ctx, packets, port, every=10, pause=0.01)
    a.stop(ctx)

    relayed = 0
    while True:
        try:
            listener.recvfrom(2048)
            relayed += 1
        except BlockingIOError:
            break
    listener.close()
    assert relayed == len(packets)

    folder = only_session(ctx)
    assert [e["event"] for e in read_rows(folder) if e["event"] != "position"][:2] == ["session_start", "race_start"]
    (lap,) = events(folder, "lap")
    assert lap["lap_time"] == 21.5 and lap["lap"] == 0
    (finish,) = events(folder, "finish")
    assert finish["race_position"] == 2
    assert events(folder, "session_end")
    m = meta(folder)
    assert m["stats"]["laps"] == 1 and m["stats"]["best_lap"] == 21.5
    assert m["stats"]["bad_packets"] == 1
    assert 1.1 < m["stats"]["distance_km"] < 1.35  # two times round 628 m
    assert abs(m["stats"]["top_speed_kmh"] - 108.0) < 0.5
    t = pq.read_table(folder / "heat.parquet")
    rows = t.to_pylist()
    assert rows and all(r["player"] == "me" for r in rows)
    assert {"speed_kmh", "gear", "throttle", "brake"} <= set(t.column_names)
    assert m["axes"]["up"] == "y"


def test_forza_breaks_the_route_at_a_fast_travel(tmp_path):
    ctx = context(tmp_path)
    a = racing.ForzaAdapter(id="forza-test", game="Forza Horizon 6", port=free_port())
    a.start(ctx)
    for i in range(30):
        a.feed(forza_packet(pos=(i * 0.5, 0, 0), vel=(30, 0, 0)))
    a.feed(forza_packet(pos=(5000, 0, 3000), vel=(30, 0, 0)))
    a.stop(ctx)
    (jump,) = events(only_session(ctx), "teleport")
    assert jump["x"] == 5000 and jump["metres"] > 5000


def test_forza_leaves_a_port_held_by_another_app_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(racing, "port_holder", lambda port: "FH Companion.exe (pid 4)")
    u = racing.UdpPort(free_port())
    reason = u.open()
    assert reason and "FH Companion" in reason and not u.ready


def test_the_port_is_given_up_when_another_app_starts_listening(monkeypatch):
    holder: list[str | None] = [None]
    monkeypatch.setattr(racing, "port_holder", lambda port: holder[0])
    u = racing.UdpPort(free_port(), recheck=0.01)
    assert u.try_open("test")
    holder[0] = "FH Companion.exe (pid 4)"
    time.sleep(0.02)
    assert not u.try_open("test") and not u.ready and u.yielded == 1


def test_forza_waits_before_it_takes_the_port(tmp_path):
    ctx = context(tmp_path)
    a = racing.ForzaAdapter(id="forza-test", game="Forza Horizon 6", port=free_port(), grace=30.0)
    a.start(ctx)
    a.poll(ctx)
    assert not a._udp.ready
    a.stop(ctx)


# ------------------------------------------------------------------ Assetto Corsa games


class Pages:
    """The three pages, made the way the game makes them."""

    def __init__(self, tag: str, graphics: bool = True):
        self.names = (f"Local\\heat3d_test_{tag}_physics", f"Local\\heat3d_test_{tag}_graphics" if graphics else None,
                      f"Local\\heat3d_test_{tag}_static")
        self.physics = mmap.mmap(-1, kunos.PHYSICS_SIZE, tagname=self.names[0])
        self.graphics = mmap.mmap(-1, kunos.GRAPHICS_SIZE, tagname=self.names[1]) if graphics else None
        self.static = mmap.mmap(-1, 256, tagname=self.names[2])
        self.static[68 : 68 + 20] = "test car".encode("utf-16-le").ljust(20, b"\0")
        self.static[134 : 134 + 20] = "test track".encode("utf-16-le")
        self.packet = 0

    def write(self, pos, speed, *, laps=0, last_ms=0, status=2, position=1):
        self.packet += 1
        p = bytearray(kunos.PHYSICS_SIZE)
        struct.pack_into("<i", p, 0, self.packet)
        struct.pack_into("<f", p, 4, 0.8)
        struct.pack_into("<i", p, 16, 4)  # third gear
        struct.pack_into("<f", p, 28, speed * 3.6)
        for w in range(4):
            struct.pack_into("<3f", p, kunos.TYRE_CONTACT_POINT + 12 * w, *pos)
        self.physics[:] = bytes(p)
        if self.graphics is not None:
            g = bytearray(kunos.GRAPHICS_SIZE)
            struct.pack_into("<i", g, 4, status)
            struct.pack_into("<i", g, kunos.G_COMPLETED_LAPS, laps)
            struct.pack_into("<i", g, 136, position)
            struct.pack_into("<i", g, kunos.G_LAST_LAP_MS, last_ms)
            self.graphics[:] = bytes(g)

    def close(self):
        for m in (self.physics, self.graphics, self.static):
            if m is not None:
                m.close()


@windows
def test_acc_counts_laps_by_the_game_and_skips_replays(tmp_path):
    pages = Pages(f"acc{os.getpid()}")
    ctx = context(tmp_path)
    a = racing.KunosAdapter(id="acc-test", game="Assetto Corsa Competizione", flavour="acc", names=pages.names)
    try:
        a.start(ctx)
        for i, (p, v) in enumerate(circle(150, 100.0, 40.0, laps=2)):
            laps = 0 if i <= 150 else 1
            pages.write(p, 40.0, laps=laps, last_ms=95_250 if laps else 0, position=3)
            a.poll(ctx)
            if i % 15 == 0:
                time.sleep(0.01)
        for i in range(20):  # a replay afterwards: not driving
            pages.write((500 + i, 0, 0), 30.0, status=1)
            a.poll(ctx)
        a.stop(ctx)
    finally:
        pages.close()
    folder = only_session(ctx)
    (lap,) = events(folder, "lap")
    assert lap["lap_time"] == 95.25 and lap["lap"] == 0
    (start,) = events(folder, "session_start")
    assert start["track"] == "test track" and start["car"] == "test car"
    m = meta(folder)
    assert m["stats"]["laps"] == 1 and m["stats"]["race_position"] == 3
    assert m["track"] == "test track"
    rows = pq.read_table(folder / "heat.parquet").to_pylist()
    assert rows and max(r["x"] for r in rows) <= 100.01  # nothing from the replay
    assert {r["gear"] for r in rows if r["event"] == "position" and r["gear"] is not None} == {3}


@windows
def test_ac_rally_ends_a_stage_where_the_car_stops(tmp_path):
    pages = Pages(f"acr{os.getpid()}", graphics=False)
    ctx = context(tmp_path)
    a = racing.KunosAdapter(id="acr-test", game="Assetto Corsa Rally", flavour="acrally", names=pages.names)
    try:
        a.start(ctx)

        def at(seconds):
            # The adapter's clock, wound on: a stage of 50 s, then 10 s standing still.
            a._clock0 = time.perf_counter() - seconds

        for k in range(501):
            at(k * 0.1)
            pages.write((k * 1.0, 0.0, k * 1.0), 14.0)
            a.poll(ctx)
        for k in range(1, 101):
            at(50.0 + k * 0.1)
            pages.write((500.0, 0.0, 500.0), 0.0)
            a.poll(ctx)
        a.stop(ctx)
    finally:
        pages.close()
    folder = only_session(ctx)
    (finish,) = events(folder, "finish")
    assert 49.0 < finish["stage_time"] < 59.0
    assert meta(folder)["stats"]["stages"] == 1


@windows
def test_kunos_waits_for_the_game_to_make_its_pages(tmp_path):
    ctx = context(tmp_path)
    names = (f"Local\\heat3d_absent_{os.getpid()}_p", None, f"Local\\heat3d_absent_{os.getpid()}_s")
    a = racing.KunosAdapter(id="ac-test", game="Assetto Corsa EVO", flavour="acevo", names=names)
    a.start(ctx)
    a.poll(ctx)
    from heat3d_watch.adapters.simracing_shm import SharedView

    # Not created by the look: the game would otherwise find a read-only mapping of the watcher's.
    assert SharedView.open(names[0]) is None
    a.stop(ctx)
    assert not (ctx.data / "sessions").exists() or not list((ctx.data / "sessions").glob("*/*"))


# ------------------------------------------------------------------ rally games


def wrc_packet(*, pos, speed, stage_time, distance, length=1000.0) -> bytes:
    b = bytearray(rally.WRC_LENGTH)
    values = {
        "position": pos, "velocity": (0.0, 0.0, speed), "forward": (0.0, 0.0, 1.0), "left": (1.0, 0.0, 0.0),
        "up": (0.0, 1.0, 0.0), "speed": speed, "stage_time": stage_time, "stage_distance": distance,
        "stage_length": length, "throttle": 1.0, "gear_index": 3, "gear_neutral": 0, "gear_reverse": 9,
    }
    for offset, fmt, name in rally.WRC_FIELDS:
        if name in values:
            v = values[name]
            struct.pack_into("<" + fmt, b, offset, *(v if isinstance(v, tuple) else (v,)))
    return bytes(b)


def test_wrc_records_a_stage_to_the_finish(tmp_path):
    port = free_port()
    ctx = context(tmp_path)
    a = racing.RallyAdapter(id="wrc-test", game="EA SPORTS WRC", flavour="wrc", port=port)
    opened(a, ctx)
    packets = [wrc_packet(pos=(0.0, 0.0, 0.0), speed=0.0, stage_time=0.0, distance=0.0)]
    for k in range(1, 101):
        packets.append(wrc_packet(pos=(0.0, 0.0, k * 10.0), speed=25.0, stage_time=k * 0.4, distance=k * 10.0))
    packets.append(wrc_packet(pos=(0.0, 0.0, 1000.0), speed=5.0, stage_time=40.5, distance=1000.0))
    pump(a, ctx, packets, port, every=10, pause=0.01)
    a.stop(ctx)
    folder = only_session(ctx)
    names = [r["event"] for r in read_rows(folder) if r["event"] != "position"]
    assert names[0] == "session_start" and "finish" in names and "restart" not in names
    (finish,) = events(folder, "finish")
    assert finish["stage_time"] == 40.0
    m = meta(folder)
    assert m["stats"]["stages"] == 1 and m["stats"]["best_stage"] == 40.0
    assert m["stats"]["top_speed_kmh"] == 90.0
    assert (folder / "heat.parquet").exists()


def test_wrc_calls_a_stage_given_up_halfway_a_restart(tmp_path):
    ctx = context(tmp_path)
    a = racing.RallyAdapter(id="wrc-test", game="EA SPORTS WRC", flavour="wrc", port=free_port())
    a.start(ctx)
    for k in range(1, 60):
        a.feed(wrc_packet(pos=(0.0, 0.0, k * 10.0), speed=25.0, stage_time=k * 0.4, distance=k * 10.0))
    a.feed(wrc_packet(pos=(0.0, 0.0, 0.0), speed=0.0, stage_time=0.0, distance=0.0))
    a.stop(ctx)
    folder = only_session(ctx)
    assert events(folder, "restart") and not events(folder, "finish")


def test_dirt_counts_a_stage_when_its_lap_counter_moves(tmp_path):
    ctx = context(tmp_path)
    a = racing.RallyAdapter(id="dirt-test", game="DiRT Rally 2.0", flavour="dirtrally2", port=free_port())
    a.start(ctx)
    DR = rally.DR
    for k in range(1, 102):
        f = [0.0] * 66
        f[DR["lap_time"]] = k * 0.5
        f[DR["lap_distance"]] = k * 8.0
        f[DR["position"] + 2] = k * 8.0
        f[DR["speed"]] = 16.0
        f[DR["forward"] + 2] = 1.0
        f[DR["track_length"]] = 900.0  # the counter, not the distance, ends it
        f[DR["total_laps"]] = 1.0
        f[DR["laps_completed"]] = 1.0 if k == 101 else 0.0
        f[DR["last_lap_time"]] = 50.5 if k == 101 else 0.0
        a.feed(struct.pack("<66f", *f))
    a.stop(ctx)
    folder = only_session(ctx)
    (finish,) = events(folder, "finish")
    assert finish["stage_time"] == 50.5


def test_rbr_finds_its_up_axis_from_the_stage(tmp_path):
    ctx = context(tmp_path)
    a = racing.RallyAdapter(id="rbr-test", game="Richard Burns Rally", flavour="rbr", port=free_port())
    a.start(ctx)
    for k in range(1, 200):
        b = bytearray(rally.RBR_LENGTH)
        struct.pack_into("<f", b, 12, k * 0.2)  # race time
        struct.pack_into("<f", b, 20, 2000.0 - k * 5)  # distance to the end
        struct.pack_into("<3f", b, 64, k * 3.0, k * 4.0, 1.0 + 0.01 * k)  # a stage on the x-y plane, z up
        a.feed(bytes(b))
    a.stop(ctx)
    folder = only_session(ctx)
    m = meta(folder)
    assert m["axes"]["up"] == "z"
    assert json.loads(pq.read_metadata(folder / "heat.parquet").metadata[b"heat3d"])["axes"]["up"] == "z"
    # Speed from the positions over the stage clock: 5 m every 0.2 s.
    assert m["stats"]["top_speed_kmh"] == pytest.approx(25 * 3.6, rel=0.01)


# ------------------------------------------------------------------ BeamNG


def beamng_packet(t, pos, vel) -> bytes:
    from heat3d_capture.telemetry import beamng

    floats = [t, *pos, *vel, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0] + [0.5, 0.0, 0.0, 0.0, 0.0, 3000.0, 3.0, 0.0, 0.0] + [0.0] * 16
    assert len(floats) == beamng.FLOATS
    return beamng.MAGIC + struct.pack(f"<{beamng.FLOATS}f", *floats)


def test_beamng_records_a_reset_as_a_respawn(tmp_path):
    ctx = context(tmp_path)
    a = racing.BeamNGAdapter(id="beamng-test", game="BeamNG.drive", port=free_port())
    a.start(ctx)
    for k in range(50):
        a.feed(beamng_packet(10.0 + k * 0.1, (0.0, -k * 2.0, 0.0), (0.0, -20.0, 0.0)))
    a.feed(beamng_packet(0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    a.feed(b"not beamng")
    a.stop(ctx)
    folder = only_session(ctx)
    names = [r["event"] for r in read_rows(folder) if r["event"] != "position"]
    assert names[0] == "session_start" and "respawn" in names and names[-1] == "session_end"
    # BeamNG's -Y forwards turned into the viewer's +Z: (x, z, -y); the reset jumps back from there.
    (jump,) = events(folder, "teleport")
    assert jump["from_z"] == pytest.approx(98.0) and jump["from_y"] == 0 and jump["z"] == 0
    assert meta(folder)["stats"]["bad_packets"] == 1


# ------------------------------------------------------------------ switching the games' output on


def test_dirt_setup_changes_only_its_udp_line_and_keeps_a_backup(tmp_path, monkeypatch):
    monkeypatch.setattr(racing, "documents", lambda: tmp_path)
    config = tmp_path / "My Games" / "DiRT Rally 2.0" / "hardwaresettings" / "hardware_settings_config.xml"
    config.parent.mkdir(parents=True)
    original = '<?xml version="1.0"?>\n<hardware_settings_config>\n  <motion_platform>\n    <dbox enabled="true" />\n    <udp enabled="false" extradata="0" ip="127.0.0.1" port="20777" delay="1" />\n  </motion_platform>\n</hardware_settings_config>\n'
    config.write_text(original, encoding="utf-8")
    a = racing.RALLY[1]
    ctx = context(tmp_path)
    plan = a.setup(ctx, apply=False)
    assert any('enabled="false" -> "true"' in line for line in plan)
    assert config.read_text(encoding="utf-8") == original
    a.setup(ctx, apply=True)
    text = config.read_text(encoding="utf-8")
    assert '<udp enabled="true" extradata="3" ip="127.0.0.1" port="20777" delay="1" />' in text
    assert '<dbox enabled="true" />' in text
    backup = config.with_name(config.name + ".heat3d-backup")
    assert backup.read_text(encoding="utf-8") == original
    assert a.setup(ctx, apply=True) == []  # already on
    assert backup.read_text(encoding="utf-8") == original


def test_wrc_setup_switches_the_session_update_packet_on(tmp_path, monkeypatch):
    monkeypatch.setattr(racing, "documents", lambda: tmp_path)
    config = tmp_path / "My Games" / "WRC" / "telemetry" / "config.json"
    config.parent.mkdir(parents=True)
    doc = {"udp": {"packets": [{"structure": "wrc", "packet": "session_update", "ip": "127.0.0.1", "port": 20777,
                                "frequencyHz": 60, "bEnabled": False}]}}
    config.write_text(json.dumps(doc), encoding="utf-8")
    a = racing.RALLY[0]
    assert a.setup(context(tmp_path), apply=True)
    after = json.loads(config.read_text(encoding="utf-8"))["udp"]["packets"][0]
    assert after["bEnabled"] is True and after["frequencyHz"] == 60
    assert a.setup(context(tmp_path), apply=True) == []


def test_every_racing_adapter_says_where_from_and_what(tmp_path):
    for a in racing.ADAPTERS:
        assert a.id and a.game and a.exes and a.source and a.records and a.verified, a.id
        assert a.missing() is None or sys.platform != "win32"


# ------------------------------------------------------------------ against real laps


def _fh_laps() -> list[Path]:
    root = Path(os.environ.get("LOCALAPPDATA", "")) / "FHCompanion" / "laps"
    if not root.is_dir():
        return []
    return [p for p in sorted(root.rglob("*.json")) if p.name != "course.json"][:40]


@pytest.mark.local
def test_forza_on_real_fh_companion_laps_replayed_as_data_out(tmp_path):
    """
    FH Companion's own laps (positions and speeds the game sent) are turned back
    into Data Out packets and driven through the adapter: the distance it adds
    up has to be the lap's length.
    """
    from heat3d_capture.telemetry.fhcompanion import read_lap

    laps = []
    for path in _fh_laps():
        try:
            lap = read_lap(path)
        except Exception:  # noqa: BLE001
            continue
        if len(lap) > 50 and lap.metres > 500:
            laps.append(lap)
        if len(laps) == 3:
            break
    if not laps:
        pytest.skip("no FH Companion laps on this machine")
    for n, lap in enumerate(laps):
        ctx = context(tmp_path / str(n))
        a = racing.ForzaAdapter(id="forza-local", game="Forza Horizon 6", port=free_port())
        a.start(ctx)
        speeds = lap.channels["speed_ms"]
        previous = None
        for i, (x, y, z) in enumerate(lap.positions):
            # Velocity along the path, so the layout check (|velocity| == speed) holds as in the game.
            if previous is None:
                direction = (1.0, 0.0, 0.0)
            else:
                dx, dy, dz = x - previous[0], y - previous[1], z - previous[2]
                norm = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
                direction = (dx / norm, dy / norm, dz / norm)
            previous = (x, y, z)
            s = float(speeds[i])
            stamp = int(float(lap.channels["elapsed"][i]) * 1000)
            a.feed(forza_packet(pos=(float(x), float(y), float(z)), vel=tuple(s * c for c in direction), lap=0, stamp=stamp))
        a.feed(forza_packet(pos=tuple(map(float, lap.positions[-1])), vel=(0.0, 0.0, 0.0), lap=1, last_lap=lap.seconds))
        a.stop(ctx)
        folder = only_session(ctx)
        m = meta(folder)
        assert m["stats"]["laps"] == 1 and m["stats"]["best_lap"] == round(lap.seconds, 3)
        assert m["stats"]["distance_km"] * 1000 == pytest.approx(lap.metres, rel=0.05)
        assert m["stats"]["top_speed_kmh"] == pytest.approx(float(max(speeds)) * 3.6, rel=0.01)
        assert not events(folder, "teleport")
        assert (folder / "heat.parquet").exists()
