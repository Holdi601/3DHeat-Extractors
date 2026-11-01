"""ETS2/ATS: the RenCloud plugin's revision 12 layout, re-derived from its header, and a drive in shared memory."""

from __future__ import annotations

import ctypes as C
import json
import mmap
import os
import sys
from dataclasses import replace

import pytest

from heat3d_watch.adapters.base import Context
from heat3d_watch.adapters.ets2 import ADAPTERS, READ, SPECIAL, STRINGS, decode
from heat3d_watch.config import Settings
from heat3d_watch.session import read_rows

S = 64  # stringsize


class Map(C.Structure):
    """scsTelemetryMap_t of scs-telemetry-common.hpp (PLUGIN_REVID 12), zones 1-12, field by field.

    Natural alignment, as the plugin is compiled; the header marks each zone's
    start offset, and the asserts below hold this transcription to them.
    """

    _fields_ = [
        ("sdkActive", C.c_bool), ("placeHolder", C.c_char * 3), ("paused", C.c_bool), ("placeHolder2", C.c_char * 3),
        ("time", C.c_uint64), ("simulatedTime", C.c_uint64), ("renderTime", C.c_uint64),
        ("multiplayerTimeOffset", C.c_int64),
        # zone 2 @ 40
        ("telemetry_plugin_revision", C.c_uint32), ("version_major", C.c_uint32), ("version_minor", C.c_uint32),
        ("game", C.c_uint32), ("telemetry_version_game_major", C.c_uint32), ("telemetry_version_game_minor", C.c_uint32),
        ("time_abs", C.c_uint32),
        ("config_ui", C.c_uint32 * 9),
        ("truck_ui_head", C.c_uint32 * 4), ("truck_wheelSubstance", C.c_uint32 * 16),
        ("hshifterPosition", C.c_uint32 * 32), ("hshifterBitmask", C.c_uint32 * 32),
        ("gameplay_ui", C.c_uint32 * 3), ("buffer_ui", C.c_char * 48),
        # zone 3 @ 500
        ("restStop", C.c_int32), ("gear", C.c_int32), ("gearDashboard", C.c_int32), ("hshifterResulting", C.c_int32 * 32),
        ("jobDeliveredEarnedXp", C.c_int32), ("buffer_i", C.c_char * 56),
        # zone 4 @ 700
        ("scale", C.c_float),
        ("config_f_head", C.c_float * 12), ("truckWheelRadius", C.c_float * 16), ("gearRatiosForward", C.c_float * 24),
        ("gearRatiosReverse", C.c_float * 8), ("unitMass", C.c_float),
        ("speed", C.c_float), ("truck_f_1", C.c_float * 26), ("truckOdometer", C.c_float), ("truck_f_2", C.c_float * 3),
        ("truck_f_wheels", C.c_float * 96),
        ("jobDeliveredCargoDamage", C.c_float), ("jobDeliveredDistanceKm", C.c_float), ("refuelAmount", C.c_float),
        ("cargoDamage", C.c_float), ("buffer_f", C.c_char * 28),
        # zone 5 @ 1500
        ("config_b", C.c_bool * 66), ("truck_b", C.c_bool * 47), ("gameplay_b", C.c_bool * 2), ("buffer_b", C.c_char * 25),
        # zone 6 @ 1640
        ("config_fv", C.c_float * 57), ("truck_fv", C.c_float * 18), ("buffer_fv", C.c_char * 60),
        # zone 7 @ 2000
        ("truck_fp", C.c_float * 12), ("buffer_fp", C.c_char * 152),
        # zone 8 @ 2200
        ("coordinateX", C.c_double), ("coordinateY", C.c_double), ("coordinateZ", C.c_double),
        ("rotationX", C.c_double), ("rotationY", C.c_double), ("rotationZ", C.c_double), ("buffer_dp", C.c_char * 52),
        # zone 9 @ 2300
        ("truckBrandId", C.c_char * S), ("truckBrand", C.c_char * S), ("truckId", C.c_char * S), ("truckName", C.c_char * S),
        ("cargoId", C.c_char * S), ("cargo", C.c_char * S), ("cityDstId", C.c_char * S), ("cityDst", C.c_char * S),
        ("compDstId", C.c_char * S), ("compDst", C.c_char * S), ("citySrcId", C.c_char * S), ("citySrc", C.c_char * S),
        ("compSrcId", C.c_char * S), ("compSrc", C.c_char * S), ("shifterType", C.c_char * 16),
        ("truckLicensePlate", C.c_char * S), ("truckLicensePlateCountryId", C.c_char * S),
        ("truckLicensePlateCountry", C.c_char * S), ("jobMarket", C.c_char * 32),
        ("fineOffence", C.c_char * 32), ("gameplay_s", C.c_char * (8 * S)), ("buffer_s", C.c_char * 20),
        # zone 10 @ 4000
        ("jobIncome", C.c_uint64), ("buffer_ull", C.c_char * 192),
        # zone 11 @ 4200
        ("jobCancelledPenalty", C.c_int64), ("jobDeliveredRevenue", C.c_int64), ("fineAmount", C.c_int64),
        ("tollgatePayAmount", C.c_int64), ("ferryPayAmount", C.c_int64), ("trainPayAmount", C.c_int64),
        ("buffer_ll", C.c_char * 52),
        # zone 12 @ 4300
        ("special_b", C.c_bool * 10), ("buffer_special", C.c_char * 90),
    ]


def test_the_transcribed_header_lands_on_the_zones_it_marks():
    zones = {"telemetry_plugin_revision": 40, "restStop": 500, "scale": 700, "config_b": 1500, "config_fv": 1640,
             "truck_fp": 2000, "coordinateX": 2200, "truckBrandId": 2300, "jobIncome": 4000,
             "jobCancelledPenalty": 4200, "special_b": 4300}
    for name, offset in zones.items():
        assert getattr(Map, name).offset == offset, name
    assert C.sizeof(Map) == READ
    # ... and the offsets the adapter reads are the fields they claim to be.
    used = {"time_abs": 64, "jobDeliveredEarnedXp": 640, "speed": 948, "truckOdometer": 1056,
            "jobDeliveredDistanceKm": 1460, "fineOffence": 3436, "truckBrand": STRINGS["truck_brand"],
            "truckName": STRINGS["truck_name"], "cargo": STRINGS["cargo"], "cityDst": STRINGS["city_dst"],
            "compDst": STRINGS["company_dst"], "citySrc": STRINGS["city_src"], "compSrc": STRINGS["company_src"]}
    for name, offset in used.items():
        assert getattr(Map, name).offset == offset, name


def snapshot(**kw) -> bytes:
    m = Map()
    m.sdkActive = kw.get("active", True)
    m.paused = kw.get("paused", False)
    m.telemetry_plugin_revision = kw.get("revision", 12)
    m.version_major, m.version_minor, m.game = 1, 50, 1
    m.coordinateX, m.coordinateY, m.coordinateZ = kw.get("pos", (0.0, 0.0, 0.0))
    m.rotationX = kw.get("heading", 0.25)
    m.speed = kw.get("speed", 0.0)
    m.truckOdometer = kw.get("odometer", 1000.0)
    m.truckName = b"Truck"
    m.cargo, m.citySrc, m.cityDst = b"Cargo", b"Town A", b"Town B"
    m.jobDeliveredRevenue, m.jobDeliveredEarnedXp, m.jobDeliveredDistanceKm = 12000, 500, 321.5
    m.fineAmount = 250
    m.fineOffence = b"speeding"
    for i, name in enumerate(SPECIAL):
        m.special_b[i] = name in kw.get("flags", ())
    return bytes(m)


def test_decode_reads_the_fields():
    d = decode(snapshot(pos=(1000.5, 50.25, -2000.75), speed=20.0, flags=("on_job",)))
    assert (d["x"], d["y"], d["z"]) == (1000.5, 50.25, -2000.75) and d["heading"] == 0.25
    assert d["speed"] == 20.0 and d["truck_name"] == "Truck" and d["city_dst"] == "Town B"
    assert d["on_job"] and not d["fined"] and d["revenue"] == 12000 and d["game"] == "ets2"
    with pytest.raises(ValueError):
        decode(snapshot(revision=11))
    with pytest.raises(ValueError):
        decode(b"\0" * 100)


@pytest.mark.skipif(sys.platform != "win32", reason="named shared memory as the plugin makes it")
def test_a_drive_read_from_shared_memory(tmp_path):
    name = f"Local\\heat3d-test-scs-{os.getpid()}"
    mem = mmap.mmap(-1, 32 * 1024, tagname=name)  # standing in for the plugin
    a = replace(ADAPTERS[0], name=name)
    ctx = Context(Settings(data=str(tmp_path)), tmp_path)
    a.start(ctx)

    def frame(**kw):
        mem.seek(0)
        mem.write(snapshot(**kw))
        a.poll(ctx)

    frame(pos=(100.0, 10.0, 200.0), flags=("on_job",))
    folder = a._session.folder
    for i in range(1, 5):
        frame(pos=(100.0 + i * 2, 10.0, 200.0), speed=20.0, flags=("on_job",))
    frame(pos=(108.0, 10.0, 200.0), flags=("on_job", "fined"))
    frame(pos=(108.0, 10.0, 200.0), flags=("on_job", "fined"))  # still set: one fine, not two
    frame(pos=(5000.0, 0.0, 9000.0), flags=("on_job", "ferry"))  # across the sea
    frame(pos=(5003.0, 0.0, 9000.0), flags=("job_delivered",), speed=10.0)
    frame(pos=(5004.0, 0.0, 9000.0), active=False)
    a.stop(ctx)
    mem.close()
    rows = read_rows(folder)
    assert [r["event"] for r in rows if r["event"] != "position"] == [
        "spawn", "job_start", "fine", "teleport", "spawn", "ferry", "delivered", "session_end",
    ]
    pos = [r for r in rows if r["event"] == "position"]
    assert len(pos) == 5 and pos[0]["x"] == 102.0 and pos[0]["heading"] == 90.0 and pos[0]["speed_kmh"] == 72.0
    teleport = next(r for r in rows if r["event"] == "teleport")
    assert teleport["x"] == 108.0 and teleport["session"] == "drive-1"
    assert rows[-1]["session"] == "drive-2"
    job = next(r for r in rows if r["event"] == "job_start")
    assert job["from_city"] == "Town A" and job["to_city"] == "Town B" and job["cargo"] == "Cargo"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    st = meta["stats"]
    assert st["deliveries"] == 1 and st["revenue"] == 12000 and st["xp"] == 500 and st["fines"] == 1
    assert st["fines_total"] == 250 and st["max_speed_kmh"] == 72.0
    assert st["distance_km"] == pytest.approx(0.011)  # 8 m driven, then 3 m: the ferry is not distance
    assert meta["axes"]["up"] == "y" and meta["plugin_revision"] == 12


def test_no_plugin_means_nothing_happens(tmp_path):
    a = replace(ADAPTERS[1], name=f"Local\\heat3d-test-absent-{os.getpid()}")
    ctx = Context(Settings(data=str(tmp_path)), tmp_path)
    a.start(ctx)
    a.poll(ctx)
    a.stop(ctx)
    assert not (tmp_path / "sessions").exists() or not any((tmp_path / "sessions").rglob("meta.json"))
