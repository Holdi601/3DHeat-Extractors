"""The Assetto Corsa family recorder, on synthetic shared-memory pages."""

import json
import math
import struct

from heat3d_capture.telemetry import kunos


def physics_page(*, x=10.0, y=2.0, z=-5.0, speed_kmh=108.0, gas=1.0, brake=0.0, gear=4, steer=0.1, packet=7, full=True):
    page = bytearray(kunos.PHYSICS_SIZE if full else kunos.PHYSICS_AC1_SIZE)
    struct.pack_into("<i", page, 0, packet)
    struct.pack_into("<f", page, 4, gas)
    struct.pack_into("<f", page, 8, brake)
    struct.pack_into("<i", page, 16, gear)
    struct.pack_into("<i", page, 20, 6500)
    struct.pack_into("<f", page, 24, steer)
    struct.pack_into("<f", page, 28, speed_kmh)
    struct.pack_into("<3f", page, 44, 1.5, 0.1, -0.5)
    struct.pack_into("<4f", page, 152, 85, 86, 90, 91)  # core temperature
    struct.pack_into("<4f", page, 184, 0.02, 0.03, 0.04, 0.05)  # suspension travel
    struct.pack_into("<4f", page, 348, 400, 410, 300, 310)  # brake temperature
    struct.pack_into("<f", page, 364, 0.0)
    for i, (dx, dz) in enumerate([(1, 1.4), (-1, 1.4), (1, -1.4), (-1, -1.4)]):
        struct.pack_into("<3f", page, kunos.TYRE_CONTACT_POINT + i * 12, x + dx, y - 0.3, z + dz)
    if full:
        struct.pack_into("<4f", page, 640, -0.06, -0.06, 0.12, 0.24)  # slip ratio
        struct.pack_into("<f", page, 564, 0.57)  # brake bias
        struct.pack_into("<i", page, 676, 1)  # ABS acting
        struct.pack_into("<4f", page, 740, 0.9, 0.9, 0.95, 0.95)  # pad life
    return bytes(page)


def test_decodes_the_shared_layout():
    raw = kunos.decode_physics(physics_page())
    frame = kunos.frame_from(raw, 1.5)
    assert frame["x"] == 10.0 and frame["z"] == -5.0
    assert abs(frame["y"] - 1.7) < 1e-6
    assert abs(frame["speed"] - 30.0) < 1e-5
    assert frame["gear"] == 3
    assert abs(frame["latAcc"] - 1.5 * kunos.G) < 1e-4
    assert frame["tireTemp.RR"] == 91
    assert frame["brakeTemp.FL"] == 400
    # Slip ratio normalised: 0.12 is the edge of grip, and so reads 1.
    assert abs(frame["slipRatio.RL"] - 1.0) < 1e-5
    assert abs(frame["slipRatioRaw.RR"] - 0.24) < 1e-6
    # The aids and settings, under the viewer's names.
    assert abs(frame["brakeBias"] - 0.57) < 1e-6
    assert frame["absActive"] == 1 and frame["tcActive"] == 0
    assert abs(frame["padLife.RL"] - 0.95) < 1e-6


def test_first_game_page_has_no_slip():
    frame = kunos.frame_from(kunos.decode_physics(physics_page(full=False)), 0.0)
    assert "slipRatio.FL" not in frame
    assert "brakeBias" not in frame and "absActive" not in frame
    assert frame["tireTemp.FL"] == 85


def test_no_position_no_frame():
    page = bytearray(kunos.PHYSICS_SIZE)
    assert kunos.frame_from(kunos.decode_physics(bytes(page)), 0.0) is None


def circle_frames(laps=2.5, radius=100.0, speed=30.0, hz=50):
    circumference = 2 * math.pi * radius
    n = int(laps * circumference / speed * hz)
    for i in range(n):
        t = i / hz
        a = (speed * t) / radius
        page = physics_page(x=radius * math.sin(a), z=radius * (1 - math.cos(a)), speed_kmh=speed * 3.6, packet=i + 1)
        yield kunos.frame_from(kunos.decode_physics(page), t)


def test_laps_by_the_line_when_there_is_no_counter():
    splitter = kunos.LineSplitter()
    laps = []
    for frame in circle_frames():
        laps += splitter.feed(frame)
    laps += splitter.flush()
    # Two whole laps from the starting point, and the half lap after them.
    complete = [lap for lap in laps if lap.complete]
    assert len(laps) == 3
    assert len(complete) == 2
    period = 2 * math.pi * 100 / 30
    assert abs(laps[0].seconds - period) < 0.1
    assert abs(complete[0].seconds - period) < 0.1


def test_laps_by_the_game_counter():
    splitter = kunos.CounterSplitter()
    laps = []
    for i, frame in enumerate(circle_frames(laps=2.2)):
        lap_number = int(frame["time"] / (2 * math.pi * 100 / 30))
        laps += splitter.feed(frame, lap_number, 20944 if lap_number else 0)
    assert len(laps) == 2
    assert laps[1].complete and laps[1].seconds == 20.944


def test_writes_a_heat3d_lap(tmp_path):
    lap = kunos.KLap(number=1, frames=list(circle_frames(laps=1.0)), complete=True)
    path = kunos.write(lap, tmp_path, game="acc", track="Monza", car="Ferrari 296 GT3")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["format"] == "heat3d-lap"
    assert doc["game"] == "assetto-corsa-competizione"
    assert doc["track"] == "Monza" and doc["car"]["name"] == "Ferrari 296 GT3"
    ch = doc["channels"]
    assert ch["time"][0] == 0 and ch["distance"][-1] > 600
    assert set(ch["suspension.FL"]) == {0.0} or max(ch["suspension.FL"]) <= 1.0
    assert "steerRaw" not in ch and ch["steer"][0] == 0.1


def test_reads_the_static_page():
    page = bytearray(256)
    page[68 : 68 + 2 * 12] = "porsche_992\x00".encode("utf-16-le")
    page[134 : 134 + 2 * 6] = "monza\x00".encode("utf-16-le")
    assert kunos.read_static(bytes(page), "acc") == ("monza", "porsche_992")
    evo = bytearray(256)
    evo[136:141] = b"Imola"
    assert kunos.read_static(bytes(evo), "acevo")[0] == "Imola"


def test_drift_angle_aids_and_kelvin_by_game():
    page = bytearray(physics_page())
    # The car's own velocity: 3 m/s sideways at 30 forwards.
    struct.pack_into("<3f", page, kunos.LOCAL_VELOCITY, 3.0, 0.0, 30.0)
    struct.pack_into("<f", page, kunos.TC_ACTING, 0.4)
    # AC Rally sends temperatures in kelvin.
    struct.pack_into("<4f", page, 348, 673.15, 683.15, 573.15, 583.15)
    raw = kunos.decode_physics(bytes(page))
    rally = kunos.frame_from(raw, 1.0, "acrally")
    assert abs(rally["latVel"] - 3.0) < 1e-6 and abs(rally["longVel"] - 30.0) < 1e-6
    assert abs(rally["brakeTemp.FL"] - 400.0) < 1e-3
    # ACC reports the aids acting in those slots; AC keeps its settings there.
    assert kunos.frame_from(raw, 1.0, "acc")["tcActive"] == 1.0
    assert kunos.frame_from(raw, 1.0, "ac")["tcActive"] == 0
    # Celsius elsewhere stays as it is.
    assert abs(kunos.frame_from(raw, 1.0, "acc")["brakeTemp.FL"] - 673.15) < 1e-3


def test_the_lap_says_what_its_slip_means(tmp_path):
    lap = kunos.KLap(number=1, frames=list(circle_frames(laps=1.0)), complete=True)
    doc = json.loads(kunos.write(lap, tmp_path, game="acrally", track=None, car=None).read_text(encoding="utf-8"))
    assert doc["meta"] == {"slip": "scaled", "discipline": "rally"}
