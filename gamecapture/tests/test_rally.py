"""The rally games' packets, built from their documented layouts, into stage files."""

from __future__ import annotations

import json
import math
import struct

from heat3d_capture.telemetry import rally


def wrc_packet(*, t: float, x: float, speed: float, heading: float = 0.0, slide: float = 0.0,
               stage_distance: float = 0.0, stage_length: float = 1000.0, wheel_speed: float | None = None,
               handbrake: float = 0.0) -> bytes:
    """A default `wrc` session_update packet: a car heading `heading` (rad from +Z), sliding `slide` rad."""
    buf = bytearray(rally.WRC_LENGTH)
    fwd = (math.sin(heading), 0.0, math.cos(heading))
    # X is left in EA's frame: left of a car heading +Z is +X.
    left = (math.cos(heading), 0.0, -math.sin(heading))
    travel = heading + slide
    vel = (speed * math.sin(travel), 0.0, speed * math.cos(travel))
    values = {
        "packet_uid": int(t * 100),
        "game_total_time": t,
        "game_delta_time": 0.01,
        "gear_index": 3,
        "gear_neutral": 0,
        "gear_reverse": 10,
        "speed": speed,
        "transmission_speed": speed,
        "position": (x, 5.0, t * speed),
        "velocity": vel,
        "acceleration": (0.0, 0.0, 0.0),
        "left": left,
        "forward": fwd,
        "up": (0.0, 1.0, 0.0),
        "hub_position": (0.05, 0.05, 0.05, 0.05),
        "hub_velocity": (0.0, 0.0, 0.0, 0.0),
        "cp_forward_speed": (wheel_speed or speed,) * 4,
        "brake_temperature": (300.0, 300.0, 400.0, 400.0),
        "rpm_max": 8000.0,
        "rpm_idle": 900.0,
        "rpm": 6000.0,
        "throttle": 1.0,
        "brake": 0.0,
        "clutch": 0.0,
        "steering": 0.25,
        "handbrake": handbrake,
        "stage_time": t,
        "stage_distance": stage_distance,
        "stage_length": stage_length,
    }
    for offset, fmt, name in rally.WRC_FIELDS:
        v = values[name]
        struct.pack_into("<" + fmt, buf, offset, *(v if isinstance(v, tuple) else (v,)))
    return bytes(buf)


def test_wrc_packet_decodes_into_the_car_frame():
    raw = rally.decode_wrc(wrc_packet(t=1.0, x=10.0, speed=20.0, heading=0.3, slide=0.2))
    frame = rally.frame_wrc(raw)
    # The velocity points 0.2 rad off the nose: the drift angle comes back.
    assert math.isclose(math.atan2(abs(frame["latVel"]), frame["longVel"]), 0.2, abs_tol=1e-4)
    assert math.isclose(frame["longVel"], 20.0 * math.cos(0.2), rel_tol=1e-5)
    # Heading from the forward vector.
    assert math.isclose(frame["yaw"], 0.3, abs_tol=1e-5)
    # BL BR FL FR in the packet, FL FR RL RR in the file.
    assert frame["brakeTemp.FL"] == 400.0 and frame["brakeTemp.RL"] == 300.0
    assert frame["steer"] == 0.25 and frame["gear"] == 3


def test_wrc_stage_opens_on_the_clock_and_closes_at_the_finish(tmp_path):
    rec = rally.RallyRecorder(tmp_path, game="wrc")
    written = []
    for k in range(0, 60):
        t = k * 0.5
        written += rec.feed(wrc_packet(t=t, x=0.0, speed=30.0, stage_distance=t * 30.0, stage_length=870.0))
    assert len(written) == 1
    doc = json.loads(written[0].read_text())
    assert doc["format"] == "heat3d-lap" and doc["game"] == "ea-sports-wrc"
    assert doc["meta"]["discipline"] == "rally"
    assert doc["lap"]["complete"] is True
    assert "wheelSurfaceSpeed.FL" in doc["channels"] and "latVel" in doc["channels"]
    # Suspension comes out as 0..1 of the travel, beside the metres.
    assert "suspension.FL" in doc["channels"]


def test_wrc_restart_keeps_the_attempt_as_partial(tmp_path):
    rec = rally.RallyRecorder(tmp_path, game="wrc")
    out = []
    for k in range(1, 40):
        out += rec.feed(wrc_packet(t=k * 0.5, x=0.0, speed=30.0, stage_distance=k * 15.0, stage_length=5000.0))
    # Back to the start line.
    out += rec.feed(wrc_packet(t=0.0, x=0.0, speed=0.0, stage_distance=0.0, stage_length=5000.0))
    assert len(out) == 1
    assert json.loads(out[0].read_text())["lap"]["complete"] is False


def dirt_packet(*, lap_time: float, distance: float, length: float = 800.0, laps_completed: float = 0.0,
                speed: float = 25.0, last_lap: float = 0.0) -> bytes:
    f = [0.0] * 66
    f[rally.DR["total_time"]] = lap_time + 5
    f[rally.DR["lap_time"]] = lap_time
    f[rally.DR["lap_distance"]] = distance
    f[rally.DR["position"] : rally.DR["position"] + 3] = [1.0, 2.0, distance]
    f[rally.DR["speed"]] = speed
    f[rally.DR["velocity"] : rally.DR["velocity"] + 3] = [0.0, 0.0, speed]
    f[rally.DR["forward"] : rally.DR["forward"] + 3] = [0.0, 0.0, 1.0]
    f[rally.DR["side"] : rally.DR["side"] + 3] = [1.0, 0.0, 0.0]
    for k in range(4):
        f[rally.DR["wheel_patch_speed"] + k] = speed * (1.2 if k < 2 else 1.0)
        f[rally.DR["brake_temp"] + k] = 100.0 + k
    f[rally.DR["throttle"]] = 1.0
    f[rally.DR["gear"]] = 4
    f[rally.DR["rpm10"]] = 650
    f[rally.DR["laps_completed"]] = laps_completed
    f[rally.DR["total_laps"]] = 1
    f[rally.DR["track_length"]] = length
    f[rally.DR["last_lap_time"]] = last_lap
    return struct.pack("<66f", *f)


def test_dirt_rally_2_stage(tmp_path):
    rec = rally.RallyRecorder(tmp_path, game="dirtrally2")
    out = []
    for k in range(0, 50):
        t = k * 0.7
        d = t * 25.0
        finished = d >= 800
        out += rec.feed(dirt_packet(lap_time=t, distance=d, laps_completed=1.0 if finished else 0.0, last_lap=32.0 if finished else 0.0))
    assert len(out) == 1
    doc = json.loads(out[0].read_text())
    assert doc["lap"]["complete"] and doc["lap"]["seconds"] == 32.0
    ch = doc["channels"]
    # RL RR FL FR in the packet: the rears are the first two.
    assert ch["wheelSurfaceSpeed.RL"][5] == ch["speed"][5] * 1.2
    assert ch["brakeTemp.FL"][0] == 102.0
    assert ch["rpm"][0] == 6500.0


def rbr_packet(*, t: float, y: float, steps: int, distance_to_end: float) -> bytes:
    buf = bytearray(rally.RBR_LENGTH)
    struct.pack_into("<I", buf, 0, steps)
    struct.pack_into("<f", buf, 12, t)
    struct.pack_into("<f", buf, 20, distance_to_end)
    struct.pack_into("<f", buf, 28, 1.0)
    struct.pack_into("<i", buf, 44, 4)
    # RBR's up axis is not documented: here the stage runs along y and x,
    # and z holds the height.
    struct.pack_into("<3f", buf, 64, 0.5 * y, y, 3.0 + 0.01 * y)
    struct.pack_into("<3f", buf, 88, 30.0, 1.0, 0.0)
    return bytes(buf)


def test_rbr_puts_up_in_y_and_speed_from_positions(tmp_path):
    rec = rally.RallyRecorder(tmp_path, game="rbr")
    out = []
    for k in range(1, 60):
        t = k * 0.5
        out += rec.feed(rbr_packet(t=t, y=t * 30.0, steps=k, distance_to_end=800 - t * 30.0))
    assert len(out) == 1
    ch = json.loads(out[0].read_text())["channels"]
    # Height varied least: it is now y.
    assert max(ch["y"]) - min(ch["y"]) < 20
    # 30 m/s along y, 15 along x: about 33.5 m/s from the positions.
    assert abs(ch["speed"][10] - math.hypot(30, 15)) < 1.0
    assert ch["gear"][0] == 3
