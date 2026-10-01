"""BeamNG's heat3d protocol packets, as the Lua mod packs them, into lap files."""

from __future__ import annotations

import json
import math
import struct

from heat3d_capture.telemetry import beamng


def packet(*, t: float, pos, vel, fwd, up=(0.0, 0.0, 1.0), speeds=(20.0, 20.0, 24.0, 24.0), material=1.0) -> bytes:
    floats = [t, *pos, *vel, *fwd, *up, 0.1, 1.0, 0.0, 0.0, 0.0, 0.2, 6000.0, 3.0, 0.0, 0.0]
    floats += list(speeds)  # surface speed
    floats += [s / 0.33 for s in speeds]  # rotation
    floats += [3000.0] * 4  # load
    floats += [material] * 4
    assert len(floats) == beamng.FLOATS
    return beamng.MAGIC + struct.pack(f"<{beamng.FLOATS}f", *floats)


def test_packet_layout_matches_the_lua_struct():
    # char[4] then 39 floats, as the mod's getStructDefinition declares.
    assert beamng.LENGTH == 4 + 4 * 39
    lua = (beamng.MOD / "lua" / "vehicle" / "protocols" / "heat3d.lua").read_text(encoding="utf-8")
    declared = lua[lua.index("[[") : lua.index("]]")]
    count = 0
    for line in declared.splitlines():
        line = line.strip().rstrip(";")
        if not line.startswith("float"):
            continue
        for name in line[len("float") :].split(","):
            count += int(name[name.index("[") + 1 : name.index("]")]) if "[" in name else 1
    assert count == beamng.FLOATS


def test_world_turns_to_y_up_and_the_car_frame_comes_out():
    # BeamNG: Z up. The car points along world +Y and slides 0.2 rad to its side.
    speed = 25.0
    slide = 0.2
    fwd = (0.0, 1.0, 0.0)
    vel = (speed * math.sin(slide), speed * math.cos(slide), 0.0)
    frame = beamng.frame_from(beamng.decode(packet(t=1.0, pos=(10.0, 20.0, 5.0), vel=vel, fwd=fwd)))
    # (x, y, z) -> (x, z, -y): height is y now.
    assert (frame["x"], frame["y"], frame["z"]) == (10.0, 5.0, -20.0)
    assert math.isclose(math.atan2(abs(frame["latVel"]), frame["longVel"]), slide, abs_tol=1e-5)
    assert math.isclose(frame["speed"], speed, rel_tol=1e-6)
    assert frame["wheelSurfaceSpeed.RL"] == 24.0 and frame["groundContact.FL"] == 1.0


def test_off_the_ground_when_no_material():
    frame = beamng.frame_from(beamng.decode(packet(t=1.0, pos=(0, 0, 0), vel=(0, 10, 0), fwd=(0, 1, 0), material=-1.0)))
    assert frame["groundContact.FL"] == 0.0


def test_a_lap_round_a_circle_is_written(tmp_path):
    rec = beamng.Recorder(tmp_path)
    out = []
    radius = 100.0
    speed = 30.0
    for k in range(0, 1400):
        t = k * 0.05
        a = speed * t / radius
        pos = (radius * math.sin(a), radius * (1 - math.cos(a)), 0.0)
        vel = (speed * math.cos(a), speed * math.sin(a), 0.0)
        fwd = (math.cos(a), math.sin(a), 0.0)
        out += rec.feed(packet(t=t, pos=pos, vel=vel, fwd=fwd))
    assert out, "a lap closes when the car comes back past where it started"
    doc = json.loads(out[0].read_text())
    assert doc["game"] == "beamng" and doc["lap"]["complete"]
    assert abs(doc["lap"]["metres"] - 2 * math.pi * radius) < 25
    # Centripetal acceleration from the velocity: v^2 / r sideways.
    lat = doc["channels"]["latAcc"]
    assert abs(abs(lat[len(lat) // 2]) - speed**2 / radius) < 0.5
