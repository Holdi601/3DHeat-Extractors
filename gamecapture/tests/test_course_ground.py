"""
Does the extracted ground sit under the recorded laps?

This is the check the whole race pipeline rests on, and it is the only one that
can catch the failure that matters. Everything else — the archive index, the
vertex format, the tile grid — can be wrong in a way that still produces a
handsome mesh; what it produces is a handsome mesh of somewhere else. Nothing
about the geometry says so. The telemetry does: the car drove on that ground, so
the ground has to be under the car.

It is also a real measurement rather than a tolerance someone picked. Across
the 38 courses of one recorded library the car sits a median of 0.28 to 0.47 m
above the surface directly beneath it - that is ride height - and there is no
room in it for an error of even a couple of metres, let alone the tile-sized
ones that are easy to make. The surface is sampled from every triangle of the
export, bridges and expressway decks included, and "beneath" means the highest
surface not above the car: under a deck, the car is on the street.

A course with jumps fails the last two checks by design. On a cross-country
course the car is up to 26 m in the air for a seventh of the lap, over
ground that is there, and positions alone cannot tell a jump from a car
floating over a hole.

Skips unless both halves are present, since they are a cut of someone's own play
data and a cut of a game they own:

    python -m heat3d_capture laps "Lakeside Circuit" -o courses/lakeside_circuit
    python -m heat3d_gamefiles course "Lakeside Circuit" -o courses/lakeside_circuit.glb
    HEAT3D_COURSE=courses/lakeside_circuit pytest tests/test_course_ground.py
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest


COURSE = os.environ.get("HEAT3D_COURSE", "")
STEM = Path(COURSE) if COURSE else None
present = bool(
    STEM and STEM.with_suffix(".glb").exists() and STEM.with_suffix(".json").exists()
)

pytestmark = pytest.mark.skipif(
    not present,
    reason="set HEAT3D_COURSE to the stem of an exported course and its .glb",
)


#: Component type -> numpy type, for index buffers.
INDEX_TYPES = {5121: "<u1", 5123: "<u2", 5125: "<u4"}


def read_glb(path: Path):
    """
    Every triangle in the file, as (n, 3, 3) corners.

    Every part: the writer gives each its own vertex array, so reading only the
    first accessor reads one terrain tile. And not only the ground: an
    expressway deck or a bridge is a placed structure in the export, and a car
    driving over it is on it.
    """
    raw = path.read_bytes()
    json_length = struct.unpack_from("<I", raw, 12)[0]
    document = json.loads(raw[20 : 20 + json_length])
    binary = raw[28 + json_length :]
    accessors, views = document["accessors"], document["bufferViews"]
    blocks = []
    for mesh in document["meshes"]:
        for primitive in mesh["primitives"]:
            spec = accessors[primitive["attributes"]["POSITION"]]
            view = views[spec["bufferView"]]
            corners = np.frombuffer(
                binary, dtype="<f4", count=spec["count"] * 3,
                offset=view["byteOffset"] + spec.get("byteOffset", 0),
            ).reshape(-1, 3)
            spec = accessors[primitive["indices"]]
            view = views[spec["bufferView"]]
            faces = np.frombuffer(
                binary, dtype=INDEX_TYPES[spec["componentType"]], count=spec["count"],
                offset=view["byteOffset"] + spec.get("byteOffset", 0),
            ).reshape(-1, 3)
            blocks.append(corners[faces.astype(np.int64)].astype(np.float64))
    return np.concatenate(blocks)


def surface(triangles: np.ndarray, spacing: float = 0.5) -> np.ndarray:
    """Points over every triangle, about `spacing` apart."""
    edges = np.stack(
        [
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ],
        axis=1,
    ).max(axis=1)
    steps = np.clip(np.ceil(edges / spacing).astype(int), 1, 200)
    out = []
    for n in np.unique(steps):
        group = triangles[steps == n]
        u, v = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = (u + v) <= n
        a = (u[keep] / n)[None, :, None]
        b = (v[keep] / n)[None, :, None]
        out.append(
            (group[:, None, 0] + a * (group[:, None, 1] - group[:, None, 0])
             + b * (group[:, None, 2] - group[:, None, 0])).reshape(-1, 3)
        )
    return np.concatenate(out)


@pytest.fixture(scope="module")
def route():
    return json.loads(STEM.with_suffix(".json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ground(route):
    """
    The surface of everything near the driving, sampled every half metre, and
    an index of it in the plane.

    Sampled from the triangles rather than read from the vertices: a road is a
    few long triangles, and its vertices can be ten metres apart.
    """
    from scipy.spatial import cKDTree

    triangles = read_glb(STEM.with_suffix(".glb"))
    line = np.array(route["positions"])
    centres = triangles.mean(axis=1)
    reach = np.linalg.norm(triangles - centres[:, None, :], axis=2).max(axis=1)
    distance, _ = cKDTree(line[:, [0, 2]]).query(centres[:, [0, 2]])
    points = surface(triangles[distance <= reach + 3.0])
    return points, cKDTree(points[:, [0, 2]]), triangles


#: How far above the car a surface may be and still count as the one it is on:
#: a vertex a little above the wheels on a cambered road is still that road.
SLACK = 0.5


def heights(ground, points):
    """
    Height of each point above the surface directly beneath it: the highest
    sample within 0.6 m in the plane that is not above the point.

    Not above the point, because roads cross over roads. Under an expressway
    deck the highest surface is the deck, twenty-five metres over the car, and
    the car is on the street below it.
    """
    samples, index, _triangles = ground
    near = index.query_ball_point(points[:, [0, 2]], r=0.6)
    delta = np.full(len(points), np.nan)
    for i, found in enumerate(near):
        if not found:
            continue
        rise = points[i, 1] - samples[found, 1]
        below = rise[rise > -SLACK]
        if len(below):
            delta[i] = below.min()
    known = np.isfinite(delta)
    return delta[known], known


class TestTheCarIsOnTheGround:
    def test_the_racing_line_sits_just_above_the_extracted_surface(self, ground, route):
        line = np.array(route["positions"])

        delta, known = heights(ground, line)

        assert known.mean() > 0.9, "much of the lap found no ground beneath it"
        # A car's ride height, not a tolerance. Anything outside this is a
        # misplaced tile, a wrong scale, or the wrong stretch of map entirely -
        # and every one of those still looks like terrain.
        assert -1.0 < float(np.median(delta)) < 1.5, (
            f"median height above ground is {np.median(delta):+.2f} m"
        )
        assert float(np.std(delta)) < 0.6, (
            f"height above ground varies by {np.std(delta):.2f} m along the lap"
        )

    def test_no_part_of_the_lap_is_metres_off(self, ground, route):
        """
        The median can be right while a corner is wrong. A tile placed one grid
        step out moves 512 metres of map, and the laps that cross it come out
        floating or buried while the rest are fine.
        """
        line = np.array(route["positions"])

        delta, _known = heights(ground, line)

        assert float(np.abs(delta).max()) < 3.0, (
            f"worst point is {delta[np.abs(delta).argmax()]:+.1f} m off the ground"
        )

    def test_the_ground_covers_the_course(self, ground, route):
        _samples, _index, triangles = ground
        bounds = route["bounds"]
        low = triangles.reshape(-1, 3).min(axis=0)
        high = triangles.reshape(-1, 3).max(axis=0)

        assert low[0] <= bounds["min"][0] + 0.5
        assert low[2] <= bounds["min"][2] + 0.5
        assert high[0] >= bounds["max"][0] - 0.5
        assert high[2] >= bounds["max"][2] - 0.5
