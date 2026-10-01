"""
The map export's frame and parts: Source2Viewer's glTF metres back to the
game's inches, then into the frame the demo tables are drawn in - and, where
CS2, Source2Viewer and a demo are all on this machine, the players standing
on the floor of the exported map.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from heat3d_cs2 import glb
from heat3d_cs2.mapexport import INCH, parts_from, surface_of, to_game, to_viewer


def test_source2viewer_metres_to_the_game_and_on_to_the_viewer():
    game = np.array([[100.0, 200.0, -160.0]])
    # Source2Viewer writes (Y, Z, X) * 0.0254.
    gl = np.array([[200.0 * INCH, -160.0 * INCH, 100.0 * INCH]])
    assert np.allclose(to_game(gl), game)
    # Drawn as the demo tables are: (X, Z, -Y), a rotation - not a mirror.
    assert np.allclose(to_viewer(game), [[100.0, -160.0, -200.0]])
    m = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    assert np.isclose(np.linalg.det(m), 1.0)


def test_clips_and_sky_are_left_out_and_surfaces_named():
    assert surface_of("physics_npcclip_playerclip") is None
    assert surface_of("physics_csgo_grenadeclip") is None
    assert surface_of("physics_sky") is None
    assert surface_of("physics_group_concrete") == "concrete"
    assert surface_of("physics_passbullets_chainlink") == "chainlink"
    assert surface_of("physics_group") == "world"


def test_ground_faces_up_structure_does_not():
    # A floor quad and a wall quad of one surface, in Source2Viewer's frame.
    floor = np.array([[0, 0, 0], [0, 0, 100], [100, 0, 100], [100, 0, 0]], float) * INCH
    wall = np.array([[0, 0, 0], [0, 100, 0], [0, 100, 100], [0, 0, 100]], float) * INCH
    # Wound so the floor's normal points up in the game; the same quad wound
    # the other way faces down, a ceiling, which is structure.
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    parts = {
        n: (p, t)
        for n, p, t in parts_from(
            [
                ("physics_group_concrete", floor, tris),
                ("physics_group_concrete", wall, tris),
                ("physics_group_concrete", floor, tris[:, ::-1]),
            ]
        )
    }
    assert set(parts) == {"ground:concrete", "structure:concrete"}
    assert len(parts["ground:concrete"][1]) == 2
    assert len(parts["structure:concrete"][1]) == 4


def test_glb_round_trip(tmp_path):
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]], float)
    glb.write(tmp_path / "x.glb", [("ground:floor", p, np.array([[0, 1, 2]]))])
    doc, blob = glb.read(tmp_path / "x.glb")
    (name, positions, tris), = glb.meshes(doc, blob)
    assert name == "ground:floor" and len(tris) == 1
    assert np.allclose(positions[tris[0]], p)


DEMO = os.environ.get("HEAT3D_CS2_DEMO")


@pytest.mark.skipif(not DEMO or not os.environ.get("HEAT3D_SOURCE2VIEWER"), reason="set HEAT3D_CS2_DEMO and HEAT3D_SOURCE2VIEWER")
def test_players_stand_on_the_exported_floor(tmp_path):
    """
    The check that the frame is right: every sampled player has ground under
    them, a median of under a few units below their feet. A mirrored map
    leaves half of them over nothing.
    """
    from heat3d_cs2.demo import read_part
    from heat3d_cs2.mapexport import export_map
    from heat3d_cs2.match import build_match

    match = build_match([read_part(DEMO)], Path(DEMO).stem)
    export_map(match.map, tmp_path / "map.glb")
    doc, blob = glb.read(tmp_path / "map.glb")
    tris = []
    for name, p, t in glb.meshes(doc, blob):
        if name.startswith("ground:"):
            game = np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1)
            tris.append(game[t])
    tris = np.concatenate(tris)
    pos = match.rows[match.rows["event"] == "position"].sample(300, random_state=1)[["x", "y", "z"]].to_numpy()
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    heights = []
    for x, y, z in pos:
        v0, v1, v2 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2], np.array([x, y]) - a[:, :2]
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        ok = np.abs(den) > 1e-9
        u = np.where(ok, (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / np.where(ok, den, 1), -1)
        v = np.where(ok, (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / np.where(ok, den, 1), -1)
        inside = (u >= 0) & (v >= 0) & (u + v <= 1)
        floor = a[inside, 2] + u[inside] * (b[inside, 2] - a[inside, 2]) + v[inside] * (c[inside, 2] - a[inside, 2])
        floor = floor[floor <= z + 4]
        if len(floor):
            heights.append(z - floor.max())
    assert len(heights) >= 0.98 * len(pos)
    assert np.median(heights) < 5
