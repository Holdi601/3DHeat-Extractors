"""
The whole `course` verb, as someone runs it: a lap in, a textured course out.

Nothing is stubbed. A track folder is written the way a game ships one - an
archive with a terrain tile, a guardrail model, a kerb model, their placement
files and a swatch, a contents list, and an `AssetManifest.xml` beside it -
and the command line is given a lap and that folder and nothing else. What
comes back has to carry the ground, the placed barrier in its own texture, and
the kerb as ground.
"""

from __future__ import annotations

import struct

import numpy as np

from heat3d_gamefiles.cli import main
from heat3d_gamefiles.forzaterrain import NAME_STEP, TILE_METRES
from heat3d_gamefiles.forzatech import SLOT_BASE_COLOUR

from .forza_fixtures import (
    build_contents,
    build_minizip,
    build_modelbin,
    build_textured_model,
    build_texture,
)
from .test_forza_geometry import grid
from .test_forzamaterial import material
from .test_forzaplacement import ONE, UP, pgeo
from .test_glb import read_back

GUID = "0123abcd-0000-4000-8000-000000000000"
ARMCO_HASH = 777
KERB_HASH = 778


def rail_model(texture_hash: int) -> bytes:
    """A 4 m by 1 m upright panel, textured by the hash its material binds."""
    positions = np.array([[0, 0, 0], [4, 0, 0], [4, 1, 0], [0, 1, 0]], dtype=np.float64)
    return build_textured_model(
        positions=positions,
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        normals=np.tile([0.0, 0.0, 1.0], (4, 1)),
        uvs=np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float64),
        submeshes=[(0, 0x2, 0, 2)],
        materials=[material((SLOT_BASE_COLOUR, texture_hash))],
    )


def red_swatch() -> bytes:
    block = struct.pack("<HHI", 0xF800, 0xF800, 0)
    return build_texture(block * 4, width=8, height=8, format_code=0)


def write_track(folder):
    folder.mkdir(parents=True)
    origin = (NAME_STEP / NAME_STEP * TILE_METRES, 100.0, 0.0)
    points, faces = grid(5, size=TILE_METRES, origin=origin)
    ground = build_modelbin(
        positions=points,
        faces=faces,
        scale=(512.0, 512.0, 512.0),
        bias=(origin[0] + 256.0, origin[1], origin[2] + 256.0),
    )
    files = {
        f"scene\\tbheightfield\\autoterrain_x{NAME_STEP}_z0_cluster000.i.modelbin": ground,
        "tracks\\t\\scene\\models\\barriers\\bar_armco\\bar_armco_cluster000.i.modelbin": rail_model(ARMCO_HASH),
        "tracks\\t\\scene\\models\\roads\\road_gen_rum_round_asan_cluster000.i.modelbin": rail_model(KERB_HASH),
        "tracks\\t\\scene\\proc\\cellsize\\100\\6_1\\c100_barriers_a.pgeo": pgeo(
            {"bar_armco_3D": [((640.0, 101.0, 150.0), (1.0, 0.0, 0.0), UP, ONE),
                              ((644.0, 101.0, 150.0), (1.0, 0.0, 0.0), UP, ONE)]},
            low=(600, 0, 100), high=(700, 200, 200),
        ),
        "tracks\\t\\scene\\proc\\cellsize\\100\\6_1\\c100_props_a.pgeo": pgeo(
            {"road_gen_rum_round_asan_3D": [((650.0, 100.1, 160.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0), ONE)]},
            low=(600, 0, 100), high=(700, 200, 200), category=b"PROPS",
        ),
        f"tracks\\t\\textures\\bar_armco_bclr_{GUID}_quality1.pb": red_swatch(),
        f"tracks\\t\\textures\\kerb_bclr_{GUID}_quality1.pb": red_swatch(),
    }
    names = list(files)
    build_minizip(folder / "GeoChunk0.minizip", [files[n] for n in names])
    build_contents(folder / "ChunkContentsMiniZip0.txt", names)
    (folder / "AssetManifest.xml").write_text(
        "<AssetManifest>\n  <Textures>\n"
        f'    <Texture Source="tracks\\t\\textures\\bar_armco_bclr_{GUID}.swatch" SourceHash="{ARMCO_HASH}" />\n'
        f'    <Texture Source="tracks\\t\\textures\\kerb_bclr_{GUID}.swatch" SourceHash="{KERB_HASH}" />\n'
        "  </Textures>\n</AssetManifest>\n",
        encoding="utf-8",
    )
    return folder


def write_lap(path):
    t = np.linspace(0, 2 * np.pi, 200)
    x = 650 + 30 * np.cos(t)
    z = 150 + 30 * np.sin(t)
    rows = "\n".join(f"{a:.3f},100.5,{b:.3f}" for a, b in zip(x, z))
    path.write_text("x,y,z\n" + rows + "\n", encoding="utf-8")
    return path


def test_a_lap_and_a_track_folder_are_all_it_needs(tmp_path, capsys):
    track = write_track(tmp_path / "Tracks" / "Test")
    lap = write_lap(tmp_path / "lap.csv")
    out = tmp_path / "course.glb"

    assert main(["course", str(lap), "--track", str(track), "-o", str(out), "--margin", "40"]) == 0
    report = capsys.readouterr().out
    assert "placed on it" in report

    document, _binary = read_back(out)
    names = [mesh.get("name", "") for mesh in document["meshes"]]
    assert any(n.startswith("ground:terrain") for n in names), names
    barrier = [i for i, n in enumerate(names) if n.startswith("structure:barriers bar_armco_bclr")]
    assert barrier, names
    kerb = [n for n in names if n.startswith("ground:kerbs")]
    assert kerb, names

    # The barrier is drawn with its own texture, through UVs.
    primitive = document["meshes"][barrier[0]]["primitives"][0]
    assert "TEXCOORD_0" in primitive["attributes"]
    material = document["materials"][primitive["material"]]
    assert "baseColorTexture" in material["pbrMetallicRoughness"]


def test_placed_models_can_be_left_out(tmp_path):
    track = write_track(tmp_path / "Tracks" / "Test")
    lap = write_lap(tmp_path / "lap.csv")
    out = tmp_path / "course.glb"

    assert main(["course", str(lap), "--track", str(track), "-o", str(out), "--no-placed"]) == 0
    document, _binary = read_back(out)
    names = [mesh.get("name", "") for mesh in document["meshes"]]
    assert not any(n.startswith("structure:") or n.startswith("ground:kerbs") for n in names)


def test_a_whole_library_one_file_a_course(tmp_path, capsys):
    """
    `courses` writes one file per course folder, skips what it has already
    written, and ignores a folder that is not a course.
    """
    track = write_track(tmp_path / "Tracks" / "Test")
    library = tmp_path / "laps"
    for name in ("Test Circuit (course_650_150_to_650_150)", "Other Sprint (course_640_140_to_660_160)", "unfinished"):
        folder = library / name / "A" / "car" / "tune" / "untagged"
        folder.mkdir(parents=True)
        write_lap(folder / "lap1.csv")
    out = tmp_path / "all"

    assert main(["courses", "--library", str(library), "--track", str(track), "-o", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["other_sprint.glb", "test_circuit.glb"]

    first = (out / "test_circuit.glb").stat().st_mtime_ns
    capsys.readouterr()
    assert main(["courses", "--library", str(library), "--track", str(track), "-o", str(out)]) == 0
    assert "kept" in capsys.readouterr().out
    assert (out / "test_circuit.glb").stat().st_mtime_ns == first


def test_only_some_courses(tmp_path):
    track = write_track(tmp_path / "Tracks" / "Test")
    library = tmp_path / "laps"
    for name in ("Test Circuit (course_1_1_to_1_1)", "Other Sprint (course_2_2_to_2_2)"):
        folder = library / name
        folder.mkdir(parents=True)
        write_lap(folder / "lap1.csv")
    out = tmp_path / "some"
    assert main(["courses", "--library", str(library), "--track", str(track), "-o", str(out), "--only", "sprint"]) == 0
    assert [p.name for p in out.iterdir()] == ["other_sprint.glb"]
