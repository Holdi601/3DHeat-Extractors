"""
The BeamNG course export, against an install built the way the game ships one.

The conventions here - which way a Torque quaternion turns, that the rows of
`rotationMatrix` are an object's axes, the terrain's height scale - were
settled against the shipped game (see `beamng.py`, `beamngshape.py`); these
tests hold the code to them, and run the whole `course` verb on a lap.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from heat3d_gamefiles.beamng import (
    Files,
    Materials,
    axis_angle,
    find_install,
    list_levels,
    open_level,
    resolve,
    torque_objects,
    transform_of,
)
from heat3d_gamefiles.beamngcourse import (
    category,
    decal_kind,
    find_level,
    from_viewer,
    spline,
    surface_kind,
    to_viewer,
)
from heat3d_gamefiles.beamngshape import read_shape
from heat3d_gamefiles.cli import main
from heat3d_gamefiles.lapinput import describe, read

from .beamng_fixtures import (
    LEVEL,
    MAX_HEIGHT,
    ORIGIN,
    SQUARE,
    build_install,
    cdae,
    lap_document,
    road_lap,
    terrain_file,
)
from .test_glb import read_back


@pytest.fixture
def install(tmp_path):
    return build_install(tmp_path / "game")


@pytest.fixture
def files(install):
    return Files.of_install(install, mods=False)


class TestShapes:
    def test_a_shape_reads_back_as_written(self):
        positions = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        shape = read_shape(cdae(positions, [[0, 1, 2], [0, 2, 3]], uvs=[[0, 0], [1, 0], [1, 1], [0, 1]], materials=("m",)))
        (piece,) = shape.pieces()
        assert piece.material == "m"
        assert len(piece.faces) == 2
        assert np.allclose(np.sort(piece.positions, axis=0), np.sort(np.array(positions, float), axis=0))
        assert piece.uvs is not None

    def test_a_compressed_shape_reads_too(self):
        shape = read_shape(cdae([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], compressed=True))
        assert len(shape.pieces()[0].faces) == 1

    def test_node_rotations_turn_the_way_torque_turns_them(self):
        # A quarter turn about z, as Torque stores it, takes +x to -y: the
        # textbook reading would take it to +y, and that is 38 cm off on the
        # game's own shapes where this convention matches them to the mm.
        s = np.sin(np.pi / 4)
        shape = read_shape(cdae([[1, 0, 0], [1, 0.01, 0], [1, 0, 0.01]], [[0, 1, 2]], node_rotation=(0, 0, s, s)))
        point = shape.pieces()[0].positions
        assert np.allclose(point[np.argmin(np.abs(point[:, 2]) + np.abs(point[:, 1] + 1))], [0, -1, 0], atol=0.02)

    def test_the_full_model_is_the_largest_level_and_collision_is_never_drawn(self):
        full = [[0, 1, 2], [0, 2, 3]]
        shape = read_shape(
            cdae(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                full,
                levels=((50.0, None), (10.0, [[0, 1, 2]])),
                collision=[[0, 1, 3]],
            )
        )
        levels = shape.levels()
        assert [float(shape.details[d]["size"]) for d in levels] == [50.0, 10.0]
        assert len(shape.pieces()[0].faces) == 2
        assert shape.triangle_count(levels[1]) == 1


class TestTheLevel:
    def test_the_install_and_its_level_are_found(self, install, files):
        assert find_install((install,), defaults=False) == install
        assert list_levels(files) == [LEVEL]

    def test_textures_and_shapes_resolve_as_the_game_resolves_them(self, files):
        # A shape named .dae is drawn from its compiled .cdae.
        assert resolve(files, f"/levels/{LEVEL}/art/shapes/rail.DAE").endswith("rail.cdae")
        assert resolve(files, f"/levels/{LEVEL}/art/shapes/nothing.dae") is None

    def test_terrain_heights_are_scaled_by_max_height(self, files):
        level = open_level(files, LEVEL)
        (terrain,) = level.terrains
        _bytes, heights = terrain_file()
        x = np.array([ORIGIN[0] + 10 * SQUARE, ORIGIN[0] + 40.5 * SQUARE])
        y = np.array([ORIGIN[1] + 3 * SQUARE, ORIGIN[1] + 20 * SQUARE])
        expected = ORIGIN[2] + np.array([heights[3, 10], (heights[20, 40] + heights[20, 41]) / 2])
        assert np.allclose(terrain.height(x, y), expected, atol=MAX_HEIGHT / 65535 * 2)
        assert np.isnan(terrain.height(np.array([500.0]), np.array([0.0]))[0])

    def test_rows_of_a_rotation_matrix_are_the_objects_axes(self):
        matrix = transform_of({"rotationMatrix": [0, 1, 0, -1, 0, 0, 0, 0, 1], "position": [1, 2, 3]})
        # Local +x is world +y.
        assert np.allclose(np.array([1, 0, 0, 1]) @ matrix[:, :3], [1, 3, 3])

    def test_an_axis_angle_turns_as_the_shipped_prefabs_do(self):
        assert np.allclose(axis_angle([0, 0, 1], 90)[0], [0, -1, 0], atol=1e-9)
        matrix = transform_of({"rotation": "0 0 1 90", "scale": "1 1 2"})
        assert np.allclose(np.array([0, 0, 1, 1]) @ matrix[:, :3], [0, 0, 2])

    def test_torquescript_objects_and_glued_numbers(self):
        objects = torque_objects(
            'new SimGroup(g) { new TSStatic() { position = "1 2 3"; rotationMatrix = "1 0 0 0 1 0 0 0 1-0.0"; }; };'
        )
        (group,) = objects
        (child,) = group["children"]
        assert child["class"] == "TSStatic"
        assert transform_of(child)[3, :3].tolist() == [1, 2, 3]

    def test_a_prefab_places_its_children_relative_to_itself(self, files):
        level = open_level(files, LEVEL)
        prefab = [p for p in level.placed if p.source == "prefab"]
        assert len(prefab) == 1
        matrix = prefab[0].matrix
        # The child at (0, 0, 11) of a prefab at (-10, -8, 0); groupPosition
        # is an editor pivot and is not subtracted.
        assert np.allclose(matrix[3, :3], [-10, -8, 11])
        # Turned 90 degrees and doubled in height.
        assert np.allclose(np.array([0, 0, 1, 0]) @ matrix[:, :3], [0, 0, 2])

    def test_forest_items_are_placed_by_their_type(self, files):
        level = open_level(files, LEVEL)
        forest = [p for p in level.placed if p.source == "forest"]
        assert [p.shape for p in forest] == [f"levels/{LEVEL}/art/shapes/rock.dae"]

    def test_materials_come_from_the_level_and_the_shared_art(self, files):
        materials = Materials(files, LEVEL)
        assert materials.get("rail_mat") is not None
        assert materials.terrain["asphalt"].ground == "ASPHALT"
        assert materials.texture(materials.get("test_road"), "opacity").endswith("road_o.data.png")


class TestWhatThingsAre:
    @pytest.mark.parametrize(
        ("name", "kind"),
        [
            ("line_white", "markings"),
            ("road_asphalt_2lane", "road"),
            ("m_dirt_road_flows", "overlay"),
            ("road_gravel", "loose"),
            ("track_rubber", "tyremarks"),
            ("hr_grasslines_road_d", "overlay"),
            ("road_edge_grass", "overlay"),
        ],
    )
    def test_road_decals(self, name, kind):
        assert decal_kind(name) == kind

    @pytest.mark.parametrize(
        ("shape", "annotation", "kind"),
        [
            ("/art/shapes/objects/guardrail1.dae", "", "barriers"),
            ("/levels/x/art/shapes/signs/sign_stop.dae", "TRAFFIC_SIGNS", "signs"),
            ("/levels/x/art/shapes/trees/tree_oak_a.dae", "", "trees"),
            ("/levels/x/art/shapes/rocks/s_rock_large.dae", "", "rocks"),
            ("/levels/x/art/shapes/race/kerb_red.dae", "", "kerbs"),
            ("/levels/x/art/shapes/hr_backdrop_terrain.dae", "", "backdrop"),
            ("/levels/x/art/shapes/buildings/bld_shop.dae", "", "buildings"),
        ],
    )
    def test_placed_models(self, shape, annotation, kind):
        assert category(shape, annotation) == kind

    def test_a_models_driving_surface_by_its_ground_type(self):
        class M:
            ground, annotation = "ASPHALT", ""

        assert surface_kind("asphalt_light", M(), "props") == "road"
        assert surface_kind("lines_usa", M(), "props") == "markings"
        assert surface_kind("skidmarks", None, "props") == "tyremarks"
        # The top of a concrete barrier is not a road.
        assert surface_kind("asphalt_light", M(), "barriers") is None

    def test_a_road_spline_runs_through_its_nodes(self):
        nodes = np.array([[0, 0, 0, 4], [10, 0, 0, 4], [20, 5, 0, 6]], float)
        points = spline(nodes)
        assert np.allclose(points[0], nodes[0]) and np.allclose(points[-1], nodes[-1])
        assert np.min(np.linalg.norm(points[:, :2] - nodes[1, :2], axis=1)) < 1e-6
        # At most about a metre apart: each span between nodes is cut into
        # whole pieces, evenly in the curve's parameter.
        gaps = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
        assert gaps.max() < 1.05 and gaps.min() > 0.4


class TestTheLap:
    def test_frames_round_trip(self):
        points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
        assert np.allclose(from_viewer(to_viewer(points)), points)
        # The recorder's rotation: BeamNG's +y is the viewer's -z.
        assert np.allclose(to_viewer(np.array([[0.0, 1.0, 0.0]])), [[0, 0, -1]])

    def test_a_recorded_lap_says_which_game(self, tmp_path):
        path = tmp_path / "lap.json"
        path.write_text(json.dumps(lap_document(road_lap())), encoding="utf-8")
        assert describe(path) == {"game": "beamng"}
        driven = read(path)
        assert driven.game == "beamng"
        assert len(driven) == 120

    def test_a_csv_with_its_description_line(self, tmp_path):
        path = tmp_path / "run.csv"
        path.write_text('# {"game":"trackmania","track":"A01"}\nx,y,z\n0,0,0\n1,0,1\n', encoding="utf-8")
        assert describe(path) == {"game": "trackmania", "track": "A01"}
        assert len(read(path)) == 2

    def test_the_level_is_found_from_the_lap(self, files):
        (match,) = find_level(files, road_lap())
        assert match.level == LEVEL and match.score > 0.95 and match.by == "terrain"

    def test_a_lap_off_the_terrain_is_found_by_the_roads(self, files):
        # Along the AI road, lifted 30 m clear of any terrain the test level has.
        x = np.linspace(-30, 30, 50)
        lap = np.stack([x, np.full_like(x, 20.0), 12 + (x + 30) / 20 + 0.5], axis=1)
        lap[:, :2] += [0.0, 0.0]
        lap[:, 2] += 0.0
        far = lap.copy()
        far[:, 2] += 30.0
        (match,) = find_level(files, far)
        assert match.score == 0.0
        (match,) = find_level(files, lap)
        assert match.score > 0.9


class TestTheCourse:
    def test_a_lap_is_all_it_needs(self, tmp_path, install, capsys):
        lap = tmp_path / "lap.json"
        lap.write_text(json.dumps(lap_document(road_lap())), encoding="utf-8")
        out = tmp_path / "course.glb"
        assert main(["course", str(lap), "--install", str(install), "-o", str(out), "--margin", "30", "--texture-size", "512"]) == 0
        report = capsys.readouterr().out
        assert f"on {LEVEL}" in report
        document, _binary = read_back(out)
        names = [m["name"] for m in document["meshes"]]
        assert any(n.startswith("ground:terrain") for n in names), names
        assert any(n.startswith("ground:road") and "track_asphalt" not in n for n in names), names
        # The modelled asphalt is ground, in the tile's own image.
        assert any(n.startswith("ground:road") and n.endswith("track_asphalt") for n in names), names
        assert any(n.startswith("structure:barriers rail_mat") for n in names), names
        assert any(n.startswith("structure:rocks") for n in names), names
        assert "water:water" in names
        # No invisible road was drawn as anything.
        assert not any("invisible" in n for n in names)

    def test_positions_are_in_the_recorders_frame(self, tmp_path, install):
        lap = tmp_path / "lap.json"
        lap.write_text(json.dumps(lap_document(road_lap())), encoding="utf-8")
        out = tmp_path / "course.glb"
        assert main(["course", str(lap), "--install", str(install), "-o", str(out), "--margin", "30", "--no-textures"]) == 0
        document, binary = read_back(out)
        accessors = document["accessors"]
        rail = next(m for m in document["meshes"] if m["name"].startswith("structure:barriers"))
        bounds = []
        for mesh in [rail]:
            pos = accessors[mesh["primitives"][0]["attributes"]["POSITION"]]
            bounds.append((pos["min"], pos["max"]))
        # Both rails in the viewer's frame: the static one runs along BeamNG's
        # +y from (5, 6), which is the viewer's -z from (5, -6).
        (low, high), = bounds
        assert low[0] <= 5.01 and high[0] >= 4.99
        assert low[2] <= -9.9 and high[2] >= -6.01

    def test_without_textures_paint_is_geometry_and_ground_is_flat(self, tmp_path, install):
        lap = tmp_path / "lap.json"
        lap.write_text(json.dumps(lap_document(road_lap())), encoding="utf-8")
        out = tmp_path / "course.glb"
        assert main(["course", str(lap), "--install", str(install), "-o", str(out), "--margin", "30", "--no-textures"]) == 0
        document, _ = read_back(out)
        assert "images" not in document

    def test_another_games_lap_says_why_there_is_no_course(self, tmp_path, capsys):
        path = tmp_path / "evo.json"
        doc = lap_document(road_lap())
        doc["game"] = "assetto-corsa-evo"
        path.write_text(json.dumps(doc), encoding="utf-8")
        assert main(["course", str(path)]) == 1
        assert "obfuscated" in capsys.readouterr().err


COLLADA = """<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit meter="0.01" name="centimeter"/><up_axis>Y_UP</up_axis></asset>
  <library_materials><material id="m1" name="panel_mat"/></library_materials>
  <library_geometries>
    <geometry id="g_full"><mesh>
      <source id="p"><float_array count="12">0 0 0 400 0 0 400 100 0 0 100 0</float_array>
        <technique_common><accessor source="#p_a" count="4" stride="3"/></technique_common></source>
      <source id="t"><float_array count="8">0 0 1 0 1 1 0 1</float_array>
        <technique_common><accessor source="#t_a" count="4" stride="2"/></technique_common></source>
      <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
      <triangles material="sym" count="2"><input semantic="VERTEX" source="#v" offset="0"/>
        <input semantic="TEXCOORD" source="#t" offset="1" set="0"/><p>0 0 1 1 2 2 0 0 2 2 3 3</p></triangles>
    </mesh></geometry>
    <geometry id="g_low"><mesh>
      <source id="p2"><float_array count="9">0 0 0 400 0 0 400 100 0</float_array>
        <technique_common><accessor source="#p2_a" count="3" stride="3"/></technique_common></source>
      <vertices id="v2"><input semantic="POSITION" source="#p2"/></vertices>
      <triangles material="sym" count="1"><input semantic="VERTEX" source="#v2" offset="0"/><p>0 1 2</p></triangles>
    </mesh></geometry>
  </library_geometries>
  <library_visual_scenes><visual_scene id="s">
    <node name="base00"><node name="start01">
      <node name="panel50"><instance_geometry url="#g_full"><bind_material><technique_common>
        <instance_material symbol="sym" target="#m1"/></technique_common></bind_material></instance_geometry></node>
      <node name="panel10"><instance_geometry url="#g_low"/></node>
      <node name="Colmesh-1"><instance_geometry url="#g_low"/></node>
    </node></node>
  </visual_scene></library_visual_scenes>
</COLLADA>
"""


class TestColladaFallback:
    def test_units_axes_uvs_and_levels_as_the_importer_applies_them(self):
        from heat3d_gamefiles.beamngcollada import read_collada

        shape = read_collada(COLLADA.encode())
        assert shape.levels() == [0, 1]
        (piece,) = shape.pieces()
        assert piece.material == "panel_mat"
        # Centimetres to metres, and Y-up turned to Z-up: the panel stands 1 m tall.
        assert np.allclose(piece.positions.max(axis=0), [4, 0, 1])
        assert np.allclose(piece.positions.min(axis=0), [0, 0, 0])
        # v runs down the image, as in the compiled shapes.
        corner = np.argmin(np.linalg.norm(piece.positions - [0, 0, 0], axis=1))
        assert np.allclose(piece.uvs[corner], [0, 1])
        assert shape.triangle_count(1) == 1


INSTALLED = find_install()


@pytest.mark.skipif(INSTALLED is None, reason="BeamNG.drive is not installed")
class TestTheInstalledGame:
    """On the real archives, where the game is on this machine."""

    def test_compiled_and_source_shapes_agree(self):
        """
        Every shape ships as `.dae` and as the `.cdae` the game compiled from
        it; read both ways, the full model has to land in the same place.
        """
        from heat3d_gamefiles.beamngcollada import read_collada

        files = Files.of_install(INSTALLED, mods=False)
        pairs = [k for k in files.names(prefix="art/shapes/", suffix=".cdae") if k[:-5] + ".dae" in files][:60]
        assert pairs, "no shape pairs in the shared archive"
        agree = 0
        for key in pairs:
            compiled = read_shape(files.read(key)).pieces()
            source = read_collada(files.read(key[:-5] + ".dae")).pieces()
            if not compiled or not source:
                continue
            a = np.concatenate([p.positions for p in compiled])
            b = np.concatenate([p.positions for p in source])
            if np.abs(a.min(axis=0) - b.min(axis=0)).max() < 0.01 and np.abs(a.max(axis=0) - b.max(axis=0)).max() < 0.01:
                agree += 1
        assert agree >= 0.9 * len(pairs)

    def test_every_level_opens(self):
        files = Files.of_install(INSTALLED, mods=False)
        opened = 0
        for name in list_levels(files):
            try:
                level = open_level(files, name, terrain=False)
            except Exception as exc:  # noqa: BLE001 - which level and why, in the failure
                if "has no scene" in str(exc):
                    continue
                raise AssertionError(f"{name}: {exc}") from exc
            # An empty grid is a level too; most are not empty.
            opened += bool(level.placed or level.decals)
        assert opened >= len(list_levels(files)) // 2
