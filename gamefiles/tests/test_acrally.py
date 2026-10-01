"""
Assetto Corsa Rally: the Unreal 5 readers under the course export, and the
export's own reasoning - the frame a lap is read in, the ground's tiles.

The formats were settled against the shipped game (see each module); here
they are held to it with data built to the same layouts, and, where the game
is installed, checked on the real archives too.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from heat3d_gamefiles.acrally import find_paks, list_stages, read_stage, rotator_matrix, texture_volume, transform
from heat3d_gamefiles.acrallycourse import (
    _score,
    describe_frame,
    frames,
    ground_kind,
)
from heat3d_gamefiles.iostore import Container, IoStoreError, Store
from heat3d_gamefiles.unrealtexture import _morton, bulk_map
from heat3d_gamefiles.unversioned import PropertyError, read_header, read_properties, schema, CHAINS
from heat3d_gamefiles.zen import Package, Packages

from .unreal_fixtures import unversioned, write_container, zen_package

SMC = "/Script/Engine.StaticMeshComponent"


class TestContainers:
    def test_files_are_found_by_path_and_read_back(self, tmp_path):
        big = bytes(range(256)) * 600  # more than one block
        write_container(tmp_path, "pakchunk0-Windows", {"a.uasset": b"hello", "Levels/b.umap": big})
        store = Store(tmp_path)
        assert store.read_file("../../../acr/Content/a.uasset") == b"hello"
        assert store.read_file("../../../acr/content/levels/b.umap") == big
        assert store.names["../../../acr/content/levels/b.umap"].endswith("Levels/b.umap")

    def test_an_encrypted_container_is_refused_not_guessed(self, tmp_path):
        path = write_container(tmp_path, "c", {"a.uasset": b"x"})
        data = bytearray(path.read_bytes())
        data[80] |= 2  # the encrypted flag
        path.write_bytes(bytes(data))
        with pytest.raises(IoStoreError, match="encrypted"):
            Container.open(path)

    def test_a_folder_without_containers_says_so(self, tmp_path):
        with pytest.raises(IoStoreError):
            Store(tmp_path)


class TestPackages:
    def test_names_exports_and_numbered_imports(self):
        data = zen_package(
            ["None", "Comp", "SM_Tile"],
            [{"name": 1, "cls": 3 << 62, "data": b"abcd"}],
            imported=[("/Game/Tiles/SM_Tile_Terrain", 1), ("/Game/Other", 0)],
        )
        package = Package(data, "x")
        assert package.names == ["None", "Comp", "SM_Tile"]
        assert [e.name for e in package.exports] == ["Comp"]
        assert package.export_data(package.exports[0]) == b"abcd"
        # A package name numbered 1 is the name with `_0` - the tile's file.
        assert package.imported_packages == ["/Game/Tiles/SM_Tile_Terrain_0", "/Game/Other"]

    def test_the_bulk_map_is_found_before_the_import_hashes(self):
        data = zen_package(["None"], [], hashes=[7, 8], bulk=[(0, 100, 0x10501), (100, 16, 0x48)])
        assert bulk_map(Package(data, "x")) == [(0, 100, 0x10501), (100, 16, 0x48)]


class TestProperties:
    def test_a_static_mesh_component_by_its_schema(self):
        slots = [4, 163, 164]
        values = struct.pack("<i", -3) + struct.pack("<3d", 100.0, 200.0, 300.0) + struct.pack("<3d", 0.0, 90.0, 0.0)
        tail = bytes(20)
        props, end = read_properties(unversioned(slots) + values + tail, SMC, ["None"])
        assert props == {"StaticMesh": -3, "RelativeLocation": (100.0, 200.0, 300.0), "RelativeRotation": (0.0, 90.0, 0.0)}
        assert len(unversioned(slots) + values + tail) - end == 20

    def test_zero_values_carry_no_bytes(self):
        data = unversioned([4, 163], zero={163}) + struct.pack("<i", -1)
        props, _ = read_properties(data, SMC, ["None"])
        assert props["RelativeLocation"] == 0 and props["StaticMesh"] == -1

    def test_the_build_s_extra_slots_refuse_a_value(self):
        slots = schema(CHAINS[SMC])
        unknown = next(i for i, (name, _) in enumerate(slots) if name == "(unknown)")
        with pytest.raises(PropertyError, match="uncalibrated"):
            read_properties(unversioned([unknown]) + bytes(8), SMC, ["None"])

    def test_the_schema_puts_the_transform_where_the_game_does(self):
        names = [name for name, _ in schema(CHAINS[SMC])]
        # Checked on every component of every stage: 4, 159, 163, 164, 165.
        assert names.index("StaticMesh") == 4
        assert names.index("AttachParent") == 159
        assert names[163:166] == ["RelativeLocation", "RelativeRotation", "RelativeScale3D"]

    def test_a_header_with_a_long_gap(self):
        present, end = read_header(unversioned([2, 300]), 0)
        assert [slot for slot, _ in present] == [2, 300]


class TestTransforms:
    def test_yaw_turns_x_towards_y(self):
        assert np.allclose(rotator_matrix(0.0, 90.0, 0.0)[0], [0, 1, 0], atol=1e-12)

    def test_pitch_lifts_x(self):
        assert np.allclose(rotator_matrix(90.0, 0.0, 0.0)[0], [0, 0, 1], atol=1e-12)

    def test_a_transform_scales_then_turns_then_moves(self):
        m = transform((10.0, 0.0, 0.0), (0.0, 90.0, 0.0), (2.0, 1.0, 1.0))
        assert np.allclose(np.array([1.0, 0.0, 0.0, 1.0]) @ m[:, :3], [10, 2, 0])


class TestTheFrame:
    def test_there_are_eight_and_each_puts_height_up(self):
        all_frames = frames()
        assert len(all_frames) == 8
        for f in all_frames:
            assert np.allclose(np.array([0.0, 0.0, 100.0, 1.0]) @ f[:, :3], [0, 1, 0])

    def test_the_right_frame_scores_best(self):
        # A winding road on a grid, and the lap along it read in frame 5.
        t = np.linspace(0, 6, 300)
        road = np.stack([t * 10000, np.sin(t) * 8000 + t * 3000, np.full_like(t, 5000.0)], axis=1)
        grid = {}
        for x, y, z in road:
            grid[(int(np.floor(x / 200)), int(np.floor(y / 200)))] = z
        frame = frames()[5]
        lap = np.c_[road + [0, 0, 50], np.ones(len(road))] @ frame[:, :3]
        scores = [_score(lap, grid, f) for f in frames()]
        assert int(np.argmax(scores)) == 5 and scores[5] > 0.95
        assert describe_frame(frame).startswith("x = ")

    def test_ground_tiles_are_road_or_terrain(self):
        assert ground_kind("/Game/Environments/S/MeshesTrack/Splits/SM_STrack0101_Road_0") == "road"
        assert ground_kind("/Game/Environments/S/MeshesTrack/Splits/SM_STrack0101_Terrain_0") == "terrain"
        assert ground_kind("/Game/Environments/S/MeshesTrack/SM_STrack0101") == "road"
        assert ground_kind("/Game/Environments/S/MeshesTerrain/SM_STerrain0101") == "terrain"
        assert ground_kind("/Game/Props/SM_Fence") is None


class TestVirtualTextures:
    def test_tiles_are_addressed_by_morton_code(self):
        assert [_morton(x, y) for x, y in ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (3, 3))] == [0, 1, 2, 3, 4, 15]


PAKS = find_paks()


@pytest.fixture(scope="module")
def packages():
    return Packages(Store(PAKS))


@pytest.mark.skipif(PAKS is None, reason="Assetto Corsa Rally is not installed")
class TestTheInstalledGame:
    """On the real archives, where the game is on this machine."""

    def test_every_stage_reads_every_static_mesh_component(self, packages):
        stages = list_stages(packages)
        assert stages
        stage = read_stage(packages, stages[0])
        assert stage.read > 100 and not stage.unread
        assert any(ground_kind(p.mesh) == "road" for p in stage.placed)

    def test_the_baked_ground_colour_is_placed_on_the_world(self, packages):
        from heat3d_gamefiles.unrealtexture import VirtualTexture

        found = 0
        for name in list_stages(packages):
            volume = texture_volume(packages, name)
            if volume is None:
                continue
            assert volume.size[0] > 50_000 and volume.size[1] > 50_000  # kilometres
            texture = VirtualTexture(packages, volume.texture)
            assert texture.region(0.4, 0.4, 0.6, 0.6, 64).any()
            found += 1
            break
        assert found
