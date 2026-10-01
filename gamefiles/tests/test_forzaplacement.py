"""
Placed models: where the copies stand, which level of detail, which surfaces.

The placement record was decoded by checks against the game's own data, not by
reading a spec, and each wrong reading tried along the way is a test here:
positions as a fraction of the box, two's-complement signs, the axis vectors as
rows. The last test runs against an installed game and a recorded lap library
when there is one, and checks the thing that matters - that the kerbs lie on
the ground beside the line the cars drove.
"""

from __future__ import annotations

import dataclasses
import os
import struct

import numpy as np
import pytest

import heat3d_gamefiles.forzaplacement as placement
from heat3d_gamefiles.forzamaterial import Manifest
from heat3d_gamefiles.forzatech import SLOT_BASE_COLOUR, Geometry
from heat3d_gamefiles.forzaplacement import (
    KERB_PART,
    Level,
    ModelShelf,
    Track,
    cell_of,
    choose_levels,
    fixed,
    parse_pgeo,
    place,
    place_in_box,
)

from .forza_fixtures import build_track


def sign_magnitude(value: float) -> int:
    return (0x80000000 if value < 0 else 0) | int(round(abs(value) * 65536))


def pgeo(
    models: dict[str, list[tuple]],
    *,
    low=(0.0, 0.0, 0.0),
    high=(100.0, 10.0, 100.0),
    category: bytes = b"BARRIERS",
    variants: tuple[bytes, ...] = (),
) -> bytes:
    """A placement file: header, box, category, a variant table, then models."""
    title = b"c100_0_0 section"
    out = struct.pack("<I", len(title)) + title
    out += struct.pack("<III", 0, 13, 15)
    out += struct.pack("<8f", *low, 0.0, *high, 1.0)
    out += struct.pack("<I", len(category)) + category
    for name in variants:
        # A bare variant name followed by numbers that would read as a count.
        out += struct.pack("<I", len(name)) + name + struct.pack("<II", 2, 17) + bytes(40)
    for name, copies in models.items():
        raw = name.encode()
        out += struct.pack("<I", len(raw)) + raw + struct.pack("<I", len(copies))
        for position, x_axis, y_axis, scale, *variant in copies:
            record = bytearray(placement.INSTANCE)
            struct.pack_into("<3I", record, 0, *(sign_magnitude(v) for v in position))
            struct.pack_into("<9f", record, 12, *x_axis, *y_axis, *scale)
            struct.pack_into("<I", record, placement.VARIANT_AT, variant[0] if variant else 0)
            out += bytes(record)
    return out


UP = (0.0, 1.0, 0.0)
EAST = (1.0, 0.0, 0.0)
ONE = (1.0, 1.0, 1.0)


class TestTheRecord:
    def test_fixed_point_is_sign_magnitude(self):
        """Bit 31 is the sign. Read as two's complement, -1 m is 32 km away."""
        assert fixed(0x00010000) == 1.0
        assert fixed(0x80010000) == -1.0
        assert fixed(0x80008000) == -0.5
        assert fixed(0x0100_0000) == 256.0

    def test_positions_and_axes_come_back(self):
        found = parse_pgeo(pgeo({"bar_armco_3D": [((12.5, -3.25, 40.0), EAST, UP, ONE)]}))
        instances = found.models["bar_armco_3D"]
        assert instances.shape == (1, 13)
        assert np.allclose(instances[0, 0:3], (12.5, -3.25, 40.0))
        assert np.allclose(instances[0, 3:6], EAST)
        assert np.allclose(instances[0, 6:9], UP)

    def test_each_copy_says_which_variant_it_wears(self):
        """Byte 60: 0, 2, 2, 3, 4 across the Lakeside billboards; 0 on every armco."""
        found = parse_pgeo(
            pgeo({"sgn_billboard_3D": [((1, 0, 1), EAST, UP, ONE, v) for v in (0, 2, 2, 3, 4)]})
        )
        assert found.models["sgn_billboard_3D"][:, 12].tolist() == [0, 2, 2, 3, 4]

    def test_the_box_and_title_are_read(self):
        found = parse_pgeo(pgeo({}, low=(1, 2, 3), high=(4, 5, 6)))
        assert found.title == "c100_0_0 section"
        assert np.allclose(found.low, (1, 2, 3)) and np.allclose(found.high, (4, 5, 6))

    def test_variant_names_are_not_models(self):
        """
        `PRP_GBL_CRCT_TYRES_02_D` in the table before the models reads as a
        record with a count after it. Taken for one, a tenth of its "copies"
        land at infinity and the rest outside the file's box.
        """
        found = parse_pgeo(
            pgeo({"prp_tyres_02_a_3D": [((5, 0, 5), EAST, UP, ONE)]}, variants=(b"PRP_TYRES_02_D",))
        )
        assert list(found.models) == ["prp_tyres_02_a_3D"]

    def test_copies_with_numbers_that_are_not_numbers_are_dropped(self):
        nan = float("nan")
        found = parse_pgeo(
            pgeo({"cone_3D": [((1, 0, 1), EAST, UP, ONE), ((2, 0, 2), (nan, 0, 0), UP, ONE)]})
        )
        assert len(found.models["cone_3D"]) == 1

    def test_a_count_that_does_not_fit_is_not_a_record(self):
        data = pgeo({"cone_3D": [((1, 0, 1), EAST, UP, ONE)]})
        name_at = data.index(b"cone_3D")
        broken = data[: name_at + 7] + struct.pack("<I", 1_000) + data[name_at + 11 :]
        assert parse_pgeo(broken).models == {}

    def test_too_short_is_refused(self):
        with pytest.raises(placement.UnsupportedForza):
            parse_pgeo(b"\x01\x00")

    def test_the_cell_comes_from_the_path(self):
        name = "tracks\\mainmap\\scene\\proc\\cellsize\\100\\27_48\\c100_barriers_sec3.pgeo"
        assert cell_of(name) == (100, 27, 48, "barriers")
        assert cell_of("tracks\\mainmap\\scene\\models\\x.pgeo") is None


def triangle_level() -> Level:
    return Level(
        positions=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
        normals=np.eye(3, dtype=np.float32),
        uvs=None,
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        texture=np.array([-1], dtype=np.int32),
    )


class TestPuttingCopiesInTheWorld:
    def test_the_axes_are_columns_and_the_third_is_their_cross_product(self):
        # The model's x points south (-z), its y up; x cross y is then east.
        instances = np.array([[10, 0, 5, 0, 0, -1, 0, 1, 0, 2, 2, 2]], dtype=np.float64)
        positions, normals, faces = place(triangle_level(), instances)
        assert np.allclose(positions, [[10, 0, 3], [10, 2, 5], [12, 0, 5]])
        assert np.allclose(normals, [[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        assert faces.tolist() == [[0, 1, 2]]

    def test_consecutive_guardrails_meet_end_to_start(self):
        """
        The check that decided the reading: a 4 m segment's far end lands on the
        next segment's origin. With the axes read as rows it lands metres away.
        """
        rail = Level(
            positions=np.array([[0, 0, 0], [4, 0, 0], [4, 1, 0]], dtype=np.float32),
            normals=np.tile([0, 0, 1], (3, 1)).astype(np.float32),
            uvs=None,
            faces=np.array([[0, 1, 2]], dtype=np.int32),
            texture=np.array([-1], dtype=np.int32),
        )
        heading = np.array([0.6, 0.0, 0.8])
        first = np.array([100.0, 5.0, 200.0])
        second = first + 4 * heading
        instances = np.array(
            [[*first, *heading, *UP, *ONE], [*second, *heading, *UP, *ONE]], dtype=np.float64
        )
        positions, _normals, faces = place(rail, instances)
        assert np.allclose(positions[1], positions[3], atol=1e-4)
        assert faces.tolist() == [[0, 1, 2], [3, 4, 5]]


class TestChoosingLevels:
    COSTS = {
        "tyres": (820, [1402, 907, 502, 145]),
        "armco": (950, [960, 760, 188, 162]),
        "kerb": (78, [84, 60, 48, 6]),
    }

    def total(self, chosen):
        return sum(c * levels[chosen[n]] for n, (c, levels) in self.COSTS.items())

    def test_everything_at_its_finest_when_it_fits(self):
        assert set(choose_levels(self.COSTS, 10_000_000).values()) == {0}

    def test_the_costliest_steps_down_until_it_fits(self):
        chosen = choose_levels(self.COSTS, 1_000_000, pinned={"kerb"})
        assert self.total(chosen) <= 1_000_000
        assert chosen["kerb"] == 0
        # Tyres cost most at the finest level, so they give way first.
        assert chosen["tyres"] >= chosen["armco"]

    def test_pinned_models_never_step_down(self):
        chosen = choose_levels(self.COSTS, 0, pinned={"kerb"})
        assert chosen == {"tyres": 3, "armco": 3, "kerb": 0}


class FakeTrack:
    """Just what `ModelShelf` asks of a track: its clusters and their bytes."""

    def __init__(self, clusters: dict[str, list[str]]):
        self.clusters = {stem: [(0, i, name) for i, name in enumerate(names)] for stem, names in clusters.items()}

    def read(self, number, entry):
        return b""


class FakeTextures:
    def __init__(self, coverage: dict[str, float]):
        self.coverage_of = coverage
        self.shelf = self

    def resolve(self, name):
        return None

    def coverage(self, name):
        return np.full((32, 32), self.coverage_of.get(name, 1.0), dtype=np.float32)

    def image(self, name):
        return np.zeros((4, 4, 4), dtype=np.uint8)


def quad(material: int, lods: int, *, x: float = 0.0, uv=True) -> Geometry:
    """Two triangles making one unit square, with one material and LOD mask."""
    positions = np.array([[x, 0, 0], [x + 1, 0, 0], [x + 1, 1, 0], [x, 1, 0]], dtype=np.float32)
    uvs = {0: np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)} if uv else {}
    return Geometry(
        positions=positions,
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        lods=np.array([lods, lods], dtype=np.uint16),
        material_index=np.array([material, material], dtype=np.uint16),
        normals=np.tile([0, 0, 1], (4, 1)).astype(np.float32),
        uvs=uvs,
    )


SOLID = "Game:\\Media\\tracks\\mainmap\\materials\\environment\\environment_standard.materialbin"
CUTOUT = "Game:\\Media\\tracks\\mainmap\\materials\\environment\\environment_alphatest.materialbin"
DECAL = "Game:\\Media\\tracks\\mainmap\\materials\\environment\\deferred_decal_standard.materialbin"


@pytest.fixture
def shelf(monkeypatch):
    """
    A model shelf over fake clusters: `models[name] = (geometry, shaders)`, each
    material bound to a texture of its own name.
    """
    models: dict[str, tuple[Geometry, list[str]]] = {}
    names = {1000 + i: f"tex{i}_bclr" for i in range(8)}
    manifest = Manifest(names)
    monkeypatch.setattr(placement, "read_model", lambda name, data: name)
    monkeypatch.setattr(placement, "read_geometry", lambda model, lod=None: models[model][0])
    monkeypatch.setattr(placement, "material_paths", lambda model: models[model][1])
    monkeypatch.setattr(
        placement,
        "material_textures",
        lambda model, hashes: [[(SLOT_BASE_COLOUR, 1000 + i)] for i in range(len(models[model][1]))],
    )

    def make(clusters: dict[str, list[str]], coverage: dict[str, float] | None = None) -> ModelShelf:
        return ModelShelf(FakeTrack(clusters), manifest, FakeTextures(coverage or {}))

    make.models = models
    return make


class TestLevelsOfDetail:
    def test_the_breakable_pieces_do_not_join_the_finest_level(self, shelf):
        shelf.models["a_c0"] = (quad(0, 0x2), [SOLID])
        shelf.models["a_c1"] = (quad(0, 0xFFFF, x=0.01), [SOLID])
        shelf.models["a_c2"] = (quad(0, 0x4), [SOLID])
        model = shelf({"armco": ["a_c0", "a_c1", "a_c2"]}).load("armco_3D")
        assert [lv.triangles for lv in model.levels] == [2, 2]

    def test_a_model_with_no_marked_level_uses_its_pieces(self, shelf):
        shelf.models["b_c0"] = (quad(0, 0xFFFF), [SOLID])
        shelf.models["b_c1"] = (quad(0, 0xFFFF, x=2), [SOLID])
        model = shelf({"bench": ["b_c0", "b_c1"]}).load("bench_3D")
        assert [lv.triangles for lv in model.levels] == [4]

    def test_the_texture_follows_the_material(self, shelf):
        shelf.models["c_c0"] = (quad(1, 0x2), [SOLID, SOLID])
        model = shelf({"sign": ["c_c0"]}).load("sign_3D")
        assert [model.textures[i] for i in model.levels[0].texture[:, 0]] == ["tex1_bclr", "tex1_bclr"]

    def test_an_unknown_model_is_none(self, shelf):
        assert shelf({}).load("nothing_3D") is None


def printed(variants, lods=0x2, x=0.0) -> Geometry:
    """A quad whose one submesh has a table of (summer, winter) variants."""
    return dataclasses.replace(
        quad(variants[0][0], lods, x=x),
        submesh=np.zeros(2, dtype=np.int32),
        variants=(tuple(variants),),
    )


class TestVariants:
    def test_each_variant_has_its_own_texture(self, shelf):
        shelf.models["v_c0"] = (printed([(0, 0xFFFF), (1, 0xFFFF), (2, 0xFFFF)]), [SOLID] * 3)
        model = shelf({"board": ["v_c0"]}).load("board_3D")
        level = model.levels[0]
        assert level.texture.shape == (2, 3)
        assert [model.textures[i] for i in level.texture[0]] == ["tex0_bclr", "tex1_bclr", "tex2_bclr"]

    def test_winter_takes_the_winter_material_where_there_is_one(self, shelf):
        shelf.models["w_c0"] = (printed([(0, 3), (1, 0xFFFF)]), [SOLID] * 4)
        models = shelf({"board": ["w_c0"]})
        models.season = "winter"
        model = models.load("board_3D")
        assert [model.textures[i] for i in model.levels[0].texture[0]] == ["tex3_bclr", "tex1_bclr"]

    def test_a_part_with_fewer_variants_repeats_its_own(self, shelf):
        """The frame of a billboard has one material whichever print it holds."""
        shelf.models["f_c0"] = (printed([(0, 0xFFFF), (1, 0xFFFF)]), [SOLID] * 3)
        shelf.models["f_c1"] = (printed([(2, 0xFFFF)], x=1), [SOLID] * 3)
        level = shelf({"board": ["f_c0", "f_c1"]}).load("board_3D").levels[0]
        assert level.texture.shape == (4, 2)
        assert level.texture[2].tolist() == level.texture[3].tolist()
        assert level.texture[2, 0] == level.texture[2, 1]

    def test_copies_wear_the_variant_their_record_names(self, tmp_path, shelf, monkeypatch):
        shelf.models["b_c0"] = (printed([(0, 0xFFFF), (1, 0xFFFF)]), [SOLID, SOLID])
        files = {
            "tracks\\t\\scene\\proc\\cellsize\\100\\0_0\\c100_signs_a.pgeo": pgeo(
                {"sgn_board_3D": [((10.0, 0.0, 10.0), EAST, UP, ONE, 0), ((20.0, 0.0, 10.0), EAST, UP, ONE, 1),
                                  ((30.0, 0.0, 10.0), EAST, UP, ONE, 1)]},
                category=b"SIGNS",
            ),
            "tracks\\t\\scene\\models\\signs\\sgn_board_cluster000.i.modelbin": b"x",
        }
        folder = build_track(tmp_path / "Track", files)
        monkeypatch.setattr(placement, "read_geometry", lambda model, lod=None: shelf.models["b_c0"][0])
        monkeypatch.setattr(placement, "material_paths", lambda model: [SOLID, SOLID])
        monkeypatch.setattr(
            placement, "material_textures", lambda model, hashes: [[(SLOT_BASE_COLOUR, 1000)], [(SLOT_BASE_COLOUR, 1001)]]
        )
        monkeypatch.setattr(placement.Manifest, "read", classmethod(lambda cls, folder: Manifest({1000: "print_a_bclr", 1001: "print_b_bclr"})))

        class Shelf:
            def __init__(self, folder):
                pass

            def resolve(self, name):
                return name

            def best(self, name, largest=256):
                return np.zeros((4, 4, 4), dtype=np.uint8)

            def close(self):
                pass

        monkeypatch.setattr("heat3d_gamefiles.forzatexture.TextureShelf", Shelf)
        with Track(folder) as track:
            found = place_in_box(track, (0.0, 0.0), (50.0, 50.0))
        a = found.parts["structure:signs print_a_bclr"]
        b = found.parts["structure:signs print_b_bclr"]
        assert len(a.faces) == 2 and len(b.faces) == 4
        assert a.positions[:, 0].min() == pytest.approx(10.0)
        assert b.positions[:, 0].min() == pytest.approx(20.0)


class TestSeeThroughSurfaces:
    def load(self, shelf, coverage):
        shelf.models["f_c0"] = (quad(0, 0x2), [CUTOUT])
        shelf.models["f_c1"] = (quad(1, 0x2, x=1), [SOLID, SOLID])
        return shelf({"fence": ["f_c0", "f_c1"]}, {"tex0_bclr": coverage})

    def test_a_mostly_holes_surface_is_left_out_whole(self, shelf):
        """
        Both triangles of the chain-link quad go, not one: judged a triangle at
        a time, a fence turns into a row of teeth.
        """
        models = self.load(shelf, 0.45)
        model = models.load("fence_3D")
        assert model.levels[0].triangles == 2  # only the solid quad
        assert {model.textures[i] for i in model.levels[0].texture[:, 0]} == {"tex1_bclr"}

    def test_a_mostly_solid_cut_out_is_kept_whole(self, shelf):
        assert self.load(shelf, 0.93).load("fence_3D").levels[0].triangles == 4

    def test_alpha_that_is_nearly_all_zero_is_not_a_mask(self, shelf):
        """A warning sign that samples 0.01 is a sign whose alpha means something else."""
        assert self.load(shelf, 0.01).load("fence_3D").levels[0].triangles == 4

    def test_a_model_that_is_only_holes_is_reported_as_such(self, shelf):
        shelf.models["n_c0"] = (quad(0, 0x2), [CUTOUT])
        models = shelf({"net": ["n_c0"]}, {"tex0_bclr": 0.3})
        assert models.load("net_3D") is None
        assert "net" in models.see_through

    def test_decals_draw_nothing_solid(self, shelf):
        shelf.models["d_c0"] = (quad(0, 0x2), [DECAL])
        shelf.models["d_c1"] = (quad(1, 0x2, x=1), [DECAL, SOLID])
        model = shelf({"shed": ["d_c0", "d_c1"]}).load("shed_3D")
        assert model.levels[0].triangles == 2

    def test_vertices_without_uvs_are_drawn_in_flat_colour(self, shelf):
        geometry = quad(0, 0x2)
        geometry.uvs[0][2] = np.nan
        shelf.models["u_c0"] = (geometry, [SOLID])
        level = shelf({"odd": ["u_c0"]}).load("odd_3D").levels[0]
        assert level.texture[:, 0].tolist() == [-1, -1]
        assert np.isfinite(level.uvs).all()


class TestAWholeBox:
    def test_kerbs_barriers_and_the_box_edge(self, tmp_path, shelf, monkeypatch):
        """
        End to end over real placement files: two cells, one kerb listed by both
        (a model on a cell boundary is), one guardrail inside the box and one
        outside it.
        """
        shelf.models["kerb_c0"] = (quad(0, 0x2), [SOLID])
        shelf.models["rail_c0"] = (quad(0, 0x2), [SOLID])
        kerb = ((10.0, 1.0, 10.0), EAST, UP, ONE)
        files = {
            "tracks\\t\\scene\\proc\\cellsize\\100\\0_0\\c100_props_a.pgeo": pgeo(
                {"road_gen_rum_round_asan_3D": [kerb]}, category=b"PROPS"
            ),
            "tracks\\t\\scene\\proc\\cellsize\\100\\1_0\\c100_props_a.pgeo": pgeo(
                {"road_gen_rum_round_asan_3D": [kerb]}, category=b"PROPS"
            ),
            "tracks\\t\\scene\\proc\\cellsize\\100\\0_0\\c100_barriers_a.pgeo": pgeo(
                {"bar_armco_3D": [((20.0, 1.0, 20.0), EAST, UP, ONE), ((90.0, 1.0, 90.0), EAST, UP, ONE)]}
            ),
            "tracks\\t\\scene\\proc\\cellsize\\100\\0_0\\c100_grasstemplate_a.pgeo": pgeo(
                {"grass_3D": [((5.0, 1.0, 5.0), EAST, UP, ONE)]}
            ),
            "tracks\\t\\scene\\models\\roads\\road_gen_rum_round_asan_cluster000.i.modelbin": b"x",
            "tracks\\t\\scene\\models\\barriers\\bar_armco\\bar_armco_cluster000.i.modelbin": b"x",
        }
        folder = build_track(tmp_path / "Track", files)
        monkeypatch.setattr(
            placement,
            "read_geometry",
            lambda model, lod=None: shelf.models["kerb_c0" if "rum" in model else "rail_c0"][0],
        )
        monkeypatch.setattr(placement, "material_paths", lambda model: [SOLID])
        with Track(folder) as track:
            found = place_in_box(track, (0.0, 0.0), (50.0, 50.0), textures=False)
        assert found.counts == {KERB_PART: {"road_gen_rum_round_asan": 1}, "structure:barriers": {"bar_armco": 1}}
        assert set(found.parts) == {KERB_PART, "structure:barriers"}
        kerb_part = found.parts[KERB_PART]
        assert kerb_part.texture is None and kerb_part.colour == placement.FALLBACK_COLOURS[KERB_PART]
        assert np.allclose(kerb_part.positions.min(axis=0), (10.0, 1.0, 10.0))
        assert found.triangles() == 4


# ---------------------------------------------------------------------------
# Against an installed game and a recorded lap library, when there are both.

def _courses(limit: int = 4) -> list[str]:
    """`HEAT3D_COURSE`, or the recorded circuits with the most laps."""
    if os.environ.get("HEAT3D_COURSE"):
        return [os.environ["HEAT3D_COURSE"]]
    import re

    from heat3d_gamefiles.lapinput import default_library

    try:
        folders = [d for d in default_library().iterdir() if d.is_dir()]
    except OSError:
        return []
    circuits = []
    for folder in folders:
        # A circuit starts where it finishes: `Name (course_x_z_to_x_z)`.
        ends = re.search(r"course_(-?\d+)_(-?\d+)_to_(-?\d+)_(-?\d+)", folder.name)
        if not ends:
            continue
        x0, z0, x1, z1 = map(int, ends.groups())
        if abs(x1 - x0) + abs(z1 - z0) <= 50:
            laps = sum(1 for p in folder.rglob("*.json") if p.name != "course.json")
            circuits.append((-laps, str(folder)))
    return [folder for _, folder in sorted(circuits)[:limit]]


def _real_courses():
    """(driven, track, low, high) for each course that is installed as well as recorded."""
    from heat3d_gamefiles.forzainstall import find_track
    from heat3d_gamefiles.lapinput import UnreadableLap, read

    found = []
    for course in _courses():
        try:
            driven = read(course)
            low, high = driven.box(40)
            found.append((driven, find_track(low, high), low, high))
        except (FileNotFoundError, UnreadableLap, OSError, ValueError):
            continue
    return found


REAL = _real_courses()


@pytest.mark.skipif(not REAL, reason="no Forza install with a recorded lap library here")
def test_real_kerbs_lie_on_the_ground_beside_the_driven_line():
    """
    Measured on a recorded circuit: kerbs sit 3 cm above the terrain (median),
    face up, and are within five metres of where the cars drove - on it at the
    apexes, since that is what kerbs are for. Barriers are never on the line.

    A street circuit can have no kerb models at all, so the first of the most
    driven circuits that has any is the one checked; none having any is a fault.
    """
    from heat3d_gamefiles.forzaterrain import extract_from_track

    for driven, track, low, high in REAL:
        placed = placement.place_in_track(track, low, high, textures=False)
        if KERB_PART in placed.parts:
            break
    else:
        pytest.fail(f"no kerbs placed around any of {len(REAL)} recorded circuits")
    terrain = extract_from_track(track, low, high)

    def nearest(points, cloud):
        # Brute force in blocks: a few thousand kerb vertices against the lap.
        out = np.empty(len(points))
        for start in range(0, len(points), 512):
            block = points[start : start + 512]
            d = ((block[:, None, :] - cloud[None, :, :]) ** 2).sum(axis=2)
            out[start : start + 512] = np.sqrt(d.min(axis=1))
        return out

    kerbs = placed.parts[KERB_PART]
    ground = terrain.positions[:, [0, 2]]
    sample = kerbs.positions[:: max(1, len(kerbs.positions) // 1500)]
    nearest_ground = np.array(
        [np.argmin(((ground - p[[0, 2]]) ** 2).sum(axis=1)) for p in sample]
    )
    height = sample[:, 1] - terrain.positions[nearest_ground, 1]
    assert abs(np.median(height)) < 0.3
    assert (kerbs.normals[:, 1] > 0.7).mean() > 0.95
    line = np.asarray(driven.positions)[:, [0, 2]][::4]
    assert np.median(nearest(sample[:, [0, 2]], line)) < 5.0

    barriers = placed.parts["structure:barriers"]
    rails = barriers.positions[:: max(1, len(barriers.positions) // 1500)]
    assert np.percentile(nearest(rails[:, [0, 2]], line), 2) > 2.0


class TestTheBudgetHolds:
    """
    A city block is 380,000 building pieces; at their coarsest they are still
    forty million triangles. The budget has to hold anyway, and what gives way
    must be what matters least to a lap: what stands furthest from it.
    """

    def test_route_distance_is_to_the_nearest_point_of_the_driving(self):
        route = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        d = placement.route_distance(np.array([[10.0, 5.0], [30.0, 0.0], [0.0, 0.0]]), route)
        assert np.allclose(d, [5.0, 10.0, 0.0])

    def test_distant_copies_start_coarse(self):
        costs = {("rail", 0): (10, [100, 50, 10]), ("rail", 1): (10, [100, 50, 10]), ("rail", 2): (10, [100, 50, 10])}
        chosen = choose_levels(costs, 10**9, start={("rail", 0): 0, ("rail", 1): 1, ("rail", 2): 99})
        assert chosen == {("rail", 0): 0, ("rail", 1): 1, ("rail", 2): 2}

    class Sized:
        def __init__(self, size):
            self.size = size

    def copies(self, n):
        out = np.zeros((n, 13))
        out[:, 9:12] = 1.0
        return out

    def test_the_farthest_copies_go_first_and_kerbs_never(self):
        house = self.Sized(10.0)
        entries = {
            ("house", 0): (house, self.copies(3), np.array([1.0, 2.0, 3.0])),
            ("house", 2): (house, self.copies(3), np.array([300.0, 100.0, 200.0])),
            ("kerb", 2): (self.Sized(8.0), self.copies(3), np.array([500.0, 500.0, 500.0])),
        }
        costs = {name: (3, [10]) for name in entries}
        chosen = {name: 0 for name in entries}
        out = placement.Placed()
        keep = placement._fit(entries, costs, chosen, {("kerb", 2)}, 60, out)
        # 90 triangles against 60: three houses go, the furthest three.
        assert keep[("house", 2)].tolist() == [False, False, False]
        assert keep[("house", 0)].all() and keep[("kerb", 2)].all()
        assert out.dropped == 3 and out.considered == 9

    def test_something_small_goes_before_something_big_and_nearer_than_it(self):
        """A facade twelve metres away outlives an air conditioner at nine."""
        entries = {
            ("facade", 0): (self.Sized(20.0), self.copies(1), np.array([12.0])),
            ("aircon", 0): (self.Sized(0.5), self.copies(1), np.array([9.0])),
        }
        costs = {name: (1, [10]) for name in entries}
        out = placement.Placed()
        keep = placement._fit(entries, costs, {name: 0 for name in entries}, set(), 10, out)
        assert keep[("facade", 0)].all() and not keep[("aircon", 0)].any()

    def test_nothing_goes_when_it_fits(self):
        entries = {("house", 0): (object(), np.zeros((2, 13)), np.array([1.0, 2.0]))}
        out = placement.Placed()
        keep = placement._fit(entries, {("house", 0): (2, [10])}, {("house", 0): 0}, set(), 100, out)
        assert keep[("house", 0)].all() and out.dropped == 0

    def test_what_stands_beside_the_road_outlives_bigger_things_further_off(self):
        """A barrier five metres from the line is kept before a building at sixty."""
        entries = {
            ("barrier", 0): (self.Sized(3.0), self.copies(1), np.array([5.0])),
            ("building", 2): (self.Sized(50.0), self.copies(1), np.array([60.0])),
        }
        costs = {name: (1, [10]) for name in entries}
        out = placement.Placed()
        keep = placement._fit(entries, costs, {name: 0 for name in entries}, set(), 10, out)
        assert keep[("barrier", 0)].all() and not keep[("building", 2)].any()


class TestPlaceholderColour:
    def test_a_placeholder_block_takes_the_textures_own_colour(self):
        image = np.zeros((10, 10, 4), dtype=np.uint8)
        image[:, :, 3] = 255
        image[:5, :, :3] = (120, 130, 140)  # facade
        image[5:, :, :3] = (255, 64, 0)  # the block the shader tints
        out = placement.unmark(image)
        assert (out[5:, :, :3] == (120, 130, 140)).all()
        assert (out[:5] == image[:5]).all()

    def test_a_little_orange_is_left_alone(self):
        """A traffic cone's stripe is not a placeholder block."""
        image = np.full((20, 20, 4), 200, dtype=np.uint8)
        image[0, :3, :3] = (255, 64, 0)
        assert (placement.unmark(image) == image).all()


class TestTheAutomaticBudget:
    def test_it_keeps_every_copy_with_room_for_detail(self):
        costs = {("aircon", 2): (7_000, [900, 400, 122]), ("kerb", 0): (100, [84, 6])}
        budget = placement.auto_budget(costs, pinned={("kerb", 0)})
        # Every air conditioner at its coarsest, every kerb at its finest.
        least = 7_000 * 122 + 100 * 84
        assert budget == max(placement.DEFAULT_BUDGET, int(least * placement.HEADROOM))

    def test_it_never_goes_below_the_default(self):
        assert placement.auto_budget({("cone", 0): (10, [500, 20])}) == placement.DEFAULT_BUDGET

    def test_with_it_nothing_is_left_out(self):
        costs = {("aircon", 2): (100_000, [900, 400, 122])}
        chosen = choose_levels(costs, placement.auto_budget(costs), set(), {("aircon", 2): 99})
        entries = {("aircon", 2): (TestTheBudgetHolds.Sized(1.0), np.zeros((100_000, 13)), np.full(100_000, 70.0))}
        out = placement.Placed()
        keep = placement._fit(entries, costs, chosen, set(), placement.auto_budget(costs), out)
        assert keep[("aircon", 2)].all() and out.dropped == 0
