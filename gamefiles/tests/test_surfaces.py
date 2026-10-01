"""
Telling a track's surfaces apart, as far as a material name can.

This exists because the first export did not, and the complaint was exact: you
could not tell road from kerb from grass from barrier. The first answer was the
material name on each triangle, and it was wrong for the biggest surfaces:
around a circuit 94% of the ground carries a road-shader material, because the
terrain near a road is drawn with it and asphalt or grass is decided per pixel.
So road, verge and terrain come from elsewhere (the vertex layout and the tile's
asphalt mask), and what is tested here is the part a name *can* decide - and
that it says so, rather than guessing, when it cannot.
"""

from __future__ import annotations

import re

import pytest

from heat3d_gamefiles.surfaces import (
    MARKINGS,
    OTHER,
    ROAD,
    SURFACES,
    TERRAIN,
    TYREMARKS,
    VERGE,
    WATER,
    classify,
    part_name,
)


class TestWhatANameDecides:
    @pytest.mark.parametrize(
        "material,expected",
        [
            ("whiteline_decal_damage_a", "markings"),
            ("yellowline_decal_damage_b", "markings"),
            ("decal_d1n1m1p1_alphatest", "markings"),
            ("tyremark_decal_dirt", "tyremarks"),
            ("skidmarks_heavy", "tyremarks"),
            ("PLN_Lake_RicePaddy", "water"),
            ("river_shallow_a", "water"),
        ],
    )
    def test_the_names_a_real_circuit_ships(self, material, expected):
        assert classify(material).key == expected

    def test_a_tyre_mark_is_not_painted_as_a_marking(self):
        """
        `tyremark_decal_dirt` contains `decal`, so the order of the patterns is
        what decides it. Backwards, every rubbered-in braking zone is drawn as
        white paint.
        """
        assert classify("tyremark_decal_dirt") is TYREMARKS
        assert classify("whiteline_decal_damage_a") is MARKINGS


class TestWhatANameDoesNotDecide:
    @pytest.mark.parametrize(
        "material",
        ["uber_megaroad", "uber_megaroad_edge", "junctionundermesh", "TERR_Mega", "Terr_Mega_Deform"],
    )
    def test_road_and_terrain_materials_say_look_elsewhere(self, material):
        """
        The road shader draws road, verge and grass alike, so its name is not
        evidence of any of them. `other` is the signal that the vertex layout and
        the asphalt mask have to decide - never a quiet guess of road.
        """
        assert classify(material) is OTHER

    def test_something_unrecognised_is_labelled_unrecognised(self):
        assert classify("SomeNewMaterialNobodyHasSeen") is OTHER

    def test_an_empty_material_does_not_crash_it(self):
        assert classify("") is OTHER

    def test_the_surfaces_decided_elsewhere_have_no_pattern(self):
        for surface in (ROAD, VERGE, TERRAIN):
            assert surface.pattern is None


class TestWhatTheViewerIsTold:
    def test_every_part_name_matches_the_pattern_the_viewer_enforces(self):
        """
        `classify.ts` takes a part's class from its name only when it begins with
        ground, structure or water followed by a colon or a bar. Anything else
        falls to a heuristic written for Unreal landscapes, which is not what a
        Forza circuit looks like.
        """
        for surface in (*SURFACES, OTHER):
            assert re.match(r"^(ground|structure|water)\s*[:|]", part_name(surface))

    def test_water_is_water_and_the_rest_is_ground(self):
        assert part_name(WATER) == "water:water"
        assert part_name(ROAD) == "ground:road"
        assert part_name(MARKINGS) == "ground:markings"
