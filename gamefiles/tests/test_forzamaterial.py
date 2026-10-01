"""
Which texture colours a placed model: material hashes, the manifest, seasons.

The rule this protects is that the texture comes from what the model's
material says, not from the model's name. `bar_rur_armco_crct_02_c` ships no
texture of its own name; its material binds `bar_rur_armco_crct_02_a`'s. A
lookup by name finds nothing for it - and for a model whose name happens to
prefix another texture's, finds the wrong thing.
"""

from __future__ import annotations

import struct

import numpy as np

from heat3d_gamefiles.forzamaterial import (
    Manifest,
    colour_texture,
    in_season,
    kind,
    season_of,
    strip_guid,
)
from heat3d_gamefiles.forzatech import SLOT_BASE_COLOUR, material_textures, read_model

from .forza_fixtures import build_container

GUID = "b2d95eb8-c667-4674-846c-5cdb38d4aca3"
NORMAL_SLOT = 0x9E5DB468
OTHER_SLOT = 0x2AB799BA

MANIFEST = f"""﻿<?xml version="1.0" encoding="utf-8"?>
<AssetManifest>
    <Materials>
    </Materials>
    <Textures>
      <Texture Source="tracks\\mainmap\\textures\\roads\\assets\\swatches\\road_gen_rum_round_asan_bclr_{GUID}.swatch" SourceHash="3130211576" PVSTextureIndex="1" />
      <Texture Source="tracks\\mainmap\\textures\\roads\\assets\\swatches\\road_gen_rum_round_asan_nrml_{GUID}.swatch" SourceHash="2214538317" PVSTextureIndex="2" />
      <Texture Source="x\\tex_gbl_frame_cty_01_a_bclr_9f5427d8-eb3a-4b47-b789-e708dab10c72.swatch" SourceHash="11" PVSTextureIndex="3" />
      <Texture Source="x\\tex_gbl_inf_swatchtint_a_swatchtint_2b921cc1-2448-44c3-af52-1a7d0cb1736a.swatch" SourceHash="12" PVSTextureIndex="4" />
      <Texture Source="x\\mainmap_x0_z0_xsummerx_diff_12345678-9abc-def0-3a12-0067e70067e7.swatch" SourceHash="13" PVSTextureIndex="5" />
      <Texture Source="x\\roc_alp_mid_surface_a_flat_xsummerx_diff_fec9c3d6-acb8-45b7-8df5-3b34e2b429c3.swatch" SourceHash="14" PVSTextureIndex="6" />
    </Textures>
    <Models>
      <Texture Source="x\\after_the_list.swatch" SourceHash="99" PVSTextureIndex="7" />
    </Models>
</AssetManifest>
"""


def manifest(tmp_path) -> Manifest:
    (tmp_path / "AssetManifest.xml").write_text(MANIFEST, encoding="utf-8")
    return Manifest.read(tmp_path)


def material(*slots: tuple[int, int], name: str = "Material__29", misalign: int = 1) -> bytes:
    """
    A `MatI` payload listing texture slots the way a real one does.

    The records are `u32 slot, u32, u32 hash`, 33 bytes apart in the file, so
    they land at every alignment; `misalign` bytes in front keep the fixture
    from being accidentally four-byte aligned.
    """
    body = bytearray(b"\x00" * misalign)
    body += b"Name" + name.encode() + b"\x00"
    body += b"Game:\\Media\\tracks\\mainmap\\materials\\environment\\environment_spline.materialbin\x00"
    for slot, value in slots:
        body += struct.pack("<III", slot, 0x0022_37DE, value) + b"\x00" * 21
    return bytes(body)


class TestTheManifest:
    def test_reads_every_texture_by_its_hash(self, tmp_path):
        found = manifest(tmp_path)
        assert found.name(3130211576) == f"road_gen_rum_round_asan_bclr_{GUID}"
        assert found.name(2214538317) == f"road_gen_rum_round_asan_nrml_{GUID}"

    def test_stops_at_the_end_of_the_texture_list(self, tmp_path):
        """The file is 90 MB of model records after the textures; none are read."""
        assert manifest(tmp_path).name(99) is None

    def test_a_track_without_one_has_none(self, tmp_path):
        assert Manifest.read(tmp_path) is None

    def test_the_hashes_come_sorted_for_searching(self, tmp_path):
        hashes = manifest(tmp_path).hashes()
        assert list(hashes) == sorted(hashes)


class TestNames:
    def test_the_kind_is_the_word_before_the_guid(self):
        assert kind(f"road_gen_rum_round_asan_bclr_{GUID}") == "bclr"
        assert kind("led_flipbook_4x8_emis_w2omze7") == "emis"

    def test_strip_guid(self):
        assert strip_guid(f"kerb_xsummerx_bclr_{GUID}") == "kerb_xsummerx_bclr"

    def test_season_of(self):
        assert season_of("armco_xwinterx_bclr") == "winter"
        assert season_of("kerb_winter_bclr") == "winter"
        assert season_of("kerb_bclr") is None


class TestReadingAMaterial:
    def test_finds_the_hashes_at_any_alignment(self, tmp_path):
        known = manifest(tmp_path).hashes()
        model = read_model(
            "kerb.modelbin",
            build_container(
                [
                    ("MatI", material((SLOT_BASE_COLOUR, 3130211576), (NORMAL_SLOT, 2214538317), misalign=1)),
                    ("MatI", material((SLOT_BASE_COLOUR, 11), misalign=3)),
                ]
            ),
        )
        slots = material_textures(model, known)
        assert slots == [
            [(SLOT_BASE_COLOUR, 3130211576), (NORMAL_SLOT, 2214538317)],
            [(SLOT_BASE_COLOUR, 11)],
        ]

    def test_four_bytes_that_are_not_a_known_hash_are_ignored(self, tmp_path):
        known = manifest(tmp_path).hashes()
        model = read_model("m.modelbin", build_container([("MatI", material((SLOT_BASE_COLOUR, 123456)))]))
        assert material_textures(model, known) == [[]]

    def test_accepts_a_plain_set_too(self):
        model = read_model("m.modelbin", build_container([("MatI", material((SLOT_BASE_COLOUR, 42)))]))
        assert material_textures(model, {42, 7}) == [[(SLOT_BASE_COLOUR, 42)]]


class TestChoosingTheColour:
    def test_the_standard_slot_wins(self, tmp_path):
        found = manifest(tmp_path)
        slots = [(NORMAL_SLOT, 2214538317), (SLOT_BASE_COLOUR, 3130211576)]
        assert colour_texture(slots, found) == f"road_gen_rum_round_asan_bclr_{GUID}"

    def test_another_shader_is_read_by_the_texture_kind(self, tmp_path):
        """Signage and building shaders put their colour under other slot ids."""
        found = manifest(tmp_path)
        assert colour_texture([(0x1234, 2214538317), (OTHER_SLOT, 11)], found).startswith(
            "tex_gbl_frame_cty_01_a_bclr"
        )
        assert colour_texture([(0x1234, 14)], found).startswith("roc_alp_mid_surface")

    def test_tint_palettes_and_terrain_maps_colour_nothing(self, tmp_path):
        found = manifest(tmp_path)
        assert colour_texture([(0x1, 12), (0x2, 13)], found) is None

    def test_no_colour_texture_is_none(self, tmp_path):
        assert colour_texture([(NORMAL_SLOT, 2214538317)], manifest(tmp_path)) is None


class TestSeasons:
    def available(self, *names):
        full = {n: n + "_" + GUID for n in names}
        return lambda bare: full.get(bare)

    def test_a_summer_texture_is_swapped_for_winter(self):
        lookup = self.available("armco_xwinterx_bclr")
        assert in_season(f"armco_xsummerx_bclr_{GUID}", "winter", lookup) == f"armco_xwinterx_bclr_{GUID}"

    def test_an_unmarked_summer_texture_finds_its_winter_pair(self):
        """The kerb's pair is `_bclr` and `_winter_bclr`."""
        lookup = self.available("kerb_winter_bclr")
        assert in_season(f"kerb_bclr_{GUID}", "winter", lookup) == f"kerb_winter_bclr_{GUID}"

    def test_summer_asked_of_summer_is_unchanged(self):
        name = f"kerb_bclr_{GUID}"
        assert in_season(name, "summer", self.available()) == name

    def test_no_pair_keeps_what_there_is(self):
        name = f"armco_xsummerx_bclr_{GUID}"
        assert in_season(name, "winter", self.available()) == name

    def test_winter_back_to_an_unmarked_summer(self):
        lookup = self.available("kerb_bclr")
        assert in_season(f"kerb_winter_bclr_{GUID}", "summer", lookup) == f"kerb_bclr_{GUID}"


def test_the_manifest_hashes_are_uint32():
    assert Manifest({1: "a", 2**32 - 1: "b"}).hashes().dtype == np.uint32
