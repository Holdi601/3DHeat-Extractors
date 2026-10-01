"""
Reading ForzaTech texture swatches, and baking a tile's ground from them.

The format field is the thing most worth pinning down, because it was wrong
once in a way that looked right: byte 47 of the header is 4 in every texture,
and every texture first decoded happened to be BC7, so "4 means BC7" passed
until a kerb came out as noise. The format is the u32 at byte 72.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from heat3d_gamefiles.forzatech import UnsupportedForza
from heat3d_gamefiles.forzatexture import (
    TextureShelf,
    asphalt_weight,
    bake_ground,
    read_info,
    read_texture,
)

from .forza_fixtures import build_texture, build_track

BC1, BC7, RGBA8 = 0, 9, 13


def bc1_block(colour0: int, colour1: int, indices: int = 0) -> bytes:
    """One 4x4 BC1 block: two RGB565 end points and sixteen 2-bit indices."""
    return struct.pack("<HHI", colour0, colour1, indices)


RED_565 = 0xF800
WHITE_565 = 0xFFFF


class TestTheHeader:
    def test_size_mips_and_guid_are_read_from_their_places(self):
        data = build_texture(bytes(8), width=4, height=4, format_code=BC1, mips=3, guid=b"G" * 16)
        info = read_info(data)
        assert (info.width, info.height, info.mips, info.guid) == (4, 4, 3, b"G" * 16)

    def test_the_format_is_the_word_at_72_not_byte_47(self):
        """Byte 47 is 4 in the fixture, as in every real file; the format is not."""
        data = build_texture(bytes(8), width=4, height=4, format_code=BC1)
        assert data[44 + 47] == 4
        assert read_info(data).code == BC1
        assert read_info(data).format_name == "BC1"

    def test_something_that_is_not_a_texture_is_refused(self):
        with pytest.raises(UnsupportedForza):
            read_info(b"nope" + bytes(200))


class TestDecoding:
    def test_bc1_is_half_a_byte_a_pixel(self):
        """The kerb's format: a 4x4 block of solid red decodes to red."""
        data = build_texture(bc1_block(RED_565, WHITE_565), width=4, height=4, format_code=BC1)
        image = read_texture(data)
        assert image.shape == (4, 4, 4)
        assert (image[:, :, 0] > 240).all() and (image[:, :, 1] < 16).all()

    def test_bc1_indices_pick_the_second_end_point(self):
        # Index 1 everywhere: every pixel is colour1.
        data = build_texture(bc1_block(RED_565, WHITE_565, 0x55555555), width=4, height=4, format_code=BC1)
        assert (read_texture(data)[:, :, :3] > 240).all()

    def test_rgba8_is_read_as_it_is(self):
        pixels = np.arange(2 * 3 * 4, dtype=np.uint8)
        data = build_texture(pixels.tobytes(), width=3, height=2, format_code=RGBA8)
        assert np.array_equal(read_texture(data), pixels.reshape(2, 3, 4))

    def test_a_block_format_smaller_than_a_block_still_decodes(self):
        """A 2x2 mip is stored as one whole 4x4 block."""
        data = build_texture(bc1_block(RED_565, WHITE_565), width=2, height=2, format_code=BC1)
        assert read_texture(data).shape == (2, 2, 4)

    def test_too_few_pixels_is_an_error_not_noise(self):
        data = build_texture(bytes(4), width=4, height=4, format_code=BC1)
        with pytest.raises(UnsupportedForza):
            read_texture(data)

    def test_an_unidentified_format_is_refused_rather_than_guessed(self):
        data = build_texture(bytes(16), width=4, height=4, format_code=6)
        with pytest.raises(UnsupportedForza, match="not been identified"):
            read_texture(data)


def solid_bc1(size: int, colour: int) -> bytes:
    return bc1_block(colour, colour) * ((size // 4) * (size // 4))


class TestTheShelf:
    @pytest.fixture
    def track(self, tmp_path):
        stem = "tracks\\mainmap\\textures\\kerb_bclr_0123abcd-0000-0000-0000-000000000000"
        return build_track(
            tmp_path / "MainMap",
            {
                f"{stem}_quality1.pb": build_texture(solid_bc1(8, RED_565), width=8, height=8, format_code=BC1),
                f"{stem}_quality2.pb": build_texture(solid_bc1(32, RED_565), width=32, height=32, format_code=BC1),
                f"{stem}_quality3.pb": build_texture(solid_bc1(128, RED_565), width=128, height=128, format_code=BC1),
                "tracks\\mainmap\\scene\\something.modelbin": b"not a texture",
            },
        )

    def test_a_name_resolves_with_or_without_its_guid(self, track):
        with TextureShelf(track) as shelf:
            full = shelf.resolve("kerb_bclr")
            assert full == "kerb_bclr_0123abcd-0000-0000-0000-000000000000"
            assert shelf.resolve(full) == full
            assert shelf.resolve("no_such_texture") is None
            assert sorted(shelf.qualities("kerb_bclr")) == [1, 2, 3]

    def test_best_stops_at_the_first_level_big_enough_and_shrinks_it(self, track):
        with TextureShelf(track) as shelf:
            image = shelf.best("kerb_bclr", largest=16)
            # quality2 (32 px) is the first at least 16 across; shrunk to 16.
            assert image.shape[:2] == (16, 16)
            assert (image[:, :, 0] > 240).all()

    def test_best_takes_the_largest_there_is_when_none_is_big_enough(self, track):
        with TextureShelf(track) as shelf:
            assert shelf.best("kerb_bclr", largest=4096).shape[:2] == (128, 128)

    def test_a_missing_texture_is_none(self, track):
        with TextureShelf(track) as shelf:
            assert shelf.best("no_such_texture") is None


class TestBakingTheGround:
    def maps(self, sub_blue: int, gloss_green: int, size: int = 8) -> dict:
        main = np.full((size, size, 4), 128, dtype=np.uint8)
        sub = np.zeros((size, size, 4), dtype=np.uint8)
        sub[:, :, 2] = sub_blue
        gloss = np.zeros((size, size, 4), dtype=np.uint8)
        gloss[:, :, 1] = gloss_green
        return {"main": main, "sub": sub, "gloss": gloss}

    def test_asphalt_needs_both_maps_to_agree(self):
        """
        Measured on the driven line: submap blue 239, gloss green 147. Beside it:
        165 and 215. Either map alone speckles; the product does not.
        """
        assert asphalt_weight(self.maps(239, 147)).min() > 0.99
        assert asphalt_weight(self.maps(165, 215)).max() < 0.01
        assert asphalt_weight(self.maps(239, 215)).max() < 0.01
        assert asphalt_weight(self.maps(165, 147)).max() < 0.01

    def test_the_bake_is_an_rgb_image_of_the_tile(self):
        baked = bake_ground(self.maps(239, 147))
        assert baked.shape[:2] == (8, 8)
        assert baked.shape[2] >= 3
        assert baked.dtype == np.uint8

    def test_asphalt_bakes_grey_and_grass_green(self):
        road = bake_ground(self.maps(239, 147)).astype(int)
        grass = bake_ground(self.maps(165, 215)).astype(int)
        # Grey: channels close together. Green: green well above red and blue.
        assert np.abs(road[:, :, 0] - road[:, :, 1]).max() < 20
        assert (grass[:, :, 1] > grass[:, :, 0] + 10).all()
        assert (grass[:, :, 1] > grass[:, :, 2] + 10).all()
