"""
ForzaTech textures: the `.pb` swatches.

A swatch is the same chunked container as a model — it opens with `burG` — holding
one chunk of block-compressed pixels behind a 96-byte header. The header carries
the texture's own GUID, which is the one in its file name, its size, the number
of mip levels in this file, and a format code.

The format is a u32 at byte 72 of the header, in ForzaTech's own numbering
rather than DXGI's, and the u32 at byte 84 is the size of the pixels that
follow. So each format here was identified the only way available: the payload
size fixes the bit rate, and decoding a real file every way that bit rate allows
keeps the one that yields a picture. Format 9 decodes as BC7 into a terrain tile
with the circuit visible in it; format 0 is half a byte a pixel and decodes as
BC1 into a red and white kerb, and as noise under BC4. Format 13 is four bytes a
pixel, plain RGBA. Codes not yet seen are refused rather than guessed at.

Byte 47, which an earlier version of this took for the format, is 4 in every one
of 3,000 textures sampled; it only looked like a format because every texture
first tried was BC7.

Why the terrain needs these
---------------------------
A terrain tile ships a whole-tile texture, `mainmap_x…_z…_<season>_diff`, in five
resolutions split across two archives — the largest 2048 pixels across for 512
metres of ground, which is 25 cm a pixel. It maps onto the tile by position
alone, so a terrain mesh can be coloured with no UVs at all.

Decoding is Pillow's. Its BCn decoder is the reason there is no C extension of
this project's own here, and it is already a dependency of anything that writes
an image.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass

from .forzatech import UnsupportedForza, read_model

#: Where the texture header begins, and how long it is.
HEADER_AT = 44
HEADER_BYTES = 96
HEADER_MAGIC = b"HCXT"

#: ForzaTech format code to DXGI format, for the codes that have been checked
#: against a real file. See the module docstring for what "checked" meant.
FORMATS: dict[int, tuple[int, float, str]] = {
    # code: (DXGI format, bytes per pixel, name)
    0: (71, 0.5, "BC1"),
    9: (98, 1.0, "BC7"),
    13: (28, 4.0, "RGBA8"),
}

#: Where in the header the format and the pixel byte count are.
FORMAT_AT = 72
PAYLOAD_AT = 84


@dataclass(frozen=True)
class TextureInfo:
    width: int
    height: int
    mips: int
    code: int
    guid: bytes

    @property
    def format_name(self) -> str:
        known = FORMATS.get(self.code)
        return known[2] if known else f"unknown ({self.code})"


def read_info(data: bytes) -> TextureInfo:
    """The header of a `.pb`, without decoding the pixels."""
    if data[:4] != b"burG":
        raise UnsupportedForza("not a ForzaTech texture: no burG container")
    header = data[HEADER_AT : HEADER_AT + HEADER_BYTES]
    if len(header) < HEADER_BYTES or header[:4] != HEADER_MAGIC:
        raise UnsupportedForza("the texture header is not where it should be")
    width, height = struct.unpack_from("<II", header, 32)
    mips = header[46]
    (code,) = struct.unpack_from("<I", header, FORMAT_AT)
    return TextureInfo(
        width=width, height=height, mips=mips, code=code, guid=header[16:32]
    )


def _dds(width: int, height: int, dxgi: int, payload: bytes) -> bytes:
    """Wrap block-compressed pixels in a DDS with a DX10 header, for Pillow."""
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000
    head = struct.pack("<4sI", b"DDS ", 124)
    head += struct.pack("<IIIIII", flags, height, width, len(payload), 0, 1)
    head += b"\x00" * 44
    head += struct.pack("<II4sIIIII", 32, 0x4, b"DX10", 0, 0, 0, 0, 0)
    head += struct.pack("<IIIII", 0x1000, 0, 0, 0, 0)
    head += struct.pack("<IIIII", dxgi, 3, 0, 1, 0)
    return head + payload


def read_texture(data: bytes):
    """
    The top mip of a `.pb`, as an (height, width, 4) uint8 array.

    Only the largest level is decoded. A file holds either one level or a tail
    of small ones; the caller who wants detail asks for the file holding the
    level it wants, which is how the game streams them too.
    """
    import numpy as np
    from PIL import Image

    info = read_info(data)
    known = FORMATS.get(info.code)
    if known is None:
        raise UnsupportedForza(
            f"texture format {info.code} has not been identified; refusing to guess"
        )
    dxgi, per_pixel, _name = known
    model = read_model("texture.pb", data)
    pixels = model.raw(model.chunks[0])
    if dxgi == 28:
        need = info.width * info.height * 4
    else:
        # Block formats store whole 4x4 blocks, however small the image.
        need = int(-(-info.width // 4) * 4 * -(-info.height // 4) * 4 * per_pixel)
    if len(pixels) < need:
        raise UnsupportedForza(
            f"a {info.width}x{info.height} texture needs {need} bytes and has "
            f"{len(pixels)}"
        )
    if dxgi == 28:
        flat = np.frombuffer(pixels[:need], dtype=np.uint8)
        return flat.reshape(info.height, info.width, 4).copy()
    image = Image.open(io.BytesIO(_dds(info.width, info.height, dxgi, pixels[:need])))
    return np.asarray(image.convert("RGBA"))


# Tile maps
# ---------
# Three textures per terrain tile, each covering its 512-metre square exactly:
#
#   mainmap_x…_z…_<season>_diff     R, G: detail normal (x and z)   A: low on asphalt
#   submap_x…_z…_<season>_diff      layer weights; B high on asphalt
#   autoglossf0_x…_z…_<season>_glos G low on asphalt, B rises in woodland
#
# None of them is a colour photograph, despite the `diff` in two names: they are
# the inputs a terrain shader blends its material layers with. What they give is
# *where* each kind of ground is, at 25 cm, which is the detail the geometry
# alone does not have.
#
# The channel meanings above were measured, not assumed: sampled under the 13,456
# lap samples that cross one circuit's tile and compared with the ground five,
# fifteen and sixty metres away. Submap B is 239 on the driven line and 165 beside
# it; gloss G is 147 against 215; mainmap A is 60 against 90. The normal's R and G
# correlate +0.57 and +0.61 with the x and z of the decoded mesh normals.
#
# Orientation: pixel row 0 is the tile's north edge (largest z) and column 0 its
# west edge (smallest x). Found by projecting those laps every way a square can
# be flipped or turned; one of the eight puts them on the asphalt mask's darkest
# pixels and the other seven do not.

TILE_MAPS = {
    "main": "mainmap_x{x}_z{z}_x{season}x_diff",
    "sub": "submap_x{x}_z{z}_x{season}x_diff",
    "gloss": "autoglossf0_x{x}_z{z}_x{season}x_glos",
}

#: Largest to smallest. The largest two live in a different archive from the rest.
QUALITIES = (5, 4, 3, 2, 1)


class TextureShelf:
    """
    Every texture in a track's archives, findable by name.

    Built once per export because the tile maps for one tile are spread across
    two archives - the three smallest resolutions in one, the two largest in
    another - and asking each archive in turn per tile would open them hundreds
    of times.
    """

    def __init__(self, folder, chunks=(0, 1, 2, 3)):
        from pathlib import Path

        from .minizip import open_chunk

        self.folder = Path(folder)
        self._archives = {}
        #: Stem (the name up to `_quality<n>.pb`) -> quality -> where.
        self._stems: dict[str, dict[int, tuple[int, int]]] = {}
        self._prefix: dict[str, str | None] = {}
        for number in chunks:
            if not (self.folder / f"GeoChunk{number}.minizip").exists():
                continue
            archive, contents = open_chunk(self.folder, number)
            self._archives[number] = archive
            for entry, name in enumerate(contents.names):
                if not name.endswith(".pb"):
                    continue
                leaf = name.rsplit("\\", 1)[-1].lower()
                stem, _, quality = leaf[:-3].rpartition("_quality")
                if stem and quality.isdigit():
                    self._stems.setdefault(stem, {})[int(quality)] = (number, entry)

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def resolve(self, stem: str) -> str | None:
        """
        The full stem of a swatch, from its name with or without the GUID.

        The names run `<name>_<guid>_quality<n>.pb`, and the GUID is the texture's
        own identity rather than anything a caller would know - a tile map is
        asked for by tile and season alone. A name that is already whole is
        returned as it is.
        """
        stem = stem.lower()
        if stem in self._stems:
            return stem
        if stem not in self._prefix:
            start = stem + "_"
            self._prefix[stem] = next((s for s in self._stems if s.startswith(start)), None)
        return self._prefix[stem]

    def qualities(self, stem: str) -> dict[int, tuple[int, int]]:
        full = self.resolve(stem)
        return self._stems.get(full, {}) if full else {}

    def find(self, stem: str, quality: int) -> tuple[int, int] | None:
        """Where one resolution of a swatch is, or None."""
        return self.qualities(stem).get(quality)

    def read(self, stem: str, quality: int):
        where = self.find(stem, quality)
        if where is None:
            return None
        number, entry = where
        return read_texture(self._archives[number].read(entry))

    def best(self, stem: str, *, largest: int = 512):
        """
        A swatch at the first resolution at least `largest` pixels across,
        shrunk to fit - or the biggest there is, if none is that large.

        Resolutions are tried smallest first, and each is only a header read
        before it is decoded: the finest level of a building's texture can be
        16 MB, and a viewer that packs a hundred textures into one atlas has no
        use for it.
        """
        import numpy as np
        from PIL import Image

        found = self.qualities(stem)
        chosen = None
        for quality in sorted(found):
            number, entry = found[quality]
            data = self._archives[number].read(entry)
            try:
                info = read_info(data)
            except UnsupportedForza:
                continue
            chosen = data
            if max(info.width, info.height) >= largest:
                break
        if chosen is None:
            return None
        image = read_texture(chosen)
        height, width = image.shape[:2]
        if max(height, width) > largest:
            scale = largest / max(height, width)
            size = (max(4, round(width * scale)), max(4, round(height * scale)))
            image = np.asarray(Image.fromarray(image).resize(size, Image.BOX))
        return image

    def tile_maps(self, x: int, z: int, *, season: str = "summer", quality: int = 5):
        """
        The three maps for one tile, at the best resolution available up to
        `quality`, all resampled to the size of the largest.
        """
        import numpy as np
        from PIL import Image

        found = {}
        for key, pattern in TILE_MAPS.items():
            stem = pattern.format(x=x, z=z, season=season)
            for level in QUALITIES:
                if level > quality:
                    continue
                image = self.read(stem, level)
                if image is not None:
                    found[key] = image
                    break
        if "main" not in found:
            return None
        size = found["main"].shape[0]
        for key, image in found.items():
            if image.shape[0] != size:
                found[key] = np.asarray(
                    Image.fromarray(image).resize((size, size), Image.BILINEAR)
                )
        return found


#: Linear RGB, 0..1. Chosen to read at a glance rather than to match the game's
#: palette, which lives in material layers this does not decode.
GROUND_COLOURS = {
    "asphalt": (0.20, 0.20, 0.21),
    "grass": (0.30, 0.44, 0.18),
    "rough": (0.40, 0.42, 0.22),
    "woodland": (0.13, 0.23, 0.10),
}


def _smoothstep(edge0: float, edge1: float, value):
    import numpy as np

    t = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def to_srgb(linear):
    """Linear 0..1 to 8-bit sRGB, which is what glTF expects of a colour texture."""
    import numpy as np

    linear = np.clip(linear, 0.0, 1.0)
    srgb = np.where(
        linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1 / 2.4) - 0.055
    )
    return (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def asphalt_weight(maps: dict):
    """
    How much of each pixel is asphalt, 0..1, from a tile's maps.

    Both maps have to agree. Either one alone leaves a speckle of grey through
    the grass wherever block compression rings at the edge of a patch, and
    taking the larger of the two - which the first version did - adds both
    speckles together. The thresholds sit between the values measured on and
    off the driven line.
    """
    import numpy as np

    sub = maps.get("sub", maps["main"]).astype(np.float32)
    gloss = maps.get("gloss", maps["main"]).astype(np.float32)
    return _smoothstep(188.0, 228.0, sub[:, :, 2]) * _smoothstep(200.0, 168.0, gloss[:, :, 1])


def bake_ground(maps: dict, *, light=(-0.45, 0.75, 0.48)):
    """
    One colour image for a tile, from its three maps.

    An interpretation and not a reconstruction, and it says so here because the
    difference matters to anyone using it for more than looking: the game's own
    colours come from material layers blended by these maps, and those layers
    are not decoded. What this keeps is where each kind of ground is - asphalt,
    grass kept short near the track, rougher ground further off, woodland - at
    the maps' full 25 cm, shaded by their detail normals so kerb lips, banks and
    ruts read as relief.

    The thresholds sit between the values measured on and off the driven line
    (see the section comment above), not at arbitrary points.
    """
    import numpy as np

    main = maps["main"].astype(np.float32)
    gloss = maps.get("gloss", maps["main"]).astype(np.float32)

    asphalt = asphalt_weight(maps)
    woodland = _smoothstep(4.0, 24.0, gloss[:, :, 2])
    # Kept grass near the track is lighter in the mainmap alpha than the rough
    # ground further out; that difference is what separates the two greens.
    rough = _smoothstep(88.0, 72.0, main[:, :, 3]) * (1.0 - asphalt)

    colour = np.empty(main.shape[:2] + (3,), dtype=np.float32)
    colour[:] = GROUND_COLOURS["grass"]
    colour += (np.asarray(GROUND_COLOURS["rough"]) - colour) * rough[:, :, None]
    colour += (np.asarray(GROUND_COLOURS["woodland"]) - colour) * woodland[:, :, None]
    colour += (np.asarray(GROUND_COLOURS["asphalt"]) - colour) * asphalt[:, :, None]

    nx = main[:, :, 0] / 127.5 - 1.0
    nz = main[:, :, 1] / 127.5 - 1.0
    ny = np.sqrt(np.clip(1.0 - nx * nx - nz * nz, 0.0, 1.0))
    direction = np.asarray(light, dtype=np.float32)
    direction /= np.linalg.norm(direction)
    lambert = np.clip(nx * direction[0] + ny * direction[1] + nz * direction[2], 0.0, 1.0)
    colour *= (0.55 + 0.6 * lambert)[:, :, None]
    return to_srgb(colour)
