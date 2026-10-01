"""
Cooked Unreal 5 textures: the pixels of one level of a `Texture2D`.

A texture export is its properties and then its cooked platform data: the
pixel format's name, a skip offset, the size, the format again as a string,
and a table of mip levels - each a bulk data index and its width, height and
depth. The bulk data indices point into the package header's bulk data map:
per entry an offset, a size and flags saying whether the bytes are inline in
the export or in the package's `.ubulk` chunk. Large levels are in the
`.ubulk`, the small tail inline.

The platform data is found by its own shape - width, height, slice count, and
the format string - rather than by skipping the properties, so this does not
depend on the texture class's property layout. Checked on a stage's satellite
image: 4096 x 4096 DXT5, thirteen levels whose sizes in the bulk map are
exactly those of each level's blocks.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass

import numpy as np

from .iostore import Store
from .zen import Package, Packages

#: Unreal pixel formats this decodes: DXGI format, bytes per 4x4 block (or per
#: pixel for the uncompressed ones, marked by a negative count).
FORMATS = {
    "PF_DXT1": (71, 8),
    "PF_DXT3": (74, 16),
    "PF_DXT5": (77, 16),
    "PF_BC4": (80, 8),
    "PF_BC5": (83, 16),
    "PF_BC7": (98, 16),
    "PF_B8G8R8A8": (87, -4),
    "PF_R8G8B8A8": (28, -4),
    "PF_G8": (61, -1),
}


class TextureError(Exception):
    """A texture this cannot read, and why."""


@dataclass
class Mip:
    index: int
    width: int
    height: int


def bulk_map(package: Package) -> list[tuple[int, int, int]]:
    """
    The package's bulk data map: (offset, size, flags) per entry. It sits
    just before the imported export hashes, its byte size in the 8 bytes
    before it.
    """
    data, end = package.data, package.hashes_at
    for start in range(end, 7, -1):
        if (end - start) % 32:
            continue
        (size,) = struct.unpack_from("<q", data, start - 8)
        if size == end - start:
            out = []
            for i in range(size // 32):
                offset, _dup, length, flags, _pad = struct.unpack_from("<qqqII", data, start + 32 * i)
                out.append((offset, length, flags))
            return out
        if end - start > 1 << 20:
            break
    return []


def _platform(data: bytes, names: list[str]):
    """The pixel format and mip table of a texture export's cooked data."""
    for name in names:
        if name not in FORMATS:
            continue
        text = name.encode("ascii") + b"\0"
        tag = struct.pack("<i", len(text)) + text
        at = data.find(tag)
        while at >= 12:
            width, height, _packed = struct.unpack_from("<iii", data, at - 12)
            if 0 < width <= 16384 and 0 < height <= 16384:
                cursor = at + len(tag) + 4  # the format string, then the first mip to serialize
                (count,) = struct.unpack_from("<i", data, cursor)
                cursor += 4
                if 0 < count <= 16:
                    mips = []
                    for i in range(count):
                        index, w, h, _depth = struct.unpack_from("<iiii", data, cursor + 16 * i)
                        mips.append(Mip(index, w, h))
                    return name, mips
            at = data.find(tag, at + 1)
    return None


def _dds(width: int, height: int, dxgi: int, payload: bytes) -> bytes:
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000
    head = struct.pack("<4sI", b"DDS ", 124)
    head += struct.pack("<IIIIII", flags, height, width, len(payload), 0, 1)
    head += b"\x00" * 44
    head += struct.pack("<II4sIIIII", 32, 0x4, b"DX10", 0, 0, 0, 0, 0)
    head += struct.pack("<IIIII", 0x1000, 0, 0, 0, 0)
    head += struct.pack("<IIIII", dxgi, 3, 0, 1, 0)
    return head + payload


def read_texture(packages: Packages, path: str, *, largest: int = 4096) -> np.ndarray:
    """A texture as (height, width, 4) uint8, its first level no larger than `largest`."""
    from PIL import Image

    package = packages.open(path)
    if package is None:
        raise TextureError(f"{path}: not found")
    export = next((e for e in package.exports if package.class_name(e).endswith(".Texture2D")), None)
    if export is None:
        raise TextureError(f"{path}: not a 2D texture")
    data = package.export_data(export)
    found = _platform(data, package.names)
    if found is None:
        raise TextureError(f"{path}: no cooked pixels in a known format")
    name, mips = found
    entries = bulk_map(package)
    chosen = next((m for m in mips if max(m.width, m.height) <= largest), mips[-1])
    if not 0 <= chosen.index < len(entries):
        raise TextureError(f"{path}: level {chosen.width}x{chosen.height} has no bulk entry")
    offset, length, flags = entries[chosen.index]
    if flags & 0x40:  # inline in the export
        payload = data[offset : offset + length]
    else:
        store: Store = packages.store
        bulk = store.bulk(path)
        if bulk is None:
            raise TextureError(f"{path}: its bulk data is not in any container")
        payload = bulk[offset : offset + length]
    if len(payload) < length:
        raise TextureError(f"{path}: level {chosen.width}x{chosen.height} is cut short")
    dxgi, per = FORMATS[name]
    if per < 0:
        mode = {-4: "RGBA", -1: "L"}[per]
        image = Image.frombytes(mode, (chosen.width, chosen.height), payload[: chosen.width * chosen.height * -per])
        if name == "PF_B8G8R8A8":
            b, g, r, a = image.split()
            image = Image.merge("RGBA", (r, g, b, a))
    else:
        blocks = -(-chosen.width // 4) * -(-chosen.height // 4) * per
        with Image.open(io.BytesIO(_dds(chosen.width, chosen.height, dxgi, payload[:blocks]))) as im:
            image = im.convert("RGBA")
    return np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()


# --------------------------------------------------------------------------
# Streaming virtual textures


def _morton(x: int, y: int) -> int:
    """Tile address: the bits of x and y interleaved, x in the even bits."""
    out = 0
    for bit in range(16):
        out |= ((x >> bit) & 1) << (2 * bit) | ((y >> bit) & 1) << (2 * bit + 1)
    return out


class _Cursor:
    def __init__(self, data: bytes, at: int) -> None:
        self.data, self.at = data, at

    def u32(self) -> int:
        (value,) = struct.unpack_from("<I", self.data, self.at)
        self.at += 4
        return value

    def array(self) -> list[int]:
        count = self.u32()
        values = list(struct.unpack_from(f"<{count}I", self.data, self.at))
        self.at += 4 * count
        return values

    def text(self) -> str:
        (length,) = struct.unpack_from("<i", self.data, self.at)
        self.at += 4
        if length < 0:
            raw = self.data[self.at : self.at - 2 * length]
            self.at -= 2 * length
            return raw.decode("utf-16-le").rstrip(chr(0))
        raw = self.data[self.at : self.at + length]
        self.at += length
        return raw.decode("latin-1").rstrip(chr(0))


class VirtualTexture:
    """
    A baked streaming virtual texture (`VirtualTexture2D`): square tiles with
    a border, per level of detail, addressed by Morton code, stored in chunks
    of the package's bulk data. Layout as `FVirtualTextureBuiltData::Serialize`
    writes it in Unreal 5.6; checked on a stage's 16384-pixel base colour,
    whose 64 x 64 tiles of 256 pixels fill its first chunk to the byte.
    """

    def __init__(self, packages: Packages, path: str) -> None:
        package = packages.open(path)
        if package is None:
            raise TextureError(f"{path}: not found")
        export = next((e for e in package.exports if package.class_name(e).endswith("VirtualTexture2D")), None)
        if export is None:
            raise TextureError(f"{path}: not a virtual texture")
        self.packages, self.path = packages, path
        data = package.export_data(export)
        found = _platform_header(data, package.names)
        if found is None:
            raise TextureError(f"{path}: no virtual texture data")
        r = _Cursor(data, found)
        if r.u32() != 1:  # cooked
            raise TextureError(f"{path}: not cooked")
        self.layers = r.u32()
        r.u32()
        r.u32()  # width and height in blocks
        self.tile = r.u32()
        self.border = r.u32()
        self.layer_offsets = r.array()
        self.mips = r.u32()
        self.width, self.height = r.u32(), r.u32()
        self.chunk_of_mip = r.array()
        self.base_of_mip = r.array()
        self.offsets = []
        for _ in range(r.u32()):
            w, h, _max = r.u32(), r.u32(), r.u32()
            self.offsets.append((w, h, r.array(), r.array()))
        r.array()
        r.array()
        r.array()  # the legacy tile tables
        self.formats = [r.text() for _ in range(self.layers)]
        r.at += 16 * self.layers  # fallback colours
        chunks = r.u32()
        self.entries = bulk_map(package)
        # Chunk n is bulk entry n: the first entry of a 64 x 64-tile level is
        # 4 + 4096 tiles of 34,848 bytes, the first chunk's size to the byte.
        if len(self.entries) < chunks:
            raise TextureError(f"{path}: {chunks} chunks and {len(self.entries)} bulk entries")
        self.chunk_bulk = list(range(chunks))
        self._chunks: dict[int, bytes] = {}
        self._tiles: dict[tuple[int, int, int], np.ndarray | None] = {}
        if self.formats[0] not in FORMATS:
            raise TextureError(f"{path}: layer format {self.formats[0]}")

    def _chunk(self, number: int) -> bytes:
        if number not in self._chunks:
            offset, length, _flags = self.entries[self.chunk_bulk[number]]
            bulk = self.packages.store.bulk(self.path)
            if bulk is None:
                raise TextureError(f"{self.path}: its bulk data is not in any container")
            self._chunks[number] = bulk[offset : offset + length]
        return self._chunks[number]

    def tile_pixels(self, mip: int, x: int, y: int) -> np.ndarray | None:
        """One tile's colour without its border, (tile, tile, 3), or None where there is none."""
        key = (mip, x, y)
        if key in self._tiles:
            return self._tiles[key]
        from PIL import Image

        width, height, addresses, offsets = self.offsets[mip]
        result = None
        if x < width and y < height:
            address = _morton(x, y)
            block = int(np.searchsorted(addresses, address, side="right")) - 1
            if block >= 0 and offsets[block] != 0xFFFFFFFF:
                index = offsets[block] + address - addresses[block]
                size = self.layer_offsets[-1]
                start = self.base_of_mip[mip] + index * size
                chunk = self._chunk(self.chunk_of_mip[mip])
                payload = chunk[start : start + self.layer_offsets[0]]
                side = self.tile + 2 * self.border
                dxgi, per = FORMATS[self.formats[0]]
                if per > 0 and len(payload) >= (side // 4) * (side // 4) * per:
                    with Image.open(io.BytesIO(_dds(side, side, dxgi, payload))) as im:
                        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
                    b = self.border
                    result = rgb[b : b + self.tile, b : b + self.tile]
        self._tiles[key] = result
        return result

    def region(self, u0: float, v0: float, u1: float, v1: float, size: int) -> np.ndarray:
        """
        The texture between (u0, v0) and (u1, v1), resampled to `size` square,
        from the level whose pixels are nearest that size. Outside, zeros.
        """
        from PIL import Image

        span = max(u1 - u0, v1 - v0)
        wanted = size / max(span, 1e-9)
        mip = 0
        while mip + 1 < len(self.offsets) and (self.width >> (mip + 1)) >= wanted:
            mip += 1
        level_w, level_h = max(1, self.width >> mip), max(1, self.height >> mip)
        px0, py0 = int(np.floor(u0 * level_w)), int(np.floor(v0 * level_h))
        px1, py1 = int(np.ceil(u1 * level_w)), int(np.ceil(v1 * level_h))
        canvas = np.zeros((max(1, py1 - py0), max(1, px1 - px0), 3), np.uint8)
        for ty in range(max(0, py0 // self.tile), min((level_h - 1) // self.tile, (py1 - 1) // self.tile) + 1):
            for tx in range(max(0, px0 // self.tile), min((level_w - 1) // self.tile, (px1 - 1) // self.tile) + 1):
                pixels = self.tile_pixels(mip, tx, ty)
                if pixels is None:
                    continue
                gx, gy = tx * self.tile - px0, ty * self.tile - py0
                sx0, sy0 = max(0, -gx), max(0, -gy)
                dx0, dy0 = max(0, gx), max(0, gy)
                w = min(self.tile - sx0, canvas.shape[1] - dx0)
                h = min(self.tile - sy0, canvas.shape[0] - dy0)
                if w > 0 and h > 0:
                    canvas[dy0 : dy0 + h, dx0 : dx0 + w] = pixels[sy0 : sy0 + h, sx0 : sx0 + w]
        return np.array(Image.fromarray(canvas).resize((size, size), Image.BILINEAR), dtype=np.uint8)


def _platform_header(data: bytes, names: list[str]) -> int | None:
    """Where a virtual texture's built data starts: after `..., format, 0 mips, virtual = 1`."""
    for name in names:
        if name not in FORMATS:
            continue
        text = name.encode("ascii") + bytes(1)
        tag = struct.pack("<i", len(text)) + text
        at = data.find(tag)
        while at >= 0:
            cursor = at + len(tag)
            _first, count, virtual = struct.unpack_from("<iii", data, cursor)
            if count == 0 and virtual == 1:
                return cursor + 12
            at = data.find(tag, at + 1)
    return None
