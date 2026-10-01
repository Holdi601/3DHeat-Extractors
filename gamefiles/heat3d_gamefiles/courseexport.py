"""
Turning an extracted piece of Forza terrain into the file someone looks at.

The extraction says what every triangle is and where it came from; this decides
how that is shown. Three things are drawn differently:

- **ground and road**, one pair of parts per tile, each carrying that tile's
  baked ground texture through planar UVs. Per tile because each tile has its own
  image, and a vertex belongs to exactly one tile - models do not share vertices
  - so one UV per vertex is enough.
- **markings and tyre marks**, which are geometry in the game rather than paint,
  and are kept as their own parts in a flat colour so they can be switched off.
- **water**, likewise flat.

Road and ground are separate parts even though both use the same image. The
viewer classifies and filters by part, and "show me only the racing surface" is
the first thing anyone analysing laps wants to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .forzaterrain import NAME_STEP, TILE_METRES, Extraction
from .glb import Part
from .surfaces import BY_KEY, part_name


def tile_origin(key: tuple[int, int]) -> tuple[float, float]:
    """A tile's corner with the smallest x and z, in world metres."""
    return key[0] / NAME_STEP * TILE_METRES, key[1] / NAME_STEP * TILE_METRES


def vertex_tiles(extraction: Extraction) -> np.ndarray:
    """
    The tile each vertex belongs to.

    Every triangle knows its tile, and every vertex belongs to triangles of only
    one tile, so the answer is well defined - which is asserted rather than
    assumed, because if it ever fails the UVs would quietly point into the wrong
    image.
    """
    owner = np.full(len(extraction.positions), -1, dtype=np.int32)
    tile_of = extraction.tile_of
    for column in range(3):
        owner[extraction.faces[:, column]] = tile_of
    for column in range(3):
        if np.any(owner[extraction.faces[:, column]] != tile_of):
            raise ValueError("a vertex is shared between tiles; planar UVs would be ambiguous")
    return owner


def planar_uvs(extraction: Extraction) -> np.ndarray:
    """
    Per-vertex UVs into each vertex's own tile image.

    u runs west to east and v north to south, because image row 0 is the tile's
    northern edge - found against recorded laps, see `forzatexture`.
    """
    owner = vertex_tiles(extraction)
    origins = np.array([tile_origin(k) for k in extraction.tile_keys], dtype=np.float64)
    uv = np.zeros((len(extraction.positions), 2), dtype=np.float32)
    used = owner >= 0
    x0 = origins[owner[used], 0]
    z0 = origins[owner[used], 1]
    points = extraction.positions[used].astype(np.float64)
    uv[used, 0] = (points[:, 0] - x0) / TILE_METRES
    uv[used, 1] = 1.0 - (points[:, 2] - z0) / TILE_METRES
    # Triangles clipped at a tile's edge can put a vertex a hair outside it.
    return np.clip(uv, 0.0, 1.0)


@dataclass
class Plan:
    """What will be written: the parts, and what each holds."""

    parts: list[Part] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    textured: bool = False


#: Drawn in this order, least important first, since vertex colour is per vertex
#: and a corner shared by two parts takes the colour of the later one.
FLAT = ("water", "tyremarks", "markings")


#: Above this, a road-strip triangle is asphalt; below it, verge. The strips
#: are built wider than the carriageway - 30% of their area around the test
#: circuit is shoulder - and the tile's asphalt mask is what says where the asphalt ends.
ASPHALT = 0.5


def plan_parts(extraction: Extraction, baked: dict | None) -> Plan:
    """
    Split an extraction into parts, textured where an image exists for the tile.

    Road strips are split into asphalt and verge by the tile's asphalt mask at
    each triangle's centre. A tile with no maps - not found, or textures turned
    off - keeps its strips whole as road and gets flat colours, so missing maps
    cost detail and never cost geometry.
    """
    plan = Plan()
    surfaces = extraction.surfaces
    if surfaces is None:
        plan.parts.append(Part(name="ground:terrain", faces=extraction.faces))
        plan.counts["terrain"] = len(extraction.faces)
        return plan

    surfaces = surfaces.copy()
    tile_of = extraction.tile_of if extraction.tile_of is not None else np.zeros(len(surfaces), np.int32)
    for index, key in enumerate(extraction.tile_keys):
        tile = baked.get(key) if baked else None
        if tile is None:
            continue
        strip = (tile_of == index) & (surfaces == "road")
        if strip.any():
            where = np.flatnonzero(strip)
            verge = asphalt_at(extraction, strip, tile.asphalt, key) < ASPHALT
            surfaces[where[verge]] = "verge"

    for index, key in enumerate(extraction.tile_keys):
        tile = baked.get(key) if baked else None
        image = tile.image if tile is not None else None
        in_tile = tile_of == index
        for kind in ("terrain", "verge", "road"):
            chosen = in_tile & (surfaces == kind)
            if not chosen.any():
                continue
            surface = BY_KEY[kind]
            plan.parts.append(
                Part(
                    name=f"{part_name(surface)} x{key[0]} z{key[1]}",
                    faces=extraction.faces[chosen],
                    colour=surface.colour,
                    texture=image,
                )
            )
            plan.counts[surface.label] = plan.counts.get(surface.label, 0) + int(chosen.sum())
            plan.textured = plan.textured or image is not None

    for kind in FLAT:
        chosen = surfaces == kind
        if not chosen.any():
            continue
        surface = BY_KEY[kind]
        plan.parts.append(
            Part(name=part_name(surface), faces=extraction.faces[chosen], colour=surface.colour)
        )
        plan.counts[surface.label] = int(chosen.sum())
    return plan


@dataclass
class Baked:
    """One tile's ground image, and the asphalt mask it was made from."""

    image: np.ndarray
    asphalt: np.ndarray


def bake_tiles(track, keys, *, size: int = 2048, season: str = "summer") -> dict:
    """
    The ground image for every tile in `keys`, at up to `size` pixels across.

    `size` picks the resolution: 2048 is the game's largest, 25 cm a pixel;
    1024 halves each file's side for a quarter of the bytes. The asphalt mask
    is kept at full resolution whatever the image size, since it decides
    geometry and the image only decides looks.
    """
    from PIL import Image

    from .forzatexture import TextureShelf, asphalt_weight, bake_ground

    quality = {2048: 5, 1024: 4, 512: 3, 256: 2}.get(size, 5)
    baked = {}
    with TextureShelf(track) as shelf:
        for key in keys:
            maps = shelf.tile_maps(key[0], key[1], season=season, quality=5)
            if maps is None:
                continue
            image = bake_ground(maps)
            if image.shape[0] > size:
                image = np.asarray(Image.fromarray(image).resize((size, size), Image.LANCZOS))
            baked[key] = Baked(image=image, asphalt=asphalt_weight(maps))
    return baked


def asphalt_at(extraction: Extraction, chosen: np.ndarray, mask: np.ndarray, key) -> np.ndarray:
    """The asphalt mask's value at the centroid of each chosen triangle."""
    x0, z0 = tile_origin(key)
    centres = extraction.positions[extraction.faces[chosen]].astype(np.float64).mean(axis=1)
    size = mask.shape[0]
    col = np.clip(((centres[:, 0] - x0) / TILE_METRES * size).astype(int), 0, size - 1)
    row = np.clip(((1.0 - (centres[:, 2] - z0) / TILE_METRES) * size).astype(int), 0, size - 1)
    return mask[row, col]


#: How far painted layers - lane markings, rubbered-in tyre marks - are lifted
#: off the surface they are painted on. They ship as their own triangles lying
#: exactly in the road, which the game draws with a depth bias; drawn as they
#: are, road and paint fight over every pixel and the paint comes out in
#: fragments. Five centimetres settles it at any distance someone reads a
#: marking from, and is nothing against a car's ride height.
DECAL_LIFT = 0.05
DECALS = ("markings", "tyre marks")


def lift_decals(parts, positions, normals, *, lift: float = DECAL_LIFT):
    """
    The parts, with every painted layer given vertices of its own lifted along
    the surface normal, and those new vertices' positions and normals.

    Its own vertices because the paint shares corners with the road under it,
    and lifting a shared corner would lift the road too.
    """
    kept, extra_positions, extra_normals = [], [], []
    offset = len(positions)
    for part in parts:
        kind = part.name.split(":", 1)[-1]
        if part.name.startswith("ground:") and kind in DECALS and len(part.faces):
            used, inverse = np.unique(part.faces, return_inverse=True)
            up = normals[used]
            extra_positions.append((positions[used] + up * lift).astype(np.float32))
            extra_normals.append(up.astype(np.float32))
            part = Part(
                name=part.name,
                faces=inverse.reshape(-1, 3).astype(np.int32) + offset,
                colour=part.colour,
                texture=part.texture,
            )
            offset += len(used)
        kept.append(part)
    if not extra_positions:
        return kept, np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    return kept, np.concatenate(extra_positions), np.concatenate(extra_normals)
