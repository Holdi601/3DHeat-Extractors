"""
Cutting a piece of Forza terrain out of the archive, by world coordinates.

This is the step that makes the rest of the ForzaTech reading useful. A track's
terrain is tens of thousands of meshes across four archives; nobody wants all of
it, and nobody can say which of it they want by name. What they can say is
*where* - and a lap of telemetry says exactly that, in the same world
coordinates the terrain is stored in.

How the tiles are found
-----------------------
The archive has no names inside it, but the manifest beside it does, and the
terrain's are a spatial index in disguise::

    scene\\tbheightfield\\autoterrain_x-8184_z13299_cluster012.i.modelbin

The two numbers are a grid coordinate, stepping by 1023. The world is not in
those units, though - the tile they name spans 512 metres, which is the trap in
this file. `-8184 / 1023` is tile -8, and tile -8 begins at -4096 metres. Take
the name as metres and every lookup misses by a factor of two, which is the kind
of wrong that still returns terrain: plenty of it, from the wrong side of the
map.

Nothing here has to trust that arithmetic, though, and it does not. Each mesh
carries its own world-space scale and bias, so a tile is checked against the box
after it is read, and the grid is only used to avoid reading all 62,295 of them.

The variants
------------
A tile appears up to three times - plain, `_cb` and `_ul` - and the three cover
the same ground with the same triangle count. Merging them triples the geometry
and leaves three coincident surfaces that shade badly, so one is chosen per
tile.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .forzatech import Geometry, UnsupportedForza, read_geometry, read_model

from .minizip import Contents, MiniZip, open_chunk

#: `autoterrain_x{X}_z{Z}[_variant]_cluster{N}.i.modelbin`
TILE = re.compile(
    r"autoterrain_x(-?\d+)_z(-?\d+)(?:_([a-z]+))?_cluster(\d+)\.i\.modelbin$",
    re.IGNORECASE,
)

#: What the grid coordinate in the name steps by.
NAME_STEP = 1023

#: What a tile is actually worth, in metres.
TILE_METRES = 512.0

#: Which variant of a tile to take, best first. The plain one is the terrain
#: surface; the other two are the same ground under a different shader.
VARIANTS = ("", "cb", "ul")


@dataclass(frozen=True)
class Tile:
    """One grid square of terrain, and the entries that make it up."""

    x: int
    z: int
    variant: str
    entries: list[int] = field(default_factory=list)

    @property
    def origin(self) -> tuple[float, float]:
        """The tile's corner, in world metres."""
        return (self.x / NAME_STEP * TILE_METRES, self.z / NAME_STEP * TILE_METRES)

    def touches(self, low: tuple[float, float], high: tuple[float, float]) -> bool:
        x0, z0 = self.origin
        return (
            x0 < high[0]
            and x0 + TILE_METRES > low[0]
            and z0 < high[1]
            and z0 + TILE_METRES > low[1]
        )


def index_tiles(contents: Contents) -> dict[tuple[int, int, str], Tile]:
    """Group a manifest's terrain meshes by grid square and variant."""
    tiles: dict[tuple[int, int, str], Tile] = {}
    for entry, name in enumerate(contents.names):
        found = TILE.search(name)
        if not found:
            continue
        key = (int(found.group(1)), int(found.group(2)), (found.group(3) or "").lower())
        tile = tiles.get(key)
        if tile is None:
            tile = tiles[key] = Tile(x=key[0], z=key[1], variant=key[2])
        tile.entries.append(entry)
    return tiles


def choose_variant(
    tiles: dict[tuple[int, int, str], Tile], x: int, z: int
) -> Tile | None:
    for variant in VARIANTS:
        tile = tiles.get((x, z, variant))
        if tile is not None:
            return tile
    return None




#: The UV set only road surface carries. Terrain has one UV set, or none; the
#: road strips carry three, and the second is what tells them apart.
ROAD_UV = 1


def surface_keys(geometry: Geometry) -> np.ndarray:
    """
    What each triangle is: road, markings, water or terrain.

    Road is decided by vertex layout rather than by material name, and that is
    the lesson of the first attempt. Around a circuit 94% of the ground carries a
    road-shader material - the terrain near roads is drawn with it, and whether a
    given square metre shows asphalt or grass is decided per pixel. Triangle
    materials therefore cannot say where the road is. The vertices can: road
    surface is built as strips with three UV sets, running across and along the
    carriageway, and nothing else has them.

    Checked against 46 recorded laps of one circuit: 99.2% of 16,920 samples
    land on triangles this calls road, 0.7% on markings that lie on the road, and
    0.1% on terrain - a car running wide.

    Markings and water still come from the material, because for those the
    material is specific: a decal submesh is nothing but the decal.
    """
    from .surfaces import classify

    count = len(geometry.faces)
    keys = np.full(count, "terrain", dtype=object)
    uv = geometry.uvs.get(ROAD_UV)
    if uv is not None and count:
        keys[np.isfinite(uv[geometry.faces][:, :, 0]).all(axis=1)] = "road"
    if geometry.materials:
        for index, name in enumerate(geometry.materials):
            kind = classify(name).key
            if kind in ("markings", "tyremarks", "water"):
                keys[index] = kind
    return keys


@dataclass
class Extraction:
    """What came out, and what it cost."""

    positions: np.ndarray
    faces: np.ndarray
    #: Unit normals per vertex, as the game stores them, or None.
    normals: np.ndarray | None = None
    #: One surface key per triangle, parallel to `faces`.
    surfaces: np.ndarray | None = None
    #: Which tile each triangle came from, as an index into `tile_keys`.
    tile_of: np.ndarray | None = None
    #: (x, z) of each tile, in the grid units its file names use.
    tile_keys: list[tuple[int, int]] = field(default_factory=list)
    tiles: int = 0
    meshes: int = 0
    skipped: int = 0

    def __len__(self) -> int:
        return len(self.faces)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if not len(self.positions):
            return np.zeros(3), np.zeros(3)
        return self.positions.min(axis=0), self.positions.max(axis=0)

    def summary(self) -> str:
        low, high = self.bounds
        return (
            f"{len(self.positions):,} vertices, {len(self.faces):,} triangles "
            f"from {self.meshes:,} meshes across {self.tiles} tiles\n"
            f"  x {low[0]:9.1f} .. {high[0]:9.1f}\n"
            f"  y {low[1]:9.1f} .. {high[1]:9.1f}\n"
            f"  z {low[2]:9.1f} .. {high[2]:9.1f}"
            + (f"\n  {self.skipped} meshes could not be read" if self.skipped else "")
        )


def _empty(tiles: int = 0, skipped: int = 0) -> Extraction:
    return Extraction(
        positions=np.zeros((0, 3), dtype=np.float32),
        faces=np.zeros((0, 3), dtype=np.int32),
        tiles=tiles,
        skipped=skipped,
    )


def clip(geometry: Geometry, low: tuple[float, float], high: tuple[float, float]):
    """
    Keep the triangles that reach into the box.

    Returns the kept vertices, the renumbered faces, the mask of kept triangles
    and the indices of the kept vertices - the last two so that everything else
    carried per triangle or per vertex can follow.

    A triangle is kept when its own footprint overlaps the box, which is not the
    same as having a corner inside it and is not the same as having all three.

    All three is wrong at every edge: a triangle with one corner in is part of
    the surface there, and dropping it leaves a fringe of holes right where the
    box was drawn, which is exactly where someone is looking.

    Any corner is wrong in the other direction, and less obviously. A triangle
    can cover the whole box without putting a single vertex in it — terrain
    ships with LOD meshes whose triangles are tens of metres across, and asking
    for a few metres of road inside one of them gets nothing back at all.
    """
    points = geometry.positions
    corners = points[geometry.faces]
    keep = (
        (corners[:, :, 0].max(axis=1) >= low[0])
        & (corners[:, :, 0].min(axis=1) <= high[0])
        & (corners[:, :, 2].max(axis=1) >= low[1])
        & (corners[:, :, 2].min(axis=1) <= high[1])
    )
    if keep.all():
        return points, geometry.faces, keep, np.arange(len(points))
    if not keep.any():
        return points[:0], geometry.faces[:0], keep, np.zeros(0, dtype=np.int64)
    faces = geometry.faces[keep]
    used = np.unique(faces)
    renumber = np.zeros(len(points), dtype=np.int32)
    renumber[used] = np.arange(len(used), dtype=np.int32)
    return points[used], renumber[faces], keep, used


def _narrow(points, faces, kept, used, corridor):
    """`clip`'s result cut down to the corridor, in the same four parts."""
    inside = corridor.triangles(points, faces)
    if inside.all():
        return points, faces, kept, used
    faces = faces[inside]
    local = np.unique(faces)
    renumber = np.zeros(len(points), dtype=np.int32)
    renumber[local] = np.arange(len(local), dtype=np.int32)
    narrowed = kept.copy()
    narrowed[np.flatnonzero(kept)[~inside]] = False
    return points[local], renumber[faces], narrowed, used[local]


def extract(
    archive: MiniZip,
    contents: Contents,
    low: tuple[float, float],
    high: tuple[float, float],
    *,
    tiles: dict[tuple[int, int, str], Tile] | None = None,
    corridor=None,
) -> Extraction:
    """
    Every terrain triangle inside a world-space box, as one mesh.

    With a `corridor` (see `corridor.Corridor`), only what lies in it as well:
    a tile the band does not touch is not read at all, and a triangle outside
    it is dropped.

    `low` and `high` are (x, z) in the game's metres. Height is not bounded:
    a box that clipped by altitude would cut the ground out from under a hill.

    Only the finest level of detail is kept. A tile carries four over the same
    ground, and exporting all of them put the same square metre in the file four
    times - which is what made the first road-and-verge split cover more area
    than the box it was cut from.
    """
    if high[0] <= low[0] or high[1] <= low[1]:
        raise ValueError(f"the box {low} .. {high} is empty")
    if tiles is None:
        tiles = index_tiles(contents)

    wanted: list[Tile] = []
    seen: set[tuple[int, int]] = set()
    for (x, z, _variant) in tiles:
        if (x, z) in seen:
            continue
        seen.add((x, z))
        chosen = choose_variant(tiles, x, z)
        if chosen is None or not chosen.touches(low, high):
            continue
        if corridor is not None:
            x0, z0 = x / NAME_STEP * TILE_METRES, z / NAME_STEP * TILE_METRES
            if not corridor.touches((x0, z0), (x0 + TILE_METRES, z0 + TILE_METRES)):
                continue
        wanted.append(chosen)

    points_out: list[np.ndarray] = []
    normals_out: list[np.ndarray | None] = []
    faces_out: list[np.ndarray] = []
    keys_out: list[np.ndarray] = []
    tile_out: list[np.ndarray] = []
    tile_keys: list[tuple[int, int]] = []
    meshes = skipped = 0
    for tile in sorted(wanted, key=lambda t: (t.x, t.z)):
        tile_index = len(tile_keys)
        tile_keys.append((tile.x, tile.z))
        for entry in tile.entries:
            try:
                geometry = read_geometry(
                    read_model(contents.names[entry], archive.read(entry)), lod=0
                )
            except (UnsupportedForza, ValueError):
                # One mesh in an unknown format must not cost the extraction;
                # the count is reported so a run that loses most of a region
                # says so rather than quietly returning a thin surface.
                skipped += 1
                continue
            if not len(geometry.faces):
                continue
            points, faces, kept, used = clip(geometry, low, high)
            if len(faces) and corridor is not None:
                points, faces, kept, used = _narrow(points, faces, kept, used, corridor)
            if not len(faces):
                continue
            points_out.append(points)
            normals_out.append(
                geometry.normals[used] if geometry.normals is not None else None
            )
            faces_out.append(faces)
            keys_out.append(surface_keys(geometry)[kept])
            tile_out.append(np.full(len(faces), tile_index, dtype=np.int32))
            meshes += 1

    if not faces_out:
        return _empty(tiles=len(wanted), skipped=skipped)

    offsets = np.cumsum([0] + [len(p) for p in points_out])
    normals = None
    if all(n is not None for n in normals_out):
        normals = np.concatenate(normals_out).astype(np.float32)
        # A vertex whose group carried no normal is NaN; the writer computes
        # those from the triangles rather than exporting a hole in the shading.
        if np.isnan(normals).any():
            normals = None
    return Extraction(
        positions=np.concatenate(points_out).astype(np.float32),
        faces=np.concatenate(
            [f + start for f, start in zip(faces_out, offsets[:-1])]
        ).astype(np.int32),
        normals=normals,
        surfaces=np.concatenate(keys_out),
        tile_of=np.concatenate(tile_out),
        tile_keys=tile_keys,
        tiles=len(wanted),
        meshes=meshes,
        skipped=skipped,
    )


def merge(parts: list[Extraction]) -> Extraction:
    """Join extractions from several archives into one mesh."""
    parts = [p for p in parts if len(p.faces)]
    if not parts:
        return _empty()
    if len(parts) == 1:
        return parts[0]
    offsets = np.cumsum([0] + [len(p.positions) for p in parts])
    tile_keys: list[tuple[int, int]] = []
    tile_of = []
    for part in parts:
        remap = []
        for key in part.tile_keys:
            if key not in tile_keys:
                tile_keys.append(key)
            remap.append(tile_keys.index(key))
        remap = np.asarray(remap, dtype=np.int32)
        tile_of.append(
            remap[part.tile_of] if part.tile_of is not None else np.zeros(len(part.faces), np.int32)
        )
    normals = None
    if all(p.normals is not None for p in parts):
        normals = np.concatenate([p.normals for p in parts])
    return Extraction(
        positions=np.concatenate([p.positions for p in parts]),
        faces=np.concatenate([p.faces + start for p, start in zip(parts, offsets[:-1])]),
        normals=normals,
        surfaces=np.concatenate(
            [p.surfaces if p.surfaces is not None else np.full(len(p.faces), "terrain", dtype=object) for p in parts]
        ),
        tile_of=np.concatenate(tile_of),
        tile_keys=tile_keys,
        tiles=sum(p.tiles for p in parts),
        meshes=sum(p.meshes for p in parts),
        skipped=sum(p.skipped for p in parts),
    )


def extract_from_track(
    folder: str | Path,
    low: tuple[float, float],
    high: tuple[float, float],
    *,
    chunks: tuple[int, ...] = (0, 1, 2, 3),
    corridor=None,
) -> Extraction:
    """
    The same, across a track folder's archives.

    Terrain lives in one of them in every track looked at so far, but which one
    is not fixed, and an archive with no terrain costs only its index to open.
    """
    folder = Path(folder)
    parts: list[Extraction] = []
    for number in chunks:
        if not (folder / f"GeoChunk{number}.minizip").exists():
            continue
        archive, contents = open_chunk(folder, number)
        try:
            parts.append(extract(archive, contents, low, high, corridor=corridor))
        finally:
            archive.close()
    return merge(parts)
