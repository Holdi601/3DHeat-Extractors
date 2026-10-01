"""
A BeamNG course: the ground and everything on it along one lap's driving.

The same file the Forza `course` export writes - ground per 512 m tile with a
baked texture, road and verge as their own parts, placed models by category -
built from what a BeamNG level is made of:

- **Ground** is the terrain heightfield, cut to a band either side of the
  driving. Each square is classed by its layer's ground model (`ASPHALT` is
  road, `DIRT`, `GRAVEL`, `SAND`, `MUD` are loose ground) and then by the
  road decals painted over it, which is where most of BeamNG's roads are:
  drawn on the terrain, not modelled.
- **The ground texture** is baked per tile the way the game draws it: each
  terrain layer's detail texture blended by the layer map, then every road
  decal painted over it in the game's order (descending `renderPriority`:
  asphalt bases sit at 12-16, markings at 10, rubber at 4 across the shipped
  levels) with its own texture and opacity map. Markings and tyre marks are
  paint here, not geometry, so they are in the texture rather than parts.
- **Placed models** - static objects, prefabs, forest items - come from their
  compiled shapes, classed by the level's own annotations (`GUARD_RAIL`,
  `TRAFFIC_SIGNS`, `BUILDINGS`, `POLE`, `ROCK`) and their names. A model
  whose texture repeats is drawn in that texture's average colour: the
  viewer packs textures into an atlas and cannot repeat one.
- **Water** - the sea plane, water blocks and rivers - as flat parts.

Positions go out in the viewer's frame, `(x, z, -y)` of BeamNG's Z-up world:
the rotation the telemetry recorder applies, so a recorded lap sits on the
course. It is a rotation, not a mirror, so nothing comes out mirrored.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .beamng import (
    BeamNGError,
    DecalRoad,
    Files,
    Level,
    Materials,
    Terrain,
    TerrainMaterial,
    Textures,
    list_levels,
    numbers,
    open_level,
    read_terrain,
    resolve,
    terrain_blocks,
)
from .beamngshape import Shape, UnreadableShape, read_shape
from .corridor import Corridor
from .glb import Part, compute_normals, write_glb
from .surfaces import BY_KEY, WATER, part_name

#: Metres a baked ground texture covers, as for Forza.
TILE = 512.0

#: Road decals are sampled this often along their spline.
SPLINE_STEP = 1.0

#: How far a lap's samples may sit above the terrain and still count as on
#: it: a car's reference point is its body, not its tyres.
ON_GROUND = (-1.0, 3.0)

#: Pixels of texture a placed model's image is kept at. Hundreds of them go
#: into one atlas in the viewer.
MODEL_TEXTURE = 256

#: Lifted off the ground they sit on, as Forza's paint is.
WATER_LIFT = 0.0


# --------------------------------------------------------------------------
# Frames


def to_viewer(points: np.ndarray) -> np.ndarray:
    """BeamNG's Z-up world to the viewer's Y-up one: `(x, z, -y)`."""
    points = np.asarray(points, dtype=np.float64)
    return np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=1)


def from_viewer(points: np.ndarray) -> np.ndarray:
    """The viewer's frame back to BeamNG's: `(x, -z, y)`."""
    points = np.asarray(points, dtype=np.float64)
    return np.stack([points[:, 0], -points[:, 2], points[:, 1]], axis=1)


def plane(points: np.ndarray) -> np.ndarray:
    """World points as the corridor reads them: `(x, height, y)`."""
    return np.stack([points[:, 0], points[:, 2], points[:, 1]], axis=1)


# --------------------------------------------------------------------------
# Which level


@dataclass
class Match:
    level: str
    #: Share of the lap's samples lying on that level's terrain, or roads.
    score: float
    #: `terrain` or `roads`: what the score counts.
    by: str = "terrain"


def find_level(files: Files, route: np.ndarray) -> list[Match]:
    """
    Every level, ranked by how much of a lap lies on its terrain.

    A lap is on a level when its samples sit on that level's ground: inside
    the terrain's square and a little above its height. The extent is read
    from the terrain file's first five bytes, so a level the lap is nowhere
    near costs nothing; the heights are read only for the rest.
    """
    if len(route) == 0:
        return []
    pick = np.linspace(0, len(route) - 1, min(len(route), 500)).astype(np.int64)
    sample = route[pick]
    matches = []
    for name in list_levels(files):
        best = 0.0
        for block in terrain_blocks(files, name):
            source = resolve(files, str(block.get("terrainFile") or ""))
            if not source:
                continue
            head = files.head(source, 5)
            size = int.from_bytes(head[1:5], "little") if len(head) == 5 else 0
            position = numbers(block.get("position")) or [0.0, 0.0, 0.0]
            square = (numbers(block.get("squareSize")) or [1.0])[0]
            span = (size - 1) * square
            inside = (
                (sample[:, 0] >= position[0])
                & (sample[:, 0] <= position[0] + span)
                & (sample[:, 1] >= position[1])
                & (sample[:, 1] <= position[1] + span)
            )
            if inside.mean() < 0.3:
                continue
            terrain = read_terrain(files, block)
            if terrain is None:
                continue
            above = sample[:, 2] - terrain.height(sample[:, 0], sample[:, 1])
            on = np.isfinite(above) & (above > ON_GROUND[0]) & (above < ON_GROUND[1])
            best = max(best, float(on.mean()))
        matches.append(Match(name, best))
    # Not every level is all heightmap: one shipped desert runs on past its
    # terrain as rock sheets, and its roads with it. Where no terrain
    # holds the lap, its roads - the AI's included - can.
    if not matches or max(m.score for m in matches) < 0.5:
        for match in matches:
            roads = _on_roads(files, match.level, sample)
            if roads > match.score:
                match.score, match.by = roads, "roads"
    matches.sort(key=lambda m: -m.score)
    return matches


def _on_roads(files: Files, name: str, sample: np.ndarray) -> float:
    """Share of a lap's samples lying on one of a level's road splines."""
    from .beamng import _flatten, json_objects, scene_files, torque_objects

    starts, ends = [], []
    low, high = sample[:, :2].min(axis=0) - 50, sample[:, :2].max(axis=0) + 50
    for source in scene_files(files, name):
        text = files.read(source).decode("utf-8", "replace")
        if "DecalRoad" not in text:
            continue
        objects = torque_objects(text) if source.endswith(".mis") else json_objects(text)
        for obj in _flatten(objects):
            if obj.get("class") != "DecalRoad":
                continue
            nodes = np.array([numbers(n)[:4] for n in obj.get("nodes") or [] if len(numbers(n)) >= 4])
            if len(nodes) < 2:
                continue
            if (nodes[:, :2].max(axis=0) < low).any() or (nodes[:, :2].min(axis=0) > high).any():
                continue
            starts.append(nodes[:-1])
            ends.append(nodes[1:])
    if not starts:
        return 0.0
    a, b = np.concatenate(starts), np.concatenate(ends)
    on = np.zeros(len(sample), bool)
    for first in range(0, len(sample), 64):
        p = sample[first : first + 64]
        d = b[None, :, :2] - a[None, :, :2]
        t = np.clip(
            np.einsum("pmk,pmk->pm", p[:, None, :2] - a[None, :, :2], d) / np.maximum((d**2).sum(axis=2), 1e-9),
            0.0,
            1.0,
        )
        nearest = a[None, :, :3] + t[..., None] * (b[None, :, :3] - a[None, :, :3])
        flat = np.linalg.norm(p[:, None, :2] - nearest[..., :2], axis=2)
        reach = np.maximum((a[None, :, 3] + b[None, :, 3]) / 4 + 4.0, 6.0)
        above = p[:, None, 2] - nearest[..., 2]
        on[first : first + 64] = ((flat <= reach) & (above > -2.0) & (above < 4.0)).any(axis=1)
    return float(on.mean())


# --------------------------------------------------------------------------
# What things are


_TYRE = re.compile(r"rubber|skid|tire_?track|tyre_?track|tread_?mark", re.I)
_OVERLAY = re.compile(
    r"crack|patch|repair|damage|variation|edge|erosion|leaf|litter|gutter|grass|puddle|stain|oil|trace|bank|flow|leak",
    re.I,
)
_MARKING = re.compile(r"line|marking|crossing|zebra|parking|arrow|stripe|checker|chevron", re.I)
_LOOSE = re.compile(r"dirt|gravel|mud|sand|snow|loose|dust|soil|clay|rally", re.I)
_PAVED = re.compile(r"asphalt|road|concrete|cobble|tarmac|pavement|street|paved|brick", re.I)
_GREEN = re.compile(r"grass|forest|rock|moss|leaves|cliff|stone", re.I)


def decal_kind(name: str) -> str:
    """What a road decal paints: `road`, `loose`, `markings`, `tyremarks` or `overlay`."""
    if _TYRE.search(name):
        return "tyremarks"
    if _OVERLAY.search(name):
        return "overlay"
    if _MARKING.search(name):
        return "markings"
    if _LOOSE.search(name):
        return "loose"
    if _PAVED.search(name):
        return "road"
    return "overlay"


def ground_kind(layer: str, material: TerrainMaterial | None) -> str:
    """A terrain layer as a surface: `road`, `loose` or `terrain`."""
    text = f"{material.ground if material else ''} {material.annotation if material else ''} {layer}"
    if _PAVED.search(text):
        return "road"
    if _GREEN.search(text):
        return "terrain"
    if _LOOSE.search(text):
        return "loose"
    return "terrain"


#: Part names for placed models, as the Forza export writes them.
_CATEGORY_PART = {
    "kerbs": "ground:kerbs",
    "road": "ground:road",
    "verge": "ground:verge",
    "barriers": "structure:barriers",
    "signs": "structure:signs",
    "buildings": "structure:buildings",
    "infrastructure": "structure:infrastructure",
    "bridges": "structure:bridges",
    "rocks": "structure:rocks",
    "props": "structure:props",
    "trees": "structure:trees",
}
CATEGORY_COLOUR = {
    "kerbs": (0.60, 0.20, 0.18),
    "road": (0.22, 0.22, 0.23),
    "verge": (0.45, 0.44, 0.42),
    "barriers": (0.50, 0.50, 0.50),
    "signs": (0.70, 0.70, 0.70),
    "buildings": (0.55, 0.52, 0.48),
    "infrastructure": (0.45, 0.45, 0.47),
    "bridges": (0.45, 0.45, 0.45),
    "rocks": (0.30, 0.28, 0.25),
    "props": (0.45, 0.35, 0.25),
    "trees": (0.10, 0.20, 0.05),
}
#: How each kind of modelled driving surface is named and coloured.
SURFACE_PART = {
    "road": "ground:road",
    "loose": "ground:gravel and dirt",
    "verge": "ground:verge",
    "kerbs": "ground:kerbs",
    "markings": "ground:markings",
    "tyremarks": "ground:tyre marks",
}
SURFACE_LABEL = {k: v.split(":", 1)[1] for k, v in SURFACE_PART.items()}
SURFACE_COLOUR = {
    "road": (0.22, 0.22, 0.23),
    "loose": (0.50, 0.42, 0.30),
    "verge": (0.45, 0.44, 0.42),
    "kerbs": (0.60, 0.20, 0.18),
    "markings": (0.90, 0.90, 0.88),
    "tyremarks": (0.20, 0.17, 0.16),
}
#: Paint lifted off what it is painted on, where it is drawn as geometry.
DECAL_LIFT = 0.03

DEFAULT_CATEGORIES = ("kerbs", "road", "verge", "barriers", "signs", "buildings", "infrastructure", "bridges", "rocks", "props")

_VEGETATION = re.compile(
    r"/(trees?|foliage|groundcover|vegetation|plants?|bushes|grass)/|(^|[/_])(tree|bush|shrub|fern|grass|plant|flower|clover|daisies|hedge|vine|palm|ivy|weed|reed|leaves|sapling|conifer|pine|oak|beech|aspen|birch|fir)",
    re.I,
)


def category(shape: str, annotation: str) -> str:
    """
    Which part a placed model goes into. `backdrop` is scenery painted round
    the edge of the world - one circuit's hillside is three kilometres across - which
    is never part of a course however near its bounds come.
    """
    path = shape.replace("\\", "/").lower()
    name = path.rsplit("/", 1)[-1]
    tag = annotation.upper()
    if re.search(r"backdrop|skybox|sky_?dome|horizon|distant|matte", path):
        return "backdrop"
    if re.search(r"kerb|curb|rumble", name):
        return "kerbs"
    if tag in ("ASPHALT", "STREET", "SPEED_BUMP"):
        return "road"
    if tag in ("SIDEWALK", "COBBLESTONE"):
        return "verge"
    if tag == "NATURE" or _VEGETATION.search(path):
        return "trees"
    if tag == "ROCK" or re.search(r"rock|cliff|boulder|quarry", path):
        return "rocks"
    if tag in ("GUARD_RAIL", "ROADBLOCK") or re.search(
        r"guard_?rail|barrier|armco|fence|wall|tire_?stack|tire_?wall|tyre|tire|cone|bollard|jersey|railing|catch|blocker",
        name,
    ):
        return "barriers"
    if tag == "TRAFFIC_SIGNS" or re.search(r"sign|billboard|banner|flag|arrow|marker", name):
        return "signs"
    if re.search(r"bridge|overpass|tunnel|viaduct", name):
        return "bridges"
    if tag == "POLE" or re.search(r"pole|light|lamp|electric|power|cable|pylon|antenna|mast|transformer", name):
        return "infrastructure"
    if tag == "BUILDINGS" or re.search(r"/build|bld_|house|garage|shed|hall|tower|stand|pit|tent|gazebo|roof|warehouse|factory", path):
        return "buildings"
    return "props"


# --------------------------------------------------------------------------
# Road decals


def spline(nodes: np.ndarray, looped: bool = False, step: float = SPLINE_STEP) -> np.ndarray:
    """
    A centripetal Catmull-Rom through a road's nodes, about `step` apart.

    Rows are x, y, z, width, as the nodes are. Centripetal, because the
    uniform kind loops back on itself where nodes bunch up at a hairpin.
    """
    points = np.asarray(nodes, dtype=np.float64)
    if looped and len(points) > 2:
        points = np.vstack([points, points[:1]])
    if len(points) < 2:
        return points
    if looped and len(points) > 3:
        before, after = points[-2], points[1]
    else:
        before, after = 2 * points[0] - points[1], 2 * points[-1] - points[-2]
    ext = np.vstack([before, points, after])
    out = []
    for i in range(len(points) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        length = float(np.linalg.norm(p2[:3] - p1[:3]))
        count = max(1, int(np.ceil(length / step)))
        t = np.arange(count) / count
        d01 = max(np.linalg.norm(p1[:3] - p0[:3]) ** 0.5, 1e-4)
        d12 = max(length**0.5, 1e-4)
        d23 = max(np.linalg.norm(p3[:3] - p2[:3]) ** 0.5, 1e-4)
        t0, t1 = 0.0, d01
        t2, t3 = t1 + d12, t1 + d12 + d23
        tt = (t1 + t * (t2 - t1))[:, None]
        a1 = (t1 - tt) / (t1 - t0) * p0 + (tt - t0) / (t1 - t0) * p1
        a2 = (t2 - tt) / (t2 - t1) * p1 + (tt - t1) / (t2 - t1) * p2
        a3 = (t3 - tt) / (t3 - t2) * p2 + (tt - t2) / (t3 - t2) * p3
        b1 = (t2 - tt) / (t2 - t0) * a1 + (tt - t0) / (t2 - t0) * a2
        b2 = (t3 - tt) / (t3 - t1) * a2 + (tt - t1) / (t3 - t1) * a3
        out.append((t2 - tt) / (t2 - t1) * b1 + (tt - t1) / (t2 - t1) * b2)
    out.append(points[-1:])
    return np.concatenate(out)


@dataclass
class Ribbon:
    """A road decal laid out: its two edges and the distance along it."""

    decal: DecalRoad
    kind: str
    left: np.ndarray
    right: np.ndarray
    along: np.ndarray


def ribbon(decal: DecalRoad, kind: str, terrains: list[Terrain]) -> Ribbon | None:
    """
    A decal's edges in the world. Decals are projected onto what lies under
    them, so an edge takes the terrain's height where the terrain is close to
    the road's own - and the road's where it is not, on a bridge.
    """
    samples = spline(decal.nodes, decal.looped)
    if len(samples) < 2:
        return None
    flat = samples[:, :2]
    tangent = np.gradient(flat, axis=0)
    length = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.maximum(length, 1e-9)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    half = np.maximum(samples[:, 3:4], 0.05) / 2
    left = np.c_[flat + normal * half, samples[:, 2]]
    right = np.c_[flat - normal * half, samples[:, 2]]
    for edge in (left, right):
        for terrain in terrains:
            ground = terrain.height(edge[:, 0], edge[:, 1])
            near = np.isfinite(ground) & (np.abs(ground - edge[:, 2]) < 1.5)
            edge[near, 2] = ground[near]
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(samples[:, :3], axis=0), axis=1))])
    return Ribbon(decal=decal, kind=kind, left=left, right=right, along=along)


def ribbon_triangles(r: Ribbon) -> tuple[np.ndarray, np.ndarray]:
    """The ribbon as triangles, (n, 3, 3) corners and (n, 3, 2) UVs."""
    n = len(r.left)
    v = r.along / max(r.decal.texture_length, 1e-3)
    l0, l1, r0, r1 = r.left[:-1], r.left[1:], r.right[:-1], r.right[1:]
    ul0 = np.c_[np.zeros(n - 1), v[:-1]]
    ul1 = np.c_[np.zeros(n - 1), v[1:]]
    ur0 = np.c_[np.ones(n - 1), v[:-1]]
    ur1 = np.c_[np.ones(n - 1), v[1:]]
    corners = np.concatenate([np.stack([l0, r0, r1], axis=1), np.stack([l0, r1, l1], axis=1)])
    uvs = np.concatenate([np.stack([ul0, ur0, ur1], axis=1), np.stack([ul0, ur1, ul1], axis=1)])
    return corners, uvs


# --------------------------------------------------------------------------
# Rasterising, for the classes and the baked texture


def raster(corners: np.ndarray, uvs: np.ndarray | None, shape: tuple[int, int], limit: int = 30_000_000):
    """
    Pixels inside triangles, given in pixel coordinates (column, row): their
    columns, rows, triangle and interpolated UV. Vectorised over triangles,
    each expanded to the pixels of its bounding box.
    """
    height, width = shape
    if not len(corners):
        empty = np.zeros(0, np.int64)
        return empty, empty, empty, np.zeros((0, 2))
    low = np.floor(corners.min(axis=1)).astype(np.int64)
    high = np.ceil(corners.max(axis=1)).astype(np.int64)
    low = np.maximum(low, 0)
    high = np.minimum(high, [width - 1, height - 1])
    w = high[:, 0] - low[:, 0] + 1
    h = high[:, 1] - low[:, 1] + 1
    ok = (w > 0) & (h > 0)
    out_c, out_r, out_t, out_uv = [], [], [], []
    index = np.flatnonzero(ok)
    counts = (w * h)[index]
    start = 0
    while start < len(index):
        # Chunks bounded in pixels, so one long road cannot exhaust memory.
        total = np.cumsum(counts[start:])
        stop = start + max(1, int(np.searchsorted(total, limit)))
        chosen = index[start:stop]
        n = (w * h)[chosen]
        tri = np.repeat(chosen, n)
        local = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
        col = low[tri, 0] + local % w[tri]
        row = low[tri, 1] + local // w[tri]
        a, b, c = corners[tri, 0], corners[tri, 1], corners[tri, 2]
        px = np.stack([col, row], axis=1).astype(np.float64)
        v0, v1, v2 = b - a, c - a, px - a
        d00 = np.einsum("ij,ij->i", v0, v0)
        d01 = np.einsum("ij,ij->i", v0, v1)
        d11 = np.einsum("ij,ij->i", v1, v1)
        d20 = np.einsum("ij,ij->i", v2, v0)
        d21 = np.einsum("ij,ij->i", v2, v1)
        denom = d00 * d11 - d01 * d01
        good = np.abs(denom) > 1e-12
        denom = np.where(good, denom, 1.0)
        bv = (d11 * d20 - d01 * d21) / denom
        bw = (d00 * d21 - d01 * d20) / denom
        bu = 1.0 - bv - bw
        inside = good & (bu >= -1e-6) & (bv >= -1e-6) & (bw >= -1e-6)
        out_c.append(col[inside])
        out_r.append(row[inside])
        out_t.append(tri[inside])
        if uvs is not None:
            t_in = tri[inside]
            uv = (
                bu[inside, None] * uvs[t_in, 0]
                + bv[inside, None] * uvs[t_in, 1]
                + bw[inside, None] * uvs[t_in, 2]
            )
            out_uv.append(uv)
        start = stop
    if not out_c:
        empty = np.zeros(0, np.int64)
        return empty, empty, empty, np.zeros((0, 2))
    return (
        np.concatenate(out_c),
        np.concatenate(out_r),
        np.concatenate(out_t),
        np.concatenate(out_uv) if out_uv else np.zeros((0, 2)),
    )


# --------------------------------------------------------------------------
# The ground


@dataclass
class Ground:
    """Terrain triangles in the band, in BeamNG's frame."""

    positions: np.ndarray
    normals: np.ndarray
    faces: np.ndarray
    #: Per face: `road`, `loose`, `verge` or `terrain`.
    kinds: np.ndarray
    #: Per face, into `tiles`.
    tile_of: np.ndarray
    tiles: list[tuple[int, int]]
    #: Per vertex, into its tile's image.
    uvs: np.ndarray

    def summary(self) -> str:
        names, counts = np.unique(self.kinds, return_counts=True)
        total = max(1, int(counts.sum()))
        lines = [f"  {int(c):9,} triangles  {c / total:5.1%}  {BY_KEY[str(n)].label}" for n, c in sorted(zip(names, counts), key=lambda p: -p[1])]
        return "\n".join(lines)


def _overrides(terrain: Terrain, ribbons: list[Ribbon]) -> dict[int, str]:
    """
    Terrain squares a road or loose-ground decal covers, and which, by the
    order the game draws them in: the last one painted decides.
    """
    marked: dict[int, str] = {}
    size = terrain.size
    for r in ribbons:
        if r.kind not in ("road", "loose"):
            continue
        corners, _ = ribbon_triangles(r)
        grid = np.stack(
            [
                (corners[..., 0] - terrain.position[0]) / terrain.square - 0.5,
                (corners[..., 1] - terrain.position[1]) / terrain.square - 0.5,
            ],
            axis=2,
        )
        col, row, _, _ = raster(grid, None, (size - 1, size - 1))
        for key in np.unique(row * size + col).tolist():
            marked[key] = r.kind
    return marked


def build_ground(level: Level, materials: Materials, corridor: Corridor, ribbons: list[Ribbon]) -> Ground | None:
    """The terrain inside the band, classed, one vertex set per tile."""
    positions, normals, faces, kinds, tiles_of, uvs = [], [], [], [], [], []
    tile_index: dict[tuple[int, int], int] = {}
    offset = 0
    for terrain in level.terrains:
        size, square = terrain.size, terrain.square
        px, py, pz = terrain.position
        scale = terrain.max_height / 65535.0
        low, high = corridor.low, corridor.high
        i0 = max(0, int(np.floor((low[0] - px) / square)))
        i1 = min(size - 2, int(np.ceil((high[0] - px) / square)))
        j0 = max(0, int(np.floor((low[1] - py) / square)))
        j1 = min(size - 2, int(np.ceil((high[1] - py) / square)))
        if i0 > i1 or j0 > j1:
            continue
        layer_kind = []
        for index, name in enumerate(terrain.names):
            layer_kind.append(ground_kind(name, materials.terrain.get(name.lower())))
        override = _overrides(terrain, ribbons)
        for start in range(j0, j1 + 1, 256):
            stop = min(j1, start + 255)
            jj, ii = np.mgrid[start : stop + 1, i0 : i1 + 1]
            ii, jj = ii.ravel(), jj.ravel()
            cx = px + (ii + 0.5) * square
            cy = py + (jj + 0.5) * square
            keep = corridor.contains(np.stack([cx, cy], axis=1))
            keep &= terrain.layers[jj, ii] != 255
            if not keep.any():
                continue
            ii, jj, cx, cy = ii[keep], jj[keep], cx[keep], cy[keep]
            tx = np.floor(cx / TILE).astype(np.int64)
            ty = np.floor(cy / TILE).astype(np.int64)
            tile_ids = np.empty(len(ii), np.int64)
            for k, key in enumerate(zip(tx.tolist(), ty.tolist())):
                if key not in tile_index:
                    tile_index[key] = len(tile_index)
                tile_ids[k] = tile_index[key]
            v00 = jj * size + ii
            corners = np.stack([v00, v00 + 1, v00 + size, v00 + size + 1], axis=1)
            keys = tile_ids[:, None] * (size * size) + corners
            unique, inverse = np.unique(keys.ravel(), return_inverse=True)
            inverse = inverse.reshape(-1, 4)
            vid = unique % (size * size)
            vtile = unique // (size * size)
            vi, vj = vid % size, vid // size
            x = px + vi * square
            y = py + vj * square
            z = pz + terrain.heights[vj, vi].astype(np.float64) * scale
            positions.append(np.stack([x, y, z], axis=1))
            hx = (terrain.heights[vj, np.minimum(vi + 1, size - 1)] - terrain.heights[vj, np.maximum(vi - 1, 0)]) * scale
            hy = (terrain.heights[np.minimum(vj + 1, size - 1), vi] - terrain.heights[np.maximum(vj - 1, 0), vi]) * scale
            n = np.stack([-hx / (2 * square), -hy / (2 * square), np.ones(len(vi))], axis=1)
            normals.append(n / np.linalg.norm(n, axis=1, keepdims=True))
            keys_list = list(tile_index.items())
            origin = np.zeros((len(tile_index), 2))
            for key, number in keys_list:
                origin[number] = (key[0] * TILE, key[1] * TILE)
            u = (x - origin[vtile, 0]) / TILE
            v = 1.0 - (y - origin[vtile, 1]) / TILE
            uvs.append(np.stack([u, v], axis=1))
            a, b, c, d = (inverse[:, k] + offset for k in range(4))
            # Counter-clockwise seen from above, the diagonal alternating as
            # Torque's terrain does.
            flip = ((ii + jj) & 1) == 1
            first = np.where(flip[:, None], np.stack([a, b, d], axis=1), np.stack([a, b, c], axis=1))
            second = np.where(flip[:, None], np.stack([a, d, c], axis=1), np.stack([b, d, c], axis=1))
            faces.append(np.concatenate([first, second]))
            layer = terrain.layers[jj, ii]
            kind = np.array([layer_kind[k] if k < len(layer_kind) else "terrain" for k in layer.tolist()], dtype=object)
            if override:
                square_key = (jj * size + ii).tolist()
                for k, key in enumerate(square_key):
                    found = override.get(key)
                    if found is not None:
                        kind[k] = found
            kinds.append(np.concatenate([kind, kind]))
            tiles_of.append(np.concatenate([tile_ids, tile_ids]))
            offset += len(unique)
    if not faces:
        return None
    return Ground(
        positions=np.concatenate(positions),
        normals=np.concatenate(normals),
        faces=np.concatenate(faces),
        kinds=np.concatenate(kinds).astype(str),
        tile_of=np.concatenate(tiles_of),
        tiles=[key for key, _ in sorted(tile_index.items(), key=lambda kv: kv[1])],
        uvs=np.concatenate(uvs),
    )


# --------------------------------------------------------------------------
# Baking the ground texture


@dataclass
class Painter:
    """Images a bake draws with, each shrunk to the pixel size of the bake."""

    materials: Materials
    textures: Textures
    _cache: dict = field(default_factory=dict)

    def tiled(self, path: str | None, metres: float, pixel: float) -> np.ndarray | None:
        """A repeating texture at `metres` a repeat, one texel a bake pixel."""
        if not path:
            return None
        texels = max(4, int(round(metres / pixel)))
        key = ("tiled", path.lower(), texels)
        if key not in self._cache:
            image = self.textures.rgba(path, 2048)
            if image is not None:
                from PIL import Image

                image = np.asarray(
                    Image.fromarray(image).resize((texels, texels), Image.BOX), dtype=np.float32
                ) / 255.0
            self._cache[key] = image
        return self._cache[key]

    def stretch(self, path: str | None) -> np.ndarray | None:
        """A colour map laid over a whole terrain, as floats."""
        if not path:
            return None
        key = ("stretch", path.lower())
        if key not in self._cache:
            image = self.textures.rgba(path, 4096)
            self._cache[key] = None if image is None else image.astype(np.float32) / 255.0
        return self._cache[key]

    def surface(self, material, texels: float):
        """
        A model's repeating texture with one texel a bake pixel - `texels`
        across one unit of its UVs - and its alpha, or None.
        """
        path = self.materials.texture(material, "colour")
        mask_path = self.materials.texture(material, "opacity")
        if not path and not mask_path:
            return None
        texels = int(min(2048, max(4, round(texels))))
        key = ("surface", material.name.lower(), texels)
        if key in self._cache:
            return self._cache[key]
        from PIL import Image

        image = self.textures.rgba(path, 2048) if path else None
        if image is None and mask_path:
            # A mask and a tint and no picture: rubber laid down as a shade.
            mask = self.textures.rgba(mask_path, 2048)
            if mask is not None:
                image = np.full(mask.shape, 255, np.uint8)
        found = None
        if image is not None:
            h, w = image.shape[:2]
            scale = texels / max(h, w)
            shape = (max(1, round(w * scale)), max(1, round(h * scale)))
            rgba = np.asarray(Image.fromarray(image).resize(shape, Image.BOX), dtype=np.float32) / 255.0
            alpha = rgba[..., 3]
            mask = self.textures.rgba(mask_path, 2048) if mask_path else None
            if mask is not None:
                alpha = alpha * np.asarray(Image.fromarray(mask).resize(shape, Image.BOX), dtype=np.float32)[..., 0] / 255.0
            alpha = np.clip(alpha * min(material.opacity, 1.0) * material.colour[3], 0.0, 1.0)
            found = (rgba[..., :3] * np.array(material.colour[:3], np.float32), alpha)
        self._cache[key] = found
        return found

    def decal(self, material_name: str, width: float, length: float, pixel: float):
        """A decal's colour and alpha at bake resolution, or None if it draws nothing."""
        across = max(2, int(round(width / pixel)))
        along = max(2, int(round(length / pixel)))
        key = ("decal", material_name.lower(), across, along)
        if key in self._cache:
            return self._cache[key]
        from PIL import Image

        material = self.materials.get(material_name)
        found = None
        if material is not None:
            colour_path = self.materials.texture(material, "colour")
            opacity_path = self.materials.texture(material, "opacity")
            image = self.textures.rgba(colour_path, 2048) if colour_path else None
            mask = self.textures.rgba(opacity_path, 2048) if opacity_path else None
            if image is not None or mask is not None:
                if image is not None:
                    rgba = np.asarray(Image.fromarray(image).resize((across, along), Image.BOX), dtype=np.float32) / 255.0
                    rgb = rgba[..., :3]
                    alpha = rgba[..., 3]
                else:
                    rgb = np.ones((along, across, 3), np.float32)
                    alpha = np.ones((along, across), np.float32)
                if mask is not None:
                    m = np.asarray(Image.fromarray(mask).resize((across, along), Image.BOX), dtype=np.float32) / 255.0
                    alpha = alpha * m[..., 0]
                tint = np.array(material.colour[:3], np.float32)
                rgb = rgb * tint
                alpha = np.clip(alpha * min(material.opacity, 1.0) * material.colour[3], 0.0, 1.0)
                if alpha.max() > 0.01:
                    found = (rgb, alpha)
        self._cache[key] = found
        return found


def _repeat(image: np.ndarray, metres: float, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """A texture repeating every `metres`, at the pixels `xs` by `ys`, row 0 north."""
    h, w = image.shape[:2]
    ci = np.floor(xs / metres * w).astype(np.int64) % w
    ri = np.floor(-ys / metres * h).astype(np.int64) % h
    return image[ri[:, None], ci[None, :], :3]


def _stretched(image: np.ndarray, origin: tuple[float, float], metres: float, xs, ys) -> np.ndarray:
    """A texture laid once over `metres` from `origin`, bilinear, row 0 north."""
    h, w = image.shape[:2]
    u = ((xs - origin[0]) / metres) % 1.0 * w - 0.5
    v = (1.0 - ((ys - origin[1]) / metres) % 1.0) * h - 0.5
    c0 = np.floor(u).astype(np.int64)
    r0 = np.floor(v).astype(np.int64)
    fu = (u - c0).astype(np.float32)[None, :, None]
    fv = (v - r0).astype(np.float32)[:, None, None]
    c1, r1 = (c0 + 1) % w, (r0 + 1) % h
    c0, r0 = c0 % w, r0 % h
    top = image[r0[:, None], c0[None, :], :3] * (1 - fu) + image[r0[:, None], c1[None, :], :3] * fu
    bottom = image[r1[:, None], c0[None, :], :3] * (1 - fu) + image[r1[:, None], c1[None, :], :3] * fu
    return top * (1 - fv) + bottom * fv


def _layer_colour(painter: "Painter", terrain: Terrain, name: str, material, xs, ys, pixel: float) -> np.ndarray:
    """
    A terrain layer as the game colours it: the base map, then the detail and
    macro overlays, each `base * lerp(1, 2 * overlay, strength)` - grey 0.5 is
    neutral, which is what those maps average.
    """
    size = (len(ys), len(xs))
    if material is None:
        shade = np.array(BY_KEY[ground_kind(name, None)].colour, np.float32) ** (1 / 2.2)
        return np.broadcast_to(shade, (*size, 3)).copy()
    extent = terrain.size * terrain.square
    base = painter.stretch(painter.materials.texture(material, "base"))
    if base is not None:
        colour = _stretched(base, terrain.position[:2], material.base_size or extent, xs, ys)
    else:
        shade = np.array(BY_KEY[ground_kind(name, material)].colour, np.float32) ** (1 / 2.2)
        colour = np.broadcast_to(shade, (*size, 3)).copy()
    for which, metres, strength in (
        ("detail", material.detail_size, material.detail_strength),
        ("macro", material.macro_size, material.macro_strength),
    ):
        overlay = painter.tiled(painter.materials.texture(material, which), metres, pixel)
        if overlay is None or strength <= 0:
            continue
        sample = _repeat(overlay, metres, xs, ys)
        if base is None:
            # No base map to modulate: the overlay is the colour.
            colour = sample.copy()
            continue
        colour = colour * (1.0 + strength * (2.0 * sample - 1.0))
    return np.clip(colour, 0.0, 1.0)


def bake_tile(
    key: tuple[int, int],
    size: int,
    level: Level,
    painter: Painter,
    ribbons: list[Ribbon],
    surfaces: list = (),
) -> np.ndarray:
    """One tile's ground texture, 8-bit sRGB, row 0 its northern edge."""
    pixel = TILE / size
    x0, y0 = key[0] * TILE, key[1] * TILE
    xs = x0 + (np.arange(size) + 0.5) * pixel
    ys = y0 + TILE - (np.arange(size) + 0.5) * pixel
    image = np.full((size, size, 3), 0.35, np.float32)
    for terrain in level.terrains:
        gx = (xs - terrain.position[0]) / terrain.square
        gy = (ys - terrain.position[1]) / terrain.square
        cols = (gx >= 0) & (gx <= terrain.size - 1)
        rows = (gy >= 0) & (gy <= terrain.size - 1)
        if not cols.any() or not rows.any():
            continue
        gxc = np.clip(gx, 0, terrain.size - 1.000001)
        gyc = np.clip(gy, 0, terrain.size - 1.000001)
        i0 = np.floor(gxc).astype(np.int64)
        j0 = np.floor(gyc).astype(np.int64)
        fx = (gxc - i0).astype(np.float32)
        fy = (gyc - j0).astype(np.float32)
        layers = terrain.layers
        corner = [
            (layers[np.ix_(j0, i0)], (1 - fy)[:, None] * (1 - fx)[None, :]),
            (layers[np.ix_(j0, i0 + 1)], (1 - fy)[:, None] * fx[None, :]),
            (layers[np.ix_(j0 + 1, i0)], fy[:, None] * (1 - fx)[None, :]),
            (layers[np.ix_(j0 + 1, i0 + 1)], fy[:, None] * fx[None, :]),
        ]
        present = np.unique(np.concatenate([c[0].ravel() for c in corner]))
        colour = np.zeros((size, size, 3), np.float32)
        weight_sum = np.zeros((size, size), np.float32)
        for layer in present.tolist():
            if layer == 255 or layer >= len(terrain.names):
                continue
            weight = sum(w * (ids == layer) for ids, w in corner).astype(np.float32)
            material = painter.materials.terrain.get(terrain.names[layer].lower())
            colour += _layer_colour(painter, terrain, terrain.names[layer], material, xs, ys, pixel) * weight[..., None]
            weight_sum += weight
        inside = (rows[:, None] & cols[None, :]) & (weight_sum > 0)
        image[inside] = colour[inside] / weight_sum[inside, None]
    # Then what is painted over it, in the game's order: road decals on the
    # terrain, driving surface modelled as meshes, what is painted on those
    # (lines, rubber), and last the decals that project over models too.
    def decals(over_objects: bool) -> None:
        for r in ribbons:
            if r.decal.over_objects != over_objects:
                continue
            found = painter.decal(r.decal.material, float(np.median(r.decal.nodes[:, 3])), r.decal.texture_length, pixel)
            if found is None:
                continue
            rgb, alpha = found
            corners, uvs = ribbon_triangles(r)
            grid = np.stack([(corners[..., 0] - x0) / pixel - 0.5, (y0 + TILE - corners[..., 1]) / pixel - 0.5], axis=2)
            lo = grid.min(axis=(0, 1))
            hi = grid.max(axis=(0, 1))
            if hi[0] < 0 or hi[1] < 0 or lo[0] > size - 1 or lo[1] > size - 1:
                continue
            col, row, _, uv = raster(grid, uvs, (size, size))
            if not len(col):
                continue
            th, tw = alpha.shape
            ci = np.clip((uv[:, 0] * tw).astype(np.int64), 0, tw - 1)
            ri = (np.floor(uv[:, 1] * th).astype(np.int64)) % th
            a = alpha[ri, ci][:, None]
            image[row, col] = image[row, col] * (1 - a) + rgb[ri, ci] * a

    decals(False)
    for piece in sorted(surfaces, key=lambda s: s.overlay):
        _paint_surface(image, piece, painter, x0, y0, pixel, size)
    decals(True)
    return (np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8)


def _cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The z of the cross product of 2D vectors: twice the signed area."""
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _uv_scale(piece: "SurfacePiece") -> float:
    """Metres of ground one unit of a model's UVs covers, from its own triangles."""
    corners = piece.positions[piece.faces]
    uv = piece.uvs[piece.faces]
    world = np.abs(_cross2(corners[:, 1, :2] - corners[:, 0, :2], corners[:, 2, :2] - corners[:, 0, :2])).sum()
    flat = np.abs(_cross2(uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0])).sum()
    return float(np.sqrt(world / flat)) if flat > 1e-12 else 1.0


def _paint_surface(image: np.ndarray, piece: "SurfacePiece", painter: Painter, x0: float, y0: float, pixel: float, size: int) -> None:
    """A model's driving surface seen from above, in its own repeating texture."""
    corners = piece.positions[piece.faces][..., :2]
    grid = np.stack([(corners[..., 0] - x0) / pixel - 0.5, (y0 + TILE - corners[..., 1]) / pixel - 0.5], axis=2)
    lo = grid.min(axis=(0, 1))
    hi = grid.max(axis=(0, 1))
    if hi[0] < 0 or hi[1] < 0 or lo[0] > size - 1 or lo[1] > size - 1:
        return
    material = painter.materials.get(piece.material)
    tint = np.array(material.colour[:3], np.float32) if material else np.ones(3, np.float32)
    look = None
    if piece.uvs is not None and material is not None:
        look = painter.surface(material, _uv_scale(piece) / pixel)
    if look is not None:
        rgb, alpha = look
        col, row, _, uv = raster(grid, piece.uvs[piece.faces], (size, size))
        if not len(col):
            return
        th, tw = alpha.shape
        ci = (np.floor(uv[:, 0] * tw).astype(np.int64)) % tw
        ri = (np.floor(uv[:, 1] * th).astype(np.int64)) % th
        colour = rgb[ri, ci]
        a = alpha[ri, ci][:, None] if piece.overlay else np.ones((len(col), 1), np.float32)
    elif piece.colours is not None:
        col, row, _, colour = raster(grid, piece.colours[piece.faces], (size, size))
        if not len(col):
            return
        colour = colour.astype(np.float32) * tint
        # Paint with neither a picture nor a mask cannot say where it is thin.
        a = np.full((len(col), 1), 0.5 if piece.overlay else 1.0, np.float32)
    else:
        col, row, _, _ = raster(grid, None, (size, size))
        if not len(col):
            return
        mean = painter.textures.mean(painter.materials.texture(material)) if material else None
        flat = np.array(mean, np.float32) * tint if mean else np.array(SURFACE_COLOUR[piece.kind], np.float32)
        colour = np.broadcast_to(flat, (len(col), 3))
        a = np.full((len(col), 1), 0.5 if piece.overlay else 1.0, np.float32)
    image[row, col] = image[row, col] * (1 - a) + colour * a


# --------------------------------------------------------------------------
# Placed models


#: Ground models whose surface the game treats as road, by `groundType`.
_ROAD_GROUND = re.compile(r"^(asphalt\w*|concrete\w*|cobblestone|tarmac|street|paved\w*|brick\w*)$", re.I)
_LOOSE_GROUND = re.compile(r"^(dirt\w*|gravel\w*|sand\w*|mud\w*|snow\w*|soil\w*|clay\w*)$", re.I)

#: Shape categories whose up-facing surfaces may be driving surface. Not
#: buildings, barriers, signs: the top of a concrete wall is not a road.
_SURFACE_CATEGORIES = ("road", "kerbs", "verge", "props", "bridges")

#: Surface kinds painted over what is under them rather than being ground.
OVERLAYS = ("markings", "tyremarks")


def surface_kind(material_name: str, material, shape_category: str) -> str | None:
    """
    Whether a model's triangles of one material are driving surface, and
    which: `road`, `loose`, `kerbs`, `verge`, or the overlays `markings` and
    `tyremarks`. BeamNG's circuits are models as often as they are terrain -
    a raceway's track is one mesh of `asphalt_light` with its lines, rubber
    and rumble strips as meshes of their own on top - and the material's
    ground type says so where its name does not.
    """
    if shape_category not in _SURFACE_CATEGORIES:
        return None
    name = material_name or ""
    ground = (material.ground if material else "") or ""
    annotation = (material.annotation if material else "") or ""
    text = f"{name} {ground} {annotation}"
    if re.search(r"skid|rubber|tire_?mark|tyre_?mark|tread", text, re.I):
        return "tyremarks"
    if re.search(r"line|marking|zebra|arrow|stripe|crossing", name, re.I):
        return "markings"
    if re.search(r"rumble|kerb|curb", text, re.I):
        return "kerbs"
    if _ROAD_GROUND.match(ground) or re.search(r"asphalt|tarmac|track_edge|road_?surface", name, re.I):
        return "road"
    if _LOOSE_GROUND.match(ground):
        return "loose"
    if re.search(r"sidewalk", text, re.I):
        return "verge"
    if shape_category in ("road", "kerbs", "verge"):
        return shape_category
    return None


@dataclass
class SurfacePiece:
    """Driving surface from a model, in the world, to be baked and drawn as ground."""

    kind: str
    material: str
    positions: np.ndarray
    faces: np.ndarray
    #: The model's own UVs, repeating as the game repeats them.
    uvs: np.ndarray | None
    colours: np.ndarray | None

    @property
    def overlay(self) -> bool:
        return self.kind in OVERLAYS


@dataclass
class Placement:
    """Placed models gathered into parts, and the driving surface among them."""

    parts: dict[str, dict] = field(default_factory=dict)
    surfaces: list[SurfacePiece] = field(default_factory=list)
    copies: dict[str, int] = field(default_factory=dict)
    missing: dict[str, int] = field(default_factory=dict)
    #: Copies left out to fit the triangle budget.
    dropped: int = 0
    budget: int = 0

    def triangles(self) -> int:
        return sum(sum(len(f) for f in part["faces"]) for part in self.parts.values())

    def surface_triangles(self) -> int:
        return sum(len(s.faces) for s in self.surfaces)


def _invisible(material_name: str, material, materials: Materials) -> bool:
    """Collision walls and the like: a material the game never draws."""
    if re.search(r"invisible|collision|nodraw", material_name or "", re.I):
        return True
    path = materials.texture(material) if material else None
    return bool(path and re.search(r"invisible", path.rsplit("/", 1)[-1], re.I))


def _subset(positions: np.ndarray, faces: np.ndarray, *extra):
    """The vertices `faces` use, renumbered, and the same rows of `extra`."""
    used, inverse = np.unique(faces, return_inverse=True)
    out = [positions[used], inverse.reshape(-1, 3)]
    for array in extra:
        out.append(None if array is None else array[used])
    return out


def place(
    level: Level,
    materials: Materials,
    textures: Textures,
    corridor: Corridor,
    reach: Corridor,
    route: np.ndarray,
    *,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    budget: int | None = None,
    use_textures: bool = True,
    structures: bool = True,
) -> Placement:
    """
    Every placed model whose bounds reach into the band, by category, with its
    driving surface taken out as ground. `structures=False` keeps only that
    surface - a circuit modelled as one mesh is still its track.
    """
    files = level.files
    shapes: dict[str, object | None] = {}
    out = Placement()
    positions = np.array([p.matrix[3, :3] for p in level.placed]) if level.placed else np.zeros((0, 3))
    near = reach.contains(plane(positions)) if len(positions) else np.zeros(0, bool)
    wanted = set(categories) | set(_SURFACE_CATEGORIES)
    chosen = []
    for index in np.flatnonzero(near):
        placed = level.placed[index]
        kind = category(placed.shape, placed.annotation)
        if kind not in wanted:
            continue
        path = resolve(files, placed.shape)
        if path is None:
            out.missing[placed.shape] = out.missing.get(placed.shape, 0) + 1
            continue
        if path not in shapes:
            shape = None
            if path.lower().endswith(".cdae"):
                try:
                    shape = read_shape(files.read(path))
                except (UnreadableShape, ValueError, KeyError):
                    shape = None
            else:
                from .beamngcollada import read_collada

                try:
                    shape = read_collada(files.read(path))
                except Exception:  # noqa: BLE001 - a shape that will not read is reported, not fatal
                    shape = None
            shapes[path] = shape
        shape = shapes[path]
        if shape is None:
            out.missing[placed.shape] = out.missing.get(placed.shape, 0) + 1
            continue
        # The copy's footprint in the world, from the shape's bounds.
        low, high = np.array(shape.bounds[0]), np.array(shape.bounds[1])
        box = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
        world = np.c_[box, np.ones(8)] @ placed.matrix[:, :3]
        if not corridor.touches(world[:, :2].min(axis=0), world[:, :2].max(axis=0)):
            continue
        chosen.append((placed, kind, path))

    # Levels of detail, as the Forza export picks them: copies near the
    # driving start at the full model, the next band one level coarser, the far
    # band at the coarsest; the models costing most step down until the total
    # fits the budget, and past that the furthest copies go. Kerbs and models
    # carrying driving surface stay whole - a coarse track is a wrong track.
    level_of = _choose_levels(chosen, shapes, materials, route, budget, out)

    groups: dict[tuple[str, str], dict] = {}
    surfaces: dict[tuple[str, str], dict] = {}
    piece_cache: dict[tuple[str, int], list] = {}
    for number, (placed, kind, path) in enumerate(chosen):
        shape = shapes[path]
        detail = level_of.get(number)
        if detail is None:
            continue
        cache_key = (path, detail)
        if cache_key not in piece_cache:
            piece_cache[cache_key] = shape.pieces(detail)
        matrix = placed.matrix
        linear = matrix[:3, :3]
        try:
            normal_matrix = np.linalg.inv(linear).T
        except np.linalg.LinAlgError:
            continue
        mirrored = np.linalg.det(linear) < 0
        drew = False
        for piece in piece_cache[cache_key]:
            material = materials.get(piece.material)
            if _invisible(piece.material, material, materials):
                continue
            p = np.c_[piece.positions, np.ones(len(piece.positions))] @ matrix[:, :3]
            n = piece.normals @ normal_matrix
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            faces = piece.faces[:, [0, 2, 1]] if mirrored else piece.faces
            surface = surface_kind(piece.material, material, kind)
            if surface is not None:
                edge = np.cross(p[faces[:, 1]] - p[faces[:, 0]], p[faces[:, 2]] - p[faces[:, 0]])
                length = np.maximum(np.linalg.norm(edge, axis=1), 1e-12)
                up = edge[:, 2] / length > 0.5
                # Ground is cut to the band like the terrain is: a circuit's
                # track is one model a kilometre across.
                inside = up & corridor.triangles(plane(p), faces)
                if inside.any():
                    sp, sf, suv, scol = _subset(p, faces[inside], piece.uvs, piece.colours)
                    entry = surfaces.setdefault(
                        (surface, piece.material), {"positions": [], "faces": [], "uvs": [], "colours": [], "count": 0}
                    )
                    entry["positions"].append(sp)
                    entry["faces"].append(sf + entry["count"])
                    entry["uvs"].append(suv)
                    entry["colours"].append(scol)
                    entry["count"] += len(sp)
                faces = faces[~up]
                if surface in OVERLAYS or not len(faces):
                    continue
            if not structures or kind not in categories:
                continue
            drew = True
            group = groups.setdefault(
                (kind, piece.material),
                {"positions": [], "normals": [], "uvs": [], "faces": [], "count": 0, "textured": True},
            )
            sp, sf, sn, suv = _subset(p, faces, n, piece.uvs)
            group["faces"].append(sf + group["count"])
            group["positions"].append(sp)
            group["normals"].append(sn)
            if suv is None:
                group["textured"] = False
                group["uvs"].append(np.zeros((len(sp), 2)))
            else:
                group["uvs"].append(suv)
            group["count"] += len(sp)
        if drew:
            out.copies[kind] = out.copies.get(kind, 0) + 1

    for (kind, material_name), entry in surfaces.items():
        out.surfaces.append(
            SurfacePiece(
                kind=kind,
                material=material_name,
                positions=np.concatenate(entry["positions"]),
                faces=np.concatenate(entry["faces"]),
                uvs=None if any(u is None for u in entry["uvs"]) else np.concatenate(entry["uvs"]),
                colours=None if any(c is None for c in entry["colours"]) else np.concatenate(entry["colours"]),
            )
        )

    for (kind, material_name), group in groups.items():
        material = materials.get(material_name)
        path = materials.texture(material) if material else None
        uv = np.concatenate(group["uvs"])
        fits = group["textured"] and len(uv) and float(np.mean((uv >= -0.02).all(axis=1) & (uv <= 1.02).all(axis=1))) > 0.98
        texture = textures.rgba(path, MODEL_TEXTURE) if (use_textures and path and fits) else None
        colour = None
        if texture is None:
            mean = textures.mean(path) if path else None
            tint = np.array(material.colour[:3]) if material else np.ones(3)
            colour = tuple(float(c) for c in (np.array(mean) * tint if mean else np.array(CATEGORY_COLOUR[kind])))
        name = f"{_CATEGORY_PART[kind]} {material_name or 'untextured'}"
        out.parts[name] = {
            "positions": np.concatenate(group["positions"]),
            "normals": np.concatenate(group["normals"]),
            "uvs": np.clip(uv, 0.0, 1.0) if texture is not None else None,
            "faces": group["faces"],
            "texture": None if texture is None else texture[..., :3],
            "colour": colour,
            "kind": kind,
        }
    return out


def _drawn_levels(shape) -> list[int]:
    """A shape's levels of detail that draw anything, finest first."""
    return [d for d in shape.levels() if shape.triangle_count(d) > 0]


def _choose_levels(chosen, shapes, materials: Materials, route: np.ndarray, budget: int | None, out) -> dict[int, int]:
    """The detail level each chosen copy is drawn at, by its index; absent if left out."""
    from .forzaplacement import FAR, NEAR, auto_budget, choose_levels, route_distance

    if not chosen:
        return {}
    where = np.array([placed.matrix[3, :2] for placed, _, _ in chosen])
    distance = route_distance(where, route[:, :2])
    band = np.digitize(distance, (NEAR, FAR))
    levels = {path: _drawn_levels(shapes[path]) for _, _, path in chosen}
    surface_shapes = set()
    for path, drawn in levels.items():
        if drawn and any(
            surface_kind(piece.material, materials.get(piece.material), "props") is not None
            for piece in shapes[path].pieces(drawn[0])
        ):
            surface_shapes.add(path)
    entries: dict[tuple[str, int], list[int]] = {}
    for number, (_placed, kind, path) in enumerate(chosen):
        if levels[path]:
            entries.setdefault((path, int(band[number])), []).append(number)
    costs = {
        name: (len(numbers), [shapes[name[0]].triangle_count(d) for d in levels[name[0]]])
        for name, numbers in entries.items()
    }
    pinned = {
        name
        for name, numbers in entries.items()
        if name[0] in surface_shapes or chosen[numbers[0]][1] == "kerbs"
    }
    start = {name: (0, 1, 99)[name[1]] for name in entries}
    if budget is None:
        budget = auto_budget(costs, pinned)
    picked = choose_levels(costs, budget, pinned, start)
    level_of = {}
    total = 0
    for name, numbers in entries.items():
        detail = levels[name[0]][picked[name]]
        for number in numbers:
            level_of[number] = detail
        total += len(numbers) * costs[name][1][picked[name]]
    if total > budget:
        # Still over at the coarsest: the furthest copies go, never a pinned one.
        order = sorted(
            (n for name, numbers in entries.items() if name not in pinned for n in numbers),
            key=lambda n: -distance[n],
        )
        for number in order:
            if total <= budget:
                break
            path = chosen[number][2]
            total -= shapes[path].triangle_count(level_of.pop(number))
            out.dropped += 1
    out.budget = budget
    return level_of


# --------------------------------------------------------------------------
# Water


def water_parts(level: Level, corridor: Corridor) -> tuple[np.ndarray, np.ndarray]:
    """Water surfaces inside the band, as positions and faces, BeamNG's frame."""
    positions, faces = [], []
    offset = 0
    cell = corridor.cell
    ix, iy = np.nonzero(corridor.grid)
    cx = corridor.origin[0] + (ix + 0.5) * cell
    cy = corridor.origin[1] + (iy + 0.5) * cell
    for water in level.water:
        if water.kind == "plane" and water.height is not None:
            below = np.ones(len(cx), bool)
            for terrain in level.terrains:
                ground = terrain.height(cx, cy)
                below &= ~(np.isfinite(ground) & (ground >= water.height))
            if not below.any():
                continue
            x, y = cx[below], cy[below]
            h = cell / 2
            quad = np.stack(
                [np.c_[x - h, y - h], np.c_[x + h, y - h], np.c_[x + h, y + h], np.c_[x - h, y + h]], axis=1
            )
            points = np.c_[quad.reshape(-1, 2), np.full(len(quad) * 4, water.height)]
            base = offset + np.arange(len(quad))[:, None] * 4
            faces.append(np.concatenate([base + [0, 1, 2], base + [0, 2, 3]]))
            positions.append(points)
            offset += len(points)
        elif water.kind == "block" and water.matrix is not None:
            corners = np.array([[-0.5, -0.5, 0, 1], [0.5, -0.5, 0, 1], [0.5, 0.5, 0, 1], [-0.5, 0.5, 0, 1]]) @ water.matrix[:, :3]
            if not corridor.touches(corners[:, :2].min(axis=0), corners[:, :2].max(axis=0)):
                continue
            positions.append(corners)
            faces.append(np.array([[0, 1, 2], [0, 2, 3]]) + offset)
            offset += 4
        elif water.kind == "river" and water.nodes is not None:
            decal = DecalRoad(material="", nodes=water.nodes[:, :4], texture_length=10.0, priority=0.0, order=0)
            r = ribbon(decal, "water", [])
            if r is None:
                continue
            if not corridor.touches(r.left[:, :2].min(axis=0), r.left[:, :2].max(axis=0)):
                continue
            n = len(r.left)
            points = np.concatenate([r.left, r.right])
            k = np.arange(n - 1)
            faces.append(np.concatenate([np.stack([k, n + k, n + k + 1], 1), np.stack([k, n + k + 1, k + 1], 1)]) + offset)
            positions.append(points)
            offset += len(points)
    if not positions:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    return np.concatenate(positions), np.concatenate(faces)


# --------------------------------------------------------------------------
# The export


def export(driven, target: Path, args, *, install: Path | None = None) -> int:
    """
    A lap in, the course around it out: which level is found from the lap
    unless `args.level` says, and nothing else needs saying.
    """
    from .beamng import find_install

    game = Path(install) if install else find_install()
    if game is None:
        print(
            "no BeamNG.drive install found in the Steam libraries. Name its folder "
            "with --install (the one holding gameengine.zip).",
            file=sys.stderr,
        )
        return 1
    files = Files.of_install(game)
    route = from_viewer(driven.positions)
    name = getattr(args, "level", None)
    if not name:
        matches = find_level(files, route)
        if not matches or matches[0].score < 0.3:
            near = ", ".join(f"{m.level} {m.score:.0%}" for m in matches[:3])
            print(
                "the lap does not lie on any installed level's terrain"
                + (f" (best: {near})" if near else "")
                + ". Name the level with --level.",
                file=sys.stderr,
            )
            return 1
        name = matches[0].level
        print(f"  on {name}: {matches[0].score:.0%} of the lap lies on its {matches[0].by}")
    try:
        level = open_level(files, name)
    except BeamNGError as exc:
        print(exc, file=sys.stderr)
        return 1
    materials = Materials(files, level.name)
    textures = Textures(files)
    corridor = Corridor(plane(route), args.margin)
    reach = Corridor(plane(route), args.margin + 60.0)
    print(
        f"  cutting {args.margin:,.0f} m either side of the driving: "
        f"{corridor.area / 1e6:,.2f} km2"
    )

    # Road decals in the band, in the order the game paints them.
    decals = [
        d
        for d in level.decals
        if reach.touches(d.nodes[:, :2].min(axis=0) - d.nodes[:, 3].max(), d.nodes[:, :2].max(axis=0) + d.nodes[:, 3].max())
    ]
    decals.sort(key=lambda d: (-d.priority, d.order))
    ribbons = []
    for decal in decals:
        made = ribbon(decal, decal_kind(decal.material), level.terrains)
        if made is not None:
            ribbons.append(made)

    ground = build_ground(level, materials, corridor, ribbons)
    if ground is None:
        print(f"\nno terrain of {level.name} lies along the driving.", file=sys.stderr)
        return 1

    # Models: their driving surface always, the rest unless told otherwise.
    categories = DEFAULT_CATEGORIES + (("trees",) if args.trees else ())
    placed = place(
        level,
        materials,
        textures,
        corridor,
        reach,
        route,
        categories=categories,
        budget=args.detail_budget,
        use_textures=not args.no_textures,
        structures=not args.no_placed,
    )
    bases = [s for s in placed.surfaces if not s.overlay]
    overlays = [s for s in placed.surfaces if s.overlay]

    print("\n  what the ground is made of:")
    print(ground.summary())
    if placed.surfaces:
        counts: dict[str, int] = {}
        for s in placed.surfaces:
            counts[s.kind] = counts.get(s.kind, 0) + len(s.faces)
        listed = ", ".join(f"{n:,} {SURFACE_LABEL[k]}" for k, n in sorted(counts.items(), key=lambda kv: -kv[1]))
        print(f"  and modelled: {listed} triangles")

    # Every tile with ground in it, terrain's or a model's.
    tiles = list(ground.tiles)
    tile_number = {key: n for n, key in enumerate(tiles)}
    surface_tiles = []
    for s in bases:
        centre = s.positions[s.faces].mean(axis=1)
        keys = np.stack([np.floor(centre[:, 0] / TILE), np.floor(centre[:, 1] / TILE)], axis=1).astype(np.int64)
        numbers_of = np.empty(len(keys), np.int64)
        for k, key in enumerate(map(tuple, keys.tolist())):
            if key not in tile_number:
                tile_number[key] = len(tiles)
                tiles.append(key)
            numbers_of[k] = tile_number[key]
        surface_tiles.append(numbers_of)

    painter = Painter(materials, textures)
    images: dict[int, np.ndarray] = {}
    if not args.no_textures:
        for number, key in enumerate(tiles):
            images[number] = bake_tile(key, args.texture_size, level, painter, ribbons, placed.surfaces)
        print(
            f"\n  baked {len(images)} ground textures at {args.texture_size} px "
            f"({TILE / args.texture_size * 100:.0f} cm a pixel): {len(ribbons)} road decals and "
            f"{len(placed.surfaces)} modelled surfaces painted"
        )

    positions = [to_viewer(ground.positions)]
    normals = [to_viewer(ground.normals)]
    uvs = [ground.uvs]
    parts: list[Part] = []
    for kind in ("terrain", "loose", "verge", "road"):
        mask = ground.kinds == kind
        if not mask.any():
            continue
        for number in np.unique(ground.tile_of[mask]).tolist():
            chosen = ground.faces[mask & (ground.tile_of == number)]
            surface = BY_KEY[kind]
            parts.append(
                Part(
                    name=f"{part_name(surface)} {tiles[number][0]}_{tiles[number][1]}",
                    faces=chosen,
                    colour=None if number in images else surface.colour,
                    texture=images.get(number),
                )
            )
    offset = len(ground.positions)
    textured = bool(images)

    # Modelled driving surface: per tile, in that tile's image.
    for s, tile_of in zip(bases, surface_tiles):
        for number in np.unique(tile_of).tolist():
            p, f = _subset(s.positions, s.faces[tile_of == number])
            key = tiles[number]
            u = (p[:, 0] - key[0] * TILE) / TILE
            v = 1.0 - (p[:, 1] - key[1] * TILE) / TILE
            positions.append(to_viewer(p))
            normals.append(to_viewer(compute_normals(p, f)))
            uvs.append(np.stack([u, v], axis=1))
            parts.append(
                Part(
                    name=f"{SURFACE_PART[s.kind]} {key[0]}_{key[1]} {s.material}",
                    faces=f + offset,
                    colour=None if number in images else SURFACE_COLOUR[s.kind],
                    texture=images.get(number),
                )
            )
            offset += len(p)
    # Without a texture to paint them into, lines and rubber are parts of
    # their own, lifted off the surface as Forza's are.
    if not images:
        for s in overlays:
            p = s.positions.copy()
            p[:, 2] += DECAL_LIFT
            positions.append(to_viewer(p))
            normals.append(to_viewer(compute_normals(p, s.faces)))
            uvs.append(np.zeros((len(p), 2)))
            parts.append(Part(name=f"{SURFACE_PART[s.kind]} {s.material}", faces=s.faces + offset, colour=SURFACE_COLOUR[s.kind]))
            offset += len(p)

    if placed.parts:
        print(f"\n  placed on it ({placed.triangles():,} triangles of a {placed.budget:,} budget):")
        for kind, copies in sorted(placed.copies.items(), key=lambda kv: -kv[1]):
            print(f"    {copies:7,} {kind}")
    if placed.dropped:
        print(
            f"    {placed.dropped:,} copies left out to fit the budget, the furthest from the "
            "driving first (--detail-budget raises it)"
        )
    if placed.missing:
        print(f"  {sum(placed.missing.values()):,} copies of {len(placed.missing)} models could not be read")
    for name, part in sorted(placed.parts.items()):
        positions.append(to_viewer(part["positions"]))
        normals.append(to_viewer(part["normals"]))
        uvs.append(part["uvs"] if part["uvs"] is not None else np.zeros((len(part["positions"]), 2)))
        faces = np.concatenate(part["faces"]) + offset
        textured = textured or part["texture"] is not None
        parts.append(Part(name=name, faces=faces, colour=part["colour"], texture=part["texture"]))
        offset += len(part["positions"])

    water_points, water_faces = water_parts(level, corridor)
    if len(water_faces):
        positions.append(to_viewer(water_points))
        normals.append(np.tile([0.0, 1.0, 0.0], (len(water_points), 1)))
        uvs.append(np.zeros((len(water_points), 2)))
        parts.append(Part(name=part_name(WATER), faces=water_faces + offset, colour=WATER.colour))
        offset += len(water_points)

    written = write_glb(
        target,
        np.concatenate(positions),
        parts=parts,
        normals=np.concatenate(normals),
        uvs=np.concatenate(uvs).astype(np.float32) if textured else None,
    )
    print(f"\nwrote {written} ({written.stat().st_size:,} bytes)")
    return 0
