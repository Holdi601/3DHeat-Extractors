"""
An Assetto Corsa Rally course: the stage's ground and what stands on it along
one lap's driving.

The stage is found from the lap, and so is the frame. The game's own world is
Unreal's - centimetres, Z up, left-handed - and the telemetry's is not
documented, so it is not assumed: of the eight ways the two horizontal axes
can map onto the lap's (swapped or not, each either sign), the one that puts
the most of the lap on the stage's road tiles is used, on whichever stage it
scores best. A lap that fits none of them is refused rather than drawn
somewhere plausible.

Ground is the stage's own: the track tiles split into road and terrain, and
the distant terrain beyond them, cut to a band either side of the driving.
Everything else is placed by category, as for Forza, with the same levels of
detail by distance and the same triangle budget. Models are drawn in flat
colours by category.
"""

from __future__ import annotations

import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .acrally import Placement, Stage, find_paks, list_stages, read_stage
from .corridor import Corridor
from .glb import Part, compute_normals, write_glb
from .iostore import IoStoreError, Store
from .surfaces import ROAD, TERRAIN, part_name
from .unrealmesh import MeshShelf
from .zen import Packages

#: How far above the road a lap's samples may be and count as on it, metres.
ON_ROAD = (-1.0, 3.0)
#: Cells of the road-height grid the frame is scored on, metres.
GRID = 2.0

CATEGORY_COLOUR = {
    "barriers": (0.55, 0.52, 0.48),
    "signs": (0.75, 0.72, 0.65),
    "buildings": (0.62, 0.58, 0.52),
    "infrastructure": (0.48, 0.48, 0.50),
    "rocks": (0.45, 0.43, 0.40),
    "props": (0.55, 0.45, 0.35),
    "trees": (0.20, 0.32, 0.14),
    "kerbs": (0.70, 0.25, 0.22),
}
DEFAULT_CATEGORIES = ("barriers", "signs", "buildings", "infrastructure", "rocks", "props", "kerbs")


def _named_stage(packages: Packages, track: str | None) -> str | None:
    """The stage a lap's own track name means, matched loosely, or None."""
    if not track:
        return None
    wanted = re.sub(r"[^a-z0-9]", "", track.lower())
    if not wanted:
        return None
    for stage in list_stages(packages):
        name = stage.lower()
        if wanted in name or name in wanted:
            return stage
    return None


def frames() -> list[np.ndarray]:
    """
    The candidate frames, each as the 4x4 (row vectors) taking Unreal's world
    to the lap's: Unreal Z (cm) is the lap's height (m); its X and Y become
    the lap's x and z in either order and either sign.
    """
    out = []
    for swap in (False, True):
        for sx in (1.0, -1.0):
            for sz in (1.0, -1.0):
                m = np.zeros((4, 4))
                m[3, 3] = 1.0
                m[2, 1] = 0.01  # Unreal Z -> lap y
                if not swap:
                    m[0, 0], m[1, 2] = 0.01 * sx, 0.01 * sz  # X -> x, Y -> z
                else:
                    m[1, 0], m[0, 2] = 0.01 * sx, 0.01 * sz  # Y -> x, X -> z
                out.append(m)
    return out


def describe_frame(frame: np.ndarray) -> str:
    """Which Unreal axis became the lap's x and z, and which way."""
    def axis(column: int) -> str:
        row = int(np.argmax(np.abs(frame[:2, column])))
        return ("-" if frame[row, column] < 0 else "+") + "XY"[row]

    return f"x = {axis(0)}, z = {axis(2)} (Unreal, metres)"


def ground_kind(mesh: str) -> str | None:
    """`road` or `terrain` for the stage's own ground tiles; None for the rest."""
    path = mesh.lower()
    if "/meshestrack/" in path:
        # Split tiles say which half they are; an unsplit one - the ice
        # school's frozen lake - is all driving surface.
        return "terrain" if "_terrain" in path.rsplit("/", 1)[-1] else "road"
    if "/meshesterrain/" in path:
        return "terrain"
    return None


_VEGETATION = re.compile(r"/foliage/|tree|bush|grass|plant|flower|leaves|spruce|pine|larch|birch|maple|oak|fern|shrub|hedge", re.I)


def category(mesh: str) -> str:
    name = mesh.lower().rsplit("/", 1)[-1]
    path = mesh.lower()
    if _VEGETATION.search(path):
        return "trees"
    if re.search(r"rock|cliff|boulder|stone", name):
        return "rocks"
    if re.search(r"kerb|curb", name):
        return "kerbs"
    if re.search(r"barrier|fence|wall|rail|tape|bale|tyre|tire|cone|bollard|net|guard", name):
        return "barriers"
    if re.search(r"sign|banner|board|flag|arch|gantry|advert|logo", name):
        return "signs"
    if re.search(r"house|building|bld|barn|shed|church|hut|garage|roof|tent|stand|cabin|chalet|farm", name):
        return "buildings"
    if re.search(r"pole|light|lamp|power|pylon|cable|post|mast|bridge", name):
        return "infrastructure"
    return "props"


@dataclass
class GroundImage:
    """A stage's satellite image and how it lies on the world."""

    pixels: np.ndarray
    #: (3, 2): `uv = [x, y, 1] @ mapping` for Unreal world centimetres.
    mapping: np.ndarray
    metres_per_pixel: float

    def uv(self, unreal: np.ndarray) -> np.ndarray:
        return np.c_[unreal[:, :2], np.ones(len(unreal))] @ self.mapping


_SATELLITE = re.compile(r"sat[^/]*_(b|bc|c|basecolor)\.uasset$")


def ground_image(packages: Packages, shelf: MeshShelf, stage: Stage, *, largest: int = 4096) -> GroundImage | None:
    """
    The stage's satellite image, placed on the world.

    The far terrain is drawn in it, through its first UV channel, and that
    channel is a plane in the world: fitted over every far tile whose UVs are
    in the image, it is linear to a few thousandths. The fit is the image's
    placement. A material may repeat the coordinates - one stage's far terrain
    runs to 2 - so the scale is halved until the far terrain fits the image
    once, which put that stage's road exactly on the road the image shows.
    """
    from .unrealtexture import TextureError, read_texture

    folder = f"/content/environments/{stage.name.lower()}/"
    found = sorted(p for p in packages.store.files if folder in p and _SATELLITE.search(p))
    if not found:
        return None
    points, coords = [], []
    for placed in stage.placed:
        if "/meshesterrain/" not in placed.mesh.lower():
            continue
        mesh = shelf.get(placed.mesh)
        if mesh is None or mesh.lods[0].uvs is None:
            continue
        lod = mesh.lods[0]
        uv = lod.uvs.astype(np.float64)
        if not ((uv >= -0.01).all() and (uv <= 4.01).all()):
            continue
        world = np.c_[lod.positions.astype(np.float64), np.ones(len(lod.positions))] @ placed.matrix[:, :3]
        points.append(world[:, :2])
        coords.append(uv)
    if not points:
        return None
    points, coords = np.concatenate(points), np.concatenate(coords)
    design = np.c_[points, np.ones(len(points))]
    mapping, *_ = np.linalg.lstsq(design, coords, rcond=None)
    if np.percentile(np.abs(design @ mapping - coords), 95) > 0.01:
        return None
    reach = coords.max()
    while reach > 1.05:
        mapping, reach = mapping / 2, reach / 2
    try:
        pixels = read_texture(packages, found[0], largest=largest)[..., :3]
    except (TextureError, OSError, ValueError):
        return None
    scale = float(np.abs(mapping[:2]).max())  # uv per centimetre
    return GroundImage(pixels, mapping, 1.0 / (scale * pixels.shape[1]) / 100.0)


#: Ground textures cover 512 m squares, as Forza's and BeamNG's do.
TILE_CM = 51200.0


class GroundPainter:
    """Ground images per 512 m square: the virtual texture, the satellite around it."""

    def __init__(self, packages: Packages, shelf: MeshShelf, stage: Stage, size: int) -> None:
        from .acrally import texture_volume
        from .unrealtexture import TextureError, VirtualTexture

        self.size = size
        self.volume = texture_volume(packages, stage.name)
        self.virtual = None
        if self.volume is not None:
            try:
                self.virtual = VirtualTexture(packages, self.volume.texture)
            except (TextureError, OSError, ValueError, struct.error):
                self.virtual = None
        self.satellite = ground_image(packages, shelf, stage)

    def describe(self) -> str:
        parts = []
        if self.virtual is not None:
            metres = self.volume.size[0] / 100 / self.virtual.width
            parts.append(f"the stage's baked ground colour ({metres * 100:.0f} cm a pixel)")
        if self.satellite is not None:
            parts.append(f"its satellite image ({self.satellite.metres_per_pixel:.1f} m a pixel)" if parts else f"the stage's satellite image ({self.satellite.metres_per_pixel:.1f} m a pixel)")
        return " and ".join(parts)

    def tile(self, key: tuple[int, int]) -> np.ndarray | None:
        size = self.size
        x0, y0 = key[0] * TILE_CM, key[1] * TILE_CM
        image = np.zeros((size, size, 3), np.uint8)
        if self.virtual is not None:
            (ox, oy), (sx, sy) = self.volume.origin, self.volume.size
            image = self.virtual.region((x0 - ox) / sx, (y0 - oy) / sy, (x0 + TILE_CM - ox) / sx, (y0 + TILE_CM - oy) / sy, size)
        empty = image.sum(axis=2) == 0
        if empty.any() and self.satellite is not None:
            step = TILE_CM / size
            xs = x0 + (np.arange(size) + 0.5) * step
            ys = y0 + (np.arange(size) + 0.5) * step
            gx, gy = np.meshgrid(xs, ys)
            uv = np.c_[gx[empty], gy[empty], np.ones(int(empty.sum()))] @ self.satellite.mapping
            h, w = self.satellite.pixels.shape[:2]
            col = np.clip((uv[:, 0] * w).astype(np.int64), 0, w - 1)
            row = np.clip((uv[:, 1] * h).astype(np.int64), 0, h - 1)
            inside = (uv >= 0).all(axis=1) & (uv <= 1).all(axis=1)
            fill = self.satellite.pixels[row, col]
            fill[~inside] = 0
            image[empty] = fill
        if not image.any():
            return None
        return image


@dataclass
class Fit:
    stage: str
    frame: np.ndarray
    score: float


def _road_grid(stage: Stage, shelf: MeshShelf) -> dict[tuple[int, int], float]:
    """The highest road surface in each grid cell, in Unreal's world (cm)."""
    cell = GRID * 100.0
    grid: dict[tuple[int, int], float] = {}
    for placed in stage.placed:
        if ground_kind(placed.mesh) != "road":
            continue
        mesh = shelf.get(placed.mesh)
        if mesh is None:
            continue
        lod = mesh.lods[-1]
        world = np.c_[lod.positions.astype(np.float64), np.ones(len(lod.positions))] @ placed.matrix[:, :3]
        keys = np.floor(world[:, :2] / cell).astype(np.int64)
        for (i, j), z in zip(map(tuple, keys.tolist()), world[:, 2].tolist()):
            if grid.get((i, j), -1e18) < z:
                grid[(i, j)] = z
    return grid


def _score(route: np.ndarray, grid: dict, frame: np.ndarray) -> float:
    """Share of the lap on the road when the lap is read in `frame`."""
    inverse = np.linalg.inv(frame)
    unreal = np.c_[route, np.ones(len(route))] @ inverse[:, :3]
    cell = GRID * 100.0
    keys = np.floor(unreal[:, :2] / cell).astype(np.int64)
    on = 0
    for (i, j), z in zip(map(tuple, keys.tolist()), unreal[:, 2].tolist()):
        best = None
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                h = grid.get((i + di, j + dj))
                if h is not None and (best is None or abs(z - h) < abs(z - best)):
                    best = h
        if best is not None and ON_ROAD[0] * 100 <= z - best <= ON_ROAD[1] * 100:
            on += 1
    return on / max(1, len(route))


def fit(packages: Packages, shelf: MeshShelf, route: np.ndarray, *, stage: str | None = None, report=print) -> tuple[Fit | None, list[Fit]]:
    """The stage and frame a lap fits best, and every candidate's best score."""
    pick = route[np.linspace(0, len(route) - 1, min(len(route), 400)).astype(np.int64)]
    names = [stage] if stage else list_stages(packages)
    results = []
    for name in names:
        read = read_stage(packages, name)
        grid = _road_grid(read, shelf)
        if not grid:
            continue
        scored = [(_score(pick, grid, f), f) for f in frames()]
        best = max(scored, key=lambda s: s[0])
        results.append(Fit(name, best[1], best[0]))
    results.sort(key=lambda f: -f.score)
    winner = results[0] if results and results[0].score >= 0.5 else None
    return winner, results


def export(driven, target: Path, args, *, paks: Path | None = None) -> int:
    """A lap in, the stage around it out."""
    paks = Path(paks) if paks else find_paks()
    if paks is None:
        print(
            "no Assetto Corsa Rally install found in the Steam libraries. Name its "
            "folder with --install.",
            file=sys.stderr,
        )
        return 1
    try:
        store = Store(paks)
    except IoStoreError as exc:
        print(exc, file=sys.stderr)
        return 1
    packages = Packages(store)
    shelf = MeshShelf(packages)
    route = np.asarray(driven.positions, dtype=np.float64)
    wanted = getattr(args, "level", None)
    if wanted:
        match = next((s for s in list_stages(packages) if s.lower() == wanted.lower()), None)
        if match is None:
            print(f"no stage called {wanted!r}; there are: {', '.join(list_stages(packages))}", file=sys.stderr)
            return 1
        wanted = match
    best, tried = None, []
    named = _named_stage(packages, getattr(driven, "track", None)) if not wanted else None
    if named:
        # The lap names its stage: check that one first, and only it if it fits.
        best, tried = fit(packages, shelf, route, stage=named)
        if best is not None and best.score < 0.8:
            best = None
    if best is None:
        best, tried = fit(packages, shelf, route, stage=wanted)
    if best is None:
        near = ", ".join(f"{f.stage} {f.score:.0%}" for f in tried[:3])
        print(
            "the lap does not lie on any stage's road in any frame"
            + (f" (best: {near})" if near else "")
            + ". Name the stage with --level.",
            file=sys.stderr,
        )
        return 1
    frame = best.frame
    print(f"  on {best.stage}: {best.score:.0%} of the lap lies on its road, read as {describe_frame(frame)}")

    corridor = Corridor(route, args.margin)
    reach = Corridor(route, args.margin + 60.0)
    print(f"  cutting {args.margin:,.0f} m either side of the driving: {corridor.area / 1e6:,.2f} km2")
    stage = read_stage(packages, best.stage, instanced=args.trees)
    if stage.unread:
        print(f"  {sum(stage.unread.values()):,} of {stage.read + sum(stage.unread.values()):,} components could not be read: {stage.unread}")

    positions, faces_of, names, colours, uvs, textures = [], [], [], [], [], []
    ground_counts = {"road": 0, "terrain": 0}
    offset = 0
    placed_ground = []
    placed_other = []
    for p in stage.placed:
        (placed_ground if ground_kind(p.mesh) else placed_other).append(p)

    # The ground's own colour, from above: the baked virtual texture where
    # the stage has one, the satellite image around it.
    painter = None if args.no_textures else GroundPainter(packages, shelf, stage, args.texture_size)
    if painter is not None and painter.describe():
        print(f"  ground textured from {painter.describe()}")
    elif painter is not None:
        painter = None

    # Ground: each tile cut to the band, its triangles as road or terrain, in
    # 512 m squares each with its own image.
    grouped: dict[tuple[str, tuple[int, int]], list] = {}
    for p in placed_ground:
        mesh = shelf.get(p.mesh)
        if mesh is None:
            continue
        lod = mesh.lods[0]
        unreal = np.c_[lod.positions.astype(np.float64), np.ones(len(lod.positions))] @ p.matrix[:, :3]
        world = np.c_[unreal, np.ones(len(unreal))] @ frame[:, :3]
        faces = lod.faces if np.linalg.det((p.matrix @ frame)[:3, :3]) > 0 else lod.faces[:, [0, 2, 1]]
        faces = faces[corridor.triangles(world, faces)]
        if not len(faces):
            continue
        centre = unreal[faces].mean(axis=1)[:, :2]
        keys = np.floor(centre / TILE_CM).astype(np.int64)
        kind = ground_kind(p.mesh)
        for key in np.unique(keys, axis=0):
            chosen = faces[(keys == key).all(axis=1)]
            grouped.setdefault((kind, (int(key[0]), int(key[1]))), []).append((world, chosen, unreal[:, :2]))
    images: dict[tuple[int, int], np.ndarray | None] = {}
    for (kind, key), pieces in sorted(grouped.items()):
        points, faces, flat = _join(pieces)
        if painter is not None and key not in images:
            images[key] = painter.tile(key)
        image = images.get(key)
        positions.append(points)
        uvs.append((flat - np.array(key) * TILE_CM) / TILE_CM if image is not None else np.zeros((len(points), 2)))
        faces_of.append(faces + offset)
        surface = ROAD if kind == "road" else TERRAIN
        names.append(f"{part_name(surface)} {key[0]}_{key[1]}")
        colours.append(surface.colour)
        textures.append(image)
        offset += len(points)
        ground_counts[kind] += len(faces)
    if images:
        print(f"  baked {sum(i is not None for i in images.values())} ground textures at {args.texture_size} px ({TILE_CM / 100 / args.texture_size * 100:.0f} cm a pixel)")
    if not any(ground_counts.values()):
        print(f"\nno ground of {best.stage} lies along the driving.", file=sys.stderr)
        return 1
    print("\n  what the ground is made of:")
    total = max(1, sum(ground_counts.values()))
    for kind, count in sorted(ground_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {count:9,} triangles  {count / total:5.1%}  {kind}")

    # Everything else, with levels of detail as Forza's export picks them.
    if not args.no_placed:
        categories = DEFAULT_CATEGORIES + (("trees",) if args.trees else ())
        chosen = []
        for p in placed_other:
            kind = category(p.mesh)
            if kind not in categories:
                continue
            where = (np.array([[0.0, 0.0, 0.0, 1.0]]) @ (p.matrix @ frame))[0, :3]
            if not reach.contains(where[None, :])[0]:
                continue
            chosen.append((p, kind))
        levels = _levels(chosen, shelf, frame, route, args.detail_budget)
        by_part: dict[str, list] = {}
        copies: dict[str, int] = {}
        for (p, kind), level in zip(chosen, levels):
            if level is None:
                continue
            mesh = shelf.get(p.mesh)
            lod = mesh.lods[level]
            matrix = p.matrix @ frame
            world = np.c_[lod.positions.astype(np.float64), np.ones(len(lod.positions))] @ matrix[:, :3]
            low, high = world[:, [0, 2]].min(axis=0), world[:, [0, 2]].max(axis=0)
            if not corridor.touches(low, high):
                continue
            faces = lod.faces if np.linalg.det(matrix[:3, :3]) > 0 else lod.faces[:, [0, 2, 1]]
            by_part.setdefault(kind, []).append((world, faces))
            copies[kind] = copies.get(kind, 0) + 1
        if by_part:
            triangles = sum(len(f) for pieces in by_part.values() for _, f in pieces)
            print(f"\n  placed on it ({triangles:,} triangles):")
            for kind, count in sorted(copies.items(), key=lambda kv: -kv[1]):
                print(f"    {count:7,} {kind}")
        for kind, pieces in sorted(by_part.items()):
            points, faces, _ = _join(pieces)
            positions.append(points)
            uvs.append(np.zeros((len(points), 2)))
            faces_of.append(faces + offset)
            names.append(f"structure:{kind}")
            colours.append(CATEGORY_COLOUR[kind])
            textures.append(None)
            offset += len(points)
    if shelf.failed:
        print(f"  {len(shelf.failed)} meshes could not be read")

    every = np.concatenate(positions)
    faces_all = np.concatenate(faces_of)
    normals = compute_normals(every, faces_all)
    parts = [
        Part(name=n, faces=f, colour=None if t is not None else c, texture=t)
        for n, f, c, t in zip(names, faces_of, colours, textures)
    ]
    textured = any(t is not None for t in textures)
    written = write_glb(
        target,
        every,
        parts=parts,
        normals=normals,
        uvs=np.clip(np.concatenate(uvs), 0.0, 1.0).astype(np.float32) if textured else None,
    )
    print(f"\nwrote {written} ({written.stat().st_size:,} bytes)")
    return 0


def _join(pieces):
    """
    Several meshes as one, keeping only the vertices their faces use: each
    piece is (positions, faces) or (positions, faces, uvs).
    """
    points, faces, uvs = [], [], []
    offset = 0
    for piece in pieces:
        world, chosen = piece[0], piece[1]
        uv = piece[2] if len(piece) > 2 else None
        used, inverse = np.unique(chosen, return_inverse=True)
        points.append(world[used])
        uvs.append(None if uv is None else uv[used])
        faces.append(inverse.reshape(-1, 3) + offset)
        offset += len(used)
    joined_uv = None if any(u is None for u in uvs) else np.concatenate(uvs)
    return np.concatenate(points), np.concatenate(faces), joined_uv


def _levels(chosen, shelf: MeshShelf, frame: np.ndarray, route: np.ndarray, budget: int | None) -> list[int | None]:
    """A level of detail per copy, by distance bands and the budget, as for Forza."""
    from .forzaplacement import FAR, NEAR, auto_budget, choose_levels, route_distance

    if not chosen:
        return []
    where = np.array([(np.array([0.0, 0.0, 0.0, 1.0]) @ (p.matrix @ frame))[[0, 2]] for p, _ in chosen])
    distance = route_distance(where, route[:, [0, 2]])
    band = np.digitize(distance, (NEAR, FAR))
    entries: dict[tuple[str, int], list[int]] = {}
    for number, (p, kind) in enumerate(chosen):
        mesh = shelf.get(p.mesh)
        if mesh is None or not mesh.lods:
            continue
        entries.setdefault((p.mesh.lower(), int(band[number])), []).append(number)
    costs = {
        name: (len(numbers), [len(lod.faces) for lod in shelf.get(chosen[numbers[0]][0].mesh).lods])
        for name, numbers in entries.items()
    }
    pinned = {name for name, numbers in entries.items() if chosen[numbers[0]][1] == "kerbs"}
    start = {name: (0, 1, 99)[name[1]] for name in entries}
    if budget is None:
        budget = auto_budget(costs, pinned)
    picked = choose_levels(costs, budget, pinned, start)
    out: list[int | None] = [None] * len(chosen)
    for name, numbers in entries.items():
        for number in numbers:
            out[number] = picked[name]
    return out
