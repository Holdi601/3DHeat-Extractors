"""
Assetto Corsa Rally: finding the install and reading a stage the way the game
places it.

The game is Unreal Engine 5 (5.6.1, a modified build) and ships its content
as IoStore containers, none of them encrypted. A stage is a World Partition
map: `Levels/<Stage>.umap`, which places the distant terrain, and cooked cells
under `Levels/<Stage>/_Generated_/`, which place everything else. There is no
landscape: the ground is static meshes, the stage cut into 175 m tiles split
into road (`SM_<Stage>TrackNNNN_Road_k`) and terrain (`..._Terrain_k`), and the
far ground as 2 km tiles (`MeshesTerrain`).

What places a mesh is a `StaticMeshComponent`: its mesh, and its transform -
`RelativeLocation`, `RelativeRotation`, `RelativeScale3D` - relative to the
component it is attached to, up to the actor's root, which is in the world.
A value equal to the component's template (its archetype in a Blueprint) is
not stored; the template's is used. Those properties are read with the
schema in `unversioned.py`.

Instanced components - the forests, nearly all of them - are read without
their properties, which this build has rearranged beyond the engine's headers:
the mesh from the package's dependency table (each lists exactly the one mesh
it draws), the instances from the native block after the properties, a
matrix per instance relative to the component. Such a component is taken to
sit at its actor's root, which is how the foliage actors place them.

Coordinates are Unreal's: centimetres, Z up, X forward and Y right - a
left-handed frame. The course export maps them onto the telemetry's frame.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .unversioned import CHAINS, PropertyError, read_properties
from .zen import Export, Package, Packages

GAME_FOLDER = "Assetto Corsa Rally"
_SMC = "/Script/Engine.StaticMeshComponent"
_SCENE = "/Script/Engine.SceneComponent"
_INSTANCED = (
    "/Script/Foliage.FoliageInstancedStaticMeshComponent",
    "/Script/Engine.HierarchicalInstancedStaticMeshComponent",
    "/Script/Engine.InstancedStaticMeshComponent",
)
#: Meshes that are never seen: the invisible walls that put a car back on
#: the stage, and collision proxies.
_NEVER_DRAWN = re.compile(r"respawnwall|collision|blocker|invisible|/hlod/|_hlod", re.I)


def find_paks(extra: tuple[str | Path, ...] = (), *, defaults: bool = True) -> Path | None:
    """The game's `Content/Paks` folder."""
    candidates = [Path(p) for p in extra]
    if defaults:
        from .forzainstall import steam_libraries

        for library in steam_libraries():
            candidates.append(Path(library) / "steamapps" / "common" / GAME_FOLDER)
    for folder in candidates:
        for paks in (folder, folder / "acr" / "Content" / "Paks", folder / "Content" / "Paks"):
            if paks.is_dir() and any(paks.glob("*.utoc")):
                return paks
    return None


def rotator_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Unreal's `FRotationMatrix`: rows are the turned axes (row vectors)."""
    p, y, r = (math.radians(v) for v in (pitch, yaw, roll))
    sp, cp, sy, cy, sr, cr = math.sin(p), math.cos(p), math.sin(y), math.cos(y), math.sin(r), math.cos(r)
    return np.array(
        [
            [cp * cy, cp * sy, sp],
            [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp],
            [-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp],
        ]
    )


def transform(location, rotation, scale) -> np.ndarray:
    """A relative transform as a 4x4 for row vectors: `world = [p, 1] @ M`."""
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(scale, dtype=np.float64)[:, None] * rotator_matrix(*rotation)
    matrix[3, :3] = location
    return matrix


@dataclass
class Placement:
    """One copy of a mesh in the world."""

    mesh: str
    matrix: np.ndarray
    actor: str = ""
    #: `static` or `instanced`.
    source: str = "static"


@dataclass
class Stage:
    name: str
    placed: list[Placement] = field(default_factory=list)
    #: Components that could not be read, by reason.
    unread: dict[str, int] = field(default_factory=dict)
    read: int = 0

    def note(self, reason: str) -> None:
        self.unread[reason] = self.unread.get(reason, 0) + 1


def list_stages(packages: Packages) -> list[str]:
    """
    Every level that is a stage, by its map name: one whose environment
    folder has track tiles. Not by the tiles' names - one stage names them
    after its town rather than after its map.
    """
    with_tracks = {
        p.split("/content/environments/")[1].split("/")[0]
        for p in packages.store.files
        if "/meshestrack/" in p and "/content/environments/" in p
    }
    stages = []
    for path in packages.store.files:
        if not path.endswith(".umap") or "/content/levels/" not in path or "/_generated_/" in path:
            continue
        name = path.rsplit("/", 1)[-1][:-5]
        if name in with_tracks:
            stages.append(packages.store.names.get(path, path).rsplit("/", 1)[-1][:-5])
    return sorted(stages, key=str.lower)


def stage_packages(packages: Packages, stage: str) -> list[str]:
    store = packages.store
    umap = next((p for p in store.files if p.endswith(f"/content/levels/{stage.lower()}.umap")), None)
    cells = store.find(f"/levels/{stage.lower()}/_generated_/", ".umap")
    return ([umap] if umap else []) + cells


class _Components:
    """Component transforms and properties within one package, each read once."""

    def __init__(self, packages: Packages, package: Package) -> None:
        self.packages = packages
        self.package = package
        self._values: dict[int, dict | None] = {}
        self._world: dict[int, np.ndarray | None] = {}

    def values(self, export: Export) -> dict | None:
        if export.index in self._values:
            return self._values[export.index]
        cls = self.package.class_name(export)
        out = None
        if cls in CHAINS and cls not in _INSTANCED:
            try:
                own, _end = read_properties(self.package.export_data(export), cls, self.package.names)
            except (PropertyError, struct.error, IndexError):
                own = None
            if own is not None:
                out = {**_template_values(self.packages, self.package, export), **own}
        self._values[export.index] = out
        return out

    def world(self, export: Export, depth: int = 0) -> np.ndarray | None:
        if export.index in self._world:
            return self._world[export.index]
        values = self.values(export)
        result = None
        if values is not None and depth < 32:
            local = transform(
                values.get("RelativeLocation") or (0.0, 0.0, 0.0),
                values.get("RelativeRotation") or (0.0, 0.0, 0.0),
                values.get("RelativeScale3D") or (1.0, 1.0, 1.0),
            )
            parent = values.get("AttachParent")
            if isinstance(parent, int) and parent > 0 and parent - 1 < len(self.package.exports):
                above = self.world(self.package.exports[parent - 1], depth + 1)
                result = None if above is None else local @ above
            else:
                result = local
        self._world[export.index] = result
        return result


_TEMPLATES: dict[tuple[str, int], dict] = {}


def _template_values(packages: Packages, package: Package, export: Export, depth: int = 0) -> dict:
    """What a component inherits from its template: its archetype's values, recursively."""
    if depth > 8 or export.template >> 62 != 2:
        return {}
    name, public = package.package_import(export.template)
    path = packages.file_of(name) if name else None
    if path is None:
        return {}
    key = (path, public)
    if key in _TEMPLATES:
        return _TEMPLATES[key]
    _TEMPLATES[key] = {}
    other = packages.open(path)
    target = next((e for e in other.exports if e.public_hash == public), None) if other else None
    result: dict = {}
    if target is not None:
        cls = other.class_name(target)
        if cls in CHAINS and cls not in _INSTANCED:
            try:
                own, _ = read_properties(other.export_data(target), cls, other.names)
            except (PropertyError, struct.error, IndexError):
                own = {}
            result = {**_template_values(packages, other, target, depth + 1), **own}
    _TEMPLATES[key] = result
    return result


def _mesh_path(packages: Packages, package: Package, reference) -> str | None:
    if not isinstance(reference, int) or reference == 0:
        return None
    found = packages.resolve(package, reference)
    if found is None:
        return None
    other, export = found
    if not other.class_name(export).endswith(".StaticMesh"):
        return None
    return other.path


def _dependencies(package: Package) -> list[tuple[int, ...]]:
    """Per export, the imports and exports it depends on (5.3+ dependency bundles)."""
    data = package.data
    offsets = struct.unpack_from("<10i", data, 24)
    heads, entries = offsets[4], offsets[5]
    out = []
    for i in range(len(package.exports)):
        first, *counts = struct.unpack_from("<i4I", data, heads + 20 * i)
        total = sum(counts)
        out.append(struct.unpack_from(f"<{total}i", data, entries + 4 * first) if total else ())
    return out


def _instances(data: bytes) -> np.ndarray | None:
    """
    An instanced component's per-instance matrices: the native block after its
    properties, `[1][1][128][count][count x 16 doubles]`, rows affine.
    """
    pattern = struct.pack("<iii", 1, 1, 128)
    at = data.find(pattern)
    while at >= 0:
        (count,) = struct.unpack_from("<i", data, at + 12)
        start = at + 16
        if 0 < count and start + 128 * count <= len(data):
            matrices = np.frombuffer(data, dtype="<f8", count=16 * count, offset=start).reshape(count, 4, 4)
            if np.allclose(matrices[:, :3, 3], 0.0, atol=1e-6) and np.allclose(matrices[:, 3, 3], 1.0, atol=1e-6):
                return matrices
        at = data.find(pattern, at + 1)
    return None


def read_stage(packages: Packages, stage: str, *, instanced: bool = False, within=None) -> Stage:
    """
    Every mesh a stage places. `within(low, high)` - corners in Unreal's (x, y)
    - drops cells and components wholly outside; `instanced` adds the forests.
    """
    out = Stage(name=stage)
    seen: set[tuple] = set()
    for path in stage_packages(packages, stage):
        package = packages.open(path)
        if package is None:
            out.note("package unreadable")
            continue
        components = _Components(packages, package)
        deps = _dependencies(package) if instanced else None
        for export in package.exports:
            cls = package.class_name(export)
            if cls == _SMC:
                values = components.values(export)
                if values is None:
                    out.note("properties unreadable")
                    continue
                out.read += 1
                if values.get("bVisible", 1) == 0 or values.get("bHiddenInGame", 0) == 1:
                    continue
                mesh = _mesh_path(packages, package, values.get("StaticMesh"))
                if mesh is None or _NEVER_DRAWN.search(mesh):
                    continue
                matrix = components.world(export)
                if matrix is None:
                    out.note("attachment unreadable")
                    continue
                if within is not None and not within(matrix[3, :2], matrix[3, :2]):
                    pass
                key = (mesh.lower(), tuple(np.round(matrix.ravel(), 1)))
                if key in seen:
                    continue
                seen.add(key)
                actor = package.outer_export(export)
                out.placed.append(Placement(mesh, matrix, package.class_name(actor) if actor else ""))
            elif instanced and cls in _INSTANCED:
                meshes = []
                for reference in deps[export.index]:
                    found = _mesh_path(packages, package, reference) if reference < 0 else None
                    if found:
                        meshes.append(found)
                if len(set(meshes)) != 1 or _NEVER_DRAWN.search(meshes[0]):
                    continue
                matrices = _instances(package.export_data(export))
                if matrices is None:
                    continue
                root = _actor_root(components, package, export)
                for m in matrices:
                    world = m @ root
                    key = (meshes[0].lower(), tuple(np.round(world.ravel(), 1)))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.placed.append(Placement(meshes[0], world, "instanced", "instanced"))
    return out


def _actor_root(components: _Components, package: Package, export: Export) -> np.ndarray:
    """The world transform of the root of the actor a component belongs to."""
    actor = package.outer_export(export)
    if actor is None:
        return np.eye(4)
    for other in package.exports:
        if other is export or package.outer_export(other) is not actor:
            continue
        cls = package.class_name(other)
        if cls in (_SCENE, _SMC):
            values = components.values(other)
            if values is not None and not values.get("AttachParent"):
                world = components.world(other)
                if world is not None:
                    return world
    return np.eye(4)


@dataclass
class TextureVolume:
    """A baked virtual texture and the box of the world it covers (Unreal cm)."""

    texture: str
    origin: tuple[float, float]
    size: tuple[float, float]


def texture_volume(packages: Packages, stage: str) -> TextureVolume | None:
    """
    The stage's baked ground colour: the runtime virtual texture volume in
    its map, whose box - location its corner, scale its size - the texture
    is laid over, and the streaming texture it was baked into.
    """
    from .unversioned import read_properties

    paths = stage_packages(packages, stage)
    package = packages.open(paths[0]) if paths else None
    if package is None:
        return None
    for export in package.exports:
        cls = package.class_name(export)
        if cls != "/Script/Engine.RuntimeVirtualTextureComponent":
            continue
        try:
            values, end = read_properties(package.export_data(export), cls, package.names)
        except (PropertyError, struct.error, IndexError):
            continue
        location = values.get("RelativeLocation")
        scale = values.get("RelativeScale3D")
        if not location or not scale or values.get("RelativeRotation"):
            continue
        found = packages.resolve(package, values.get("StreamingTexture") or 0)
        if found is None:
            continue
        builder, _export = found
        return TextureVolume(builder.path, (location[0], location[1]), (scale[0], scale[1]))
    return None
