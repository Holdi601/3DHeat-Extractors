"""
BeamNG.drive: finding the install, and reading a level the way the game does.

BeamNG ships its content as plain zip archives - `gameengine.zip` for the
engine's own art, `content/art_shapes.zip` and `content/assets/**` for what
levels share, one zip per level under `content/levels` - and mounts them all
into one file system. A level refers to files anywhere in it
(`/art/shapes/...`, `/assets/materials/...`), so this builds the same thing:
one case-insensitive index over every archive, with the user's mods on top.

What a level is made of, and where each part comes from, each settled against
the shipped levels rather than assumed:

- **The scene**: `levels/<name>/main/**/items.level.json`, one JSON object per
  line (a few files are a single JSON array instead). Objects carry world
  positions; `rotationMatrix` is nine numbers whose *rows* are the object's
  own axes in the world - guardrail sections chained along a road point along
  their row 0, and the up row tilts with the road's slope. Older prefab files
  give `rotation = "x y z degrees"` instead, a right-handed turn by that angle
  (checked the same way: 92% of rail chains line up with it, 19% with the
  opposite sense).
- **Prefabs**: a group of objects in a `.prefab` (TorqueScript) or
  `.prefab.json` file, placed by a `Prefab` object. Children are relative to
  the prefab's own transform; `groupPosition` is an editor pivot and is not
  subtracted (a trolley prefab sits on the ground only without it).
- **Forest items**: `levels/<name>/forest/*.forest4.json`, one line per
  instance, typed by `TSForestItemData` records in `managedItemData.json`.
  Barriers, fences and rocks are forest items as often as trees are.
- **Terrain**: a `TerrainBlock` names a `.ter` file: a version byte, the size,
  `size x size` 16-bit heights, as many layer bytes, then the layer names. The
  height is `position.z + h * maxHeight / 65535` and rows run along +Y: road
  nodes lie on it to 1-2 cm in the median on three levels, where `h / 32` or
  the other axis order miss by tens to hundreds of metres.
- **Materials**: `*.materials.json` anywhere in the file system, matched by
  `mapTo` or name. A texture named `x.color.png` ships as `x.color.dds`, or as
  `x.color.png.link` pointing into `/assets`.

Coordinates are BeamNG's: Z up, metres. The export turns them into the
viewer's Y-up frame with the same rotation the telemetry recorder uses, so a
recorded lap and the course line up.
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: The game's folder name in a Steam library.
GAME_FOLDER = "BeamNG.drive"

#: A number in a TorqueScript value. Values are space separated, except where
#: they are not: some shipped prefabs write `-0.99563-0.0`.
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


class BeamNGError(Exception):
    """Something about the install or the level that stops an export."""


# --------------------------------------------------------------------------
# The install and the file system


def find_install(extra: tuple[str | Path, ...] = (), *, defaults: bool = True) -> Path | None:
    """The game's folder: the one holding `gameengine.zip` and `content`."""
    candidates: list[Path] = [Path(p) for p in extra]
    if defaults:
        from .forzainstall import steam_libraries

        for library in steam_libraries():
            candidates.append(Path(library) / "steamapps" / "common" / GAME_FOLDER)
    for folder in candidates:
        if (folder / "gameengine.zip").exists() and (folder / "content").is_dir():
            return folder
    return None


def user_folders() -> list[Path]:
    """Where BeamNG keeps the user's own content, newest layout first."""
    local = os.environ.get("LOCALAPPDATA")
    found: list[Path] = []
    if not local:
        return found
    for base in (Path(local) / "BeamNG" / "BeamNG.drive", Path(local) / "BeamNG.drive"):
        if not base.is_dir():
            continue
        current = base / "current"
        if current.is_dir():
            found.append(current)
        versions = sorted(
            (p for p in base.iterdir() if p.is_dir() and re.fullmatch(r"\d+(\.\d+)*", p.name)),
            key=lambda p: [int(x) for x in p.name.split(".")],
            reverse=True,
        )
        found.extend(versions[:1])
    return found


def _key(path: str) -> str:
    return path.replace("\\", "/").lstrip("/").lower()


class Files:
    """
    Every archive and folder the game mounts, as one file system.

    Later additions win, as they do in the game: a mod replaces what it
    overrides. Lookups ignore case, since the levels' own references do not
    agree with the archives on it (`.DAE` for a `.dae`).
    """

    def __init__(self) -> None:
        self._zips: list[zipfile.ZipFile] = []
        self._dirs: list[Path] = []
        self._index: dict[str, tuple[int, str]] = {}

    def add_zip(self, path: str | Path) -> None:
        try:
            archive = zipfile.ZipFile(path)
        except (OSError, zipfile.BadZipFile):
            return
        number = len(self._zips)
        self._zips.append(archive)
        for name in archive.namelist():
            if not name.endswith("/"):
                self._index[_key(name)] = (number, name)

    def add_dir(self, root: str | Path) -> None:
        root = Path(root)
        number = -1 - len(self._dirs)
        self._dirs.append(root)
        for path in root.rglob("*"):
            if path.is_file():
                self._index[_key(str(path.relative_to(root)))] = (number, str(path))

    def __contains__(self, path: str) -> bool:
        return _key(path) in self._index

    def real(self, path: str) -> str | None:
        """The path as stored, or None."""
        found = self._index.get(_key(path))
        return found[1] if found else None

    def read(self, path: str) -> bytes:
        found = self._index.get(_key(path))
        if found is None:
            raise FileNotFoundError(path)
        number, name = found
        if number >= 0:
            return self._zips[number].read(name)
        return Path(name).read_bytes()

    def head(self, path: str, size: int) -> bytes:
        """The first `size` bytes, without inflating the rest."""
        found = self._index.get(_key(path))
        if found is None:
            raise FileNotFoundError(path)
        number, name = found
        if number >= 0:
            with self._zips[number].open(name) as fh:
                return fh.read(size)
        with open(name, "rb") as fh:
            return fh.read(size)

    def names(self, prefix: str = "", suffix: str = "") -> list[str]:
        """Every file under `prefix` ending in `suffix`, as index keys."""
        prefix, suffix = _key(prefix), suffix.lower()
        return sorted(k for k in self._index if k.startswith(prefix) and k.endswith(suffix))

    @classmethod
    def of_install(cls, game: Path, *, mods: bool = True) -> "Files":
        """What the game mounts: its own archives, then the user's."""
        files = cls()
        files.add_zip(game / "gameengine.zip")
        content = game / "content"
        for archive in sorted(content.rglob("*.zip")):
            # Vehicles and sounds are never part of a course.
            relative = archive.relative_to(content).parts
            if relative and relative[0] in ("vehicles",) or archive.name.startswith(("art_sound", "audio")):
                continue
            files.add_zip(archive)
        if mods:
            for user in user_folders():
                for archive in sorted((user / "mods").glob("*.zip")) if (user / "mods").is_dir() else ():
                    files.add_zip(archive)
                unpacked = user / "mods" / "unpacked"
                if unpacked.is_dir():
                    for folder in sorted(unpacked.iterdir()):
                        if folder.is_dir():
                            files.add_dir(folder)
                if (user / "levels").is_dir():
                    files.add_dir(user)
        return files


def resolve(files: Files, path: str | None, relative_to: str = "") -> str | None:
    """
    Where a path a level names actually is, or None.

    Tries what the game tries: the path as given, then relative to the file
    that named it; each as named, as the `.dds` a texture was cooked to, as
    the `.cdae` a shape was compiled to, and through a `.link` file.
    """
    if not path:
        return None
    path = path.replace("\\", "/").strip()
    bases = [path.lstrip("/")]
    if relative_to and not path.startswith("/"):
        folder = relative_to.replace("\\", "/").rsplit("/", 1)[0] if "/" in relative_to else ""
        bases.append(f"{folder}/{path.lstrip('./')}")
    for base in bases:
        for candidate in _spellings(base):
            if candidate in files:
                return files.real(candidate) or candidate
            link = candidate + ".link"
            if link in files:
                try:
                    target = json.loads(files.read(link)).get("path")
                except (ValueError, KeyError, OSError):
                    target = None
                if target and target.lstrip("/").lower() != candidate.lower():
                    found = resolve(files, target)
                    if found:
                        return found
    return None


def _spellings(path: str) -> list[str]:
    stem, dot, ext = path.rpartition(".")
    out = [path]
    if not dot:
        return out + [path + ".dds", path + ".png", path + ".cdae"]
    ext = ext.lower()
    if ext in ("png", "jpg", "jpeg", "tga", "bmp"):
        out.append(stem + ".dds")
    elif ext == "dae":
        out.insert(0, stem + ".cdae")
    elif ext == "dds":
        out += [stem + ".png"]
    return out


# --------------------------------------------------------------------------
# Scene files


def numbers(value) -> list[float]:
    """A vector however it was written: a list, or a TorqueScript string."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                return []
        return out
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(v) for v in _NUMBER.findall(str(value))]


def json_objects(text: str) -> list[dict]:
    """Objects from a scene file: one per line, or one JSON array or map."""
    objects: list[dict] = []
    lines = [line for line in text.splitlines() if line.strip()]
    try:
        for line in lines:
            value = json.loads(line)
            if isinstance(value, dict):
                objects.append(value)
        return objects
    except ValueError:
        pass
    try:
        value = json.loads(text)
    except ValueError:
        return []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, dict):
        if "class" in value:
            return [value]
        return [v for v in value.values() if isinstance(v, dict)]
    return []


_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|//[^\n]*|[A-Za-z_$%][\w$:%]*(?:\[\d+\])?|[{}();=]|[^\s{}();="]+')


def torque_objects(text: str) -> list[dict]:
    """
    The objects in a TorqueScript `.prefab` or `.mis`, as nested dicts.

    Only the object syntax - `new Class(name) { field = "value"; ... };` -
    which is all these files hold. Children sit under `"children"`.
    """
    tokens = [t for t in _TOKEN.findall(text) if not t.startswith("//")]
    at = 0

    def parse_object() -> dict | None:
        nonlocal at
        # at points just after `new`
        if at >= len(tokens):
            return None
        obj: dict = {"class": tokens[at], "children": []}
        at += 1
        if at < len(tokens) and tokens[at] == "(":
            at += 1
            name = []
            while at < len(tokens) and tokens[at] != ")":
                name.append(tokens[at])
                at += 1
            at += 1
            if name and name[0] not in (":",):
                obj["name"] = "".join(name).strip('"')
        if at < len(tokens) and tokens[at] == "{":
            at += 1
            while at < len(tokens) and tokens[at] != "}":
                if tokens[at] == "new":
                    at += 1
                    child = parse_object()
                    if child is not None:
                        obj["children"].append(child)
                    continue
                if at + 2 < len(tokens) and tokens[at + 1] == "=":
                    key, value = tokens[at], tokens[at + 2]
                    obj[key] = value[1:-1] if value.startswith('"') else value
                    at += 3
                    continue
                at += 1
            at += 1
        while at < len(tokens) and tokens[at] == ";":
            at += 1
        return obj

    objects = []
    while at < len(tokens):
        if tokens[at] == "new":
            at += 1
            obj = parse_object()
            if obj is not None:
                objects.append(obj)
        else:
            at += 1
    return objects


def transform_of(obj: dict) -> np.ndarray:
    """
    An object's placement as a 4x4 for row vectors: `world = [p, 1] @ M`.

    Rows 0-2 are its own axes scaled, row 3 its position.
    """
    matrix = np.eye(4)
    rows = numbers(obj.get("rotationMatrix"))
    if len(rows) == 9:
        matrix[:3, :3] = np.array(rows).reshape(3, 3)
    else:
        turn = numbers(obj.get("rotation"))
        if len(turn) == 4 and np.linalg.norm(turn[:3]) > 1e-9:
            matrix[:3, :3] = axis_angle(turn[:3], turn[3])
    scale = numbers(obj.get("scale"))
    if len(scale) == 1:
        scale = scale * 3
    if len(scale) == 3:
        matrix[:3, :3] *= np.array(scale)[:, None]
    position = numbers(obj.get("position") or obj.get("pos"))
    if len(position) == 3:
        matrix[3, :3] = position
    return matrix


def axis_angle(axis, degrees: float) -> np.ndarray:
    """Rows are the turned axes: a right-handed turn by `degrees` about `axis`."""
    u = np.asarray(axis, dtype=np.float64)
    u = u / np.linalg.norm(u)
    a = np.radians(degrees)
    k = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    standard = np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)
    return standard


# --------------------------------------------------------------------------
# Materials and textures


@dataclass
class Material:
    """What a surface looks like, as much of it as a course needs."""

    name: str
    colour_map: str | None = None
    colour: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    opacity_map: str | None = None
    opacity: float = 1.0
    cutout: bool = False
    annotation: str = ""
    ground: str = ""
    defined_in: str = ""


@dataclass
class TerrainMaterial:
    """
    A terrain layer's look. The colour is the base map, stretched over
    `base_size` metres - the whole terrain unless the material says otherwise,
    image row 0 at its northern edge (its green follows the grass layers:
    correlation 0.72 and 0.71 on two levels, near zero turned any other way).
    The detail and macro maps are grey overlays, 0.5 neutral, repeating every
    `detail_size` and `macro_size` metres and mixed in by their strengths.
    """

    name: str
    detail_map: str | None = None
    detail_size: float = 2.0
    detail_strength: float = 0.5
    base_map: str | None = None
    base_size: float | None = None
    macro_map: str | None = None
    macro_size: float = 60.0
    macro_strength: float = 0.3
    ground: str = ""
    annotation: str = ""
    defined_in: str = ""


def _colour(value) -> tuple[float, float, float, float] | None:
    v = numbers(value)
    if len(v) == 3:
        v = v + [1.0]
    if len(v) != 4:
        return None
    if max(v) > 1.5:  # 0..255
        v = [x / 255.0 for x in v]
    return tuple(float(x) for x in v)  # type: ignore[return-value]


def _stage_value(record: dict, *keys):
    """A value from the first stage, or from the old top-level arrays."""
    stages = record.get("Stages")
    if isinstance(stages, list) and stages and isinstance(stages[0], dict):
        for key in keys:
            if stages[0].get(key) not in (None, ""):
                return stages[0][key]
    for key in keys:
        value = record.get(key)
        if isinstance(value, list):
            value = value[0] if value else None
        if value not in (None, ""):
            return value
    return None


class Materials:
    """Every material a level can use, by the name its models give."""

    def __init__(self, files: Files, level: str) -> None:
        self.files = files
        self.by_name: dict[str, Material] = {}
        self.terrain: dict[str, TerrainMaterial] = {}
        level_root = f"levels/{level.lower()}/"
        sources = [k for k in files.names(suffix="materials.json") if not k.startswith("levels/")]
        sources += files.names(prefix=level_root, suffix="materials.json")
        for source in sources:
            try:
                records = json.loads(files.read(source))
            except (ValueError, OSError):
                continue
            if not isinstance(records, dict):
                continue
            for key, record in records.items():
                if isinstance(record, dict):
                    self._add(key, record, source)

    def _add(self, key: str, record: dict, source: str) -> None:
        kind = record.get("class")
        if kind == "TerrainMaterial":
            name = str(record.get("internalName") or record.get("name") or key)
            detail = _stage_value(record, "baseColorDetailTex", "detailMap")
            size = numbers(record.get("baseColorDetailTexSize") or record.get("detailSize"))
            macro_size = numbers(record.get("baseColorMacroTexSize") or record.get("macroSize"))
            base_size = numbers(record.get("diffuseSize"))
            detail_strength = numbers(record.get("baseColorDetailStrength") or record.get("detailStrength"))
            macro_strength = numbers(record.get("baseColorMacroStrength") or record.get("macroStrength"))
            self.terrain[name.lower()] = TerrainMaterial(
                name=name,
                detail_map=detail,
                detail_size=float(size[0]) if size and size[0] > 0 else 2.0,
                detail_strength=float(detail_strength[0]) if detail_strength else 0.5,
                base_map=_stage_value(record, "baseColorBaseTex", "diffuseMap"),
                base_size=float(base_size[0]) if base_size and base_size[0] > 0 else None,
                macro_map=_stage_value(record, "baseColorMacroTex", "macroMap"),
                macro_size=float(macro_size[0]) if macro_size and macro_size[0] > 0 else 60.0,
                macro_strength=float(macro_strength[0]) if macro_strength else 0.3,
                ground=str(record.get("groundmodelName") or ""),
                annotation=str(record.get("annotation") or ""),
                defined_in=source,
            )
            return
        if kind not in (None, "Material", "CustomMaterial"):
            return
        colour = _colour(_stage_value(record, "baseColorFactor", "diffuseColor")) or (1.0, 1.0, 1.0, 1.0)
        opacity = numbers(_stage_value(record, "opacityFactor"))
        material = Material(
            name=str(record.get("name") or key),
            colour_map=_stage_value(record, "baseColorMap", "colorMap", "diffuseMap"),
            colour=colour,
            opacity_map=_stage_value(record, "opacityMap"),
            opacity=float(opacity[0]) if opacity else 1.0,
            cutout=bool(record.get("alphaTest")) or bool(record.get("translucent")),
            annotation=str(record.get("annotation") or ""),
            ground=str(record.get("groundType") or ""),
            defined_in=source,
        )
        for name in {str(record.get("mapTo") or ""), str(record.get("name") or ""), key}:
            if name:
                self.by_name[name.lower()] = material

    def get(self, name: str) -> Material | None:
        return self.by_name.get((name or "").lower())

    def texture(self, material: Material | TerrainMaterial | None, which: str = "colour") -> str | None:
        """The resolved path of a material's texture, or None."""
        if material is None:
            return None
        if isinstance(material, TerrainMaterial):
            wanted = {"detail": material.detail_map, "base": material.base_map, "macro": material.macro_map}[which]
        else:
            wanted = material.colour_map if which == "colour" else material.opacity_map
        return resolve(self.files, wanted, material.defined_in)


class Textures:
    """Decoded images, each read once, shrunk to what an export can use."""

    def __init__(self, files: Files, largest: int = 1024) -> None:
        self.files = files
        self.largest = largest
        self._cache: dict[tuple[str, int], np.ndarray | None] = {}

    def rgba(self, path: str | None, largest: int | None = None) -> np.ndarray | None:
        """An 8-bit RGBA image, (height, width, 4), or None if unreadable."""
        if not path:
            return None
        largest = largest or self.largest
        key = (path.lower(), largest)
        if key in self._cache:
            return self._cache[key]
        image = None
        try:
            import io

            from PIL import Image

            with Image.open(io.BytesIO(self.files.read(path))) as im:
                im.draft("RGBA", (largest, largest))
                im = im.convert("RGBA")
                if max(im.size) > largest:
                    ratio = largest / max(im.size)
                    im = im.resize((max(1, round(im.size[0] * ratio)), max(1, round(im.size[1] * ratio))), Image.BILINEAR)
                image = np.asarray(im, dtype=np.uint8).copy()
        except Exception:  # noqa: BLE001 - an unreadable texture falls back to flat colour
            image = None
        self._cache[key] = image
        return image

    def mean(self, path: str | None) -> tuple[float, float, float] | None:
        """
        A texture's average colour, weighted by its alpha, as the image stores
        it (sRGB). The viewer draws a vertex colour exactly as it draws a
        texel, so a flat part in its texture's average matches its textured
        neighbours only in the same encoding; linearised, it came out at a
        third of the brightness.
        """
        image = self.rgba(path, 64)
        if image is None:
            return None
        rgb = image[..., :3].astype(np.float64) / 255.0
        alpha = image[..., 3:4].astype(np.float64) / 255.0
        weight = alpha.sum()
        mean = (rgb * alpha).sum(axis=(0, 1)) / weight if weight > 1e-6 else rgb.mean(axis=(0, 1))
        return tuple(float(c) for c in mean)  # type: ignore[return-value]


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# --------------------------------------------------------------------------
# The level


@dataclass
class Placed:
    """One copy of a model in the world."""

    shape: str
    matrix: np.ndarray
    annotation: str = ""
    #: `static`, `prefab` or `forest`: where it was placed from.
    source: str = "static"


@dataclass
class DecalRoad:
    material: str
    #: (n, 4): x, y, z, width.
    nodes: np.ndarray
    texture_length: float
    priority: float
    order: int
    looped: bool = False
    #: Painted over models too, not only the terrain.
    over_objects: bool = False


@dataclass
class Terrain:
    position: tuple[float, float, float]
    square: float
    max_height: float
    size: int
    heights: np.ndarray
    layers: np.ndarray
    names: list[str]

    @property
    def extent(self) -> tuple[tuple[float, float], tuple[float, float]]:
        x, y, _ = self.position
        span = (self.size - 1) * self.square
        return (x, y), (x + span, y + span)

    def height(self, x, y) -> np.ndarray:
        """Heights at world points, bilinear; NaN off the terrain."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        gx = (x - self.position[0]) / self.square
        gy = (y - self.position[1]) / self.square
        inside = (gx >= 0) & (gy >= 0) & (gx <= self.size - 1) & (gy <= self.size - 1)
        gx = np.clip(gx, 0, self.size - 1.000001)
        gy = np.clip(gy, 0, self.size - 1.000001)
        i0 = np.floor(gx).astype(np.int64)
        j0 = np.floor(gy).astype(np.int64)
        fx = gx - i0
        fy = gy - j0
        h = self.heights
        top = h[j0, i0] * (1 - fx) + h[j0, i0 + 1] * fx
        bottom = h[j0 + 1, i0] * (1 - fx) + h[j0 + 1, i0 + 1] * fx
        value = self.position[2] + (top * (1 - fy) + bottom * fy) * (self.max_height / 65535.0)
        return np.where(inside, value, np.nan)


@dataclass
class Water:
    kind: str
    #: For a plane, the height. For a block, its transform; for a river, nodes.
    height: float | None = None
    matrix: np.ndarray | None = None
    nodes: np.ndarray | None = None


@dataclass
class Level:
    name: str
    title: str
    files: Files
    placed: list[Placed] = field(default_factory=list)
    decals: list[DecalRoad] = field(default_factory=list)
    terrains: list[Terrain] = field(default_factory=list)
    water: list[Water] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def root(self) -> str:
        return f"levels/{self.name}"


def list_levels(files: Files) -> list[str]:
    """Every level with a scene, by folder name."""
    names: dict[str, str] = {}
    for key in files.names(prefix="levels/"):
        parts = key.split("/")
        if len(parts) < 3 or parts[1] in names:
            continue
        if parts[2] == "main" or key.endswith((".level.json", ".mis")):
            # The name as the archive spells it, for messages and file names;
            # a mod folder stores an absolute path, so look for the segment.
            real = (files.real(key) or key).replace("\\", "/")
            found = re.search(r"(?:^|/)levels/([^/]+)/", real, re.I)
            names[parts[1]] = found.group(1) if found else parts[1]
    return sorted(names.values(), key=str.lower)


def scene_files(files: Files, level: str) -> list[str]:
    root = f"levels/{level.lower()}/"
    found = [k for k in files.names(prefix=root + "main/", suffix=".level.json")]
    found += [k for k in files.names(prefix=root, suffix="main.level.json") if k.count("/") == 3]
    if not found:
        found = [k for k in files.names(prefix=root, suffix=".mis") if k.count("/") == 2]
    return sorted(set(found))


def terrain_blocks(files: Files, level: str) -> list[dict]:
    """The `TerrainBlock` objects of a level, without reading the rest."""
    blocks = []
    for source in scene_files(files, level):
        text = files.read(source).decode("utf-8", "replace")
        if "TerrainBlock" not in text:
            continue
        objects = torque_objects(text) if source.endswith(".mis") else json_objects(text)
        for obj in _flatten(objects):
            if obj.get("class") == "TerrainBlock":
                blocks.append(obj)
    return blocks


def _flatten(objects: list[dict]):
    for obj in objects:
        yield obj
        children = obj.get("children")
        if children:
            yield from _flatten(children)


def read_terrain(files: Files, block: dict) -> Terrain | None:
    """A `TerrainBlock`'s heights and layers."""
    source = resolve(files, str(block.get("terrainFile") or ""))
    if not source:
        return None
    data = files.read(source)
    if len(data) < 5:
        return None
    size = int.from_bytes(data[1:5], "little")
    cells = size * size
    if size <= 1 or len(data) < 5 + cells * 3:
        return None
    heights = np.frombuffer(data, dtype="<u2", count=cells, offset=5).reshape(size, size).astype(np.float32)
    layers = np.frombuffer(data, dtype=np.uint8, count=cells, offset=5 + cells * 2).reshape(size, size)
    names: list[str] = []
    at = 5 + cells * 3
    if at + 4 <= len(data):
        count = int.from_bytes(data[at : at + 4], "little")
        at += 4
        for _ in range(count):
            if at >= len(data):
                break
            length = data[at]
            names.append(data[at + 1 : at + 1 + length].decode("latin-1"))
            at += 1 + length
    position = numbers(block.get("position")) or [0.0, 0.0, 0.0]
    square = numbers(block.get("squareSize")) or [1.0]
    height = numbers(block.get("maxHeight")) or [2048.0]
    return Terrain(
        position=(float(position[0]), float(position[1]), float(position[2])),
        square=float(square[0]),
        max_height=float(height[0]),
        size=size,
        heights=heights,
        layers=layers,
        names=names,
    )


def open_level(files: Files, name: str, *, terrain: bool = True) -> Level:
    """Everything of a level a course export draws."""
    real = next((n for n in list_levels(files) if n.lower() == name.lower()), None)
    if real is None:
        raise BeamNGError(f"no level called {name!r}; installed: {', '.join(list_levels(files))}")
    title = real
    info = f"levels/{real}/info.json"
    if info in files:
        try:
            title = str(json.loads(files.read(info)).get("title") or real)
        except (ValueError, OSError):
            pass
        # Some titles are keys into the game's translations, not names.
        if re.fullmatch(r"levels\.[\w.]+\.title", title):
            title = real
    level = Level(name=real, title=title, files=files)
    sources = scene_files(files, real)
    if not sources:
        raise BeamNGError(f"{real} has no scene (no main/*.level.json and no .mis)")
    order = 0
    for source in sources:
        text = files.read(source).decode("utf-8", "replace")
        objects = torque_objects(text) if source.endswith(".mis") else json_objects(text)
        for obj in _flatten(objects):
            order = _take(level, obj, np.eye(4), source, order, depth=0)
    _forest(level)
    if terrain:
        for block in terrain_blocks(files, real):
            found = read_terrain(files, block)
            if found is not None:
                level.terrains.append(found)
    return level


def _take(level: Level, obj: dict, parent: np.ndarray, source: str, order: int, depth: int) -> int:
    kind = obj.get("class")
    if kind == "TSStatic":
        shape = obj.get("shapeName")
        if shape and str(obj.get("hidden", "")).lower() not in ("1", "true"):
            level.placed.append(
                Placed(
                    shape=str(shape),
                    matrix=transform_of(obj) @ parent,
                    annotation=str(obj.get("annotation") or ""),
                    source="prefab" if depth else "static",
                )
            )
    elif kind == "Prefab" and depth < 8:
        path = resolve(level.files, str(obj.get("filename") or obj.get("fileName") or ""), source)
        if path is None:
            level.skipped["prefab files missing"] = level.skipped.get("prefab files missing", 0) + 1
            return order
        matrix = transform_of(obj) @ parent
        text = level.files.read(path).decode("utf-8", "replace")
        children = torque_objects(text) if not path.lower().endswith(".json") else json_objects(text)
        for child in _flatten(children):
            order = _take(level, child, matrix, path, order, depth + 1)
    elif kind == "DecalRoad":
        nodes = [numbers(n) for n in obj.get("nodes") or []]
        nodes = [n[:4] for n in nodes if len(n) >= 4]
        if len(nodes) >= 2 and not depth:
            level.decals.append(
                DecalRoad(
                    material=str(obj.get("material") or ""),
                    nodes=np.array(nodes, dtype=np.float64),
                    texture_length=float((numbers(obj.get("textureLength")) or [5.0])[0]) or 5.0,
                    priority=float((numbers(obj.get("renderPriority")) or [10.0])[0]),
                    order=order,
                    looped=str(obj.get("looped", "")).lower() in ("1", "true"),
                    over_objects=str(obj.get("overObjects", "")).lower() in ("1", "true"),
                )
            )
            order += 1
        elif len(nodes) >= 2:
            # A road inside a prefab: its nodes are the prefab's.
            points = np.array(nodes, dtype=np.float64)
            points[:, :3] = np.c_[points[:, :3], np.ones(len(points))] @ parent[:, :3]
            level.decals.append(
                DecalRoad(
                    material=str(obj.get("material") or ""),
                    nodes=points,
                    texture_length=float((numbers(obj.get("textureLength")) or [5.0])[0]) or 5.0,
                    priority=float((numbers(obj.get("renderPriority")) or [10.0])[0]),
                    order=order,
                )
            )
            order += 1
    elif kind == "WaterPlane":
        position = numbers(obj.get("position"))
        if len(position) == 3:
            level.water.append(Water("plane", height=position[2]))
    elif kind == "WaterBlock":
        level.water.append(Water("block", matrix=transform_of(obj) @ parent))
    elif kind == "River":
        nodes = [numbers(n) for n in obj.get("nodes") or []]
        nodes = [n[:5] for n in nodes if len(n) >= 4]
        if len(nodes) >= 2:
            level.water.append(Water("river", nodes=np.array([n + [0.0] * (5 - len(n)) for n in nodes])))
    return order


def _forest(level: Level) -> None:
    """Forest instances, each typed by its item record's shape."""
    files = level.files
    kinds: dict[str, tuple[str, str]] = {}
    sources = [k for k in files.names(suffix="manageditemdata.json") if not k.startswith("levels/")]
    sources += files.names(prefix=f"{level.root.lower()}/", suffix="manageditemdata.json")
    for source in sources:
        try:
            records = json.loads(files.read(source))
        except (ValueError, OSError):
            continue
        if not isinstance(records, dict):
            continue
        for key, record in records.items():
            if not isinstance(record, dict) or not record.get("shapeFile"):
                continue
            for name in {key, str(record.get("name") or ""), str(record.get("internalName") or "")}:
                if name:
                    kinds[name.lower()] = (str(record["shapeFile"]), source)
    missing = 0
    for source in files.names(prefix=f"{level.root.lower()}/", suffix=".forest4.json"):
        for obj in json_objects(files.read(source).decode("utf-8", "replace")):
            kind = kinds.get(str(obj.get("type") or "").lower())
            if kind is None:
                missing += 1
                continue
            level.placed.append(Placed(shape=kind[0], matrix=transform_of(obj), source="forest"))
    if missing:
        level.skipped["forest items of an unknown type"] = missing
