"""
A BeamNG.drive install small enough to build in a test.

Written the way the game ships one - `gameengine.zip`, a `content` folder, one
zip per level - with every format the export reads in it: a terrain file,
scene files one JSON object a line, a TorqueScript prefab, forest instances,
materials, and compiled shapes written to BeamNG's documented `.cdae` layout.
The level is a 64 m square of grass with a strip of asphalt along x, a road
decal on it, a rail beside it, a prefab and a forest item, and the sea.
"""

from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np

LEVEL = "testlevel"
SIZE = 65
SQUARE = 1.0
ORIGIN = (-32.0, -32.0, 10.0)
MAX_HEIGHT = 50.0
#: The asphalt strip, in terrain rows: y from -4 to +4.
STRIP = (28, 36)


def _vector(values: np.ndarray | None, size: int) -> list:
    if values is None or not len(values):
        return [0, size, b""]
    data = np.ascontiguousarray(values).tobytes()
    return [len(values), size, data]


def cdae(
    positions,
    faces,
    *,
    uvs=None,
    materials=("mat",),
    material_of=None,
    node_rotation=(0.0, 0.0, 0.0, 1.0),
    node_translation=(0.0, 0.0, 0.0),
    levels=((2.0, None),),
    collision=None,
    compressed: bool = False,
) -> bytes:
    """
    A shape in BeamNG's `.cdae` layout: one object, one mesh per level of
    detail, each level's (size, faces) - `None` for the full `faces`.
    `node_rotation` is a quaternion (x, y, z, w) as Torque stores it.
    """
    import msgpack

    positions = np.asarray(positions, dtype=np.float32)
    names = ["base00", "start01", "obj2", "obj"] + [f"detail{int(s)}" for s, _ in levels]
    nodes = np.array([[0, -1, -1, 1, -1], [1, 0, -1, 2, -1], [2, 1, 0, -1, -1]], dtype="<i4")
    meshes = len(levels) + (1 if collision is not None else 0)
    objects = np.array([[3, meshes, 0, 2, -1, -1]], dtype="<i4")
    detail_rows = []
    for k, (size, _f) in enumerate(levels):
        detail_rows.append((4 + k, 0, k, size))
    if collision is not None:
        names.append("Collision-1")
        detail_rows.append((len(names) - 1, 0, len(levels), -1.0))
    detail = np.zeros(len(detail_rows), dtype=[
        ("name", "<i4"), ("subshape", "<i4"), ("od", "<i4"), ("size", "<f4"),
        ("a", "<f4"), ("b", "<f4"), ("polys", "<i4"), ("c", "<i4"), ("d", "<i4"),
        ("e", "<u4"), ("f", "<u4"), ("g", "<f4"), ("h", "<u4"),
    ])
    for i, (name, sub, od, size) in enumerate(detail_rows):
        detail[i]["name"], detail[i]["subshape"], detail[i]["od"], detail[i]["size"] = name, sub, od, size
    q = np.array(node_rotation, dtype=np.float64)
    q16 = np.round(q / np.linalg.norm(q) * 32767).astype("<i2")
    rotations = np.tile(np.array([0, 0, 0, 32767], dtype="<i2"), (3, 1))
    rotations[2] = q16
    translations = np.zeros((3, 3), dtype="<f4")
    translations[2] = node_translation
    stream = [10.0, 0, 1.0, 1.0, [0.0, 0.0, 0.0], [*positions.min(axis=0).tolist(), *positions.max(axis=0).tolist()]]
    vectors = {
        "nodes": _vector(nodes, 20), "objects": _vector(objects, 24),
        "subShapeFirstNode": _vector(np.array([0], "<i4"), 4), "subShapeFirstObject": _vector(np.array([0], "<i4"), 4),
        "subShapeNumNodes": _vector(np.array([3], "<i4"), 4), "subShapeNumObjects": _vector(np.array([1], "<i4"), 4),
        "defaultRotations": _vector(rotations, 8), "defaultTranslations": _vector(translations, 12),
    }
    for name in (
        "nodes", "objects", "subShapeFirstNode", "subShapeFirstObject", "subShapeNumNodes",
        "subShapeNumObjects", "defaultRotations", "defaultTranslations", "nodeRotations",
        "nodeTranslations", "nodeUniformScales", "nodeAlignedScales", "nodeArbitraryScaleFactors",
        "nodeArbitraryScaleRots", "groundTranslations", "groundRotations", "objectStates",
        "triggers", "details",
    ):
        if name == "details":
            stream.extend(_vector(detail, 52))
        else:
            stream.extend(vectors.get(name, [0, 4, b""]))
    stream.append(len(names))
    stream.extend(names)
    every = [f for _s, f in levels] + ([collision] if collision is not None else [])
    stream.append(len(every))
    for chosen in every:
        f = np.asarray(faces if chosen is None else chosen, dtype="<u4")
        slots = np.zeros(len(f), np.int64) if material_of is None else np.asarray(material_of)
        if chosen is not None:
            slots = np.zeros(len(f), np.int64)
        primitives, indices = [], []
        for slot in np.unique(slots):
            run = f[slots == slot].ravel()
            primitives.append((len(indices) and sum(len(i) for i in indices), len(run), 0x20000000 | int(slot)))
            indices.append(run)
        flat = np.concatenate(indices).astype("<u4")
        prim = np.array(primitives, dtype=[("s", "<i4"), ("n", "<i4"), ("m", "<u4")])
        stream += [0, 1, 1, -1, [*positions.min(axis=0).tolist(), *positions.max(axis=0).tolist()], [0.0, 0.0, 0.0], 1.0]
        stream.extend(_vector(positions, 12))
        stream.extend(_vector(None if uvs is None else np.asarray(uvs, "<f4"), 8))
        stream.extend([0, 8, b""])
        stream.extend([0, 4, b""])
        normals = np.tile(np.array([0, 0, 1], "<f4"), (len(positions), 1))
        stream.extend(_vector(normals, 12))
        stream.extend([0, 1, b""])
        stream.extend(_vector(prim, 12))
        stream.extend(_vector(flat, 4))
        stream.extend([0, 16, b""])
        stream += [len(positions), 0]
    stream.append(0)  # sequences
    stream.append(len(materials))
    for name in materials:
        stream += [name, 0, 0, 4294967295, 4294967295, 1.0, 1.0]
    body = b"".join(msgpack.packb(v, use_bin_type=True) for v in stream)
    header = {"info": "test", "compression": compressed, "bodysize": len(body)}
    if compressed:
        import zstandard

        body = zstandard.ZstdCompressor().compress(body)
    head = msgpack.packb(header, use_bin_type=True)
    return struct.pack("<II", 31, len(head)) + head + body


def png(colour, size=8, alpha=255) -> bytes:
    from PIL import Image

    image = Image.new("RGBA", (size, size), (*colour, alpha))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def terrain_file() -> tuple[bytes, np.ndarray]:
    """A gentle slope rising 0.1 m a metre towards +x, asphalt along the strip."""
    heights_m = np.zeros((SIZE, SIZE))
    for i in range(SIZE):
        heights_m[:, i] = 0.1 * i
    h = np.round(heights_m / MAX_HEIGHT * 65535).astype("<u2")
    layers = np.zeros((SIZE, SIZE), np.uint8)
    layers[STRIP[0] : STRIP[1], :] = 1
    names = [b"Grass", b"Asphalt"]
    tail = struct.pack("<I", len(names)) + b"".join(bytes([len(n)]) + n for n in names)
    return bytes([9]) + struct.pack("<I", SIZE) + h.tobytes() + layers.tobytes() + tail, heights_m


def rail_shape() -> bytes:
    """A 4 m long, 1 m high panel along local x, in its own texture."""
    positions = [[0, 0, 0], [4, 0, 0], [4, 0, 1], [0, 0, 1]]
    return cdae(positions, [[0, 1, 2], [0, 2, 3]], uvs=[[0, 1], [1, 1], [1, 0], [0, 0]], materials=("rail_mat",))


def track_shape() -> bytes:
    """A 10 m square of modelled asphalt, flat, its texture repeating."""
    positions = [[-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]]
    return cdae(positions, [[0, 1, 2], [0, 2, 3]], uvs=[[0, 0], [4, 0], [4, 4], [0, 4]], materials=("track_asphalt",))


def build_install(root: Path, *, sea: bool = True) -> Path:
    """The game folder: what `find_install` and `Files.of_install` read."""
    game = root / "BeamNG.drive"
    (game / "content" / "levels").mkdir(parents=True)
    with zipfile.ZipFile(game / "gameengine.zip", "w") as z:
        z.writestr("core/readme.txt", "engine")
    ter, _ = terrain_file()
    base = f"levels/{LEVEL}"
    scene = [
        {"class": "SimGroup", "name": "MissionGroup"},
        {
            "class": "TerrainBlock", "name": "theTerrain", "position": list(ORIGIN), "squareSize": SQUARE,
            "maxHeight": MAX_HEIGHT, "terrainFile": f"/{base}/theTerrain.ter",
        },
        {
            "class": "DecalRoad", "material": "test_road", "renderPriority": 12, "textureLength": 8,
            "position": [-20, 0, 12], "nodes": [[-20, 0, 12, 6], [0, 0, 13, 6], [20, 0, 14, 6]],
        },
        {
            "class": "DecalRoad", "material": "road_invisible", "position": [-30, 20, 12],
            "nodes": [[-30, 20, 12, 4], [30, 20, 15, 4]],
        },
        # A rail turned to run along +y: rows of the matrix are its own axes.
        {
            "class": "TSStatic", "shapeName": f"/{base}/art/shapes/rail.dae", "annotation": "GUARD_RAIL",
            "position": [5, 6, 10.5], "rotationMatrix": [0, 1, 0, -1, 0, 0, 0, 0, 1],
        },
        {"class": "Prefab", "filename": f"/{base}/art/prefabs/barriers.prefab", "position": [-10, -8, 0]},
        {"class": "TSStatic", "shapeName": f"/{base}/art/shapes/track.dae", "position": [10, 0, 11.6]},
    ]
    if sea:
        scene.append({"class": "WaterPlane", "position": [0, 0, 10.5]})
    prefab = """//--- OBJECT WRITE BEGIN ---
$ThisPrefab = new SimGroup() {
   groupPosition = "0 0 5";
   new TSStatic() {
      shapeName = "/levels/testlevel/art/shapes/rail.dae";
      position = "0 0 11";
      rotation = "0 0 1 90";
      scale = "1 1 2";
   };
};
"""
    materials = {
        "rail_mat": {"name": "rail_mat", "class": "Material", "Stages": [{"baseColorMap": "/levels/testlevel/art/shapes/rail_b.color.png"}]},
        "track_asphalt": {
            "name": "track_asphalt", "class": "Material", "groundType": "ASPHALT",
            "Stages": [{"baseColorMap": "/levels/testlevel/art/shapes/asphalt_b.color.png"}],
        },
        "rock_mat": {"name": "rock_mat", "class": "Material", "annotation": "ROCK", "Stages": [{"baseColorFactor": [0.4, 0.35, 0.3, 1]}]},
    }
    decal_materials = {
        "test_road": {
            "name": "test_road", "class": "Material",
            "Stages": [{"baseColorMap": "/levels/testlevel/art/road/road_b.color.png", "opacityMap": "/levels/testlevel/art/road/road_o.data.png"}],
        },
        "road_invisible": {"name": "road_invisible", "class": "Material", "Stages": [{"baseColorMap": "/levels/testlevel/art/road/invisible.png"}]},
    }
    terrain_materials = {
        "Grass-1": {
            "class": "TerrainMaterial", "internalName": "Grass", "groundmodelName": "GRASS",
            "baseColorBaseTex": "/levels/testlevel/art/terrains/base_b.png", "baseColorDetailTex": "/levels/testlevel/art/terrains/grass_b.png",
            "baseColorDetailTexSize": 2, "baseColorDetailStrength": [0.5, 0],
        },
        "Asphalt-2": {
            "class": "TerrainMaterial", "internalName": "Asphalt", "groundmodelName": "ASPHALT",
            "baseColorBaseTex": "/levels/testlevel/art/terrains/asphalt_base_b.png",
        },
    }
    forest_items = {"rock_a": {"name": "rock_a", "class": "TSForestItemData", "shapeFile": f"{base}/art/shapes/rock.dae"}}
    rock = cdae([[0, 0, 0], [2, 0, 0], [1, 2, 0], [1, 1, 1.5]], [[0, 1, 3], [1, 2, 3], [2, 0, 3]], materials=("rock_mat",))
    with zipfile.ZipFile(game / "content" / "levels" / f"{LEVEL}.zip", "w") as z:
        z.writestr(f"{base}/info.json", json.dumps({"title": "Test Level"}))
        z.writestr(f"{base}/main/MissionGroup/items.level.json", "\n".join(json.dumps(o) for o in scene) + "\n")
        z.writestr(f"{base}/theTerrain.ter", ter)
        z.writestr(f"{base}/art/terrains/main.materials.json", json.dumps(terrain_materials))
        z.writestr(f"{base}/art/terrains/base_b.png", png((70, 110, 50), 16))
        z.writestr(f"{base}/art/terrains/grass_b.png", png((128, 128, 128)))
        z.writestr(f"{base}/art/terrains/asphalt_base_b.png", png((60, 60, 62), 16))
        z.writestr(f"{base}/art/shapes/rail.dae", "<COLLADA/>")
        z.writestr(f"{base}/art/shapes/rail.cdae", rail_shape())
        z.writestr(f"{base}/art/shapes/track.cdae", track_shape())
        z.writestr(f"{base}/art/shapes/rock.cdae", rock)
        z.writestr(f"{base}/art/shapes/main.materials.json", json.dumps(materials))
        z.writestr(f"{base}/art/shapes/rail_b.color.png", png((200, 30, 30)))
        z.writestr(f"{base}/art/shapes/asphalt_b.color.png", png((90, 90, 95)))
        z.writestr(f"{base}/art/road/main.materials.json", json.dumps(decal_materials))
        z.writestr(f"{base}/art/road/road_b.color.png", png((40, 40, 40)))
        z.writestr(f"{base}/art/road/road_o.data.png", png((255, 255, 255)))
        z.writestr(f"{base}/art/road/invisible.png", png((0, 0, 0), alpha=0))
        z.writestr(f"{base}/art/prefabs/barriers.prefab", prefab)
        z.writestr(f"{base}/art/forest/managedItemData.json", json.dumps(forest_items))
        z.writestr(
            f"{base}/forest/rock_a.forest4.json",
            json.dumps({"type": "rock_a", "pos": [-15, 15, 12], "rotationMatrix": [1, 0, 0, 0, 1, 0, 0, 0, 1], "scale": 1}) + "\n",
        )
    return game


def lap_document(points_world: np.ndarray) -> dict:
    """A `heat3d-lap` document as the BeamNG recorder writes one: viewer frame."""
    viewer = np.stack([points_world[:, 0], points_world[:, 2], -points_world[:, 1]], axis=1)
    distance = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points_world, axis=0), axis=1))])
    return {
        "format": "heat3d-lap",
        "version": 1,
        "game": "beamng",
        "track": None,
        "lap": {"number": 1, "seconds": float(distance[-1] / 20), "complete": True},
        "channels": {
            "time": (distance / 20).tolist(),
            "distance": distance.tolist(),
            "x": viewer[:, 0].tolist(),
            "y": viewer[:, 1].tolist(),
            "z": viewer[:, 2].tolist(),
        },
    }


def road_lap() -> np.ndarray:
    """Down the asphalt strip, the car's body half a metre over the ground."""
    x = np.linspace(-25, 25, 120)
    z = ORIGIN[2] + 0.1 * (x - ORIGIN[0]) + 0.5
    return np.stack([x, np.zeros_like(x), z], axis=1)
