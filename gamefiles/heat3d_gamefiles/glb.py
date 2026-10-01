"""
Writing a mesh out as GLB.

Deliberately small. The reconstruction exporter next door has a fuller writer,
and this is not a second copy of it: that one is coupled to a reconstruction's
output and to a package that pulls in torch and Qt, and this package exists to be
the light half that only reads game files.

What is here writes positions, normals, per-vertex colour, UVs, textures and any
number of named parts. Each of those earns its place against something an earlier
version of this file got wrong:

- **Normals**, because without them a viewer shades flat and a whole racetrack
  arrives as one featureless grey shell. Taken from the archive where the game
  stores them, and computed from the triangles where it does not.
- **Parts**, because the node name is where the viewer takes a surface's class
  from — it reads the *node*, not the mesh, and only when
  it matches ``^(ground|structure|water)\\s*[:|]``.
- **Colour**, because parts alone are a list in a panel.
- **Textures**, because colour per vertex is colour per metre on open ground, and
  a kerb, a verge or a line of trees is finer than that. A part may carry an
  image; its primitive then carries UVs and no vertex colour, since glTF
  multiplies the two and a surface would come out darkened twice.

Coordinates are left exactly as the game had them: Y up, metres, world origin.
That is the whole point of reading the archive rather than reconstructing — the
numbers mean something, and rescaling them here would throw that away.
"""

from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_FLOAT = 5126
_UNSIGNED_INT = 5125
_UNSIGNED_SHORT = 5123
_UNSIGNED_BYTE = 5121
_BYTE = 5120
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_LINEAR = 9729
_LINEAR_MIPMAP_LINEAR = 9987
_CLAMP_TO_EDGE = 33071

_MAGIC = 0x46546C67
_JSON = 0x4E4F534A
_BIN = 0x004E4942

#: JPEG rather than PNG for the images. A 2048-pixel ground texture of grass and
#: tree crowns is noise to a lossless coder - several megabytes a tile as PNG,
#: under one as JPEG at a quality where the difference cannot be seen.
JPEG_QUALITY = 90


@dataclass
class Part:
    """One named run of triangles, sharing the file's vertices."""

    name: str
    faces: np.ndarray
    #: Linear RGB applied to every vertex this part uses. Parts may overlap in
    #: their vertices; the last one written wins, which is why the caller should
    #: pass them in order of increasing importance.
    colour: tuple[float, float, float] | None = None
    #: An 8-bit sRGB image, (height, width, 3). The part's primitive is drawn
    #: with it through the file's UVs instead of through vertex colour.
    texture: np.ndarray | None = None


def compute_normals(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Smooth normals, area-weighted by construction.

    The cross product of two triangle edges is twice the triangle's area, so
    accumulating it unnormalised weights each face by its size — which is what
    keeps a dense strip of road from being shouted down by the huge triangles of
    the field it runs through.
    """
    normals = np.zeros(positions.shape, dtype=np.float64)
    corners = positions[faces]
    face = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    for column in range(3):
        np.add.at(normals, faces[:, column], face)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    # A vertex no triangle uses, or one whose triangles cancel out exactly.
    # Pointing it up is arbitrary but finite, which is what matters.
    flat = length[:, 0] < 1e-12
    normals[flat] = (0.0, 1.0, 0.0)
    length[flat] = 1.0
    return (normals / length).astype(np.float32)


def _jpeg(image: np.ndarray) -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)).save(
        out, format="JPEG", quality=JPEG_QUALITY, subsampling=0
    )
    return out.getvalue()


def _normals_bytes(normals: np.ndarray) -> bytes:
    """Unit normals as signed bytes, four to a vertex: x, y, z and padding."""
    out = np.zeros((len(normals), 4), dtype=np.int8)
    out[:, :3] = np.clip(np.round(normals * 127.0), -127, 127)
    return out.tobytes()


def _colours_bytes(colours: np.ndarray) -> bytes:
    """Colours 0..1 as unsigned bytes, four to a vertex: r, g, b and padding."""
    out = np.zeros((len(colours), 4), dtype=np.uint8)
    out[:, :3] = np.clip(np.round(colours * 255.0), 0, 255)
    return out.tobytes()


def _uv_bytes(uvs: np.ndarray) -> bytes:
    """UVs 0..1 as unsigned shorts: a step of 1/65535, a fifteenth of a texel of 4096."""
    return np.round(np.clip(uvs, 0.0, 1.0) * 65535.0).astype("<u2").tobytes()


def write_glb(
    path: str | Path,
    positions: np.ndarray,
    faces: np.ndarray | None = None,
    *,
    name: str = "ground:terrain",
    parts: list[Part] | None = None,
    colours: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    uvs: np.ndarray | None = None,
    quantize: bool = True,
) -> Path:
    """
    Write a mesh, as one part or as several.

    With `quantize`, the default, normals are written as signed bytes
    (`KHR_mesh_quantization`), UVs as normalised shorts and colours as
    normalised bytes (both core glTF), and each part's indices as shorts where
    it has few enough vertices. Positions stay floats. A vertex goes from 32
    bytes, or 44 with colour, to 20 or 24: a city course with fifteen million
    placed triangles was a gigabyte, at the edge of what a browser tab will
    load. The normals lose under half a degree, the UVs a fifteenth of a texel.

    Raises rather than writing something a viewer would open and show wrongly:
    an index past the end of the vertex list renders as noise, a textured part
    with no UVs renders as one smeared pixel, and a file with no triangles
    renders as nothing at all.
    """
    path = Path(path)
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be N by 3, not {positions.shape}")
    if not np.isfinite(positions).all():
        raise ValueError("the mesh has vertices that are not finite")

    if parts is None:
        if faces is None:
            raise ValueError("give either faces or parts")
        parts = [Part(name=name, faces=faces)]
    if not parts:
        raise ValueError("nothing to write: no parts")

    checked: list[Part] = []
    for part in parts:
        block = np.ascontiguousarray(part.faces, dtype=np.uint32)
        if block.ndim != 2 or block.shape[1] != 3:
            raise ValueError(f"{part.name}: faces must be M by 3, not {block.shape}")
        if len(block) and block.max() >= len(positions):
            raise ValueError(
                f"{part.name}: an index reaches vertex {block.max()} of {len(positions)}"
            )
        if part.texture is not None and uvs is None:
            raise ValueError(f"{part.name} has a texture and the file has no UVs")
        if len(block):
            checked.append(
                Part(name=part.name, faces=block, colour=part.colour, texture=part.texture)
            )
    if not checked:
        raise ValueError("nothing to write: the mesh has no triangles")

    every = np.concatenate([p.faces for p in checked])
    if normals is None:
        normals = compute_normals(positions.astype(np.float64), every.astype(np.int64))
    normals = np.ascontiguousarray(normals, dtype=np.float32)
    if normals.shape != positions.shape or not np.isfinite(normals).all():
        raise ValueError("one finite normal per vertex is needed")

    untextured = [p for p in checked if p.texture is None]
    if colours is None and any(p.colour is not None for p in untextured):
        colours = np.full(positions.shape, 0.5, dtype=np.float32)
        for part in untextured:
            if part.colour is not None:
                colours[np.unique(part.faces)] = part.colour
    if colours is not None:
        colours = np.ascontiguousarray(colours, dtype=np.float32)
        if colours.shape != positions.shape:
            raise ValueError("one colour per vertex is needed")
    if uvs is not None:
        uvs = np.ascontiguousarray(uvs, dtype=np.float32)
        if uvs.shape != (len(positions), 2) or not np.isfinite(uvs).all():
            raise ValueError("one finite UV pair per vertex is needed")

    binary = bytearray()
    views: list[dict] = []

    def add_view(data: bytes, target: int | None, stride: int | None = None) -> int:
        binary.extend(b"\x00" * ((-len(binary)) % 4))
        view = {"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        if stride is not None:
            # Three bytes a vertex padded to four: vertex attributes have to be
            # four-byte aligned, and the stride is how a reader knows.
            view["byteStride"] = stride
        binary.extend(data)
        views.append(view)
        return len(views) - 1

    accessors: list[dict] = []

    def add_accessor(view: int, count: int, kind: str, component: int, **extra) -> int:
        accessors.append(
            {"bufferView": view, "componentType": component, "count": count, "type": kind, **extra}
        )
        return len(accessors) - 1

    # Each part gets its own compact vertex arrays: only the vertices it uses,
    # renumbered. Sharing one array across every part is legal glTF and is what
    # this wrote first - and a reader that expands each primitive's accessors,
    # which the viewer does, then holds every vertex once per part: a five-part
    # circuit arrived as 1.44 million vertices from 287 thousand.
    images: list[dict] = []
    textures: list[dict] = []
    materials: list[dict] = []
    meshes: list[dict] = []
    nodes: list[dict] = []
    image_of: dict[int, int] = {}
    for index, part in enumerate(checked):
        used, local = np.unique(part.faces, return_inverse=True)
        local = local.reshape(-1, 3).astype(np.uint32)
        points = positions[used]
        attributes = {
            "POSITION": add_accessor(
                add_view(points.tobytes(), _ARRAY_BUFFER),
                len(points),
                "VEC3",
                _FLOAT,
                min=points.min(axis=0).tolist(),
                max=points.max(axis=0).tolist(),
            ),
            "NORMAL": (
                add_accessor(
                    add_view(_normals_bytes(normals[used]), _ARRAY_BUFFER, stride=4),
                    len(used), "VEC3", _BYTE, normalized=True,
                )
                if quantize
                else add_accessor(
                    add_view(normals[used].tobytes(), _ARRAY_BUFFER), len(used), "VEC3", _FLOAT
                )
            ),
        }
        if part.texture is not None:
            key = id(part.texture)
            if key not in image_of:
                images.append(
                    {"bufferView": add_view(_jpeg(part.texture), None), "mimeType": "image/jpeg"}
                )
                textures.append({"source": len(images) - 1, "sampler": 0})
                image_of[key] = len(textures) - 1
            attributes["TEXCOORD_0"] = (
                add_accessor(
                    add_view(_uv_bytes(uvs[used]), _ARRAY_BUFFER),
                    len(used), "VEC2", _UNSIGNED_SHORT, normalized=True,
                )
                if quantize
                else add_accessor(
                    add_view(uvs[used].tobytes(), _ARRAY_BUFFER), len(used), "VEC2", _FLOAT
                )
            )
            material = {
                "name": part.name,
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": image_of[key]},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
            }
        else:
            if colours is not None:
                attributes["COLOR_0"] = (
                    add_accessor(
                        add_view(_colours_bytes(colours[used]), _ARRAY_BUFFER, stride=4),
                        len(used), "VEC3", _UNSIGNED_BYTE, normalized=True,
                    )
                    if quantize
                    else add_accessor(
                        add_view(colours[used].tobytes(), _ARRAY_BUFFER), len(used), "VEC3", _FLOAT
                    )
                )
            material = {
                "name": part.name,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [*(part.colour or (1.0, 1.0, 1.0)), 1.0]
                    if colours is None
                    else [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
            }
        materials.append(material)
        short = quantize and len(used) <= 65535
        indices_at = add_accessor(
            add_view(
                local.astype("<u2" if short else "<u4").tobytes(), _ELEMENT_ARRAY_BUFFER
            ),
            int(local.size),
            "SCALAR",
            _UNSIGNED_SHORT if short else _UNSIGNED_INT,
        )
        meshes.append(
            {
                "name": part.name,
                "primitives": [
                    {"attributes": attributes, "indices": indices_at, "material": index}
                ],
            }
        )
        nodes.append({"name": part.name, "mesh": index})

    document = {
        "asset": {"version": "2.0", "generator": "heat3d_gamefiles"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
    }
    if quantize:
        # Byte normals are the one thing core glTF does not allow; the extension
        # is what says a reader has to understand them.
        document["extensionsUsed"] = ["KHR_mesh_quantization"]
        document["extensionsRequired"] = ["KHR_mesh_quantization"]
    if images:
        document["images"] = images
        document["textures"] = textures
        document["samplers"] = [
            {
                "magFilter": _LINEAR,
                "minFilter": _LINEAR_MIPMAP_LINEAR,
                "wrapS": _CLAMP_TO_EDGE,
                "wrapT": _CLAMP_TO_EDGE,
            }
        ]

    text = json.dumps(document, separators=(",", ":")).encode("utf-8")
    text += b" " * ((-len(text)) % 4)
    binary.extend(b"\x00" * ((-len(binary)) % 4))

    with path.open("wb") as fh:
        fh.write(struct.pack("<III", _MAGIC, 2, 12 + 8 + len(text) + 8 + len(binary)))
        fh.write(struct.pack("<II", len(text), _JSON))
        fh.write(text)
        fh.write(struct.pack("<II", len(binary), _BIN))
        fh.write(bytes(binary))
    return path
