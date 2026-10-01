"""
GLB writing, against the contract the viewer actually enforces.

The other half of this is the viewer's model parser. Two details of it
decide the whole file layout, and both are easy to get wrong in a way that
produces a file which opens fine in Blender and arrives in the viewer as one
grey blob:

- A "part" is a **node**, and its label is taken from the *node* name first,
  falling back to the mesh name. Naming meshes and
  leaving nodes unnamed loses the classification.
- The label must match ``^(ground|structure|water)\\s*[:|]`` for the viewer to
  take it at its word instead of guessing from geometry. That is the difference
  between a level that renders with a solid ground and transparent structures,
  and one that is classified by heuristics built for Unreal landscape meshes.

Written by hand rather than through a glTF library on purpose. The output is a
narrow subset — one buffer, tightly packed, no animation, no skinning, no
scene graph to speak of — and a dependency that can write all of glTF would mean
carrying its version skew for the few hundred lines below. Reconstruction output
is also large enough that controlling the byte layout directly matters.

Coordinates are viewer space: **Y up**, right-handed, and by default scaled to
centimetres to match the Unreal exporter, so a capture and an engine export of
the same place load at the same size with the same camera defaults.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np

PartClass = Literal["ground", "structure", "water"]

# glTF component types, from the specification's table.
_FLOAT = 5126
_UNSIGNED_INT = 5125
_UNSIGNED_SHORT = 5123

# Buffer view targets.
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963

_GLB_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942

#: Metres to the viewer's working unit. The Unreal exporter emits centimetres and
#: the viewer's camera defaults, fly speed and far plane are all tuned for that,
#: so a reconstruction that came out in metres would load a hundred times too
#: small and frame as a dot.
METRES_TO_VIEWER = 100.0


@dataclass
class Part:
    """One named run of geometry, which becomes one node and one mesh."""

    #: Semantic class. Written into the node name as the viewer's prefix.
    cls: PartClass
    #: Human half of the label, after the colon. Free text.
    name: str
    #: (N, 3) float, metres, Y-up.
    positions: np.ndarray
    #: (M,) integer triangle indices into `positions`.
    indices: np.ndarray
    #: (N, 3) float unit normals. Computed from the triangles when omitted.
    normals: np.ndarray | None = None
    #: (N, 2) float texture coordinates. Required if `texture_png` is set.
    uvs: np.ndarray | None = None
    #: (N, 3) float vertex colours in 0..1, written as glTF `COLOR_0`.
    #:
    #: The viewer ignores these — it renders levels untextured on purpose, so the
    #: background does not compete with the heatmap it sits behind. They are
    #: written for everywhere else: Blender and every other glTF tool reads
    #: COLOR_0, and a reconstructed level is worth looking at outside this
    #: viewer too.
    colours: np.ndarray | None = None
    #: PNG bytes for this part's base colour texture.
    texture_png: bytes | None = None

    @property
    def label(self) -> str:
        return f"{self.cls}:{self.name}"


@dataclass
class _Bin:
    """The BIN chunk under construction, plus the views that point into it."""

    data: bytearray = field(default_factory=bytearray)
    views: list[dict] = field(default_factory=list)

    def add(self, raw: bytes, target: int | None) -> int:
        # Every view starts 4-byte aligned. Accessors inherit the view's offset,
        # and glTF requires each accessor to be aligned to its component size;
        # 4 satisfies every component type used here.
        while len(self.data) % 4:
            self.data.append(0)
        offset = len(self.data)
        self.data.extend(raw)
        view: dict = {"buffer": 0, "byteOffset": offset, "byteLength": len(raw)}
        if target is not None:
            view["target"] = target
        self.views.append(view)
        return len(self.views) - 1


def compute_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Area-weighted vertex normals.

    Area weighting rather than a plain average because reconstruction output has
    wildly uneven triangle sizes — marching cubes leaves slivers against large
    flat runs — and an unweighted average lets a cluster of tiny degenerate
    triangles outvote the large face they sit on. The viewer reads `upFacing`
    off these normals to decide what looks like ground, so a normal that tips
    the wrong way is not cosmetic: it reclassifies the surface.
    """
    positions = np.asarray(positions, dtype=np.float64)
    tri = np.asarray(indices, dtype=np.int64).reshape(-1, 3)

    a = positions[tri[:, 0]]
    b = positions[tri[:, 1]]
    c = positions[tri[:, 2]]
    # Cross product magnitude is twice the triangle area, so accumulating it
    # unnormalised weights each face by its size for free.
    face = np.cross(b - a, c - a)

    normals = np.zeros_like(positions)
    for col in range(3):
        np.add.at(normals, tri[:, col], face)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    # An isolated or fully degenerate vertex has no defined normal. Up is the
    # least destructive answer: it reads as floor rather than as a wall facing
    # nowhere, and the alternative (zero) is not a unit vector at all.
    degenerate = lengths[:, 0] < 1e-12
    normals[degenerate] = (0.0, 1.0, 0.0)
    lengths[degenerate] = 1.0
    return (normals / lengths).astype(np.float32)


def _validate(part: Part) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(part.positions, dtype=np.float32).reshape(-1, 3)
    indices = np.asarray(part.indices, dtype=np.int64).reshape(-1)

    if positions.shape[0] == 0:
        raise ValueError(f"part {part.label!r} has no vertices")
    if indices.size == 0 or indices.size % 3:
        raise ValueError(
            f"part {part.label!r} has {indices.size} indices, which is not whole triangles"
        )
    # Caught here rather than in the viewer, where an out-of-range index reads as
    # corrupt geometry with no indication of which part produced it.
    top = int(indices.max())
    if top >= positions.shape[0]:
        raise ValueError(
            f"part {part.label!r} indexes vertex {top} but only has {positions.shape[0]}"
        )
    if int(indices.min()) < 0:
        raise ValueError(f"part {part.label!r} has a negative index")
    if not np.isfinite(positions).all():
        raise ValueError(f"part {part.label!r} has non-finite positions")

    normals = (
        compute_normals(positions, indices)
        if part.normals is None
        else np.asarray(part.normals, dtype=np.float32).reshape(-1, 3)
    )
    if normals.shape != positions.shape:
        raise ValueError(
            f"part {part.label!r} has {normals.shape[0]} normals for {positions.shape[0]} vertices"
        )
    return positions, indices, normals


def write_glb(
    path: str | Path,
    parts: Sequence[Part],
    *,
    scale: float = METRES_TO_VIEWER,
    generator: str = "heat3d-capture",
) -> Path:
    """
    Write `parts` as a GLB the viewer will classify from its labels.

    Returns the path written, so a caller can log it without rebuilding it.
    """
    if not parts:
        raise ValueError("nothing to write: no parts")

    bin_ = _Bin()
    accessors: list[dict] = []
    meshes: list[dict] = []
    nodes: list[dict] = []
    materials: list[dict] = []
    images: list[dict] = []
    textures: list[dict] = []

    def add_accessor(raw: bytes, component: int, count: int, kind: str, target: int,
                     extra: dict | None = None) -> int:
        view = bin_.add(raw, target)
        acc: dict = {
            "bufferView": view,
            "componentType": component,
            "count": count,
            "type": kind,
        }
        if extra:
            acc.update(extra)
        accessors.append(acc)
        return len(accessors) - 1

    for part in parts:
        positions, indices, normals = _validate(part)
        scaled = (positions * float(scale)).astype(np.float32)

        # POSITION carries required min/max; the viewer derives its bounds and
        # therefore its camera framing from them, so they are not decoration.
        pos_acc = add_accessor(
            scaled.tobytes(),
            _FLOAT,
            scaled.shape[0],
            "VEC3",
            _ARRAY_BUFFER,
            {
                "min": [float(v) for v in scaled.min(axis=0)],
                "max": [float(v) for v in scaled.max(axis=0)],
            },
        )
        nrm_acc = add_accessor(
            normals.tobytes(), _FLOAT, normals.shape[0], "VEC3", _ARRAY_BUFFER
        )

        attributes = {"POSITION": pos_acc, "NORMAL": nrm_acc}

        if part.colours is not None:
            colours = np.asarray(part.colours, dtype=np.float32).reshape(-1, 3)
            if colours.shape[0] != scaled.shape[0]:
                raise ValueError(
                    f"part {part.label!r} has {colours.shape[0]} colours "
                    f"for {scaled.shape[0]} vertices"
                )
            # Float VEC3 rather than normalised bytes: the specification allows
            # both, float needs no `normalized` flag for readers to honour, and
            # the size difference is immaterial next to positions and normals.
            attributes["COLOR_0"] = add_accessor(
                np.clip(colours, 0.0, 1.0).tobytes(),
                _FLOAT,
                colours.shape[0],
                "VEC3",
                _ARRAY_BUFFER,
            )

        if part.uvs is not None:
            uvs = np.asarray(part.uvs, dtype=np.float32).reshape(-1, 2)
            if uvs.shape[0] != scaled.shape[0]:
                raise ValueError(
                    f"part {part.label!r} has {uvs.shape[0]} UVs for {scaled.shape[0]} vertices"
                )
            attributes["TEXCOORD_0"] = add_accessor(
                uvs.tobytes(), _FLOAT, uvs.shape[0], "VEC2", _ARRAY_BUFFER
            )
        elif part.texture_png is not None:
            raise ValueError(f"part {part.label!r} has a texture but no UVs")

        # 16-bit indices where they fit. Reconstruction meshes are large enough
        # that this is worth doing: it halves the index buffer, and the index
        # buffer is a third of the file.
        if int(indices.max()) < 65536:
            idx_raw = indices.astype(np.uint16).tobytes()
            idx_component = _UNSIGNED_SHORT
        else:
            idx_raw = indices.astype(np.uint32).tobytes()
            idx_component = _UNSIGNED_INT
        idx_acc = add_accessor(
            idx_raw, idx_component, int(indices.size), "SCALAR", _ELEMENT_ARRAY_BUFFER
        )

        primitive: dict = {"attributes": attributes, "indices": idx_acc, "mode": 4}

        if part.texture_png is not None:
            img_view = bin_.add(part.texture_png, None)
            images.append({"bufferView": img_view, "mimeType": "image/png"})
            textures.append({"source": len(images) - 1})
            materials.append(
                {
                    "name": f"{part.label}:mat",
                    "pbrMetallicRoughness": {
                        "baseColorTexture": {"index": len(textures) - 1},
                        "metallicFactor": 0.0,
                        "roughnessFactor": 1.0,
                    },
                    # Reconstructed surfaces are shells with no meaningful inside,
                    # and a back face culled away leaves a hole you can see the
                    # sky through when the camera clips into geometry.
                    "doubleSided": True,
                }
            )
            primitive["material"] = len(materials) - 1

        meshes.append({"name": part.label, "primitives": [primitive]})
        # The name the viewer classifies on. On the node, because that is what
        # its `bestName()` reads first; repeated on the mesh so the file is also
        # legible in tools that show mesh names instead.
        nodes.append({"name": part.label, "mesh": len(meshes) - 1})

    gltf: dict = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "accessors": accessors,
        "bufferViews": bin_.views,
        "buffers": [{"byteLength": len(bin_.data)}],
    }
    if materials:
        gltf["materials"] = materials
        gltf["textures"] = textures
        gltf["images"] = images

    json_raw = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    # Both chunks pad to 4 bytes: JSON with spaces so it stays parseable, BIN
    # with zeros. A parser that trusts the declared length and a parser that
    # walks chunk to chunk both have to land in the same place.
    json_raw += b" " * ((4 - len(json_raw) % 4) % 4)
    bin_raw = bytes(bin_.data) + b"\x00" * ((4 - len(bin_.data) % 4) % 4)

    total = 12 + 8 + len(json_raw) + 8 + len(bin_raw)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        fh.write(struct.pack("<III", _GLB_MAGIC, 2, total))
        fh.write(struct.pack("<II", len(json_raw), _CHUNK_JSON))
        fh.write(json_raw)
        fh.write(struct.pack("<II", len(bin_raw), _CHUNK_BIN))
        fh.write(bin_raw)
    return out


def label_parts(labels: Iterable[str]) -> list[str]:
    """Sanity check a set of labels against the viewer's prefix rule."""
    bad = [l for l in labels if not l.split(":")[0] in ("ground", "structure", "water")]
    if bad:
        raise ValueError(f"labels the viewer will not classify: {bad}")
    return list(labels)
