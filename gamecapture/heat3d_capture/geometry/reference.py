"""
A synthetic level with known dimensions.

The reconstruction pipeline cannot be tested end to end without a game running,
which makes it very easy for the *export* half to rot unnoticed — the half that
decides whether anything reconstructed is loadable at all. This builds a level
of exactly known size and composition through the same writer the real exporter
uses, so both sides of the file format stay under test on any machine, with no
capture hardware and no game.

Every dimension here is asserted on by the viewer's own tests. Changing one
means changing them, which is the point: the numbers are a contract, not
a fixture that happens to work.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .glb import Part, write_glb

#: Ground extent in metres. 400 m is a realistic walked area and, at the viewer's
#: centimetre scale, lands at 40 000 units — far enough out to exercise the
#: camera's far-plane widening rather than sitting comfortably inside it.
GROUND_METRES = 400.0
#: Height of each box, metres.
BUILDING_HEIGHT = 12.0
#: Footprint of each box, metres.
BUILDING_FOOTPRINT = 20.0
#: Where the boxes stand, as (x, z) of the near corner, metres.
BUILDING_ORIGINS = [(60.0, 60.0), (150.0, 220.0), (300.0, 120.0)]
#: Water square, (x, z, size) metres. Deliberately below y=0 so it reads as a
#: depression rather than a sheet floating over the ground.
WATER = (220.0, 300.0, 50.0)
WATER_DEPTH = -1.5

#: Colour given to each class, so the round trip through `COLOR_0` is checked
#: rather than merely supported.
#:
#: A real scan's colour comes from the frames it was reconstructed from; this
#: only has to be *distinct per class* and exactly representable in float32, so
#: the viewer's parser can be asserted to have read the right bytes for the
#: right vertices rather than to have read plausible ones.
PART_COLOURS = {
    "ground": (0.25, 0.5, 0.75),
    "structure": (1.0, 0.5, 0.0),
    "water": (0.0, 0.25, 1.0),
}


def _grid(size: float, y: float, divisions: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    A subdivided horizontal plane.

    Subdivided rather than two triangles because a single enormous quad is not
    what any real ground looks like, and the viewer's geometric classifier reads
    triangle area against map area — a two-triangle ground would sail through a
    test that a realistic one fails.
    """
    steps = np.linspace(0.0, size, divisions + 1, dtype=np.float32)
    xs, zs = np.meshgrid(steps, steps, indexing="ij")
    positions = np.stack(
        [xs.ravel(), np.full(xs.size, y, dtype=np.float32), zs.ravel()], axis=1
    ).astype(np.float32)

    idx: list[int] = []
    stride = divisions + 1
    for i in range(divisions):
        for j in range(divisions):
            a = i * stride + j
            b = a + 1
            c = a + stride
            d = c + 1
            # Counter-clockwise seen from above, so the normals point up.
            idx += [a, c, b, b, c, d]
    return positions, np.array(idx, dtype=np.uint32)


def _box(x: float, z: float, footprint: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """An axis-aligned box standing on y=0, with outward-facing triangles."""
    x0, x1 = x, x + footprint
    z0, z1 = z, z + footprint
    y0, y1 = 0.0, height
    positions = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1],
            [x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    faces = [
        (0, 1, 5, 4),  # -Z
        (1, 2, 6, 5),  # +X
        (2, 3, 7, 6),  # +Z
        (3, 0, 4, 7),  # -X
        (4, 5, 6, 7),  # top
        (3, 2, 1, 0),  # bottom
    ]
    idx: list[int] = []
    for a, b, c, d in faces:
        idx += [a, b, c, a, c, d]
    return positions, np.array(idx, dtype=np.uint32)


def _flat_colour(count: int, cls: str) -> np.ndarray:
    """One colour repeated per vertex, as the writer expects it."""
    return np.tile(np.array(PART_COLOURS[cls], dtype=np.float32), (count, 1))


def build_reference_parts() -> list[Part]:
    gp, gi = _grid(GROUND_METRES, 0.0)
    parts = [Part("ground", "terrain", gp, gi, colours=_flat_colour(len(gp), "ground"))]

    for n, (x, z) in enumerate(BUILDING_ORIGINS, start=1):
        bp, bi = _box(x, z, BUILDING_FOOTPRINT, BUILDING_HEIGHT)
        parts.append(
            Part(
                "structure",
                f"building_{n:02d}",
                bp,
                bi,
                colours=_flat_colour(len(bp), "structure"),
            )
        )

    wx, wz, wsize = WATER
    wp, wi = _grid(wsize, WATER_DEPTH, divisions=2)
    wp = wp + np.array([wx, 0.0, wz], dtype=np.float32)
    parts.append(Part("water", "pond", wp, wi, colours=_flat_colour(len(wp), "water")))
    return parts


def write_reference(path: str | Path) -> Path:
    """Write the reference level, returning the path."""
    return write_glb(path, build_reference_parts(), generator="heat3d-capture-reference")


if __name__ == "__main__":  # pragma: no cover - a convenience entry point
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".out/reference.glb")
    print(write_reference(target))
