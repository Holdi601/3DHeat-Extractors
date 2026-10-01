"""
The ground, by tracing rays at it.

A landscape has no mesh anyone can ask for. It is a heightmap plus a material,
turned into geometry by the renderer at a resolution that depends on where the
camera is; the scripting API offers no triangles, `CopyCollisionMeshesFromObject`
declines (it handles static meshes only), and the heightmap-to-render-target
route needs an RHI that a headless commandlet does not have.

What a landscape does have is collision, and collision answers questions. So the
ground is sampled: a regular grid of downward rays over the world, one height per
cell, triangulated. That has three things going for it beyond being the only way
that works headlessly — the resolution is chosen rather than inherited, the
result is a uniform grid that decimates predictably, and it is not specific to
landscapes at all. Any solid surface under the rays comes back, so a level whose
terrain is static meshes or some plugin's own heightfield exports the same way.

Only the landscape is loaded while this runs, so every hit is terrain: no need to
inspect what was hit, and no roofs baked into the ground.
"""
import time

import unreal

# How to get the hit point out of a HitResult, worked out on the first hit.
#
# `unreal.HitResult` exposes no named fields to Python in 5.2 — no
# `.impact_point`, and `GameplayStatics.break_hit_result` is not bound either
# (its Blueprint node is a custom thunk). What does work varies by engine
# version, so rather than pinning one way that breaks on the next version, the
# first hit tries each and the winner is remembered for the remaining quarter of
# a million rays.
_EXTRACT = None


def _find_extractor(hit):
    for name in ("impact_point", "location"):
        try:
            value = hit.get_editor_property(name)
        except Exception:  # noqa: BLE001 - property may not be exposed
            continue
        if isinstance(value, unreal.Vector):
            return lambda h, n=name: h.get_editor_property(n).z

    try:
        fields = hit.to_tuple()
    except Exception:  # noqa: BLE001 - struct may be opaque
        fields = ()
    for i, value in enumerate(fields):
        if isinstance(value, unreal.Vector):
            return lambda h, idx=i: h.to_tuple()[idx].z

    raise RuntimeError(
        "cannot read a hit location from this engine's HitResult "
        "(tried impact_point, location and the struct tuple)"
    )


def _impact_z(hit):
    global _EXTRACT
    if _EXTRACT is None:
        _EXTRACT = _find_extractor(hit)
    return _EXTRACT(hit)


def sample_grid(world, bounds, resolution, ceiling, floor, log, progress_every=32):
    """Trace `resolution`² rays down over `bounds`, returning heights.

    `bounds` is ((min_x, min_y), (max_x, max_y)) in Unreal centimetres. Returns
    (heights, hits) where `heights` is a row-major list of floats with `None`
    where nothing was hit — holes are real, and filling them in would invent
    ground that is not there.
    """
    (min_x, min_y), (max_x, max_y) = bounds
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)

    heights = [None] * (resolution * resolution)
    hits = 0
    ignore = []
    t0 = time.time()
    for j in range(resolution):
        y = min_y + span_y * (j + 0.5) / resolution
        row = j * resolution
        for i in range(resolution):
            x = min_x + span_x * (i + 0.5) / resolution
            hit = unreal.SystemLibrary.line_trace_single(
                world,
                unreal.Vector(x, y, ceiling),
                unreal.Vector(x, y, floor),
                unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,
                True,
                ignore,
                unreal.DrawDebugTrace.NONE,
                True,
            )
            if hit:
                heights[row + i] = _impact_z(hit)
                hits += 1
        if progress_every and (j + 1) % progress_every == 0:
            log(
                "  ground row {}/{} ({} hits, {:.0f}s)".format(
                    j + 1, resolution, hits, time.time() - t0
                )
            )
    return heights, hits


def triangulate(heights, resolution, bounds, scale, swap_yz=True):
    """Grid heights to triangles, skipping cells that were not fully hit.

    A cell needs all four corners: a triangle spanning a hole would be a sheer
    cliff into nothing, which reads as a wall in the viewer.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - Unreal ships numpy
        np = None

    (min_x, min_y), (max_x, max_y) = bounds
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)

    index_of = [-1] * len(heights)
    positions = []
    count = 0
    for j in range(resolution):
        y = min_y + span_y * (j + 0.5) / resolution
        for i in range(resolution):
            h = heights[j * resolution + i]
            if h is None:
                continue
            x = min_x + span_x * (i + 0.5) / resolution
            if swap_yz:
                positions.extend((x * scale, h * scale, y * scale))
            else:
                positions.extend((x * scale, y * scale, h * scale))
            index_of[j * resolution + i] = count
            count += 1

    indices = []
    for j in range(resolution - 1):
        for i in range(resolution - 1):
            a = index_of[j * resolution + i]
            b = index_of[j * resolution + i + 1]
            c = index_of[(j + 1) * resolution + i]
            d = index_of[(j + 1) * resolution + i + 1]
            if a < 0 or b < 0 or c < 0 or d < 0:
                continue
            # Wound so the surface faces up after the Y/Z swap has mirrored it.
            if swap_yz:
                indices.extend((a, b, c))
                indices.extend((b, d, c))
            else:
                indices.extend((a, c, b))
                indices.extend((b, c, d))

    if np is not None:
        return (
            np.asarray(positions, dtype=np.float32),
            np.asarray(indices, dtype=np.uint32),
        )
    return positions, indices
