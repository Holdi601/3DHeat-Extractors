"""Water surfaces, as flat polygons taken from each water body's own spline.

Water needs its own path for two reasons.

The first is that it is not geometry a level export can find. A water body renders
through a water zone's quadtree mesh, evaluated per frame from the camera; there
is no static mesh of a river anywhere in the package. A lake happens to carry a
`StaticMeshComponent` and a river carries a fan of `SplineMeshComponent`s, but a
spline mesh's geometry is the *undeformed* source — read it through the ordinary
static-mesh path and every segment of a river lands in a heap on the actor's
origin. What is authored, and what is reliable in every version, is the
`WaterSplineComponent`: a closed loop around a lake's shore, an open centreline
down a river.

The second is that water wants to look different. It is flat, it is not walkable,
and a heatmap over it means something different from a heatmap over ground — so it
goes into the file as its own part, `water:<map>`, and the viewer shades it
without relief or contours. That look already existed: it is what the ground used
to have before the terrain cues were added, and it was described at the time as
"fine for the water, wrong for the ground".

Nothing here is exact. A river's width is estimated, a shoreline is flattened to
one height, and a spline is sampled rather than evaluated. It is a backdrop for
telemetry: being in the right place and obviously water beats being correct.
"""
import math
import re

import unreal

# Water body actor classes. Blueprint subclasses are common — `BP_MyWaterBodyRiver_C`
# — so this matches anywhere in the name rather than anchoring.
WATER_CLASS = re.compile(r"WaterBody(?!Exclusion)", re.I)
# ...but a water *zone* is the renderer's quadtree host and an exclusion volume is
# a hole, and neither is a surface.
WATER_SKIP = re.compile(r"Zone|Exclusion|Brush|Buoyancy|Manager|Illusion|Sequence", re.I)

# How many segments each span between spline points becomes. Water splines carry
# few points — three or four for a lake — and a river bends between them.
SPLINE_SUBDIVISIONS = 6
# Fallback half-width for a river whose spline meshes tell us nothing, in cm.
DEFAULT_RIVER_HALF_WIDTH = 400.0
# A river cannot be wider than this fraction of its own bounding box; the guess
# from a spline mesh is occasionally a whole tiled sheet rather than one segment.
MAX_WIDTH_FRACTION = 0.5


# The sea. Its own case, because its spline describes the land rather than the
# water. See the note in `build`.
OCEAN_CLASS = re.compile(r"Ocean", re.I)


def _ocean_surface(actor):
    """The ocean's water level, in Unreal centimetres.

    The top of its bounds: an ocean body extends downward from its surface, so
    `bounds.max.z` is the waterline. Measured on the test level, the bounds run
    from far below the surface up to it, and that top is the sea level the
    terrain's beaches meet.
    """
    try:
        origin, extent = actor.get_actor_bounds(False)
        return origin.z + extent.z
    except Exception:  # noqa: BLE001 - engine version differences
        return 0.0


def is_water(cls):
    """Is this actor class a water surface worth exporting?"""
    if not cls:
        return False
    if WATER_SKIP.search(cls):
        return False
    return bool(WATER_CLASS.search(cls))


def _spline_of(actor):
    for comp in actor.get_components_by_class(unreal.SplineComponent):
        try:
            if comp.get_number_of_spline_points() >= 3:
                return comp
        except Exception:  # noqa: BLE001 - engine version differences
            continue
    return None


def _sample(spline, closed):
    """World-space points along the spline, subdivided.

    Sampled by distance rather than by point index so a long lazy bend does not
    become one straight line. `get_location_at_distance_along_spline` is the one
    accessor that has behaved identically in every version tried.
    """
    try:
        length = spline.get_spline_length()
        n = spline.get_number_of_spline_points()
    except Exception:  # noqa: BLE001 - engine version differences
        return []
    steps = max(3, min(512, (n - (0 if closed else 1)) * SPLINE_SUBDIVISIONS))
    world = unreal.SplineCoordinateSpace.WORLD
    points = []
    for i in range(steps):
        d = length * (float(i) / steps if closed else float(i) / (steps - 1))
        try:
            p = spline.get_location_at_distance_along_spline(d, world)
        except Exception:  # noqa: BLE001 - engine version differences
            return []
        points.append((p.x, p.y, p.z))
    return points


def _river_half_width(actor, points):
    """Half the width of a river, from the spline meshes that render it.

    A `SplineMeshComponent`'s source mesh is one tile of river bed, laid along its
    forward axis, so the mesh's *other* horizontal extent is the width. Scaled by
    the component, and then sanity-checked against the actor's own bounds, because
    a project that tiles a single wide sheet along the spline would otherwise
    produce a river wider than the valley it runs through.
    """
    widest = 0.0
    for comp in actor.get_components_by_class(unreal.StaticMeshComponent):
        mesh = comp.static_mesh
        if not mesh:
            continue
        try:
            box = mesh.get_bounding_box()
            scale = comp.get_world_scale3d()
            # Forward is X by convention for a spline mesh; the width is Y.
            widest = max(widest, abs(box.max.y - box.min.y) * abs(scale.y) * 0.5)
        except Exception:  # noqa: BLE001 - engine version differences
            continue
    if widest <= 0.0:
        widest = DEFAULT_RIVER_HALF_WIDTH
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        widest = min(widest, max(50.0, span * MAX_WIDTH_FRACTION))
    return widest


def _fan(points, base):
    """Triangle fan over a closed polygon, from its centroid.

    A centroid fan rather than ear clipping: a water spline has a handful of
    points and shorelines authored by hand are close to convex, so the failure
    mode is a sliver of water outside the shore on a strongly concave lake — which
    is a better trade than several hundred lines of triangulation for a backdrop.
    The centroid is used rather than the first vertex so that a mildly concave
    shape still comes out reasonable.
    """
    n = len(points)
    indices = []
    for i in range(n):
        indices.extend((base + n, base + i, base + (i + 1) % n))
    return indices


def _ribbon(points, half_width, base):
    """Two-triangle-per-segment strip along an open centreline."""
    indices = []
    for i in range(len(points) - 1):
        a = base + i * 2
        indices.extend((a, a + 1, a + 2))
        indices.extend((a + 1, a + 3, a + 2))
    return indices


def build(actors, scale, log, swap_yz=True, clip=None):
    """One flat surface per water body, as (positions, indices).

    Coordinates match the ground pass exactly: Unreal's Z becomes the viewer's Y,
    and the winding is reversed to compensate for the mirroring that swap causes,
    so the surface faces up rather than down.

    `clip` is the map's own footprint as ((min_x, min_y), (max_x, max_y)). A body
    that reaches well past it — an ocean, which can be kilometres wider than the
    map — is replaced by a rectangle covering the footprint at its surface height. The
    alternative is a sea that sets the world's extent and spreads the ground grid
    and every tile budget over open water.
    """
    positions = []
    indices = []
    kinds = {}
    for actor in actors:
        cls_name = actor.get_class().get_name()
        # The ocean is a plane over everything, and its spline does not describe
        # it. An `AWaterBodyOcean` spline marks the *island* — the hole the sea is
        # not in — so triangulating it produces a lake-shaped polygon in the
        # middle of the land. First attempt did exactly that: the report said
        # plenty of lakes and no sea, and the map still had a coastline with nothing beyond
        # it. Handled by class, from its own surface height.
        if OCEAN_CLASS.search(cls_name) and clip:
            surface = _ocean_surface(actor)
            (cx0, cy0), (cx1, cy1) = clip
            base = len(positions) // 3
            for x, y in ((cx0, cy0), (cx1, cy0), (cx1, cy1), (cx0, cy1)):
                if swap_yz:
                    positions.extend((x * scale, surface * scale, y * scale))
                else:
                    positions.extend((x * scale, y * scale, surface * scale))
            quad = (base, base + 1, base + 2, base, base + 2, base + 3)
            if swap_yz:
                for i in range(0, len(quad), 3):
                    indices.extend((quad[i], quad[i + 2], quad[i + 1]))
            else:
                indices.extend(quad)
            kinds["sea"] = kinds.get("sea", 0) + 1
            continue

        spline = _spline_of(actor)
        if spline is None:
            continue
        try:
            closed = bool(spline.is_closed_loop())
        except Exception:  # noqa: BLE001 - engine version differences
            closed = False
        points = _sample(spline, closed)
        if len(points) < 3:
            continue

        # An open sea, reduced to a rectangle over the map. Nothing is lost that
        # a viewer could have used: past the coastline it is a flat plane either
        # way, and the coastline itself comes from the terrain.
        oversized = _outgrows(points, clip)
        if oversized:
            surface = sum(p[2] for p in points) / len(points)
            (cx0, cy0), (cx1, cy1) = clip
            points = [(cx0, cy0, surface), (cx1, cy0, surface), (cx1, cy1, surface), (cx0, cy1, surface)]
            closed = True
            kinds["sea"] = kinds.get("sea", 0) + 1

        base = len(positions) // 3
        cls = actor.get_class().get_name()

        def put(x, y, z):
            if swap_yz:
                positions.extend((x * scale, z * scale, y * scale))
            else:
                positions.extend((x * scale, y * scale, z * scale))

        if closed:
            # A lake is one height. Its spline points are authored at the surface,
            # but a shoreline dragged over terrain can pick up a few metres of
            # drift, and a lake that slopes reads as a bug.
            surface = sum(p[2] for p in points) / len(points)
            for x, y, _z in points:
                put(x, y, surface)
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            put(cx, cy, surface)
            new = _fan(points, base)
            kinds["lake"] = kinds.get("lake", 0) + 1
        else:
            half = _river_half_width(actor, points)
            for i, (x, y, z) in enumerate(points):
                # Perpendicular to the local direction, in plan view.
                nx_, ny_ = _normal_at(points, i)
                put(x - nx_ * half, y - ny_ * half, z)
                put(x + nx_ * half, y + ny_ * half, z)
            new = _ribbon(points, half, base)
            kinds["river"] = kinds.get("river", 0) + 1
        # The Y/Z swap mirrors the world, so the winding has to flip with it.
        if swap_yz:
            for i in range(0, len(new), 3):
                indices.extend((new[i], new[i + 2], new[i + 1]))
        else:
            indices.extend(new)
        kinds[cls] = kinds.get(cls, 0) + 1

    log(
        "water: {} surfaces, {} triangles ({} sea, {} lakes, {} rivers)".format(
            kinds.get("lake", 0) + kinds.get("river", 0) + kinds.get("sea", 0),
            len(indices) // 3,
            kinds.get("sea", 0),
            kinds.get("lake", 0),
            kinds.get("river", 0),
        )
    )
    return positions, indices, kinds


def _outgrows(points, clip):
    """Does this body reach well past the map's own footprint?

    Half again as wide is the threshold: a lake that laps a little over the
    content bounds is still a lake and keeps its shoreline, while an ocean is
    several times the map and is only ever a plane.
    """
    if not clip:
        return False
    (cx0, cy0), (cx1, cy1) = clip
    span = max(cx1 - cx0, cy1 - cy0)
    if span <= 0:
        return False
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return max(max(xs) - min(xs), max(ys) - min(ys)) > span * 1.5


def _normal_at(points, i):
    """Unit plan-view normal to the centreline at point `i`."""
    a = points[max(0, i - 1)]
    b = points[min(len(points) - 1, i + 1)]
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0, 1.0
    return -dy / length, dx / length
