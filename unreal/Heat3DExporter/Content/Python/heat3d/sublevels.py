"""The buildings, which live in sub-levels the open map cannot instantiate.

This module exists because of a measurement that invalidated four rounds of work
on the budget. Every export of the test level came out as landscape, rock, road
and trees with no town in it, and every explanation offered — the size filter, the
per-tile share, the allowance floor, box substitution turned into deletion — was
either wrong or a second-order effect. The actual reason:

    `WorldPartitionBlueprintLibrary.load_actors()` instantiates 399 of 400
    persistent-level actors and **0 of 400** actors belonging to a sub-level.
    `pin_actors` does no better. So does the structure pass: it loads by guid and
    finds actors by label, and for every building on the map it found nothing.

Measured with a control, which is the only reason it is trustworthy — a probe
that finds nothing looks the same whether the thing is absent or the probe is
broken, and that mistake was made here twice.

What the map does expose is *descriptors*: hundreds of sub-levels with names
like `LI_House_01`, `LI_Factory_01`, `LI_School_01`, each with its actors'
**world-space** bounds. And a
sub-level can be opened as a level in its own right, where its actors are all
present — in the sub-level's own coordinates, because these are prefabs placed by
a transform rather than levels authored in place.

So the geometry is here and the placements are here; they are just in two
different places. This module joins them:

1.  **Group** descriptors by the sub-level they belong to.
2.  **Cluster** each sub-level's descriptors by position. One cluster is one
    instance: a building is tens of metres across and instances are hundreds of
    metres apart, so a coarse grid with a merge pass separates them exactly.
3.  **Open** the sub-level once and read where its actors sit locally.
4.  **Solve** each instance's transform by matching actors between the two by
    label — translation and yaw, in closed form. Measured on a factory
    sub-level instanced four times at four different angles: all of its
    thousands of actors matched per instance, median residual 2.2cm, 90th
    percentile 2.8cm.
5.  **Stamp** the building's geometry at every instance.

Reducing once and stamping N times is also the cheaper way round. The old path
asked the simplifier for 96 triangles per wall panel; this asks for a triangle
budget for a whole building, so the building keeps its walls, its floors and the
gaps between them — which is what the heatmap is read against.
"""
import math
import time

import unreal

from . import build, classify, hotspots as hotspots_module

# Smallest grid cell used to look up the nearest instance anchor. Only affects
# speed, never the result — see `cluster`.
CLUSTER_LINK_MIN = 1500.0

# Below this many matched actors a transform is not worth trusting.
#
# Four, not eight. A 2D similarity has four unknowns — two of translation, one of
# yaw, one of scale — and each matched actor supplies two equations, so four
# actors already over-determine it and leave a residual that means something.
# Eight was chosen for comfort rather than from the arithmetic, and it cost real
# buildings: one sub-level reported "19 of 19 placements unsolved: only 6 of 12
# actors matched", so every copy of it was dropped over a threshold it beat on
# every mathematical measure.
#
# The residual below is the check that actually protects against a wrong match,
# and it is the one to tighten if a building ever lands in the wrong place — a
# count of matches says nothing about whether they agree.
MIN_MATCHES = 4

# Above this residual the match is wrong rather than noisy, and stamping a
# building in the wrong place is worse than leaving it out.
MAX_RESIDUAL_CM = 500.0

# Bounds on what one building may cost, before it is instanced.
#
# The ceiling is generous because this is a whole compound with its interior, and
# because it is paid once however many times the building occurs. The floor is
# what a building needs to still read as one from inside the viewer: below about
# eight thousand triangles a house keeps its footprint and loses its roof line,
# its openings and its floors, which are the things a movement heatmap is read
# against.
# 30,000, down from the 120,000 the first working version used. That version
# spent its whole budget on six buildings, and on a map with hundreds of
# placements the thing being read is *where the buildings are*, not the moulding
# on one of them. At 30,000 a compound keeps its walls, its floor slabs, its roof
# line and its openings — enough to see which floor a player was on — and the
# budget stretches to about a hundred and forty copies instead of six.
#
# The floor came down from 6,000 when the pass stopped opening only the busiest
# forty-five sub-levels. It is a floor on *every* building, so with hundreds of
# them it was on its own asking for millions of triangles before a single
# instance was stamped — more than the whole reservation — and the effect of a
# floor that cannot be paid is that repeats get refused, which puts one copy of a
# house on the map and leaves the other fifty-odd missing. At 2,000 a small
# house still keeps its footprint, its openings and its floor slabs, which is
# what the heatmap is read against; the extent term below is what gives the
# large buildings their detail, and it still does.
#
# The ceiling came *up* from 30,000 at the same time the split stopped being
# equal. The two go together: a proportional share is pointless if a cap
# immediately takes it away again, and 30,000 was set when every building got the
# same allowance and the question was how to stop the first few spending
# everything. With shares sized by footprint, a compound of three thousand parts
# can be given what a compound of three thousand parts needs, and a shed still
# only gets a shed's worth because its footprint is a shed's.
BUILDING_TRIANGLES_MAX = 80_000
BUILDING_TRIANGLES_MIN = 2_000

# Most and least a single part of a building may cost, in triangles.
#
# Each part is reduced on its own, by the mesh cache, which spends the allowance
# through a *tolerance* — "stay within 15cm of the original" — rather than a
# triangle count. That distinction is the whole reason the ceiling can be this
# low: a wall panel or a floor slab asked to stay within 15cm of itself comes
# back as a wall panel, whatever number of triangles that takes, because the
# shape it is being asked to keep is simple.
PART_TRIANGLES_MAX = 400
PART_TRIANGLES_MIN = 24

# Triangles a boxed part costs. A box is twelve; the cache is asked for this so
# it has no room to return anything else.
BOX_TRIANGLES = 12
# Fraction of a building's budget held back so every remaining part can be boxed.
#
# The point is that a building is complete before it is detailed. Without a
# reserve the detail pass spends everything on the largest parts and the rest of
# the building does not exist — measured at under a tenth of the parts on one
# hospital, which in the viewer is a few slabs floating where a building should
# be. A quarter is enough to box thousands of parts at twelve triangles each
# while leaving the bulk for the walls and slabs that carry the shape.
BOX_TAIL_RESERVE = 0.25

# How far a building's part may move under reduction, in centimetres.
#
# Three, against the fifteen everything else gets. Fifteen is chosen to stay
# under the thickness of a *wall*, and a building is not only walls: a railing, a
# window, a sign, a stair tread and a door panel are all thinner than that, and a
# tolerance wider than a thing is thick merges its two faces into a single sheet —
# which is what the big flat quads floating at odd angles inside the buildings
# were. Measured: around a thousand assets came out of that pass as sheets.
#
# Not zero. With the coplanar merge alone — exact, but it only removes triangles
# that lie in one plane — parts cost several hundred triangles each and a 30,000
# triangle building could afford twelve of them.
ARCH_TOLERANCE_CM = 3.0

# Triangles per metre of a building's own footprint, as a ceiling on its target.
#
# 300, so a 100m compound may reach the cap and a 10m truck gets 3,000. Without
# it, ranking by traffic hands a building's whole budget to whatever players
# happen to gather around: a parked-truck sub-level came out among the busiest
# on the map — people walk to vehicles — and took a building's worth of
# triangles for a flatbed, many times over. Traffic decides what is worth
# exporting; size decides how much of it there is to draw.
#
# Raised to 600 alongside the higher ceiling. It still does its job — a 10m truck
# gets 6,000 and cannot take a compound's budget — but at 300 it, not the share,
# was what capped the large buildings: an 80m hospital was held to 24,000
# triangles for over three thousand parts, which is under eight per part and is
# why it came out as a handful of slabs.
BUILDING_TRIANGLES_PER_METRE = 600

# Parts smaller than this are not read at all. 40cm is door furniture, signage,
# pipe fittings and light fixtures — the things a building has thousands of and a
# heatmap is never read against.
PART_MIN_SIZE_CM = 40.0


class Instance(object):
    """One placement of a sub-level, as a transform and where it is."""

    __slots__ = ("transform", "actors", "centre", "residual")

    def __init__(self, transform, actors, centre, residual):
        self.transform = transform
        self.actors = actors
        self.centre = centre
        self.residual = residual


def collect(descs, map_package, min_actors):
    """`{sub_level: [(label, world_centre), ...]}` for sub-levels worth opening.

    Two passes over the descriptors rather than one, because holding a label and
    a position for all of them on a large map costs a hundred megabytes and most
    are of no interest. Counting first takes seconds.
    """
    counts = {}
    for d in descs:
        if not d.bounds.is_valid:
            continue
        place = classify.place_of(getattr(d, "actor_path", None), map_package)
        if place:
            counts[place] = counts.get(place, 0) + 1

    wanted = {name for name, n in counts.items() if n >= min_actors}
    rows = {}
    if not wanted:
        return rows, counts
    for d in descs:
        b = d.bounds
        if not b.is_valid:
            continue
        place = classify.place_of(getattr(d, "actor_path", None), map_package)
        if place not in wanted:
            continue
        centre = (
            (b.min.x + b.max.x) * 0.5,
            (b.min.y + b.max.y) * 0.5,
            (b.min.z + b.max.z) * 0.5,
        )
        if not all(_finite(v) for v in centre):
            continue
        rows.setdefault(place, []).append((str(d.label), centre))
    return rows, counts


def cluster(rows):
    """Split one sub-level's descriptors into one group per instance.

    Not by distance. Distance clustering is the obvious approach and it fails on
    the case that matters: at a fixed 200m cell, a house placed fifty times came
    out as six groups, because occupied cells chain into one another; and any
    threshold small enough to avoid that splits a town's terraced houses wrongly
    the other way, since two neighbouring houses are closer together than one
    house is wide.

    There is an exact answer available instead. Every instance of a sub-level
    contains the same actors, so a label belonging to exactly one actor per
    instance occurs exactly as many times as there are instances — and those
    occurrences are one point inside each instance. Take the number of instances
    to be the commonest label-occurrence count, pick a label with that count, and
    its positions are an anchor per instance. Everything else joins its nearest
    anchor.

    This cannot merge two instances and cannot split one, however they are
    arranged, because it never asks how far apart anything is.
    """
    positions = {}
    for label, centre in rows:
        positions.setdefault(label, []).append(centre)
    if not positions:
        return []

    # The commonest occurrence count. Ties go to the larger count: a duplicated
    # label inflates a count but nothing deflates one, so the mode is a floor.
    frequency = {}
    for occurrences in positions.values():
        n = len(occurrences)
        frequency[n] = frequency.get(n, 0) + 1
    instances = max(frequency.items(), key=lambda kv: (kv[1], kv[0]))[0]
    if instances <= 1:
        return [list(rows)]

    anchors = None
    for occurrences in positions.values():
        if len(occurrences) == instances:
            anchors = list(occurrences)
            break
    if anchors is None:
        return [list(rows)]

    # Nearest anchor, through a grid so this is not rows x anchors comparisons. The
    # grid is sized from how far apart the anchors are, so it always has a few
    # anchors within one cell of any point.
    spread = _spread(anchors)
    cell = max(CLUSTER_LINK_MIN, spread)
    index = {}
    for i, a in enumerate(anchors):
        key = (int(math.floor(a[0] / cell)), int(math.floor(a[1] / cell)))
        index.setdefault(key, []).append(i)

    groups = [[] for _ in anchors]
    for row in rows:
        c = row[1]
        key = (int(math.floor(c[0] / cell)), int(math.floor(c[1] / cell)))
        best, best_d = -1, None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for i in index.get((key[0] + dx, key[1] + dy), ()):
                    a = anchors[i]
                    d = (a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2
                    if best_d is None or d < best_d:
                        best, best_d = i, d
        if best < 0:
            # Nothing within a cell: fall back to a full scan for this one point.
            for i, a in enumerate(anchors):
                d = (a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2
                if best_d is None or d < best_d:
                    best, best_d = i, d
        groups[best].append(row)

    out = [g for g in groups if g]
    out.sort(key=lambda g: -len(g))
    return out


def _extent_metres(group):
    """The horizontal diagonal of one instance's footprint, in metres."""
    if not group:
        return 0.0
    xs = [row[1][0] for row in group]
    ys = [row[1][1] for row in group]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    return math.sqrt(dx * dx + dy * dy) / 100.0


def _spread(anchors):
    """Roughly how far apart the anchors are, for sizing the lookup grid."""
    if len(anchors) < 2:
        return CLUSTER_LINK_MIN
    xs = sorted(a[0] for a in anchors)
    ys = sorted(a[1] for a in anchors)
    span = max(xs[-1] - xs[0], ys[-1] - ys[0])
    return max(CLUSTER_LINK_MIN, span / max(1.0, math.sqrt(len(anchors))))


def solve(group, local):
    """The transform taking the sub-level's coordinates to this instance's.

    Translation and yaw only. A building is placed on the ground and yaw is the
    only rotation a level designer normally applies to one; solving for a full
    rotation from noisy bounding-box centres would fit the noise as easily as the
    signal. Closed form, so there is nothing to converge or to tune.

    Returns `(unreal.Transform, matched, median_residual_cm, reason)`, with the
    transform None and the reason set when it could not be solved.

    The reason is not decoration. A building ranked in by traffic and then not
    stamped is indistinguishable, from the report, from one that was never
    ranked — and the first is a bug while the second is a budget. That
    ambiguity is what "there must be a building here and there is not" looked
    like from the outside.
    """
    # Only labels that are unambiguous on *both* sides.
    #
    # `local` has already dropped the ones repeated inside the sub-level; this
    # drops the ones repeated inside the cluster. A label appearing twice here
    # would be paired twice against the same source position, which pulls the
    # centroid and the rotation towards whichever actors happen to be duplicated.
    times = {}
    for label, _world in group:
        times[label] = times.get(label, 0) + 1
    pairs = []
    for label, world in group:
        here = local.get(label)
        if here is not None and times[label] == 1:
            pairs.append((here, world))
    if len(pairs) < MIN_MATCHES:
        return (
            None,
            len(pairs),
            None,
            "only {} of {} actors matched the sub-level by label".format(
                len(pairs), len(group)
            ),
        )

    n = float(len(pairs))
    ca = [sum(p[0][i] for p in pairs) / n for i in range(3)]
    cb = [sum(p[1][i] for p in pairs) / n for i in range(3)]

    sxy = sum(
        (p[0][0] - ca[0]) * (p[1][1] - cb[1]) - (p[0][1] - ca[1]) * (p[1][0] - cb[0])
        for p in pairs
    )
    sxx = sum(
        (p[0][0] - ca[0]) * (p[1][0] - cb[0]) + (p[0][1] - ca[1]) * (p[1][1] - cb[1])
        for p in pairs
    )
    yaw = math.atan2(sxy, sxx)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)

    # And a uniform scale, from the same two sums.
    #
    # Assuming unit scale was an assumption, not a fact, and it cost two
    # buildings the telemetry draws a floor plan inside. A designer may place a
    # level instance scaled, and a translation-and-yaw fit to a scaled instance
    # leaves a residual proportional to the distance from the centroid — so it
    # fails worst on the largest buildings, which are the ones that matter.
    # Measured: one compound's sub-level matched all of its thousands of actors
    # and was rejected for a median error of over ten metres, which is what a few
    # per cent of scale looks like across a compound that size.
    #
    # Closed form and free: for a 2D similarity the scale is the magnitude of the
    # cross-correlation over the source's own variance, and both terms are
    # already computed above.
    var_a = sum(
        (p[0][0] - ca[0]) ** 2 + (p[0][1] - ca[1]) ** 2 for p in pairs
    )
    scale = 1.0
    if var_a > 1e-6:
        scale = math.sqrt(sxx * sxx + sxy * sxy) / var_a
    if not _finite(scale) or scale <= 1e-4 or scale > 1e4:
        scale = 1.0

    residuals = []
    for a, b in pairs:
        dx, dy, dz = a[0] - ca[0], a[1] - ca[1], a[2] - ca[2]
        residuals.append(
            math.sqrt(
                (scale * (dx * cos_y - dy * sin_y) + cb[0] - b[0]) ** 2
                + (scale * (dx * sin_y + dy * cos_y) + cb[1] - b[1]) ** 2
                + (scale * dz + cb[2] - b[2]) ** 2
            )
        )
    residuals.sort()
    median = residuals[len(residuals) // 2]
    # `not median <= MAX`, not `median > MAX`, and the difference is the whole
    # comparison: every comparison against NaN is false, so `median > MAX` lets a
    # NaN through as if it were a perfect fit. One actor with invalid bounds is
    # enough to produce one — the engine reports an empty box as ±3.4e38, which
    # turns the centroid into infinity and the residuals into NaN — and the
    # building then gets stamped at a garbage transform. Measured: a finished
    # several-hundred-MB export whose structure part spanned 7.8e34 centimetres,
    # which loads as an empty screen because the whole map is one pixel inside it.
    if not median <= MAX_RESIDUAL_CM:
        return (
            None,
            len(pairs),
            median,
            "the fit is off by {:.0f}cm across {} actors".format(median, len(pairs)),
        )

    # world = s * R * (local - ca) + cb, so the translation is cb - s*R*ca.
    tx = cb[0] - scale * (ca[0] * cos_y - ca[1] * sin_y)
    ty = cb[1] - scale * (ca[0] * sin_y + ca[1] * cos_y)
    tz = cb[2] - scale * ca[2]
    if not all(_finite(v) for v in (tx, ty, tz, yaw)):
        return (None, len(pairs), median, "the solved position is not a number")

    transform = make_transform(tx, ty, tz, yaw, scale)
    # And check the thing that was built does what it was solved to do: it must
    # take the sub-level's centroid to this instance's. Cheap, exact, and it
    # closes off the whole class of failure rather than the one instance of it
    # that was found — a transform built through a binding that ignored an
    # argument would satisfy every check above and still stamp the building in
    # the wrong place.
    landed = transform.transform_location(unreal.Vector(ca[0], ca[1], ca[2]))
    if (
        abs(landed.x - cb[0]) > 100.0
        or abs(landed.y - cb[1]) > 100.0
        or abs(landed.z - cb[2]) > 100.0
    ):
        return (None, len(pairs), median, "the built transform does not move the centroid there")
    return transform, len(pairs), median, None


def _finite(value):
    """Is this a real number a transform can be built from?

    `math.isfinite` covers NaN and both infinities; the magnitude check covers
    the values that are finite and still nonsense — an actor whose bounds the
    engine could not compute comes back at 3.4e38, and half of that is still
    finite. Nothing on a game map is a thousand kilometres from the origin.
    """
    return math.isfinite(value) and abs(value) < 1.0e8


def make_transform(tx, ty, tz, yaw_rad, scale=1.0):
    """A yaw-and-translate transform.

    Built by assignment, with an explicit quaternion, because every other form
    has a trap in it and each one cost a ten-minute export to find:

    * `FTransform.rotation` is a **Quat**, not a Rotator. Assigning a Rotator
      raises "Cannot nativize 'Rotator' as 'Quat'".
    * The struct's *constructor* is `KismetMathLibrary.MakeTransform` in
      disguise, so its keywords are `location`, `rotation`, `scale` — not the
      property names `translation`, `rotation`, `scale3d` that the same struct
      then exposes. Passing the property names raises "'translation' is an
      invalid keyword argument".
    * `unreal.MathLibrary.make_transform` is not bound at all in 5.2.

    A rotation of `yaw` about Z is `(0, 0, sin(yaw/2), cos(yaw/2))`, so this
    needs no engine maths and no Rotator argument-order convention either.
    """
    half = yaw_rad * 0.5
    transform = unreal.Transform()
    transform.translation = unreal.Vector(tx, ty, tz)
    transform.scale3d = unreal.Vector(scale, scale, scale)
    transform.rotation = unreal.Quat(0.0, 0.0, math.sin(half), math.cos(half))
    return transform


def read_building(pool, cache, settings, filt, log, target):
    """One mesh for the whole sub-level that is currently open, in its own space.

    Built as a single mesh rather than component by component because that is
    what makes the reduction affordable: a building is reduced once, to a budget
    that suits a building, and every instance of it is then free. Reducing each
    wall panel to the floor an individual object gets is what produced a town of
    unrecognisable fragments.
    """
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    level_actors = actors.get_all_level_actors()

    # Everything the building is made of, with how big each piece is, before any
    # of it is read.
    local = {}
    duplicated = set()
    candidates = []
    excluded_actors = 0
    for actor in level_actors:
        label = actor.get_actor_label()
        if isinstance(actor, unreal.LandscapeProxy):
            continue
        # By class and by name, the same as the descriptor scan. Without this the
        # volumes a level is full of are exported as geometry — see
        # `Filter.exclude_actor`.
        if filt.exclude_actor(actor.get_class().get_name(), label):
            excluded_actors += 1
            continue
        origin, _extent = actor.get_actor_bounds(False)
        # First writer wins, matching the probe that validated the solver — and
        # only if the engine could actually say where the actor is. An actor with
        # no renderable component reports an empty box as ±3.4e38, and one of
        # those in the set destroys the centroid every transform is solved from.
        if all(_finite(v) for v in (origin.x, origin.y, origin.z)):
            # Counted, not first-writer-wins.
            #
            # `setdefault` was silently lossy and it cost the two buildings the
            # telemetry draws the clearest floor plans inside. A sub-level may
            # hold many actors with the same label, and keeping the first one
            # matched every map-side descriptor of that label against a single
            # arbitrary position — so one compound's sub-level "matched" all of
            # its thousands of actors and fitted a transform to noise, failing by
            # over ten metres. A label that occurs once on each side is an
            # unambiguous pair; anything else is worse than nothing, so it is
            # thrown away below.
            seen = local.get(label)
            if seen is None:
                local[label] = (origin.x, origin.y, origin.z)
            else:
                duplicated.add(label)
        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            static_mesh = component.static_mesh
            if not static_mesh:
                continue
            asset = classify.asset_name(static_mesh.get_path_name())
            if filt.exclude_asset(asset):
                continue
            if classify.priority(label, static_mesh.get_path_name()) < 0:
                # No vegetation inside buildings: it is the least useful geometry
                # on the map and here it would be paid for once per instance.
                continue
            transforms = []
            if isinstance(component, unreal.InstancedStaticMeshComponent):
                for i in range(component.get_instance_count()):
                    try:
                        instance = component.get_instance_transform(i, True)
                    except Exception:  # noqa: BLE001 - malformed instance data
                        continue
                    if build.sane_transform(instance):
                        transforms.append(instance)
            else:
                world = component.get_world_transform()
                if build.sane_transform(world):
                    transforms.append(world)
            if not transforms:
                continue
            size = _size_of(static_mesh, transforms[0])
            if size < PART_MIN_SIZE_CM:
                continue
            candidates.append(
                (_shell_area_of(static_mesh, transforms[0]), size, static_mesh, transforms)
            )

    # Largest first, spending a triangle budget as it goes, and reduced *per part*
    # — never across the assembly.
    #
    # That last clause is the one that matters, and getting it wrong produced the
    # worst-looking geometry this exporter has ever shipped. The previous version
    # assembled a building at full detail and then asked a quadric simplifier to
    # bring the whole thing down to the target. A quadric simplifier collapses
    # edges by which collapse adds least error, and across an assembly of
    # thousands of disconnected shells the cheapest collapses are the ones that
    # bridge between separate panels: it keeps a few far-apart vertices and joins
    # them. The result is a building shot through with long thin spikes and
    # stretched slivers, which is exactly what `MeshCache.get` has a comment
    # warning about for a single bush, one scale up.
    #
    # Reducing each part on its own cannot do that. A part is one panel, the
    # simplifier only has that panel's own edges to collapse, and the cache spends
    # the allowance as a 15cm tolerance — so a wall comes back a wall. What bounds
    # the building is then how many parts it can afford rather than how hard each
    # is squashed, and the parts it cannot afford are the smallest ones: a
    # building's tail by size is its fixtures, signage and door furniture, and
    # what survives is walls, slabs, roofs, stairs and railings.
    candidates.sort(key=lambda c: -c[0])

    # Two passes: detail for what the budget affords, a box for everything else.
    #
    # One pass drops the tail, and dropping the tail is what a half-built building
    # looks like. One hospital building came out as under a tenth of its parts
    # — the budget ran out and the rest simply did not exist, so the viewer
    # showed a few slabs hanging in the air where a hospital should be.
    #
    # A wall reduced to a box is still a wall: it occludes, it has a floor and a
    # roof line, and the heat inside it reads correctly, which is the whole job.
    # A wall that is absent is a hole. So completeness is bought first and detail
    # second, and the reserve below is what guarantees the second pass has
    # something left to spend when a building has thousands of parts.
    detail_budget = int(target * (1.0 - BOX_TAIL_RESERVE))

    mesh = pool.request_mesh()
    components = 0
    instances = 0
    boxed = 0
    spent = 0
    skipped = 0
    deferred = []
    for _area, size, static_mesh, transforms in candidates:
        if spent >= detail_budget:
            deferred.append((size, static_mesh, transforms))
            continue
        # One allowance for every part, not one per size.
        #
        # Nothing is reduced to it — the read is exact — so all it does is pick a
        # cache slot, and varying it by size would mean reading the same wall
        # panel several times and storing several identical copies of it.
        piece = cache.get(
            static_mesh,
            PART_TRIANGLES_MAX,
            size,
            strip_small=False,
            allow_box=False,
            # A tight tolerance and no count reduction. Between them these are
            # what stop a part becoming a spike or a sheet; see MeshCache.get.
            tolerance_cap=ARCH_TOLERANCE_CM,
            count_limit=False,
        )
        if piece is None:
            deferred.append((size, static_mesh, transforms))
            continue
        each = piece.get_triangle_count()
        if each <= 0:
            continue
        room = detail_budget - spent
        if each > room:
            # Cannot afford even one of this part at detail. It goes to the box
            # pass rather than being lost.
            deferred.append((size, static_mesh, transforms))
            continue
        allowed = max(1, int(room // each))
        if allowed < len(transforms):
            deferred.append((size, static_mesh, transforms[allowed:]))
            transforms = transforms[:allowed]
        components += 1
        instances += len(transforms)
        spent += each * len(transforms)
        mesh = mesh.append_mesh_transformed(
            piece, transforms, unreal.Transform(), False
        )

    # The tail, as boxes, until even that runs out.
    for size, static_mesh, transforms in deferred:
        room = target - spent
        if room <= 0:
            skipped += len(transforms)
            continue
        piece = cache.get(
            static_mesh,
            BOX_TRIANGLES,
            size,
            strip_small=False,
            allow_box=True,
            tolerance_cap=None,
            count_limit=True,
        )
        if piece is None:
            skipped += len(transforms)
            continue
        each = piece.get_triangle_count()
        if each <= 0:
            skipped += len(transforms)
            continue
        allowed = int(room // each)
        if allowed <= 0:
            skipped += len(transforms)
            continue
        if allowed < len(transforms):
            skipped += len(transforms) - allowed
            transforms = transforms[:allowed]
        boxed += 1
        instances += len(transforms)
        spent += each * len(transforms)
        mesh = mesh.append_mesh_transformed(
            piece, transforms, unreal.Transform(), False
        )
    for label in duplicated:
        local.pop(label, None)
    return mesh, local, {
        "components": components,
        "boxed": boxed,
        "instances": instances,
        "parts_available": len(candidates),
        "parts_over_budget": skipped,
        "triangles": mesh.get_triangle_count(),
        "labels": len(local),
        "ambiguous_labels": len(duplicated),
        "excluded_actors": excluded_actors,
    }


def _size_of(static_mesh, transform):
    try:
        box = static_mesh.get_bounding_box()
        extent = box.max - box.min
    except Exception:  # noqa: BLE001 - engine version differences
        return 0.0
    s = transform.scale3d
    return math.sqrt(
        (extent.x * s.x) ** 2 + (extent.y * s.y) ** 2 + (extent.z * s.z) ** 2
    )


def _shell_area_of(static_mesh, transform):
    """Roughly how much surface a part contributes, in square centimetres.

    The measure parts are ranked by, and it replaced the bounding diagonal, which
    ranks the wrong things. A diagonal rewards *length*: a ten-metre handrail, a
    pipe run, a catwalk beam and a roof truss all outrank a three-metre wall
    panel, while contributing almost no surface. On a building with more parts
    than its budget can hold, that is what decides which ones exist — and what it
    decided was long thin things. The result in the viewer was a building reduced
    to floating slabs and stray beams with the walls missing, which reads as a
    half-collapsed frame rather than as a building.

    What a movement heatmap is read against is the *shell*: walls, floor slabs,
    roofs. Those are the parts with area. So: the three face areas of the
    transformed bounding box, largest two summed, which approximates the surface
    of a slab or panel however it is oriented and correctly scores a long thin rod
    near zero.
    """
    try:
        box = static_mesh.get_bounding_box()
        extent = box.max - box.min
    except Exception:  # noqa: BLE001 - engine version differences
        return 0.0
    s = transform.scale3d
    x = abs(extent.x * s.x)
    y = abs(extent.y * s.y)
    z = abs(extent.z * s.z)
    faces = sorted((x * y, y * z, x * z))
    return faces[-1] + faces[-2]


def reduce_building(mesh, triangles, log, name):
    """Tidy an assembled building without letting the simplifier cross its parts.

    Two passes, and what is *not* here is the point.

    * **Weld** coincident vertices, so panels authored to meet become one
      surface. Exact within the tolerance and it costs nothing.
    * **Merge coplanar triangles.** Also exact: adjacent triangles lying in the
      same plane become one and the shape is identical.

    There is no reduction to a triangle count, and there was, and it produced the
    worst geometry this exporter has shipped: buildings shot through with long
    thin spikes and stretched slivers, the largest compounds worst of all. A
    quadric simplifier collapses whichever edge adds least error, and across an
    assembly of thousands of shells the cheapest collapses are the ones that
    bridge between separate panels — it keeps a few far-apart vertices and joins
    them across the gap. `MeshCache.get` carries a comment warning about exactly
    this for a single bush; a whole compound is the same failure two orders of
    magnitude up.

    So the budget is spent in `read_building` instead, per part, where a
    simplifier only ever sees one panel's own edges and the allowance is spent as
    a tolerance. Nothing here is allowed to reduce across the assembly, and a
    building that still comes out over its target is left over its target — the
    caller drops a copy of it, which is a cost the picture does not pay.
    """
    before = mesh.get_triangle_count()
    if before <= 0:
        return mesh, before
    try:
        mesh = mesh.weld_mesh_edges(build.weld_options())
    except Exception as err:  # noqa: BLE001 - engine version differences
        log("  ! could not weld {}: {}".format(name, err))
    try:
        mesh = mesh.apply_simplify_to_planar(build.planar_options())
    except Exception:  # noqa: BLE001 - engine version differences
        pass
    return mesh, before


def _stamp(accum, mesh, transforms, log):
    """Append these instances if the budget still holds them. Returns triangles.

    One tile per call, so the ceiling `append` enforces is the budget still
    unspent rather than a share of a spatial tile: buildings arrive one building
    at a time, not one region at a time.
    """
    if not transforms:
        return 0
    triangles = mesh.get_triangle_count()
    room = max(0, accum.budget - accum.total.get_triangle_count())
    if triangles <= 0 or triangles > room:
        return 0
    accum.begin_tile(1)
    added = accum.append(mesh, transforms, limit=None) or 0
    accum.flush_tile()
    return added


def _fill_repeats(accum, pending, report, log):
    """Spend what is left of the budget on further copies, fairly.

    Round robin rather than in order, so the budget running out means "the last
    few copies of everything are missing" instead of "everything after the third
    building is missing". One copy at a time, because the copies are large and a
    building that no longer fits must not block the smaller ones behind it.
    """
    if not pending:
        return
    counts = {}
    progress = True
    while progress:
        progress = False
        for short, mesh, leftover, triangles in pending:
            if not leftover:
                continue
            added = _stamp(accum, mesh, leftover[:1], log)
            if not added:
                continue
            leftover.pop(0)
            counts[short] = counts.get(short, 0) + 1
            report["instances"] += 1
            report["triangles"] += added
            progress = True
    if counts:
        log(
            "   repeats: {}".format(
                ", ".join(
                    "{} +{}".format(name, n)
                    for name, n in sorted(counts.items(), key=lambda kv: -kv[1])[:10]
                )
            )
        )
    still = sum(len(leftover) for _s, _m, leftover, _t in pending)
    if still:
        log(
            "   {} further copies did not fit the {}k triangle building "
            "budget".format(still, accum.budget // 1000)
        )
    report["repeats_over_budget"] = still
    for _short, mesh, _leftover, _triangles in pending:
        mesh.reset()


def export(settings, filt, pool, cache, accum, log):
    """Add every instance of every building sub-level to `accum`.

    Must run after everything that needs the map open, because opening a
    sub-level closes it. Collects its own descriptor snapshot first, for the same
    reason.
    """
    # Proved before the map is closed, because closing it is irreversible and a
    # transform that cannot be built would otherwise surface twelve minutes in,
    # after the ground and the structures have been done and lost.
    blank = {
        "sub_levels_seen": 0,
        "sub_levels_eligible": 0,
        "sub_levels_opened": 0,
        "instances": 0,
        "triangles": 0,
        "skipped_unsolved": 0,
        "buildings": [],
    }
    # Not "does it construct" but "does it rotate": a quarter turn about Z must
    # take (100, 0, 0) to (0, 100, 0). A transform that silently ignored its
    # rotation would construct perfectly and stamp every building on the map at
    # the wrong angle, which is a worse outcome than not stamping them.
    try:
        # A quarter turn and a doubling: (100,0,0) must land at (10, 220, 30).
        # Checking the scale too, because it is applied now and a binding that
        # ignored it would stamp every building at the wrong size.
        probe = make_transform(10.0, 20.0, 30.0, math.pi * 0.5, 2.0)
        moved = probe.transform_location(unreal.Vector(100.0, 0.0, 0.0))
        if abs(moved.x - 10.0) > 1.0 or abs(moved.y - 220.0) > 1.0:
            raise RuntimeError(
                "a quarter turn at double scale took (100,0,0) to "
                "({:.1f},{:.1f},{:.1f})".format(moved.x, moved.y, moved.z)
            )
    except Exception as err:  # noqa: BLE001 - engine version differences
        log("sub-levels: skipped, transforms do not work here ({})".format(err))
        return dict(blank, error="make_transform: {}".format(err))

    map_package = str(settings.map or "").split(".", 1)[0]
    descs = unreal.WorldPartitionBlueprintLibrary.get_actor_descs()
    rows, counts = collect(descs, map_package, settings.sublevel_min_actors)
    report = {
        "sub_levels_seen": len(counts),
        "sub_levels_eligible": len(rows),
        "sub_levels_opened": 0,
        "instances": 0,
        "triangles": 0,
        "skipped_unsolved": 0,
        "buildings": [],
    }
    if not rows:
        log("sub-levels: none with {} or more actors".format(settings.sublevel_min_actors))
        return report

    # Which buildings, decided by where the players went if anyone has said.
    #
    # Ranking by actor count is what this did, and it is a guess that gets the
    # important case wrong: a hillside prefab outranks a town hall because
    # it holds more actors. With a traffic grid the order comes from the data the
    # export exists to be read against — and the buildings the telemetry draws a
    # floor plan inside are exactly the ones that must not be missing.
    # Is it building-sized? Asked in metres, from the descriptors, before anything
    # is opened.
    #
    # This replaced a threshold on actor count, which is a guess about how one
    # project authors its prefabs and therefore does not travel: at 200 actors it
    # excluded a large share of the test level's sub-levels including every
    # guard hut and shed, and any other number would be equally arbitrary
    # somewhere else. A bench is three metres across whether its prefab holds
    # four actors or four hundred.
    #
    # Clustering here rather than inside the open loop also means each sub-level
    # is clustered once and the result is carried through, which the loop below
    # used to redo.
    spots = hotspots_module.load(settings.hotspots, log)
    scored = []
    too_small = 0
    for name, entries in rows.items():
        groups = cluster(entries)
        # The *median* instance, not the largest.
        #
        # Largest was the obvious choice and it is wrong for one specific reason:
        # when the anchor clustering cannot separate a sub-level's copies, it
        # returns them all as one group, and the "extent" of that group is the
        # distance between copies scattered across the map rather than the size of
        # the building. Measured: a storage-shed sub-level was handed 80,000
        # triangles — the ceiling, for a shed of a few hundred parts — while a
        # factory of thousands of parts sat at the 2,000 floor. A merged group is
        # an outlier among the real ones, and a median ignores outliers, which is
        # the whole reason to use one.
        extents = sorted(_extent_metres(g) for g in groups)
        extent_m = extents[len(extents) // 2] if extents else 0.0
        if extent_m < settings.sublevel_min_size_m:
            too_small += 1
            continue
        traffic, bands = (0, 0)
        if spots:
            traffic, bands = spots.score_points([e[1][:2] for e in entries])
        scored.append((name, entries, traffic, bands, groups, extent_m))
    if spots:
        # Traffic first, then actor count as the tie-break — a sub-level with no
        # telemetry over it still ranks, just below everything that has some.
        scored.sort(key=lambda r: (-r[2], -len(r[1])))
        ranking = "the {} with the most traffic".format(settings.max_sublevels)
    else:
        scored.sort(key=lambda r: -len(r[1]))
        ranking = "the largest {} (no traffic grid — see tools/heat-grid.mjs)".format(
            settings.max_sublevels
        )
    log(
        "sub-levels: {} of {} are at least {:.1f}m across ({} too small, {} under {} "
        "actors); opening {}".format(
            len(scored),
            len(counts),
            settings.sublevel_min_size_m,
            too_small,
            len(counts) - len(rows),
            settings.sublevel_min_actors,
            ranking,
        )
    )
    report["sub_levels_eligible"] = len(scored)

    order = scored[: settings.max_sublevels]
    accum.expected = 1
    # A share each, rather than first come first served.
    #
    # Buildings are ordered by traffic, and the first few would otherwise spend
    # the whole budget: one factory alone came to over a million triangles across
    # its four instances. Everything after it would then be refused, which is the
    # failure this pass exists to fix, one level down.
    #
    # Each building's share of the budget, in proportion to its footprint.
    #
    # An equal share each is what this did, and equal shares are only fair when
    # the things sharing are alike. They are not: a tool shed has forty parts and
    # a hospital has three thousand, and handing both the same twenty thousand
    # triangles means the shed is exported twice over while the hospital gets
    # under a tenth of itself — a few hundred of its thousands of parts, which in
    # the viewer is a few slabs hanging in the air where a hospital should be.
    # That is the "half buildings" report, and no amount of extra budget fixes it
    # while the split is uniform, because the extra is divided equally too.
    #
    # Linear in size, not in area.
    #
    # Area is the tempting choice — it is what the building has to cover — and it
    # concentrates far too hard: squaring turns a ten-to-one spread of sizes into
    # a hundred-to-one spread of budgets, and measured across hundreds of
    # buildings it put most of them on the 2,000 floor while a handful took the
    # 80,000 ceiling. A budget in proportion to size spreads the same total over
    # the same buildings without the tail collapsing.
    # …and divided by what it will actually cost, which is per *copy*.
    #
    # The term that was missing, and its absence undid the rest. A building's
    # share buys one mesh, but the budget pays for that mesh once per placement:
    # a house with fifty copies costs fifty houses. Allocating by size alone
    # therefore promises more than the budget holds, and the overspend is
    # collected at the far end by refusing repeats — measured, hundreds of them,
    # which is hundreds of buildings missing from the map to buy detail on the
    # ones that remained. That is the wrong way round: a coarse building is still
    # a building, a missing one is a hole.
    #
    # Normalising by the total *stamped* weight makes the sum come out at the
    # budget exactly, so the sizing is fair and nothing has to be refused.
    weights = [max(1.0, e) for (_n, _e, _t, _b, _g, e) in order]
    stamped = sum(
        w * max(1, len(groups)) for w, (_n, _e, _t, _b, groups, _x) in zip(weights, order)
    ) or 1.0
    shares = [accum.budget * w / stamped for w in weights]
    editor = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    # Buildings whose remaining instances are waiting for the second pass, kept
    # in memory rather than re-derived: re-opening a level costs twenty seconds
    # and a reduced building is a few hundred thousand triangles.
    pending = []
    for index, (name, entries, traffic, bands, groups, extent_m) in enumerate(order):
        # What one copy of this building may cost. Not divided by how many copies
        # there are: they share the mesh, so the detail is bought once, and how
        # many copies the budget then holds is a separate question answered by
        # `_fill_repeats`.
        # ...and no more than its own size is worth.
        #
        # Ranking by traffic is right and it rewards the wrong things without
        # this: players walk *to* vehicles, so a parked-truck sub-level ranked
        # among the busiest on the map and a flatbed truck was handed tens of
        # thousands of triangles — a building's worth of budget, many times
        # over. Traffic says what is worth exporting; extent says how much of it
        # there is to draw.
        target = max(
            BUILDING_TRIANGLES_MIN,
            min(
                BUILDING_TRIANGLES_MAX,
                int(shares[index]),
                max(1, int(extent_m * BUILDING_TRIANGLES_PER_METRE)),
            ),
        )
        t0 = time.time()
        try:
            if not editor.load_level(name):
                log("  ! could not open {}".format(name))
                continue
        except Exception as err:  # noqa: BLE001 - a bad sub-level must not stop the export
            log("  ! could not open {}: {}".format(name, err))
            continue
        report["sub_levels_opened"] += 1

        mesh, local, stats = read_building(pool, cache, settings, filt, log, target)
        mesh, raw = reduce_building(mesh, target, log, name)
        triangles = mesh.get_triangle_count()

        placed = []
        residuals = []
        unsolved = {}
        for group in groups:
            transform, _matched, residual, why = solve(group, local)
            if transform is None:
                report["skipped_unsolved"] += 1
                unsolved[why] = unsolved.get(why, 0) + 1
                continue
            # Traffic at this particular copy, so the copies that get stamped are
            # the ones players used. A house placed fifty times across the map is
            # not fifty equally interesting houses: some are on a path everybody
            # takes and some are scenery at the edge of the world, and the budget
            # only holds a few.
            at = spots.score_points([e[1][:2] for e in group]) if spots else (0, 0)
            placed.append((at[0], transform))
            residuals.append(residual)
        placed.sort(key=lambda p: -p[0])
        transforms = [t for _n, t in placed]
        visited = sum(1 for n, _t in placed if n > 0)

        # One instance now, the rest in a second pass. See `_fill_repeats`.
        #
        # Spending each building's share as it arrives looks reasonable and gives
        # a bad answer: a house placed fifty times cannot be reduced below about
        # a hundred thousand triangles — thousands of disconnected panels have a
        # floor no simplifier goes under, and welding them does not help because
        # abutting panels do not share vertices — so fifty copies is millions of
        # triangles whatever share it is given. It took its share, got 2 copies,
        # and every building after it in the list got nothing. Which is the
        # failure this whole pass exists to fix, one level down again.
        #
        # So: every distinct building first, because a school that appears once is
        # worth more than the fiftieth copy of a house, and repeats afterwards with
        # whatever is left.
        short = name.rsplit("/", 1)[-1]
        added = 0
        if transforms and triangles > 0:
            added = _stamp(accum, mesh, transforms[:1], log)
        leftover = transforms[1:] if added else []
        if leftover:
            pending.append((short, mesh, leftover, triangles))
        else:
            mesh.reset()

        report["instances"] += 1 if added else 0
        report["triangles"] += added
        report["buildings"].append(
            {
                "level": short,
                "actors": len(entries),
                "traffic": traffic,
                "height_bands": bands,
                "placements": len(groups),
                "visited_placements": visited,
                "solved": len(residuals),
                "unsolved": len(groups) - len(residuals),
                "source_triangles": raw,
                "parts_used": stats["components"],
                "parts_available": stats["parts_available"],
                # Volumes and other non-geometry actors skipped by class. See
                # `Filter.exclude_actor` — before it existed these were exported,
                # which put a translucent box the size of its grounds around one
                # of the town's main buildings.
                "excluded_actors": stats["excluded_actors"],
                "ambiguous_labels": stats["ambiguous_labels"],
                "target": target,
                "triangles": triangles,
                "first_copy": added,
                "residual_cm": round(max(residuals), 1) if residuals else None,
                "seconds": round(time.time() - t0, 1),
            }
        )
        log(
            "  {}: {} triangles from {} over {}+{} boxed of {} parts, {} placements{}{}{} "
            "in {:.0f}s".format(
                short,
                triangles,
                raw,
                stats["components"],
                stats.get("boxed", 0),
                stats["parts_available"],
                len(residuals),
                " ({} visited)".format(visited) if spots else "",
                ", {} unsolved".format(len(groups) - len(residuals))
                if len(groups) != len(residuals)
                else "",
                "" if added else ", NOT PLACED",
                time.time() - t0,
            )
        )
        # Why a placement could not be solved, named rather than counted.
        #
        # A building ranked in by traffic and then not stamped looks exactly like
        # one that was never ranked, from the report — and the first is a bug
        # while the second is a budget. Telling them apart from the outside meant
        # a twenty-minute export per guess.
        for why, n in sorted(unsolved.items(), key=lambda kv: -kv[1]):
            log("    {} of {} placements unsolved: {}".format(n, len(groups), why))

    _fill_repeats(accum, pending, report, log)

    # Say what limited the coverage, in the words the question gets asked in.
    #
    # Every number needed to see that most eligible buildings were never opened
    # was already in this report, and it still took a round of "there must be a
    # building here and there is not" to go and read them. A count is not a
    # finding. So the report ends by naming whichever limit actually bit, or
    # saying that none did — which is the only version of this line that means
    # the export is complete.
    limits = []
    if report["sub_levels_opened"] < report["sub_levels_eligible"]:
        limits.append(
            "--sublevels {} stopped {} eligible sub-levels from being opened".format(
                settings.max_sublevels,
                report["sub_levels_eligible"] - report["sub_levels_opened"],
            )
        )
    ineligible = report["sub_levels_seen"] - report["sub_levels_eligible"]
    if ineligible > 0:
        limits.append(
            "{} sub-levels were not building-sized: under --sublevel-min-size {:.1f}m "
            "or under --sublevel-min-actors {}".format(
                ineligible, settings.sublevel_min_size_m, settings.sublevel_min_actors
            )
        )
    if report["skipped_unsolved"]:
        limits.append(
            "{} placements could not be solved (see MIN_MATCHES, MAX_RESIDUAL_CM)".format(
                report["skipped_unsolved"]
            )
        )
    if report.get("repeats_over_budget"):
        limits.append(
            "{} repeat placements did not fit the triangle budget".format(
                report["repeats_over_budget"]
            )
        )
    report["coverage_limits"] = limits
    if limits:
        log("buildings: not every building was exported —")
        for line in limits:
            log("  - {}".format(line))
    else:
        log("buildings: every eligible sub-level was opened and every placement stamped")
    return report
