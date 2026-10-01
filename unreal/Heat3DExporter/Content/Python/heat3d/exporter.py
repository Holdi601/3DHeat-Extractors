"""
The export, start to finish.

Order of operations, and why:

  1. Open the level and read the actor descriptors. Filtering happens here,
     against descriptors, because loading first and filtering second is what
     makes a big map impossible.
  2. Ground. Only the landscape is loaded for this pass, so every ray that hits
     something hit terrain.
  3. Structures, tile by tile. Each tile loads its actors, appends their
     geometry, reduces to its share of the budget, and unloads. Peak memory is
     one tile, not one map.
  4. Write one .glb and one .json beside it. The JSON is the receipt: what was
     kept, what was dropped and why, and how long each part took.
"""
import math
import os
import time

import unreal

from . import build, classify, collect, ground, sublevels, water
from . import settings as settings_module
from .glb import Part, write_glb


# Warn above this size cut, in metres. A shed is about eight metres across, so a
# cut above it is dropping whole small buildings and the smaller parts of large
# ones. Measured: the test level cut at over twenty metres produced roof slabs
# with no walls under them, and cut at a few metres produced buildings.
CUT_WARN_M = 10.0


# Where `log` mirrors everything it prints, as plain UTF-8. Set by `run`.
#
# The engine log is not a usable receipt. UnrealEditor-Cmd writes it as UTF-16,
# Python's own log lines arrive as UTF-8 inside it, and the result is a file in
# which every other line is mojibake — so the report that says which assets got
# how many triangles cannot be read without writing a decoder first. Twice that
# cost an hour, and once it produced a wrong diagnosis: a grep that found no
# trace lines was read as "the landmark never reaches the exporter" when the
# landmark was in the export all along and the grep could not see its own
# encoding.
_REPORT = []
_REPORT_PATH = [None]


def log(message):
    unreal.log("[heat3d] {}".format(message))
    _REPORT.append(message)
    path = _REPORT_PATH[0]
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    except Exception:  # noqa: BLE001 - a log must never break the export
        _REPORT_PATH[0] = None


# `HEAT3D_TRACE=name,name` narrates what happens to matching actors and assets.
#
# Comma-separated so a run can trace something known-missing *and* something
# known-present at once. That pairing matters more than it sounds: a trace that
# prints nothing looks identical whether the object never arrives or the trace
# itself is broken, and an export takes ten minutes to find out.
_TRACE = [s for s in os.environ.get("HEAT3D_TRACE", "").split(",") if s]


def _is_architecture(target):
    """Should this actor be offered before the scenery in its tile?

    Two ways to qualify, and the second is the one that finds buildings on a map
    whose assets are not helpfully named: the actor's label says so, or the actor
    was composed into a sub-level of its own. See `classify.place_of`.
    """
    if getattr(target, "place", ""):
        return True
    return bool(classify.PRIORITY_NAMES.search(target.label or ""))


def _traced(*names):
    if not _TRACE:
        return False
    for name in names:
        if not name:
            continue
        for want in _TRACE:
            if want in name:
                return True
    return False


def _load_level(map_path):
    t0 = time.time()
    subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not subsystem.load_level(map_path):
        raise RuntimeError("could not open level {}".format(map_path))
    log("opened {} in {:.0f}s".format(map_path, time.time() - t0))


def _content_bounds(targets, settings):
    """The world the actors actually occupy, not the partition grid's extent.

    World Partition reports a grid that spans tens of kilometres around content
    that spans two; sizing tiles or a ground grid from that wastes almost all of
    both.

    Enormous actors are exported but do not get a say in this. Water is the
    obvious case — an ocean is larger than the map by design — and a landmark is
    the one that cost real damage: dropping anything over 2km to protect the
    bounds also dropped the level's centrepiece, a landmark mesh and the most
    important building on it. Measuring without them keeps the bounds honest and
    keeps the geometry.
    """
    limit = settings.bounds_max_size
    lo = [float("inf")] * 2
    hi = [float("-inf")] * 2
    zlo, zhi = float("inf"), float("-inf")
    for t in (t for t in targets if not t.water and t.size <= limit):
        lo[0] = min(lo[0], t.min[0])
        lo[1] = min(lo[1], t.min[1])
        hi[0] = max(hi[0], t.max[0])
        hi[1] = max(hi[1], t.max[1])
        zlo = min(zlo, t.min[2])
        zhi = max(zhi, t.max[2])
    if lo[0] == float("inf"):
        return None
    return (tuple(lo), tuple(hi), zlo, zhi)


def _tile_size(bounds, settings, target_count):
    if settings.tile > 0:
        return settings.tile
    (min_x, min_y), (max_x, max_y) = bounds[0], bounds[1]
    span = max(max_x - min_x, max_y - min_y)
    # Few enough tiles that each still gets a meaningful share of the budget —
    # a hundredth of it reduces a town square to nothing, and every tile is
    # simplified separately, so its boundary can crack against its neighbour.
    # Large enough that one tile's raw geometry is a few hundred thousand
    # triangles rather than millions. Around fifty occupied tiles on a big map.
    tiles = max(1, min(16, int(math.sqrt(max(1, target_count) / 2000.0))))
    return max(5000.0, span / max(1, tiles))


def _instance_transforms(component):
    """World transforms for everything this component draws.

    An instanced component is one component and hundreds of placements, and
    `AppendMeshTransformed` takes them all in one call — which is why a map with
    a hundred thousand foliage instances is not a hundred thousand calls.
    """
    if isinstance(component, unreal.InstancedStaticMeshComponent):
        out = []
        for i in range(component.get_instance_count()):
            try:
                transform = component.get_instance_transform(i, True)
            except Exception:  # noqa: BLE001 - malformed instance data
                continue
            # Every placement checked, because one bad one is fatal and silent.
            # See build.sane_transform.
            if build.sane_transform(transform):
                out.append(transform)
        return out
    world = component.get_world_transform()
    return [world] if build.sane_transform(world) else []


def _mesh_size(static_mesh, transform):
    """Rough world size of one placement, for its triangle allowance.

    Returns 0 when the asset cannot say how big it is, rather than a guess. A
    static mesh's bounding box comes from its render data, and a commandlet
    started with `-nullrhi` has none for every asset — so this can legitimately
    come back empty for a mesh that is kilometres across, and the caller has to
    fall back to what the descriptor said instead of treating it as tiny.

    That is not hypothetical. It is why the level's centrepiece was missing: a
    landmark mesh with an actor bound kilometres across, its asset reported
    nothing here, the size threshold
    read that as "smaller than the minimum" and skipped it — before the mesh cache,
    before any budgeting, with no drop reason recorded anywhere because this
    check does not log one.
    """
    try:
        box = static_mesh.get_bounding_box()
        extent = box.max - box.min
    except Exception:  # noqa: BLE001 - engine version differences
        return 0.0
    s = transform.scale3d
    return math.sqrt(
        (extent.x * s.x) ** 2 + (extent.y * s.y) ** 2 + (extent.z * s.z) ** 2
    )


def run(settings):
    started = time.time()
    report = {"map": settings.map, "settings": settings.to_dict(), "steps": {}}

    # Start the plain-text mirror before anything is logged. See `log`.
    try:
        directory = os.path.dirname(os.path.abspath(settings.out))
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        _REPORT_PATH[0] = os.path.splitext(os.path.abspath(settings.out))[0] + ".log.txt"
        with open(_REPORT_PATH[0], "w", encoding="utf-8") as handle:
            handle.write("")
    except Exception:  # noqa: BLE001 - a log must never break the export
        _REPORT_PATH[0] = None

    _load_level(settings.map)
    world = unreal.EditorLevelLibrary.get_editor_world()
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

    filt = collect.Filter(settings)
    partitioned = collect.is_partitioned()
    t0 = time.time()
    if partitioned:
        targets, landscape, total = collect.collect_partitioned(settings, filt)
    else:
        targets, landscape, total = collect.collect_loaded(settings, filt)
    if settings.max_actors:
        targets = targets[: settings.max_actors]

    # Anything sitting entirely above the terrain is scenery for the sky.
    #
    # Found by measuring the finished file: after the marker boxes were gone, the
    # largest triangles left in the structure part were all in a thin band far
    # above the terrain, spanning the whole map — a cloud layer, drawn as a few
    # dozen enormous quads. Nothing a player stands on, shelters
    # behind or is blocked by is *entirely* two hundred metres above the highest
    # ground, so this needs no list of names and will do the same job on a map
    # whose clouds are called something else.
    #
    # Measured against the actor's *lowest* point, so a radar mast, a crane or a
    # cable car pylon — tall things rooted on the ground — are unaffected. Water
    # is exempt because an ocean is positioned by its own rules.
    if landscape.valid and settings.sky_clearance > 0:
        sky_floor = landscape.max[2] + settings.sky_clearance
        before = len(targets)
        targets = [t for t in targets if t.water or t.min[2] <= sky_floor]
        above = before - len(targets)
        if above:
            filt.dropped["above the terrain"] = above
            log(
                "dropped {} actors sitting entirely more than {:.0f}m above the "
                "highest ground".format(above, settings.sky_clearance / 100.0)
            )
    log(
        "kept {} of {} actors in {:.0f}s ({})".format(
            len(targets), total, time.time() - t0, "partitioned" if partitioned else "loaded"
        )
    )
    report["steps"]["collect"] = {
        "seconds": round(time.time() - t0, 1),
        "partitioned": partitioned,
        "actors_total": total,
        "actors_kept": len(targets),
        "landscape_actors": len(landscape.guids),
        "landscape_extent_m": [
            round((landscape.max[i] - landscape.min[i]) / 100.0) for i in range(3)
        ]
        if landscape.valid
        else None,
        "dropped": dict(sorted(filt.dropped.items(), key=lambda kv: -kv[1])),
        # A sample of what each reason removed, because a count cannot tell
        # grass from a building.
        "dropped_examples": filt.dropped_examples,
    }

    bounds = _content_bounds(targets, settings)
    if bounds is None:
        raise RuntimeError("nothing left to export after filtering")
    (min_x, min_y), (max_x, max_y), min_z, max_z = bounds
    report["bounds_cm"] = {
        "min": [min_x, min_y, min_z],
        "max": [max_x, max_y, max_z],
        "extent_m": [
            round((max_x - min_x) / 100.0),
            round((max_y - min_y) / 100.0),
            round((max_z - min_z) / 100.0),
        ],
    }
    log(
        "content spans {}x{}x{} m".format(
            *report["bounds_cm"]["extent_m"]
        )
    )

    # The budget, if nobody named one. It can only be decided here: it depends on
    # how big the level turns out to be, and that is not known until the
    # descriptors have been read and filtered. This is what makes the one-click
    # menu export usable on a map of any size — the editor path has no way to pass
    # a budget, and a fixed default is either wasteful on an arena or produces
    # roofs without walls on an open world.
    if settings.budget <= 0:
        extent_m = max(report["bounds_cm"]["extent_m"][0], report["bounds_cm"]["extent_m"][2])
        settings.budget = settings_module.budget_for(extent_m)
        report["settings"]["budget"] = settings.budget
        report["budget_chosen"] = True
        log(
            "budget not given: {}k triangles for a {}m level".format(
                settings.budget // 1000, extent_m
            )
        )

    if settings.dry_run:
        report["seconds"] = round(time.time() - started, 1)
        return report, None

    pool = unreal.DynamicMeshPool()
    parts = []

    # ------------------------------------------------------------- ground
    ground_budget = int(settings.budget * settings.ground_share)
    if settings.include_landscape and ground_budget > 0:
        t0 = time.time()
        if partitioned and landscape.guids:
            unreal.WorldPartitionBlueprintLibrary.load_actors(landscape.guids)
        # Two triangles per cell, so the grid side is set by the budget.
        resolution = max(16, min(2048, int(math.sqrt(ground_budget / 2.0))))
        # Over the landscape's own footprint where there is one. Stretching the
        # grid over every actor's bounds instead spends the rays on the sky.
        if landscape.valid:
            grid_bounds = landscape.footprint()
            ceiling = landscape.max[2] + 50000.0
            floor = landscape.min[2] - 50000.0
        else:
            grid_bounds = ((min_x, min_y), (max_x, max_y))
            ceiling = max_z + 100000.0
            floor = min_z - 100000.0
        region = settings.region
        if region:
            grid_bounds = (
                (max(grid_bounds[0][0], region[0]), max(grid_bounds[0][1], region[1])),
                (min(grid_bounds[1][0], region[2]), min(grid_bounds[1][1], region[3])),
            )
        heights, hits = ground.sample_grid(
            world, grid_bounds, resolution, ceiling=ceiling, floor=floor, log=log
        )
        positions, indices = ground.triangulate(
            heights, resolution, grid_bounds, settings.scale
        )
        if len(indices):
            parts.append(
                Part(classify.part_name(classify.GROUND, settings.name), positions, indices)
            )
        report["steps"]["ground"] = {
            "seconds": round(time.time() - t0, 1),
            "resolution": resolution,
            "rays": resolution * resolution,
            "hits": hits,
            "triangles": len(indices) // 3,
            "cell_metres": round(
                (grid_bounds[1][0] - grid_bounds[0][0]) / max(1, resolution) / 100.0, 1
            ),
            "from_landscape": bool(landscape.valid),
        }
        log(
            "ground: {} of {} rays hit, {} triangles in {:.0f}s".format(
                hits, resolution * resolution, len(indices) // 3, time.time() - t0
            )
        )
        if partitioned and landscape.guids:
            unreal.WorldPartitionBlueprintLibrary.unload_actors(landscape.guids)
            unreal.SystemLibrary.collect_garbage()

    # ------------------------------------------------------------- water
    #
    # Before the structures, and taken out of their target list: a water body's
    # visible surface is not in its static meshes. See water.py.
    water_targets = [t for t in targets if t.water]
    targets = [t for t in targets if not t.water]
    if water_targets and settings.include_water:
        t0 = time.time()
        if partitioned:
            unreal.WorldPartitionBlueprintLibrary.load_actors([t.guid for t in water_targets])
        wanted = {t.key for t in water_targets}
        actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        found = [
            a
            for a in actors.get_all_level_actors()
            if (a.get_actor_label() if partitioned else a.get_path_name()) in wanted
        ]
        positions, windices, kinds = water.build(
            found, settings.scale, log, clip=((min_x, min_y), (max_x, max_y))
        )
        if windices:
            parts.append(
                Part(classify.part_name(classify.WATER, settings.name), positions, windices)
            )
        report["steps"]["water"] = {
            "seconds": round(time.time() - t0, 1),
            "actors": len(water_targets),
            "surfaces": kinds.get("lake", 0) + kinds.get("river", 0) + kinds.get("sea", 0),
            "sea": kinds.get("sea", 0),
            "lakes": kinds.get("lake", 0),
            "rivers": kinds.get("river", 0),
            "triangles": len(windices) // 3,
        }
        if partitioned:
            unreal.WorldPartitionBlueprintLibrary.unload_actors([t.guid for t in water_targets])
            unreal.SystemLibrary.collect_garbage()
    elif water_targets:
        report["steps"]["water"] = {"actors": len(water_targets), "triangles": 0, "skipped": True}

    # --------------------------------------------------------- structures
    t0 = time.time()
    structure_budget = settings.budget - ground_budget
    # Spend the budget on the largest objects rather than on all of them. See
    # build.plan: below about two dozen triangles an object is a blob, and a
    # simplifier cannot merge separate objects, so "everything, tiny" is not on
    # the menu — it is "the big things, recognisable" or nothing.
    kept, allowances, cut, planned = build.plan(targets, structure_budget, settings)
    if len(kept) < len(targets):
        log(
            "budget holds the largest {} of {} actors — everything above {:.1f}m "
            "({}k triangles planned)".format(
                len(kept), len(targets), cut / 100.0, planned // 1000
            )
        )
        report["steps"]["collect"]["actors_over_budget"] = len(targets) - len(kept)
        report["steps"]["collect"]["size_cut_m"] = round(cut / 100.0, 1)
    targets = kept

    tile = _tile_size(bounds, settings, len(targets))
    buckets = {}
    for target in targets:
        cx, cy = target.centre
        key = (int((cx - min_x) // tile), int((cy - min_y) // tile))
        buckets.setdefault(key, []).append(target)

    cache = build.MeshCache(pool, log, use_boxes=settings.box_substitutes)
    # Two accumulators over the same tiles.
    #
    # Not every static mesh placed on a level is a *structure*. Roads, paths and
    # rock formations are landscape built out of meshes, and the difference
    # matters because of how the viewer draws the two: structures are translucent
    # so you can see the heat inside a building, and a cliff drawn that way is a
    # glass crystal the size of a hill. Roads drawn that way are worse — they are
    # the walk paths, the thing a movement heatmap is *about*, and they were being
    # rendered as faint cyan shells.
    #
    # So each component is classified as it is reached, and the landscape ones go
    # into a second accumulator emitted as another `ground:` part. Measured on
    # the test level, that is a sizeable share of what used to be "structures":
    # mostly rock, the rest road.
    terrain_budget = int(structure_budget * build.TERRAIN_MESH_SHARE)
    structure_budget -= terrain_budget
    # And the buildings' share, carved out before anything is spent so the total
    # still holds. They are exported from their own sub-levels in a pass of their
    # own — see heat3d/sublevels.py — and it runs last, so without reserving this
    # here it would find the budget gone.
    building_budget = (
        int(structure_budget * build.BUILDING_BUDGET_SHARE)
        if settings.include_sublevels
        else 0
    )
    structure_budget -= building_budget
    accum = build.Accumulator("structure", pool, structure_budget, log)
    terrain = build.Accumulator("ground meshes", pool, terrain_budget, log)
    buildings = build.Accumulator("buildings", pool, building_budget, log)
    # Per-tile weights from the plan, so each tile's ceiling is proportional to
    # the geometry it is expected to hold.
    tile_expected = {
        key: sum(allowances.get(id(t), build.SHELL_FLOOR) for t in group)
        for key, group in buckets.items()
    }
    accum.expected = max(1, sum(tile_expected.values()))
    terrain.expected = accum.expected
    log(
        "structures: {} actors in {} tiles of {:.0f}m, expecting ~{}k triangles".format(
            len(targets), len(buckets), tile / 100.0, accum.expected // 1000
        )
    )

    # Vegetation's ceiling, and it has to be *per tile*.
    #
    # It was one running total across the whole map, and tiles are visited in
    # spatial order, so the first tiles reached spent the entire allowance and
    # every tile after them got none. On the test level that put all the trees in
    # a band through the middle and the right-hand side of the map and left the
    # rest bare — which reads as the exporter having lost half the foliage, and is
    # really the exporter having spent it all in one place.
    #
    # Proportional to the tile's own share, so a wood is a wood wherever it is.
    foliage_share = build.FOLIAGE_BUDGET_SHARE
    spent_by_class = {}
    spent_by_asset = {}
    spent_by_band = {}
    # And by the place each actor was composed in, which is the receipt that
    # answers the question the user actually asks. "architecture 1.4M" is a number
    # nobody can check; "OldTown 800k, Factory 200k, Farmhouse 150k" is a
    # list they can hold against the map they know.
    spent_by_place = {}
    arch_assets = {}
    # Components whose asset could not report its own bounds; see _mesh_size.
    no_mesh_bounds = 0

    done = 0
    for index, (key, tile_targets) in enumerate(sorted(buckets.items())):
        accum.begin_tile(tile_expected.get(key, 0))
        terrain.begin_tile(tile_expected.get(key, 0))
        foliage_cap = max(
            build.MIN_OBJECT_TRIANGLES, int(accum.tile_share * foliage_share)
        )
        foliage_spent = 0
        # The band of this tile only architecture may spend, and the ceiling on
        # everything else. See build.ARCH_TILE_RESERVE.
        other_cap = accum.tile_share - int(accum.tile_share * build.ARCH_TILE_RESERVE)
        other_spent = 0
        # Architecture first, then largest first.
        #
        # The second half of that was the whole ordering until now, and it is why
        # buildings were missing from every export: the largest actors on a map
        # are rocks and foliage clusters, they were offered first, and a tile was
        # empty before the first house was reached. Components were already
        # sorted by priority *within* an actor, which fixed the farmyard case and
        # could not fix this one.
        tile_targets = sorted(
            tile_targets,
            key=lambda t: (
                0 if _is_architecture(t) else 1,
                -build.rank_weight(t),
            ),
        )
        for group in collect.batches(tile_targets, settings.batch):
            if partitioned:
                unreal.WorldPartitionBlueprintLibrary.load_actors([t.guid for t in group])
            # Match loaded actors back to the targets in this group. This applies
            # to both paths: on a level that is already loaded, iterating every
            # actor instead would quietly ignore the filters and export the
            # markers, volumes and grass along with everything else.
            lookup = {t.key: t for t in group}
            if _TRACE:
                for t in group:
                    if _traced(t.label, t.key):
                        log(
                            "  trace: {} is in this batch as a target "
                            "({:.0f}m, class {})".format(t.label, t.size / 100.0, t.cls)
                        )
            seen_labels = set()
            for actor in actors.get_all_level_actors():
                label = actor.get_actor_label() if partitioned else actor.get_path_name()
                if _traced(label):
                    seen_labels.add(label)
                    log(
                        "  trace: {} is loaded, class {}, matched={}".format(
                            label, actor.get_class().get_name(), label in lookup
                        )
                    )
                target = lookup.get(label)
                if target is None:
                    continue
                if isinstance(actor, unreal.LandscapeProxy):
                    if _traced(label):
                        log("  trace: {} skipped as a LandscapeProxy".format(label))
                    continue
                # No single actor may take more than its share of a tile. Without
                # this, a forest — one actor whose bounds span everything it
                # plants, and therefore first in size order — consumed whole tiles
                # and the export came out as trees with no buildings in it.
                # One allowance per accumulator, not one shared between them: an
                # actor that is a hillside of rock plus two sheds would otherwise
                # spend the whole allowance on rock and leave the sheds nothing,
                # which is the same failure as the forest taking the tile, one
                # level down.
                # A named building gets a much larger share than a rock does —
                # it is one actor, it has few placements, and it is the thing
                # the map is read against.
                actor_share = (
                    build.PER_ACTOR_TILE_SHARE_PRIORITY
                    if _is_architecture(target)
                    else build.PER_ACTOR_TILE_SHARE
                )
                room = {
                    classify.STRUCTURE: max(
                        build.MIN_OBJECT_TRIANGLES * 4,
                        int(accum.tile_share * actor_share),
                    ),
                    classify.GROUND: max(
                        build.MIN_OBJECT_TRIANGLES * 4,
                        int(terrain.tile_share * actor_share),
                    ),
                }
                # Must-have geometry first, scenery last.
                #
                # Within one actor the order used to be whatever the engine
                # returned, and when the actor ran out of room what survived was
                # whatever happened to come first. On an actor holding a
                # farmyard's worth of trees and one barn, that was the trees.
                # Sorting by `classify.priority` means the barn and its fences are
                # offered before anything that is only decoration.
                components = sorted(
                    (
                        c
                        for c in actor.get_components_by_class(unreal.StaticMeshComponent)
                        if c.static_mesh
                    ),
                    key=lambda c: -classify.priority(
                        target.label, c.static_mesh.get_path_name(), target.place
                    ),
                )
                if _traced(target.label):
                    log(
                        "  trace: {} has {} mesh components".format(
                            target.label, len(components)
                        )
                    )
                for component in components:
                    static_mesh = component.static_mesh
                    asset = classify.asset_name(static_mesh.get_path_name())
                    tr = _traced(target.label, asset)
                    # By asset name as well as by actor label. Decoration lives in
                    # components, not actors: the roots, vines and rubble that
                    # covered a whole export belong to actors whose labels say
                    # nothing about them.
                    if filt.exclude_asset(asset):
                        if tr:
                            log("  trace: {} dropped by an asset name pattern".format(asset))
                        continue
                    rank = classify.priority(
                        target.label, static_mesh.get_path_name(), target.place
                    )
                    # Vegetation is what it *is*, not how it was placed.
                    #
                    # The cap used to apply only to components of the foliage
                    # system, and a tree dropped into a level by hand is an
                    # ordinary StaticMeshActor. So half the vegetation on the map
                    # was never capped at all: measured, nearly twice as many
                    # triangles of vegetation as for every wall, fence, stair and
                    # bridge put together. For a map read against the things that
                    # constrain movement, that is backwards, and no amount of
                    # tuning the foliage share could fix it because the trees in
                    # question were not counted as foliage.
                    placed_foliage = isinstance(
                        component, unreal.FoliageInstancedStaticMeshComponent
                    )
                    vegetation = placed_foliage or rank < 0
                    if vegetation and not settings.include_foliage:
                        if tr:
                            log("  trace: {} dropped, foliage disabled".format(asset))
                        continue
                    if vegetation and foliage_spent >= foliage_cap:
                        if tr:
                            log("  trace: {} dropped, vegetation cap spent".format(asset))
                        continue
                    transforms = _instance_transforms(component)
                    if not transforms:
                        if tr:
                            log("  trace: {} has no instance transforms".format(asset))
                        continue
                    size = _mesh_size(static_mesh, transforms[0])
                    if size <= 1.0:
                        # The asset could not say how big it is — see `_mesh_size`.
                        # The descriptor can, and for a single-component actor the
                        # two are the same thing anyway.
                        size = target.size
                        no_mesh_bounds += 1
                    # The size threshold, applied a second time — and it has to
                    # know about the architecture exemption or the exemption does
                    # nothing at all.
                    #
                    # This is where around a hundred thousand actors rescued at
                    # the descriptor stage were being thrown away again. The
                    # filter exempts named architecture from `min_size` because a
                    # wall panel, a doorway and a flight of steps are all under
                    # five metres; this loop
                    # then measured the same mesh against the same five metres and
                    # skipped it. The buildings were let in through the front door
                    # and out through the back, and the report showed them as
                    # kept, because they were — kept and then dropped.
                    if vegetation:
                        floor_size = settings.foliage_min_size
                    elif rank > 0:
                        floor_size = settings.priority_min_size
                    else:
                        floor_size = settings.min_size
                    if size < floor_size:
                        if tr:
                            log(
                                "  trace: {} dropped, {:.0f}cm under the {:.0f}cm "
                                "floor".format(asset, size, floor_size)
                            )
                        continue
                    if tr:
                        log(
                            "  trace: {} reaching the mesh cache, {} placements, "
                            "rank {}".format(asset, len(transforms), rank)
                        )
                    # One component can be hundreds of placements — a fence, a
                    # row of barriers, a field of trees — and the descriptor that
                    # earned this actor its allowance said nothing about that.
                    # Damping by the square root keeps a 400-instance component
                    # from costing 400 times a single one while still giving it
                    # more than one object's worth.
                    allowance = allowances.get(
                        id(target), build.allowance_for(size, settings)
                    )
                    if len(transforms) > 1:
                        allowance = max(
                            build.SHELL_FLOOR,
                            int(allowance / math.sqrt(len(transforms))),
                        )
                    # Then the priority adjustment, which is the whole point of
                    # the ranking: a fence panel is small, it comes in rows of
                    # forty, and both of those push its allowance under the floor
                    # — so it is exactly the thing that disappears, and it is
                    # exactly the thing a movement heatmap is read against.
                    # Guaranteed the floor here, so it is never dropped for being
                    # cheap; vegetation gives up part of its share to pay for it.
                    if rank > 0:
                        # What its own size deserves, at least. The planned
                        # allowance is the budget divided across every actor a
                        # descriptor scan kept — hundreds of thousands of them on
                        # the test level, so about a hundred triangles each — and
                        # a building given a hundred triangles is a lump.
                        # `allowance_for` answers the question the plan cannot:
                        # how much does a thing this size need to still look like
                        # itself.
                        allowance = max(
                            allowance,
                            int(
                                build.allowance_for(size, settings)
                                * build.ARCH_ALLOWANCE_WEIGHT
                            ),
                            build.MIN_OBJECT_TRIANGLES,
                        )
                    elif rank < 0:
                        allowance = int(allowance * build.VEGETATION_ALLOWANCE_WEIGHT)
                    # Scraps are stripped from foliage only — a tree's loose
                    # shells are leaf cards, a building's are its wall panels —
                    # and boxes are substituted for everything *but* foliage, on
                    # the grounds that a crate standing where a tree stood is a
                    # worse answer than nothing.
                    mesh = cache.get(
                        static_mesh,
                        allowance,
                        size,
                        strip_small=vegetation,
                        allow_box=not vegetation,
                    )
                    # Landscape built out of meshes — roads, paths, rock — goes to
                    # the other accumulator and comes out as ground. Foliage never
                    # does: a tree is not terrain however rocky its name.
                    kind = (
                        classify.STRUCTURE
                        if vegetation
                        else classify.classify(
                            target.cls,
                            target.label,
                            component.get_class().get_name(),
                            static_mesh.get_path_name(),
                        )
                    )
                    into = terrain if kind == classify.GROUND else accum
                    limit = room[kind]
                    if vegetation:
                        limit = min(limit, foliage_cap - foliage_spent)
                    if rank <= 0 and kind == classify.STRUCTURE:
                        # Scenery may not spend the band held for architecture.
                        limit = min(limit, other_cap - other_spent)
                    added = into.append(mesh, transforms, limit) or 0
                    if tr:
                        log(
                            "  trace: {} -> {} triangles appended (mesh {}, limit {})".format(
                                asset,
                                added,
                                "none" if mesh is None else mesh.get_triangle_count(),
                                limit,
                            )
                        )
                    room[kind] -= added
                    if vegetation:
                        foliage_spent += added
                    if rank <= 0 and kind == classify.STRUCTURE:
                        other_spent += added
                    cls = target.cls if target else 'unknown'
                    if kind == classify.GROUND:
                        cls = "ground mesh"
                    spent_by_class[cls] = spent_by_class.get(cls, 0) + added
                    # By priority band as well, which is the question that keeps
                    # being asked and could not be answered from the report: how
                    # much of the budget reached the walls, doors and stairs, and
                    # how much went to scenery. A per-class total cannot say —
                    # almost everything on a map is a StaticMeshActor.
                    band = (
                        "architecture" if rank > 0 else "vegetation" if rank < 0 else "other"
                    )
                    spent_by_band[band] = spent_by_band.get(band, 0) + added
                    if target.place and added:
                        # The sub-level's own name, not the whole content path.
                        place = target.place.rsplit("/", 1)[-1]
                        spent_by_place[place] = spent_by_place.get(place, 0) + added
                    if rank > 0 and added:
                        name = classify.asset_name(static_mesh.get_path_name())
                        arch_assets[name] = arch_assets.get(name, 0) + added
                    # Per asset as well as per class, because "why is this one
                    # decorative arch the most detailed thing on the map" is a
                    # question the class totals cannot answer and a reader of the
                    # report will ask. Path, not label: one asset placed six
                    # hundred times is one line.
                    if added:
                        path = static_mesh.get_path_name()
                        prev = spent_by_asset.get(path)
                        if prev is None:
                            spent_by_asset[path] = [added, len(transforms)]
                        else:
                            prev[0] += added
                            prev[1] += len(transforms)
                    if room[classify.STRUCTURE] <= 0 and room[classify.GROUND] <= 0:
                        break
            if partitioned:
                unreal.WorldPartitionBlueprintLibrary.unload_actors([t.guid for t in group])
            # Collect per batch, not per tile. Unloading only drops the packages'
            # references; until a collection runs, both the actor packages and
            # the static meshes they pulled in stay resident, and a tile can be
            # dozens of batches wide. Measured on the test level: 25GB of a 64GB
            # machine with a per-tile collection, and the export was heading for
            # swap.
            unreal.SystemLibrary.collect_garbage()
            done += len(group)
        accum.flush_tile()
        terrain.flush_tile()
        if (index + 1) % 4 == 0 or index + 1 == len(buckets):
            log(
                "  tile {}/{}: {} actors done, {} instances, {} triangles, {:.0f}s".format(
                    index + 1,
                    len(buckets),
                    done,
                    accum.instances,
                    accum.total.get_triangle_count(),
                    time.time() - t0,
                )
            )

    # What the budget actually went to, by actor class. The question "why are
    # there no buildings in this export?" took a screenshot to answer; this
    # answers it from the report.
    report["steps"]["structures_by_band"] = dict(
        sorted(spent_by_band.items(), key=lambda kv: -kv[1])
    )
    # Logged, not just filed in the json. This is the one number that says whether
    # the export answers the question it is for, and for four rounds of debugging
    # it existed only inside a report nobody read: architecture getting less than
    # vegetation is "the buildings are missing", stated plainly, and it was
    # on disk the whole time.
    report["steps"]["structures_by_place"] = dict(
        sorted(spent_by_place.items(), key=lambda kv: -kv[1])[:30]
    )
    if spent_by_place:
        log(
            "   composed places: {}".format(
                ", ".join(
                    "{} {}k".format(name, value // 1000)
                    for name, value in sorted(
                        spent_by_place.items(), key=lambda kv: -kv[1]
                    )[:10]
                )
            )
        )
    log(
        "   budget by band: {}".format(
            ", ".join(
                "{} {}k".format(band, value // 1000)
                for band, value in sorted(spent_by_band.items(), key=lambda kv: -kv[1])
            )
            or "nothing spent"
        )
    )
    report["steps"]["architecture_assets"] = [
        {"asset": k, "triangles": v}
        for k, v in sorted(arch_assets.items(), key=lambda kv: -kv[1])[:20]
    ]
    report["steps"]["structures_by_class"] = dict(
        sorted(spent_by_class.items(), key=lambda kv: -kv[1])[:15]
    )
    # Refresh the drop counters. They were snapshotted straight after the
    # descriptor scan, which is before any component has been looked at, so
    # everything the asset-name filter rejects was missing from the report — a
    # filter doing real work and reporting nothing.
    report["steps"]["collect"]["dropped"] = dict(
        sorted(filt.dropped.items(), key=lambda kv: -kv[1])
    )
    # And the twenty assets that took the most, with how many placements each
    # total covers, so a single hungry asset is visible rather than inferred.
    report["steps"]["structures_by_asset"] = [
        {
            "asset": path.rsplit("/", 1)[-1].split(".")[0],
            "triangles": spent[0],
            "instances": spent[1],
            "per_instance": spent[0] // max(1, spent[1]),
        }
        for path, spent in sorted(spent_by_asset.items(), key=lambda kv: -kv[1][0])[:20]
    ]
    for row in report["steps"]["structures_by_asset"][:5]:
        log(
            "  top asset {}: {}k triangles over {} placements ({} each)".format(
                row["asset"], row["triangles"] // 1000, row["instances"], row["per_instance"]
            )
        )

    structure_mesh = accum.finish()
    report["steps"]["structures"] = accum.stats()
    report["steps"]["structures"]["seconds"] = round(time.time() - t0, 1)
    report["steps"]["structures"]["tile_metres"] = round(tile / 100.0)

    # ------------------------------------------------- buildings in sub-levels
    #
    # Last, and it has to be last: reading a sub-level means opening it, which
    # closes the map. Everything above needs the map open.
    if settings.include_sublevels and building_budget > 0:
        t0 = time.time()
        # Never fatal. Everything above it is already done and would be thrown
        # away, and this pass has more ways to fail than the rest of the exporter
        # put together: it opens levels it did not choose, on maps it has never
        # seen, through APIs that move between engine versions.
        try:
            report["steps"]["sublevels"] = sublevels.export(
                settings, filt, pool, cache, buildings, log
            )
        except Exception as err:  # noqa: BLE001 - see above
            log("  ! the sub-level pass failed: {}: {}".format(type(err).__name__, err))
            report["steps"]["sublevels"] = {
                "error": "{}: {}".format(type(err).__name__, err),
                "instances": 0,
                "triangles": 0,
                "sub_levels_opened": 0,
            }
        report["steps"]["sublevels"]["seconds"] = round(time.time() - t0, 1)
        building_mesh = buildings.finish()
        if building_mesh.get_triangle_count() > 0:
            # Into the structure part: these are buildings, and the viewer draws
            # `structure:` translucently so the heat inside one stays visible.
            structure_mesh = structure_mesh.append_mesh(
                building_mesh, unreal.Transform()
            )
        step = report["steps"]["sublevels"]
        log(
            "   buildings: {} instances of {} sub-levels, {}k triangles".format(
                step.get("instances", 0),
                step.get("sub_levels_opened", 0),
                step.get("triangles", 0) // 1000,
            )
        )
    report["steps"]["meshes"] = {
        "unique": len(cache.entries),
        "read_seconds": round(cache.read_seconds, 1),
        "simplify_seconds": round(cache.simplify_seconds, 1),
        "source_triangles": cache.source_triangles,
        # After the coplanar merge alone, which costs no shape at all: the gap
        # between this and source_triangles is free detail removal, and the gap
        # between it and kept_triangles is what was paid for in accuracy.
        "planar_triangles": cache.planar_triangles,
        # Assets replaced by their bounding box because no shape could survive
        # the allowance. A box is an honest "something this big stands here".
        "boxed_assets": cache.boxed,
        # Of those, the ones that were boxed *after* a reduction was attempted
        # and came out as sheets. See DEGENERATE_TRIANGLES.
        "boxed_degenerate": cache.degenerate,
        # And which they were. A screenshot full of cubes is a question the
        # totals cannot answer.
        "boxed_examples": [
            path.rsplit("/", 1)[-1].split(".")[0]
            for path in sorted(cache.boxed_by_asset, key=cache.boxed_by_asset.get, reverse=True)[
                :15
            ]
        ],
        # Assets too small for even a box to be worth drawing. See BOX_MIN_SIZE.
        "dropped_no_shape": cache.dropped_small,
        # Named, so "where is my fence" is answerable from the report.
        "dropped_examples": [
            path.rsplit("/", 1)[-1].split(".")[0]
            for path in sorted(
                cache.dropped_by_asset, key=cache.dropped_by_asset.get, reverse=True
            )[:15]
        ],
        "kept_triangles": cache.kept_triangles,
        "count_limited": cache.count_limited,
        "foliage_triangles": foliage_spent,
        "foliage_cap": foliage_cap,
        "failures": cache.failures,
        # Components that fell back to the descriptor for their size because the
        # asset reported none. High is normal under -nullrhi and is fine; it
        # being silently treated as zero is what lost the level's centrepiece.
        "sized_from_descriptor": no_mesh_bounds,
        # Assets whose geometry read back as astronomical numbers. See
        # `MeshCache._sane`. One of these reaching the file makes the export
        # unopenable, so the count is worth a line in the report even at zero.
        "insane_assets": cache.insane,
        "insane_examples": " ".join(cache.insane_examples),
        # Assets rejected for being a district-sized box with a dozen triangles
        # in it — marker volumes, trigger shells, light blockers. See
        # `build.MARKER_TRIANGLES`.
        "marker_assets": cache.markers,
        "marker_examples": " ".join(cache.marker_examples),
    }
    if cache.markers:
        log(
            "   {} assets dropped as marker volumes: {}".format(
                cache.markers, " ".join(cache.marker_examples[:8])
            )
        )
    if cache.insane:
        log(
            "   {} assets read back with impossible coordinates and were "
            "dropped: {}".format(
                cache.insane, " ".join(cache.insane_examples[:8])
            )
        )

    terrain_mesh = terrain.finish()
    report["steps"]["ground_meshes"] = terrain.stats()

    t0 = time.time()
    # How far from the origin a vertex may be before it is not geometry.
    #
    # From the content bounds rather than an absolute figure, because "plausible"
    # depends on the map: four times the extent of what the descriptors said is
    # here, which is loose enough that nothing real is near it and tight enough to
    # catch a placement at a hundred kilometres. See `build.read_mesh`.
    extent = max(
        abs(bounds[0][0]), abs(bounds[0][1]), abs(bounds[1][0]), abs(bounds[1][1])
    )
    limit = max(1.0e5, extent * 4.0) * (settings.scale or 1.0)
    dropped_far = 0
    if structure_mesh.get_triangle_count() > 0:
        positions, indices, dropped = build.read_mesh(
            structure_mesh, settings.scale, limit=limit
        )
        dropped_far += dropped
        parts.append(
            Part(classify.part_name(classify.STRUCTURE, settings.name), positions, indices)
        )
    if terrain_mesh.get_triangle_count() > 0:
        positions, indices, dropped = build.read_mesh(
            terrain_mesh, settings.scale, limit=limit
        )
        dropped_far += dropped
        # A second ground part. The viewer classifies on the prefix, so it joins
        # the traced terrain and is drawn opaque with the relief cues.
        parts.append(
            Part(
                classify.part_name(classify.GROUND, settings.name + " meshes"),
                positions,
                indices,
            )
        )
    report["steps"]["readback_seconds"] = round(time.time() - t0, 1)
    report["steps"]["triangles_out_of_world"] = dropped_far
    if dropped_far:
        log(
            "   dropped {} triangles further than {:.0f}m from the origin".format(
                dropped_far, limit / 100.0
            )
        )

    # -------------------------------------------------------------- write
    directory = os.path.dirname(os.path.abspath(settings.out))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    written = write_glb(
        settings.out,
        parts,
        extras={
            "generator": "3DHeat Unreal level exporter",
            "map": settings.map,
            "unrealUnits": "centimetres",
            "axis": "Y-up (Unreal Z mapped to Y)",
            "scale": settings.scale,
        },
    )
    report["output"] = written
    report["seconds"] = round(time.time() - started, 1)
    log(
        "wrote {} ({:.1f} MB, {} triangles) in {:.0f}s".format(
            settings.out,
            written["bytes"] / 1e6,
            sum(p["triangles"] for p in written["parts"]),
            report["seconds"],
        )
    )
    skipped = report["steps"]["structures"].get("instances_over_budget", 0)
    if skipped:
        # Said plainly, because a six-figure actor count in the collect line and a
        # five-figure instance count in the file is a gap someone will otherwise
        # notice much later.
        log(
            "{} instances did not fit the {} triangle budget. Raise --budget for "
            "more of the map, or --min-size to spend it on fewer, larger things.".format(
                skipped, settings.budget
            )
        )
    cut = report["steps"]["collect"].get("size_cut_m", 0)
    if cut > CUT_WARN_M:
        # The one number in this report that predicts how the export *looks*. A
        # cut above a shed means whole buildings are represented by whichever of
        # their parts happened to be biggest, and on a modular map that is the
        # roof — an export of floor plans hanging in mid-air, which reads as a
        # broken simplifier and is a budget that was too small.
        log(
            "WARNING: the size cut landed at {:.1f}m. Buildings assembled from "
            "separate actors will be missing their smaller parts — walls kept, or "
            "only a roof. Raise --budget (or crop with --region) to bring the cut "
            "under {:.0f}m.".format(cut, CUT_WARN_M)
        )
        report["warnings"] = report.get("warnings", []) + ["size cut {:.1f}m".format(cut)]
    return report, written


