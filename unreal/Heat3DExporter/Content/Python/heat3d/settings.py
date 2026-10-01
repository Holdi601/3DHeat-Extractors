"""
What to export, and the knobs that decide how much of it.

Every default here is a judgement about a map nobody has seen yet, so each one
says what it is trading. They are all overridable from the command line or a JSON
file — the JSON file exists because Windows command lines and quoted regexes are
a bad combination, and because an export worth repeating is worth writing down.
"""
import argparse
import json
import os
import shlex

# Metres, converted to Unreal centimetres on use. Unreal is always centimetres;
# the viewer is unit-agnostic as long as the level and the telemetry agree.
CM = 100.0


class Settings(object):
    def __init__(self, **kw):
        # ---- what
        self.map = kw.get("map") or ""
        self.out = kw.get("out") or ""
        self.name = kw.get("name") or ""

        # ---- how much geometry
        #
        # The lowest authored LOD is the right starting point: someone already
        # decided what this object looks like from far away, and no automatic
        # decimation beats an artist's own reduction. `lod` overrides that with a
        # fixed index when a project's lowest LODs are too crude.
        self.lod = kw.get("lod", "lowest")
        # Triangles in the finished file, before the viewer decimates again on
        # import. It is a ceiling, not a target.
        #
        # Zero means "decide from how big the level turns out to be", which is
        # the default because no single number is right for both a 200m arena and
        # a multi-kilometre open world. A fixed two million is generous for the
        # first and too mean for the second — and the way it fails on the second is not
        # obvious from the file, because what a too-small budget drops is every
        # object below the size cut, which on a modular map means a building's
        # wall panels while its roof slab, being larger, survives on its own. An
        # export of floating floor plans looks like a broken simplifier.
        #
        # See `budget_for` below for the rule. Pass `--budget` to override it;
        # anyone who names a number has already decided.
        self.budget = int(kw.get("budget", 0) or 0)
        # How that budget is split. The ground is a handful of huge triangles and
        # needs little; structures are where the detail is.
        self.ground_share = float(kw.get("ground_share", 0.35))

        # ---- what to leave out
        #
        # The single most important filter. A modern map is millions of tiny
        # props — grass, pebbles, rubbish, screws — and they are individually
        # invisible at the range this viewer is used from, while collectively
        # being most of the triangles. Three metres keeps rocks, crates, doors
        # and vehicles; it drops grass tufts and litter.
        self.min_size_m = float(kw.get("min_size_m", 4.0))
        # And a ceiling, which matters more than it sounds. Every map has a few
        # actors that enclose the whole world — a sky sphere, an ocean plane,
        # distant backdrop scenery — and they wreck everything downstream: the test
        # level's content bounds came out tens of kilometres across instead of a
        # few, which spreads the ground grid and the tile budget over a world that
        # is almost entirely empty.
        # Nothing above two kilometres is a building; terrain comes from the
        # ground pass instead. 0 disables the check.
        # Raised from 2,000, which was deleting the most important building on the
        # map. The level's centrepiece — a landmark mesh, the thing players fight
        # over — has a bounding diagonal well over two kilometres, so the ceiling
        # dropped it, and the heat was left ringing a building that was not
        # there.
        #
        # The ceiling can afford to be loose now because what it was really for is
        # handled better elsewhere: sky spheres, vista rings, backdrop cards and
        # cloud decoration are excluded by name, and nothing above the *bounds*
        # threshold below is allowed to set the world's extent whether it is
        # exported or not.
        self.max_size_m = float(kw.get("max_size_m", 5000.0))
        # Above this, an actor is exported but is not allowed to define the
        # world's extent.
        #
        # That separation is the point. The original ceiling conflated two jobs —
        # "do not draw this" and "do not let this decide how big the map is" — and
        # only the second is really about size. The test level's content bounds
        # came out tens of kilometres across instead of a few because of a handful
        # of enormous actors, and spreading the ground grid and every tile budget
        # over a world that is almost entirely empty is what that cost. Excluding
        # them from the measurement fixes it without throwing their geometry away.
        self.bounds_max_size_m = float(kw.get("bounds_max_size_m", 1500.0))
        # Foliage is held to a higher bar: it is the densest thing on any map and
        # the least useful for locating yourself. Eight metres keeps trees.
        self.foliage_min_size_m = float(kw.get("foliage_min_size_m", 8.0))
        # The floor for named architecture, which is exempt from `min_size_m`.
        #
        # A wall panel, a doorway and a flight of steps are all under the ordinary
        # threshold, and dropping them dismantles the buildings a movement heatmap
        # is read against — measured on the test level, thousands of actors below
        # 5m in a 300m box around the densest telemetry, with no building among
        # the survivors.
        # Low, but not zero: under a metre and a half nothing is part of a
        # building.
        self.priority_min_size_m = float(kw.get("priority_min_size_m", 1.5))
        # The floor for anything composed into a sub-level — see
        # `classify.place_of`. This is the exemption that actually finds the
        # buildings, so it is the lowest of the three.
        #
        # A metre. A small tower prefab is assembled from railing pieces
        # like `SM_Railing_01`, whose bounding diagonal is about 1.4m, and a
        # town is tens of thousands of pieces that size. Below a metre it is
        # bolts and light fittings, which no heatmap is read against.
        self.composed_min_size_m = float(kw.get("composed_min_size_m", 1.0))
        # How far above the highest terrain an actor may sit before it is sky.
        #
        # 200m. Clouds, sky planes, vista cards and light shafts live up there and
        # nothing a player can reach does — and unlike a list of names this keeps
        # working on a project that calls its clouds something else. Found by
        # measuring a finished export: once the marker boxes were gone, every
        # remaining large triangle was in a thin band far above the terrain,
        # spanning the whole map.
        #
        # Measured against an actor's *lowest* point, so a mast or a crane rooted
        # on the ground is unaffected however tall it is. 0 disables the check.
        self.sky_clearance_m = float(kw.get("sky_clearance_m", 200.0))
        self.include_foliage = bool(kw.get("include_foliage", True))
        # Matched against the actor's label and the asset's own name.
        #
        # The second half of that was missing for a long time, and the report
        # named the reason for every drop except the ones that never happened.
        # Decoration is placed as components of actors whose labels say nothing,
        # so the only filter it met was the size threshold, which it passes: an
        # export of a city arrived studded with roots, lianas, rubble piles and
        # lamp posts, all of them too small to keep a shape and all of them
        # therefore drawn as cubes.
        self.exclude_patterns = list(
            kw.get(
                "exclude_patterns",
                [
                    r"grass",
                    r"weed",
                    r"fern",
                    r"nettle",
                    r"clutter",
                    r"litter",
                    r"pebble",
                    r"gravel_?small",
                    r"leaf|leaves",
                    r"decal",
                    r"billboard",
                    r"impostor|imposter",
                    # Dressing: it grows on things, it is never navigable, and at
                    # the range this viewer is used from it is a pixel.
                    # Helper actors that carry a mesh but are not geometry the
                    # game draws. Named rather than classed because their classes
                    # are `Actor` and `StaticMeshActor` — nothing distinguishes
                    # them from a building except what they are called.
                    #
                    # Both of these were found by measuring the finished file:
                    # the largest triangles in the structure part turned out to be
                    # the faces of two boxes, and asking the editor what has those
                    # exact bounds turned up a map-location blueprint (a marker
                    # hundreds of metres across for the map UI, drawn as a
                    # translucent box around a district) and a light-blocker
                    # actor (a plane hundreds of metres across and one metre
                    # thick, floating above the ground).
                    r"light_?blocker|shadow_?blocker|nav_?blocker",
                    r"map_?location|location_?marker",
                    r"(^|_)roots?(_|$)",
                    r"liana|ivy|vine|creeper|moss|lichen",
                    r"(^|_)plant_",
                    r"rubble|scree",
                    r"treelog|deadwood|driftwood|branch_",
                    # Sky and distance. None of it is reachable, none of it is
                    # cover, and all of it is enormous — so in a viewer that hides
                    # heat behind geometry, a decorative cloud spiral over the
                    # middle of the map hides the data under the middle of the map.
                    #
                    # Collision cannot be used to find these, which was the first
                    # plan: measured on the test level, a decorative cloud mesh
                    # reports QUERY_AND_PHYSICS, as does a backdrop card,
                    # while a lake mesh reports NO_COLLISION. A project can set
                    # collision on anything; what it calls things is more honest.
                    r"cloud|vista|backdrop|skybox|sky_?(sphere|dome|plane)",
                    r"(^|_)card(_|$)|godray|lightshaft|(^|_)fog(_|$)|smokeplane",
                    r"debugactor|datavisualizer|anomalyplane",
                ],
            )
        )
        self.include_patterns = list(kw.get("include_patterns", []))
        # Classes that never carry useful surface geometry. Matched on the class
        # name, so a project's own `BP_Light_...` is caught by the light pattern.
        self.exclude_classes = list(
            kw.get(
                "exclude_classes",
                [
                    r"Decal",
                    r"Light$|LightActor|PointLight|SpotLight|RectLight|SkyLight|DirectionalLight",
                    r"ReflectionCapture",
                    r"Volume$",
                    r"NavLink|NavMesh|Navigation",
                    r"Niagara|Emitter|ParticleSystem",
                    # Wwise. Anchored, and it has to be: these patterns are
                    # compiled with IGNORECASE, under which `[A-Z]` matches
                    # lowercase letters too, so an unanchored `Ak[A-Z]` matched
                    # the "ake" in `WaterBodyLake` and silently dropped every
                    # lake on the map as an audio actor. The map had no water for
                    # three exports and the report blamed a class filter that had
                    # nothing to do with water.
                    r"AudioVolume|^Ak[A-Za-z]|SoundActor",
                    r"PlayerStart|TargetPoint|Note$|Trigger",
                    r"WorldDataLayers|WorldPartitionMiniMap|LevelBounds|GroupActor",
                    r"SplineActor|SplineToolActor|LandscapeSpline",
                    r"LootSpawn|SpawnPoint|SpawnArea|SpawnZone",
                    r"CameraActor|CineCamera|SceneCapture",
                    r"PostProcessVolume|BoxGIVolume",
                ],
            )
        )

        # ---- geography
        #
        # A crop, in Unreal centimetres: min_x, min_y, max_x, max_y. Exporting a
        # region is the difference between a usable file and a 400MB one on maps
        # where the telemetry only covers part of the world.
        self.region = kw.get("region")
        # Tile edge in metres. The build accumulates one tile at a time and
        # simplifies it before moving on, which is what keeps peak memory flat
        # instead of holding the whole map's triangles at once. 0 picks a tile
        # size from the world extent.
        self.tile_m = float(kw.get("tile_m", 0))
        # Actors paged in per World Partition load. Larger is faster and uses
        # more memory; this is the main memory dial on a big map.
        self.batch = int(kw.get("batch", 2048))

        # ---- sources
        self.include_landscape = bool(kw.get("include_landscape", True))
        self.include_hlod = bool(kw.get("include_hlod", False))
        # Export the buildings that live in sub-levels. See heat3d/sublevels.py.
        #
        # On by default because without it a World Partition map composed of
        # Level Instances exports its landscape and none of its buildings, which
        # is what the test level did for every run before this existed. Off is for
        # a map where nothing is composed, where it costs one descriptor scan.
        self.include_sublevels = bool(kw.get("include_sublevels", True))
        # How many sub-levels to open. Each costs a level load, so this is the
        # single biggest lever on how long an export takes.
        #
        # High enough not to be the limit. It was 45, and on the test level that
        # was *exactly* the number opened: hundreds of sub-levels seen, most of
        # them eligible, 45 opened, and every other building that had passed
        # every other filter was never looked at. The report said so plainly and
        # it still took a round of "buildings are missing" to go and read it,
        # which is the argument for this default being a number no map reaches
        # rather than a number tuned to one.
        #
        # It is not free: 45 sub-levels took over ten minutes, so an export of
        # minutes became closer to an hour. `--sublevels N` is there for when
        # that matters, and the ordering is by traffic, so a truncated run still
        # gets the buildings the telemetry is densest inside.
        self.max_sublevels = int(kw.get("max_sublevels", 1000))
        # A cheap pre-filter on actor count, not the real test. See
        # `sublevel_min_size_m` below, which is.
        #
        # Its only job is to keep the descriptor scan's second pass small: holding
        # a label and a position for all of a large map's descriptors costs a
        # hundred megabytes, and a sub-level of four actors cannot be a building
        # in any project. It was 200, which is not a pre-filter but a decision —
        # and it silently excluded a large share of the test level's sub-levels,
        # among them the guard huts, sheds and tool units that are most of what a
        # player walks into. Any threshold on actor count is a guess about how a
        # particular project authors its prefabs, which is exactly the kind of
        # guess that does not travel to the next project.
        self.sublevel_min_actors = int(kw.get("sublevel_min_actors", 6))
        # How wide a composed sub-level must be to count as a building, in metres,
        # measured across one instance's footprint.
        #
        # This is the test that decides, and it is in metres because that is the
        # question actually being asked: is this thing building-sized? A bench, a
        # lamp cluster or a pile of crates is two or three metres across however
        # many actors its prefab happens to contain, and a guard hut is four
        # whether it was authored as eight actors or eighty.
        #
        # Measured from descriptor bounding-box centres, so it reads a little
        # under the true footprint — the spread of the pieces, not the outer
        # edges. Four metres is set with that in mind: it keeps huts and sheds and
        # drops street furniture.
        self.sublevel_min_size_m = float(kw.get("sublevel_min_size_m", 4.0))
        # A traffic grid from the telemetry, written by `tools/heat-grid.mjs`.
        #
        # Everything else here decides what matters by looking at the level —
        # size, name, actor count — and those are guesses that get the important
        # case wrong: a hillside prefab outranks a town hall because it
        # holds more actors. This is the one input that knows the answer, and it
        # is the same data the viewer draws on top of the export.
        #
        # Empty means "look beside the output file", so the ordinary case needs no
        # flag; `--hotspots none` turns it off. See heat3d/hotspots.py.
        self.hotspots_path = str(kw.get("hotspots_path", "") or "")
        self.include_water = bool(kw.get("include_water", True))

        # ---- output space
        #
        # The viewer's world is Y-up; Unreal is Z-up. `scale` multiplies
        # afterwards: leave it at 1 to keep Unreal centimetres, which is what
        # telemetry exported from the same game is in, and the two then line up
        # with no further thought.
        self.scale = float(kw.get("scale", 1.0))

        # Stand a bounding box where an object was too cheap to keep its shape.
        #
        # Off. It was on, and on a dense map the result was a field of cyan cubes
        # between the buildings that hid the buildings, the ground and the heat.
        # The argument for it is still real on a sparse map, where a box is the
        # difference between "something stands here" and an empty plain — so the
        # switch exists. On anything crowded, leave it off.
        self.box_substitutes = bool(kw.get("box_substitutes", False))

        # ---- operation
        self.dry_run = bool(kw.get("dry_run", False))
        self.max_actors = int(kw.get("max_actors", 0))  # 0 = no limit; for smoke tests
        self.verbose = bool(kw.get("verbose", False))

    # --------------------------------------------------------------- helpers

    @property
    def min_size(self):
        return self.min_size_m * CM

    @property
    def max_size(self):
        return self.max_size_m * CM if self.max_size_m > 0 else float("inf")

    @property
    def foliage_min_size(self):
        return self.foliage_min_size_m * CM

    @property
    def priority_min_size(self):
        return self.priority_min_size_m * CM

    @property
    def composed_min_size(self):
        return self.composed_min_size_m * CM

    @property
    def sky_clearance(self):
        return self.sky_clearance_m * CM

    @property
    def hotspots(self):
        """The traffic grid to read, or `''` for none.

        Defaults to `hotspots.json` beside the output, which is where the tool
        writes it, so the common case needs no flag. `none` disables it.
        """
        if self.hotspots_path.lower() in ("none", "off", "-"):
            return ""
        if self.hotspots_path:
            return self.hotspots_path
        from . import hotspots as hotspots_module

        return hotspots_module.default_path(self.out)

    @property
    def bounds_max_size(self):
        return self.bounds_max_size_m * CM if self.bounds_max_size_m > 0 else float("inf")

    @property
    def tile(self):
        return self.tile_m * CM

    def report_path(self):
        base, _ = os.path.splitext(self.out)
        return base + ".json"

    def to_dict(self):
        return {
            k: v
            for k, v in vars(self).items()
            if not k.startswith("_")
        }


# Triangles per metre of a level's longest side, for the automatic budget.
#
# Set against the bar this tool is actually judged by rather than against a
# guess: the Unity pipeline it replaces shipped 300–560 MB of geometry per map.
# At 3,000 per metre a level several kilometres across comes to ~10M triangles
# and a few hundred MB, which is the same order, and the difference it buys is not
# prettiness — with over a hundred thousand eligible actors it is the difference
# between ~24 triangles per object, where nearly everything falls back to a
# bounding box, and ~70, where most things keep their own shape. "A field of weirdly placed boxes" is what the low number looks
# like from inside the viewer.
#
# The first version of this was 1,100, fitted only to keep the *size cut* under a
# shed. That is necessary and not sufficient: every object being present says
# nothing about any of them being recognisable.
#
# Raised from 3,000 once the budget was reaching the right geometry. At 3,000 the
# test level's architecture came to fewer triangles than its vegetation — so
# more triangles would have bought more trees, and the number was not the
# thing to change. With the architecture band reserved (see
# `build.ARCH_TILE_RESERVE`) the extra is spent on what a heatmap is read
# against, and 4,000 puts a level several kilometres across at well over ten
# million triangles and a few hundred MB: inside what a browser tab holds, and
# the same order as the pipeline this replaces.
#
# Raised again from 4,000 once the sub-level pass was allowed to open the whole
# map rather than the busiest forty-five of its hundreds of sub-levels. That
# change multiplies the building geometry several times over, and at 4,000 the
# buildings' reserved share was already spent to the last triangle — every
# triangle of the reservation — so the extra buildings would simply have been
# refused one by one, which is the same failure as never opening them.
#
# Building geometry is stamped per instance rather than referenced, so a house
# placed fifty times costs fifty houses in the file. That is where the size
# goes and it is the thing worth fixing properly one day (glTF has
# `EXT_mesh_gpu_instancing` for exactly this); until then the honest options are a
# bigger file or fewer buildings, and a missing building is invisible while a
# large file is merely slow to load.
BUDGET_PER_METRE = 7_000
# Below this, a level is a building or an arena and the budget stops mattering.
BUDGET_FLOOR = 2_000_000
# And a ceiling, because a budget is also a file someone has to load. 20M was
# roughly 340 MB in practice — the 450 MB estimate this comment used to carry was
# measured against an export that had not yet been through the mesh cache.
BUDGET_CEILING = 32_000_000


def budget_for(extent_m):
    """The triangle budget for a level whose longest horizontal side is this.

    Only used when `--budget` was not given. Returned rounded to something a
    human reading the log will recognise as a decision rather than a hash.
    """
    raw = max(BUDGET_FLOOR, min(BUDGET_CEILING, int(extent_m * BUDGET_PER_METRE)))
    return int(round(raw / 100_000.0) * 100_000)


def _parser():
    p = argparse.ArgumentParser(
        prog="heat3d_export",
        description="Export a decimated world-space level mesh for the 3DHeat viewer.",
    )
    p.add_argument("--config", help="JSON file with any of the options below.")
    p.add_argument("--map", help="Package path of the level, e.g. /Game/Maps/L_Example")
    p.add_argument("--out", help="Output .glb path")
    p.add_argument("--name", help="Name to record in the file (defaults to the map name)")
    p.add_argument("--lod", default=None, help="'lowest' (default) or a LOD index")
    p.add_argument("--budget", type=int, default=None, help="Triangle ceiling for the whole file")
    p.add_argument("--ground-share", type=float, default=None, dest="ground_share")
    p.add_argument("--min-size", type=float, default=None, dest="min_size_m",
                   help="Skip anything whose bounds are smaller than this, in metres")
    p.add_argument("--max-size", type=float, default=None, dest="max_size_m",
                   help="Skip anything bigger than this, in metres: sky spheres, ocean "
                        "planes and vista props. 0 to keep everything.")
    p.add_argument("--foliage-min-size", type=float, default=None, dest="foliage_min_size_m")
    p.add_argument("--sky-clearance", type=float, default=None, dest="sky_clearance_m",
                   help="Skip actors sitting entirely more than this many metres above the "
                        "highest terrain: clouds, sky planes, vista cards. 0 keeps them.")
    p.add_argument("--composed-min-size", type=float, default=None,
                   dest="composed_min_size_m",
                   help="Floor, in metres, for actors composed into a sub-level — the "
                        "pieces buildings are made of. See classify.place_of.")
    p.add_argument("--no-foliage", action="store_true")
    p.add_argument("--no-landscape", action="store_true")
    p.add_argument("--no-water", action="store_true")
    p.add_argument("--hlod", action="store_true", help="Use prebuilt HLOD proxies where they exist")
    p.add_argument("--no-sublevels", action="store_true",
                   help="Skip the buildings that live in Level Instance sub-levels")
    p.add_argument("--sublevels", type=int, default=None, dest="max_sublevels",
                   help="How many building sub-levels to open, largest first (default 30)")
    p.add_argument("--sublevel-min-actors", type=int, default=None,
                   dest="sublevel_min_actors",
                   help="Cheap pre-filter on a sub-level's actor count; the real test is "
                        "--sublevel-min-size")
    p.add_argument("--sublevel-min-size", type=float, default=None,
                   dest="sublevel_min_size_m",
                   help="How wide a composed sub-level must be to count as a building, in "
                        "metres across one instance. Keeps huts, drops street furniture.")
    p.add_argument("--hotspots", default=None, dest="hotspots_path",
                   help="Traffic grid from the telemetry, written by tools/heat-grid.mjs. "
                        "Decides which buildings and which copies of them are worth the "
                        "budget. Defaults to hotspots.json beside the output; 'none' to "
                        "ignore it.")
    p.add_argument("--region", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"),
                   help="Crop, in Unreal centimetres")
    p.add_argument("--tile", type=float, default=None, dest="tile_m")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--exclude", action="append", default=None, dest="exclude_patterns")
    p.add_argument("--include", action="append", default=None, dest="include_patterns")
    p.add_argument("--max-actors", type=int, default=None, dest="max_actors")
    p.add_argument(
        "--boxes",
        action="store_true",
        help="Stand a bounding box where an object cannot afford its shape. "
        "Off by default; useful on a sparse map, clutter on a crowded one.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def _values(args):
    """Everything the parser actually saw, as Settings keyword arguments.

    Options left at their default contribute nothing, so a command line can be
    layered over a config file without silently resetting what the file set.
    """
    out = {}
    for key in (
        "map", "out", "name", "lod", "budget", "ground_share", "min_size_m",
        "max_size_m", "foliage_min_size_m", "composed_min_size_m", "sky_clearance_m",
        "tile_m", "batch", "scale",
        "max_actors", "exclude_patterns", "include_patterns",
        "max_sublevels", "sublevel_min_actors", "sublevel_min_size_m", "hotspots_path",
    ):
        value = getattr(args, key, None)
        if value is not None:
            out[key] = value
    if getattr(args, "region", None):
        out["region"] = list(args.region)
    for flag, key, value in (
        ("no_foliage", "include_foliage", False),
        ("no_landscape", "include_landscape", False),
        ("no_water", "include_water", False),
        ("boxes", "box_substitutes", True),
        ("hlod", "include_hlod", True),
        ("no_sublevels", "include_sublevels", False),
        ("dry_run", "dry_run", True),
        ("verbose", "verbose", True),
    ):
        if getattr(args, flag, False):
            out[key] = value
    return out


def from_argv(argv):
    """Build settings from a command line, with a JSON file underneath it.

    Anything given on the command line wins over the file, and the file wins over
    the defaults — the usual layering, so a saved export can be re-run with one
    thing changed.

    The file can also come from `HEAT3D_CONFIG`, and that is how the runner
    script passes everything. Unreal's `-script="file.py a b c"` splits its value
    on spaces and does not honour quotes inside it, so any path with a space in
    it arrives as two arguments and the export dies at the parser. An environment
    variable has no quoting to get wrong.
    """
    args = _parser().parse_args(argv)
    values = {}
    config = args.config or os.environ.get("HEAT3D_CONFIG")
    if config:
        with open(config, "r", encoding="utf-8") as f:
            values.update(json.load(f))

    # A config file may carry a command line of its own under "argv" — that is
    # how the runner forwards `-Args` without re-implementing the parser in
    # PowerShell, and it keeps one definition of every option.
    extra = values.pop("argv", None)
    if extra:
        values.update(_values(_parser().parse_args(shlex.split(extra))))
    values.update(_values(args))

    s = Settings(**values)
    if not s.map:
        raise SystemExit("heat3d: --map is required (e.g. --map /Game/Maps/L_Example)")
    if not s.out:
        raise SystemExit("heat3d: --out is required (e.g. --out C:/exports/L_Example.glb)")
    if not s.name:
        s.name = s.map.rsplit("/", 1)[-1]
    return s
