"""
Which actors are worth exporting, decided before any of them is loaded.

This is where a map of many GB becomes a tractable one. World Partition keeps a
descriptor for every actor — class, label, world bounds, and the guid needed to
page it in — and that is enough to answer "is this a building or a screw?"
without touching the package. On the test level the descriptors are hundreds of
thousands of rows read in seconds, and the size filter alone removes most of them.

Non-partitioned levels take the other path: everything is already in the level,
so the same filters run over the loaded actors instead.
"""
import math
import re

import unreal

from . import classify, water


class Target(object):
    """One actor worth loading, with what the descriptor already told us.

    `key` is how a loaded actor is matched back to its target: the label for a
    partitioned level (descriptors carry labels, not paths) and the object path
    for one that is already loaded, which is unique where a label is not.
    """

    __slots__ = (
        "guid",
        "label",
        "cls",
        "min",
        "max",
        "size",
        "key",
        "scattered",
        "water",
        "place",
    )

    def __init__(self, guid, label, cls, bmin, bmax, key=None, place=""):
        self.guid = guid
        self.label = label
        self.cls = cls
        # The sub-level this actor was composed in; see classify.place_of.
        self.place = place
        self.min = bmin
        self.max = bmax
        # A "scattered" actor is one whose bounds describe a *collection* rather
        # than an object: a foliage actor's box spans every tree it plants, so its
        # diagonal can be a kilometre. Ranked by size against real objects it
        # sorts to the front of everything and takes the budget with it, which is
        # how an export of a city came out as a forest with no buildings in it.
        self.scattered = bool(re.search(r"Foliage|InstancedFoliage|ProceduralFoliage", cls or ""))
        # Water goes down its own path — see water.py. It must not reach the
        # structure pass: a river's geometry lives in spline meshes whose source
        # is undeformed, so exporting one as an ordinary static mesh piles every
        # segment of the river on top of the actor's origin.
        self.water = water.is_water(cls)
        self.key = key if key is not None else label
        d = (bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2])
        self.size = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])

    @property
    def centre(self):
        return ((self.min[0] + self.max[0]) * 0.5, (self.min[1] + self.max[1]) * 0.5)


def _compile(patterns):
    return [re.compile(p, re.I) for p in patterns]


# A descriptor's class arrives as the repr of a UClass object:
#   <Object '/Script/Engine.DecalActor' (0x000002...) Class 'Class'>
# Splitting on '.' and stripping quotes leaves the address and trailing words
# attached, which is harmless for a substring match and unreadable in a report.
_CLASS_PATH = re.compile(r"[\w/]+\.(\w+)'")


def short_class(native_class):
    """`/Script/Engine.StaticMeshActor` and friends, reduced to the name."""
    text = str(native_class)
    found = _CLASS_PATH.search(text)
    if found:
        return found.group(1)
    return text.split(".")[-1].strip("'\"> ")


class Filter(object):
    """The keep/drop decision, and a record of why things were dropped.

    The counters matter as much as the filtering: an export that kept a small
    fraction of the level's actors is alarming without the breakdown, and
    reassuring with it.
    """

    def __init__(self, settings):
        self.s = settings
        self.exclude_classes = _compile(settings.exclude_classes)
        self.exclude_patterns = _compile(settings.exclude_patterns)
        self.include_patterns = _compile(settings.include_patterns)
        self.dropped = {}
        self.dropped_examples = {}

    def _drop(self, reason, label=None):
        self.dropped[reason] = self.dropped.get(reason, 0) + 1
        # A sample of what each reason actually removed.
        #
        # Counts alone cost days here. "smaller than 5.0m" with a six-figure
        # count reads as grass and pebbles, and it was also every wall, doorway
        # and stair of every building — invisible in the report, and only
        # findable by comparing the telemetry against the export cell by cell
        # and then asking the editor what stood at the worst cell. Names in the receipt would have said it on
        # the first run.
        if label:
            examples = self.dropped_examples.setdefault(reason, [])
            if len(examples) < 25:
                examples.append(label)
        return False

    def exclude_actor(self, cls, label):
        """Should this *loaded actor* be skipped, by class or by name?

        The same two tests `keep_desc` applies to a descriptor, available to code
        that is holding an actor instead of a descriptor — which is the sub-level
        pass, and it had neither. It iterated every actor of a building's own
        level and took any static mesh component it found, so the volumes a level
        is full of came through: gi volumes, indoor volumes, occlusion volumes,
        arena bounds. On the test level that put a translucent box the size of
        a whole compound around one of its main buildings, which is not
        geometry the game has ever drawn.

        The main pass never had the problem because a descriptor scan filters by
        class before anything is loaded. This is the same rule, one path over.
        """
        for p in self.exclude_classes:
            if p.search(cls or ""):
                self.dropped["class " + (cls or "?")] = (
                    self.dropped.get("class " + (cls or "?"), 0) + 1
                )
                return True
        for p in self.exclude_patterns:
            if p.search(label or ""):
                self.dropped["name pattern"] = self.dropped.get("name pattern", 0) + 1
                return True
        return False

    def exclude_asset(self, name):
        """Should this *asset* be skipped by name?

        The settings have always documented the exclude patterns as matching
        "actor label, component name and mesh path", and until now they matched
        the label alone. That gap is why an export of a city came out studded
        with decorative roots, lianas and rubble: those are components inside
        actors whose own labels say nothing, so the only filter they ever met was
        the size threshold, which they pass.
        """
        if not name:
            return False
        for p in self.exclude_patterns:
            if p.search(name):
                self.dropped["asset name pattern"] = (
                    self.dropped.get("asset name pattern", 0) + 1
                )
                return True
        return False

    def _rescue(self, label, size, place):
        """Is this small actor a piece of something worth keeping anyway?

        The size filter is the most valuable one here and also the one that was
        quietly deleting the answer. It measures an actor's bounding diagonal, and
        a wall panel, a doorway, a flight of steps and a catwalk railing are all
        one to four metres — so a modular building was dismantled piece by piece
        at the descriptor stage, before the budget, the ranking or the priority
        list had any say.

        The exemption is **named architecture**, down to `priority_min_size`. It
        is not a licence for clutter: what comes back in still has to earn its
        triangles from the budget like anything else.

        There was a second exemption here, for anything composed into a sub-level
        (`classify.place_of`), and it is gone because it was measured. It rescued
        over a hundred thousand extra actors — the main town's walls, stairs and
        railings, exactly the geometry that is missing — and they produced almost
        no triangles between them, because those descriptors cannot be
        instantiated at all: `WorldPartitionBlueprintLibrary.load_actors` brings
        in 399 of 400
        persistent-level actors and 0 of 400 sub-level ones. Keeping them only
        diluted `plan`'s per-actor allowance by half, which made every *other*
        object worse. The sub-levels need a different handle, not a lower floor.
        """
        if (
            classify.PRIORITY_NAMES.search(label or "")
            and size >= self.s.priority_min_size
        ):
            self.dropped["kept: small but architectural"] = (
                self.dropped.get("kept: small but architectural", 0) + 1
            )
            return True
        return False

    def keep_desc(self, cls, label, bmin, bmax, size, editor_only, place=""):
        if editor_only:
            return self._drop("editor-only")
        if size <= 0.0:
            return self._drop("no bounds")

        # An explicit include list overrides everything below it: someone naming
        # a pattern has already decided.
        for p in self.include_patterns:
            if p.search(label or "") or p.search(cls or ""):
                return True

        for p in self.exclude_classes:
            if p.search(cls or ""):
                return self._drop("class " + cls, label)
        for p in self.exclude_patterns:
            if p.search(label or ""):
                return self._drop("name pattern", label)

        if size < self.s.min_size and not self._rescue(label, size, place):
            return self._drop("smaller than {}m".format(self.s.min_size_m), label)

        # Water is exempt from the size ceiling. An ocean is legitimately larger
        # than the map — kilometres wider, on the test level — and the ceiling
        # exists to catch sky spheres and vista rings, not seas. It was silently
        # eating the ocean, which is why the map had a coastline and nothing
        # beyond it.
        #
        # It cannot extend the world, either: `water.build` clips an unbounded
        # body to the content footprint, so the bounds stay the map's own.
        if size > self.s.max_size and not water.is_water(cls):
            return self._drop("larger than {}m".format(self.s.max_size_m))

        region = self.s.region
        if region:
            # Reject only when wholly outside, so a building on the boundary is
            # still exported rather than leaving a hole at the edge of the crop.
            if bmax[0] < region[0] or bmax[1] < region[1]:
                return self._drop("outside region")
            if bmin[0] > region[2] or bmin[1] > region[3]:
                return self._drop("outside region")
        return True


def _editor_only(actor):
    """Is this actor editor-only?

    `AActor::IsEditorOnly` is not bound to Python in 5.2, and the property behind
    it has been renamed across versions, so this asks in the ways that exist and
    settles for "no" — a stray editor-only mesh in the export is a much smaller
    problem than an export that will not run.
    """
    for name in ("is_editor_only_actor", "b_is_editor_only_actor"):
        try:
            return bool(actor.get_editor_property(name))
        except Exception:  # noqa: BLE001 - property may not exist
            continue
    return False


def _box(bounds):
    return (
        (bounds.min.x, bounds.min.y, bounds.min.z),
        (bounds.max.x, bounds.max.y, bounds.max.z),
    )


class Landscape(object):
    """The landscape actors, and the footprint they cover.

    The footprint is what the ground grid is traced over, and it has to come
    from the landscape rather than from everything in the level: a sky sphere or
    an ocean plane puts the world's bounds tens of kilometres out, and a grid
    stretched over that spends every ray on empty air. On the test level the
    difference was a few hundred hits versus hundreds of thousands.
    """

    def __init__(self):
        self.guids = []
        self.min = [float("inf")] * 3
        self.max = [float("-inf")] * 3

    def add(self, guid, bmin, bmax):
        self.guids.append(guid)
        for i in range(3):
            self.min[i] = min(self.min[i], bmin[i])
            self.max[i] = max(self.max[i], bmax[i])

    @property
    def valid(self):
        return bool(self.guids) and self.min[0] < self.max[0]

    def footprint(self):
        return ((self.min[0], self.min[1]), (self.max[0], self.max[1]))


def collect_partitioned(settings, filt):
    """Targets from the World Partition descriptors, plus the landscape.

    The landscape is kept separate because it is not exported as geometry at all
    — it has no scriptable mesh — but by tracing rays into its collision, which
    needs it loaded on its own.
    """
    descs = unreal.WorldPartitionBlueprintLibrary.get_actor_descs()
    targets = []
    landscape = Landscape()
    total = 0
    # The map's own package, so `place_of` can tell an actor dropped into the map
    # from one composed into a sub-level of it.
    map_package = str(settings.map or "").split(".", 1)[0]
    for d in descs:
        total += 1
        cls = short_class(d.native_class)
        bounds = d.bounds
        if not bounds.is_valid:
            filt.dropped["no bounds"] = filt.dropped.get("no bounds", 0) + 1
            continue
        bmin, bmax = _box(bounds)

        if "Landscape" in cls and "Spline" not in cls:
            landscape.add(d.guid, bmin, bmax)
            continue

        place = classify.place_of(getattr(d, "actor_path", None), map_package)
        t = Target(d.guid, str(d.label), cls, bmin, bmax, place=place)
        # A NaN diagonal compares false against every threshold, so an actor with
        # NaN bounds would otherwise pass every filter it met.
        if not t.size == t.size:
            filt.dropped["no bounds"] = filt.dropped.get("no bounds", 0) + 1
            continue
        if filt.keep_desc(
            cls, t.label, bmin, bmax, t.size, bool(d.actor_is_editor_only), place
        ):
            targets.append(t)
    return targets, landscape, total


def collect_loaded(settings, filt):
    """Targets from a level that is not partitioned: everything is already here."""
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    targets = []
    landscape = Landscape()
    map_package = str(settings.map or "").split(".", 1)[0]
    for a in actors:
        cls = a.get_class().get_name()
        origin, extent = a.get_actor_bounds(False)
        bmin = (origin.x - extent.x, origin.y - extent.y, origin.z - extent.z)
        bmax = (origin.x + extent.x, origin.y + extent.y, origin.z + extent.z)
        if isinstance(a, unreal.LandscapeProxy):
            landscape.add(None, bmin, bmax)
            continue
        path = a.get_path_name()
        place = classify.place_of(path, map_package)
        t = Target(None, a.get_actor_label(), cls, bmin, bmax, key=path, place=place)
        if filt.keep_desc(cls, t.label, bmin, bmax, t.size, _editor_only(a), place):
            targets.append(t)
    return targets, landscape, len(actors)


def is_partitioned():
    """Is the open level a World Partition level?

    Asked by trying: a non-partitioned world returns nothing rather than
    failing, and there is no other cheap scriptable signal in 5.2.
    """
    try:
        descs = unreal.WorldPartitionBlueprintLibrary.get_actor_descs()
        return bool(descs)
    except Exception:  # noqa: BLE001 - engine-dependent
        return False


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]

