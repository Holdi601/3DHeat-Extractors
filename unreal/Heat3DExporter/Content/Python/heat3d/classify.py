"""
Ground, or a thing standing on it?

The viewer draws the two differently — the ground solid, because it is what you
read a position against, and structures as a faint wireframe, because a building
drawn solid hides the heat inside it. It can guess from part names and geometry,
but the exporter does not have to guess: inside the engine the landscape is
literally a different class from a building, and that knowledge is worth carrying
across rather than throwing away and reconstructing.

So each part is named with the answer — `ground:...` or `structure:...` — and the
viewer takes an explicitly labelled part at its word.
"""
import re

GROUND = "ground"
STRUCTURE = "structure"
# Water is neither. It is a surface you read position against, like the ground,
# but it is flat, it is not walkable, and heat over it means something different
# from heat over a street — so it is its own part, and the viewer shades it
# without relief or contours.
WATER = "water"

# Class names that are the world's surface rather than something placed on it.
GROUND_CLASSES = (
    re.compile(r"Landscape(Proxy|StreamingProxy)?$", re.I),
    re.compile(r"^Landscape$", re.I),
    re.compile(r"Terrain", re.I),
    re.compile(r"VirtualHeightfieldMesh", re.I),
    re.compile(r"WaterBody|OceanActor|LakeActor|RiverActor", re.I),
)

# Meshes whose job is to be the world's surface rather than to stand on it.
#
# Matched against the *asset's own name* and the actor's label, never against the
# asset path — and that distinction is the whole reason this comment exists. A
# first version matched the full path, and a project with a content folder called
# `Environment/Ground/` had every mesh under it, buildings included, reclassified
# as terrain: `StaticMeshActor` went from millions of triangles to a few tens of
# thousands in one export and the town disappeared. A folder name describes where
# an artist filed something, not what it is.
#
# `floor`, `dirt` and `sand` were in this list and are deliberately gone. A
# building's floor slab is not the ground, and by the time you are matching those
# words against every mesh name on a map the false positives outnumber the true
# ones.
GROUND_NAMES = (
    re.compile(r"terrain|landscape|heightfield", re.I),
    re.compile(r"(^|[_\W])(road|street|asphalt|pavement|sidewalk|kerb|curb|path|trail)", re.I),
    re.compile(r"(ground|grass)_?plane", re.I),
    re.compile(r"(^|[_\W])(water|ocean|sea|river|lake|pond)(?![a-z])", re.I),
    # Rock and cliff meshes are landscape, whatever class they are placed as.
    #
    # Worth its own line because it is the biggest thing the export got wrong
    # after everything else was right. A sizeable share of the test level's
    # structure budget was rock: one cliff mesh alone took hundreds of thousands
    # of triangles over thousands of placements. Left among the structures they
    # are drawn the way buildings are drawn — translucent, so you can see the
    # heat inside a building — and a cliff you can see through is a glass
    # crystal the size of a hill. As ground they are
    # opaque, they take the relief shading, and they read as the terrain they are.
    re.compile(r"(^|[_\W])(rock|cliff|boulder|rubble|scree|gravel|mound)", re.I),
)


# Assets that have to be in the export whatever the budget says.
#
# Not a nicety. A movement heatmap is read against the things that constrain
# movement: what you can stand in, what you can shelter behind, what you cannot
# walk through. A tree is scenery — if it is missing you lose nothing about where
# people went. A fence is not: it is the reason a path bends, and without it the
# bend looks like noise.
#
# Anything matching here is guaranteed a real reduction rather than being dropped
# for being small, and is offered before anything else in its actor.
PRIORITY_NAMES = re.compile(
    r"fence|railing|handrail|barrier|barricade|gate(?!way_?arch)|palisade|barbwire|"
    r"(^|[_\W])wall|(^|[_\W])door|window|(^|[_\W])roof|stair|ladder|catwalk|"
    r"balcon|platform|scaffold|building|house|hut(?!ch)|shed|barn|hangar|bunker|"
    r"tower|silo|warehouse|bridge|pillar|column|pier|"
    # The parts a building is assembled from. These are why the list exists: a
    # wall panel, a doorway and a flight of steps are all under the size filter's
    # threshold, so a building made of them was dropped piece by piece before any
    # of the budgeting ran — thousands of actors below 5m in one 300m box around
    # the densest heat on the map, and the halls the data was inside were among
    # them.
    r"slab|panel|beam|girder|truss|(^|[_\W])step|ramp|kerb|corridor|hall(?!ow)|"
    r"partition|doorway|archway|frame_|(^|[_\W])floor|ceiling|banister|"
    r"crate_?wall|sandbag|hesco|jersey_?barrier|checkpoint",
    re.I,
)

# ...and assets that yield to them. Ranked below everything else, because the
# budget they were taking is the budget the list above needs.
VEGETATION_NAMES = re.compile(
    # `plant` included after one plant asset became the single largest consumer
    # on the map — thousands of triangles each across a couple of hundred
    # placements — purely because nothing here matched its name. Vegetation is
    # only held back if it is recognised as vegetation, so the list has to cover
    # what maps actually call things rather than what a botanist would.
    r"tree|palm|bush|shrub|foliage|flower|fern|hedge|canopy|frond|log(?!ist)|"
    r"plant(?!room)|weed|grass|vine|reed|cactus|sapling|stump",
    re.I,
)


def place_of(actor_path, map_package):
    """The sub-level an actor was composed in, or `''` for the map itself.

    This is the best signal in the whole exporter and it took far too long to
    find, because it is not in the actor's name, class or size — the three things
    every version of this file looked at. It is in its *path*.

    A World Partition descriptor carries `actor_path`. For something dropped
    straight into the map it reads
    `/Game/Maps/L_Example.L_Example:PersistentLevel.StaticMeshActor_UAID_...`; for
    something belonging to a composed place it reads
    `/Game/Prefabs/Town/LI_Market_01.SM_Wall_01`. The
    package before the dot says which level the actor was authored in, and an
    actor authored in a level of its own is, by construction, part of a structure
    somebody built as a unit.

    That is exactly the thing this exporter is for and it needs no naming
    convention to detect: measured on the test level, its main town sub-level
    alone holds a sizeable share of the map's actors, and a handful of other
    composed places — a factory, a farm, a school and the like — account for
    most of the rest of the built environment. Under the size filter
    they were being taken apart piece by piece: the pieces are wall panels,
    catwalk railings and stair flights about a metre across, and every one of
    them is clutter by every test available until you notice which level it
    lives in.
    """
    text = str(actor_path or "")
    package = text.split(".", 1)[0]
    if not package or not package.startswith("/"):
        return ""
    if map_package and package == map_package:
        return ""
    return package


def priority(actor_label, mesh_path, place=""):
    """`1` for must-have geometry, `-1` for scenery, `0` for everything else."""
    haystack = "{} {}".format(actor_label or "", asset_name(mesh_path))
    if PRIORITY_NAMES.search(haystack):
        return 1
    # Vegetation before `place`: a palm inside a town is still a palm, and a
    # composed place is usually landscaped.
    if VEGETATION_NAMES.search(haystack):
        return -1
    if place:
        return 1
    return 0


def asset_name(mesh_path):
    """`/Game/Props/Rocks/SM_Rock_01.SM_Rock_01` -> `SM_Rock_01`.

    The name only. Matching a whole path against words like "ground" reclassifies
    everything an artist happened to file under a folder of that name.
    """
    if not mesh_path:
        return ""
    return mesh_path.rsplit("/", 1)[-1].split(".")[0]


def classify(actor_class, actor_label, component_class, mesh_path):
    """Return `ground` or `structure` for one component.

    Class first, name second. A class is a fact about what the thing is; a name
    is a convention someone followed or did not.
    """
    for p in GROUND_CLASSES:
        if p.search(actor_class or "") or p.search(component_class or ""):
            return GROUND
    haystack = "{} {}".format(actor_label or "", asset_name(mesh_path))
    for p in GROUND_NAMES:
        if p.search(haystack):
            return GROUND
    return STRUCTURE


def part_name(kind, map_name):
    """The node name written into the glb.

    The prefix is the contract with the viewer's classifier; the map name after
    it is for humans reading the part list in the level panel.
    """
    return "{}:{}".format(kind, map_name)
