# Unreal level exporter

Writes a decimated, world-space copy of an Unreal level as a single `.glb` for
the 3DHeat viewer to draw underneath the telemetry.

It is a **content-only plugin** — Python, no C++ module — so it drops into any
Unreal 5.x project and needs no compilation, no engine source, and no matching
toolchain. Copy a folder, run one command.

```powershell
.\run-export.ps1 -Project C:\game\Game.uproject -Map /Game/Maps/L_Example -Out C:\exports\L_Example.glb
```

## What comes out

One `.glb` with named parts, a `.json` beside it recording what happened, and a
`.log.txt` with everything the run printed — plain UTF-8, because the engine's own
log is UTF-16 with UTF-8 runs inside it and is unreadable without a decoder.

| Part | What it is | How the viewer draws it |
|---|---|---|
| `ground:<Level>` | the walkable surface, sampled as a heightfield | solid, desaturated |
| `ground:<Level> meshes` | landscape built from meshes: roads, paths, rock | the same, so a road reads as a road |
| `structure:<Level>` | buildings, rocks, containers, vehicles | faint wireframe you can see heat through |
| `water:<Level>` | sea, lakes and rivers, as flat surfaces | flat, no relief or contours |

The names are the contract. The viewer has heuristics for telling ground from
structures, but inside the engine a landscape is literally a different class from
a building, so the exporter says which is which and the viewer takes a labelled
part at its word.

Geometry is in **Unreal world space, centimetres**, with Y and Z swapped because
the viewer's world is Y-up. Telemetry exported from the same game is in the same
centimetres, so with the viewer's default axis mapping (`up: z`, `scale: 1`) the
level and the datapoints land on top of each other with nothing to line up by
hand. `--scale 0.01` if your telemetry is in metres.

## Install

Copy `Heat3DExporter/` into your project's `Plugins/` folder. `run-export.ps1`
does that for you and keeps it up to date, so for command-line use there is
nothing to install by hand.

The plugin enables what it needs — `PythonScriptPlugin`, `EditorScriptingUtilities`
and `GeometryScripting`, all shipped with the engine — so you do not have to edit
the `.uproject`. There is no C++ module and nothing to compile.

**What to hand someone else:** the `Heat3DExporter/` folder, plus
`run-export.ps1` and this file if they want the command line. Everything else
here — `.probe/`, `.out/` — is development scratch and not part of the plugin.

Expect a **peak of around 35 GB of RAM** on a World Partition map several
kilometres across.
That is the engine holding actor packages and their static meshes, not this
plugin's own geometry, and a collection runs after every batch. On a machine with
less, lower `--batch` (halving it roughly halves the peak of the part that *is*
controllable), crop with `--region`, or raise `--min-size`. A level of ordinary
size needs nothing.

## Use

**From the editor:** *Tools → Export level for 3DHeat*. Exports the open level to
`Saved/Heat3D/<Level>.glb` and blocks while it runs.

There is no dialog of settings, so the one setting that matters is chosen for
you: with no `--budget` given, the triangle budget comes from how large the level
turns out to be, once the descriptors have been read. On a large test level that
lands the size cut at a few metres, which is the same result as the tuned command
line below. A small level gets the 2M floor and does not need more.

The dialog at the end reports the size cut and, if the level was too large for
one click to do well, says so and points at the command line. See *when something
goes wrong* for why the size cut is the number to read.

**From the command line**, which is the one worth scripting:

```powershell
# A whole map several kilometres across, as used for the numbers below.
.\run-export.ps1 -Project C:\game\Game.uproject -Map /Game/Maps/L_Example `
                 -Out C:\exports\L_Example.glb `
                 -Args '--min-size 5 --budget 4000000 --batch 1024'

# Just the part the telemetry covers, keeping smaller props
.\run-export.ps1 -Project C:\game\Game.uproject -Map /Game/Maps/L_Example `
                 -Out C:\exports\L_Example_town.glb `
                 -Args '--min-size 2 --region -120000 -120000 120000 120000'

# What would be exported, without exporting it
.\run-export.ps1 -Project C:\game\Game.uproject -Map /Game/Maps/L_Example `
                 -Out C:\exports\probe.glb -Args '--dry-run'
```

| Option | Default | What it decides |
|---|---|---|
| `--budget N` | from the level's size | Triangles in the finished file. The viewer decimates again on import; this is a ceiling, not a target. Left unset, it is 1,100 per metre of the level's longest side, clamped to 2M–6M: 3.3M for a 3km map, the 2M floor for anything under 1.8km. Fitted so that the *size cut* stays under a shed rather than to hit a triangle count — see *the budget is objects, not detail*. |
| `--ground-share F` | 0.35 | How much of the budget the ground gets. It is a handful of huge triangles and needs little. |
| `--min-size M` | 4 m | Skip anything whose bounds are smaller. **The most important dial.** A modern map is millions of tiny props that are invisible at the range this viewer is used from and are most of the triangles. |
| `--max-size M` | 2000 m | Skip anything *bigger*, which matters more than it sounds. Every map has a few actors that enclose the whole world — a sky sphere, an ocean plane, distant backdrop scenery — and they push the level's bounds tens of kilometres out, spreading the ground grid and the tile budget over a world that is almost entirely empty. Nothing above two kilometres is a building; terrain comes from the ground pass. `0` keeps everything. |
| `--foliage-min-size M` | 8 m | The same for foliage, held higher: it is the densest thing on any map and the least useful for locating yourself. Keeps trees, drops undergrowth. |
| `--no-foliage` | off | Drop foliage entirely. |
| `--sublevels N` | 30 | How many building sub-levels to open, largest first. **On a World Partition map this is where the buildings come from** — see *the buildings are in the sub-levels*. Each costs a level load, about twenty seconds. |
| `--no-sublevels` | off | Skip that pass. Faster, and on a map composed of Level Instances it exports the landscape and none of the buildings. |
| `--sublevel-min-actors N` | 200 | Ignore sub-levels smaller than this. A prefab of three actors is a bench. |
| `--hotspots PATH` | `hotspots.json` beside the output | A traffic grid from the telemetry, written by `node tools/heat-grid.mjs`. **The only input here that is not a guess** — see *rank by where the players went*. `none` to ignore it. |
| `--region MINX MINY MAXX MAXY` | whole map | Crop, in Unreal centimetres. |
| `--exclude PATTERN` | see below | Add a name pattern to skip; repeatable. |
| `--include PATTERN` | – | Keep matching actors whatever else says otherwise; repeatable. |
| `--scale F` | 1.0 | Multiply positions. Leave alone unless your telemetry is not in centimetres. |
| `--no-landscape` | off | Skip the ground pass. |
| `--batch N` | 2048 | Actors paged in at once. The main memory dial. |
| `--tile M` | auto | Tile edge in metres for the build. |
| `--max-actors N` | – | Stop after N actors. For a quick look. |
| `--dry-run` | off | Collect and report, export nothing. |

Excluded by default, by name: `grass`, `weed`, `fern`, `nettle`, `clutter`,
`litter`, `pebble`, `gravel_small`, `leaf/leaves`, `decal`, `billboard`,
`impostor`. By class: decals, lights, reflection captures, volumes, nav links,
Niagara, audio, player starts, spawn markers, cameras, splines.

## Measured on a real map

A large test level, World Partition, in an Unreal 5.2 project: hundreds of
thousands of actor descriptors, over a hundred thousand external actor packages,
many GB on disk, several kilometres of content across. Exported headless on a
desktop with `--min-size 5 --batch 1024` and the budget chosen automatically:

| | |
|---|---|
| Total | **under ten minutes**, a file of a couple of hundred MB |
| Opening the level | ~2 minutes (World Partition initialising, no actors loaded) |
| Reading every descriptor and filtering | seconds |
| Ground: millions of rays, all of them hitting, millions of triangles at ~3m cells | under half a minute |
| Ground meshes (roads, paths, rock) | millions of triangles |
| Structures | millions of triangles |
| Water | the sea, plus many lakes and rivers |
| Actors kept | a minority of the descriptors, a sizeable share of them kept below the size cut for being architecture |
| Peak memory | ~32 GB |

Loaded into the viewer with telemetry from the same game on top:
**all of the data's X and Z spans fall inside the level.**

A small non-partitioned test level (a few dozen actors) exports in **seconds**.

Where the budget goes, by actor class — the report prints this, and it is the
first thing to read when the picture looks wrong:

| | |
|---|---|
| ground meshes (roads, rock) | the largest share |
| StaticMeshActor | next |
| InstancedFoliageActor | close behind |
| Actor (blueprints) | about half that |
| a door blueprint class | a small share |
| PackedLevelActor | a small share |

## How it works

**Filter before loading.** World Partition keeps a descriptor per actor — class,
label, world bounds, and the guid to page it in — so "is this a building or a
screw?" is answered without touching the package. On a map of many GB that is the
difference between minutes and never: hundreds of thousands of descriptors read
in seconds, and the size filter alone removes most of them.

**The ground is traced, not exported.** A landscape has no mesh anyone can ask
for; it is a heightmap the renderer turns into geometry at a resolution that
depends on the camera. `CopyCollisionMeshesFromObject` declines it (static meshes
only) and the heightmap-to-render-target route needs an RHI a headless commandlet
does not have. But a landscape *does* have collision, so the exporter samples it:
a grid of downward rays, one height per cell, triangulated, skipping cells where
nothing was hit. Only the landscape is loaded while this runs, so every hit is
terrain and no roofs end up in the ground. It is also not landscape-specific —
terrain built from static meshes or some plugin's own heightfield comes out the
same way.

**Each unique mesh is read once.** A map places a few thousand distinct assets
hundreds of thousands of times. Read and reduce each asset once, reuse it per
instance, and the expensive part scales with the asset count instead of the
placement count. Each asset gets a triangle allowance from its world size — a 40m
warehouse earns more than a 3m crate — and allowances are quantised to powers of
two so near-identical sizes share one reduction.

**Instances go in one call.** `AppendMeshTransformed` takes an array of
transforms, so a foliage component with 500 placements is one call, not 500.

**Tiles keep memory flat.** Geometry accumulates per spatial tile; each tile is
reduced to its share of the budget as it completes and folded into the total. A
tile's worth of triangles is in memory at a time, not a map's, and the budget
lands where the geometry actually is instead of being spread evenly over empty
countryside.

**Reduce to a tolerance, not to a triangle count.** Three stages per asset,
cheapest and safest first. A **coplanar merge**, which is free — adjacent
triangles in the same plane become one and the shape is identical. Then a
reduction **to a geometric tolerance** derived from the object's own size: 2% is
20cm on a 10m shed and 1.2m on a 60m hangar, which is about how much detail each
can lose and still be recognisable. Only if the result is still enormous does a
triangle count get imposed.

The order matters because a triangle count is a promise about the *wrong thing*.
Asking a 40m building and a 4m crate each to become the same handful of triangles
flattens the building into a wedge, and that is what turned an early export of
this map into a pixelated mess. A tolerance says "stay within 20cm of the
original" and lets the count fall where it may.

The tolerance is capped in absolute terms as well as proportionally, because it
controls one thing besides how much detail is lost: a tolerance wider than a
wall is thick lets the simplifier merge the wall's inside face with its outside
one. Two percent of a 40m building is 80cm, walls are 20, and a building whose
walls have all become single surfaces collapses into a few large quads hanging
in the air. `TOLERANCE_CEILING_CM` is 15.

**When shape is not affordable, nothing.** Below `MIN_OBJECT_TRIANGLES` a quadric
simplifier does not produce a smaller version of an object; it produces a few
long thin spikes, because the error metric is happy to keep some far-apart
vertices and collapse everything between them. Early screenshots of this map were
a field of white shards for exactly that reason.

The first fix was to substitute the bounding box: 12 triangles, always reads as a
solid object of about the right size, an honest *something this big stands here*.
On a sparse map that is right, and `--boxes` still does it. On a crowded one it
was worse than the problem — a field of cyan cubes between the buildings that hid
the buildings, the ground and the heat, and answered a question nobody had asked.
So the default is to leave the object out, and boxes are off.

What survives instead is decided by name. Anything matching
`classify.PRIORITY_NAMES` — fence, railing, barrier, gate, wall, door, window,
roof, stair, ladder, bridge, slab, panel, beam, step, ramp, corridor, hall — is
guaranteed a real reduction, offered before anything else in its actor, and
**exempt from the size filter down to `priority_min_size_m`**. That last part
matters more than it sounds: see *the size filter was deleting the buildings*.

Vegetation is the mirror image. It is never boxed (a crate standing where a tree
stood is worse than an empty patch of ground), its allowance is multiplied by
`VEGETATION_ALLOWANCE_WEIGHT`, and it is capped per tile. Measured before that:
foliage took about as many triangles as every building on the map put together,
which for a movement heatmap is exactly inverted.

**The size filter was deleting the buildings.** `--min-size` is the most valuable
filter here and it was also quietly removing the answer. It measures an actor's
bounding diagonal, and a wall panel, a doorway, a flight of steps and a railing
are all under five metres — so a modular building was dismantled piece by piece
at the descriptor stage, before the budget, the ranking or the priority list had
any say in it. Measured in a 300m box around the densest telemetry on the test
level: thousands of actors dropped for being small, and not one building among
the survivors.

Named architecture is therefore exempt from it, down to `priority_min_size_m`
(1.5m). On the test level that recovers around a hundred thousand actors, and the
report says so — the
`kept: small but architectural` line in the drop table. Everything else still has
to clear the full threshold, and what comes back still has to earn its triangles
from the budget like anything else.

**The buildings are in the sub-levels, and nothing will load them.** This is the
one that mattered most, and every explanation before it was wrong.

On a World Partition map composed of Level Instances, the buildings are not actors
of the map. They are actors of *other levels* — on the test level, hundreds of
them, with names like `LI_House_01`, `LI_Factory_01`, `LI_School_01` — placed by
a transform. Their descriptors are visible from
the open map, with world-space bounds, which is exactly why the problem was so
hard to see: the report counted them as kept and the size filter, the budget and
the tile shares all had opinions about them. But:

    WorldPartitionBlueprintLibrary.load_actors() instantiates 399 of 400
    persistent-level actors and 0 of 400 actors belonging to a sub-level.
    pin_actors does no better. Nor do the HLOD actors load, in either layer.

Measured with a control, which is the only reason it is trustworthy: a probe that
finds nothing looks identical whether the thing is absent or the probe is broken,
and that mistake was made twice here — once reading a UTF-16 log as bytes and
concluding an asset was missing when it was in the file all along.

So the geometry and the placements are both available, in two different places,
and `heat3d/sublevels.py` joins them: group the descriptors by sub-level, split
each into one group per instance, open the sub-level once, and solve each
instance's transform by matching actors between the two by label. Translation and
yaw, in closed form. On a factory sub-level instanced four times at four
different angles, all of its thousands of actors match per instance and the
median residual is **2.2cm**.

Two details earned their keep:

* **Instances are found by anchor, not by distance.** Distance clustering is the
  obvious approach and it fails both ways: at a 200m cell a house placed fifty
  times came out as six groups, because occupied cells chain into one another,
  and any threshold small enough to stop that splits a terrace, since two
  neighbouring houses are closer together than one house is wide. Instead: a label belonging to
  one actor per instance occurs exactly as many times as there are instances, and
  those positions are one anchor inside each. Everything else joins its nearest
  anchor. That cannot merge two instances or split one, however they are arranged.
* **Every distinct building first, repeats afterwards.** A building is reduced once
  and stamped wherever it occurs, so detail is cheap and copies are not. Spending
  each building's share as it arrived gave 2 copies of the first house and nothing
  for anything after it — the same failure as everywhere else in this file, one
  level down. A school that appears once is worth more than the fiftieth copy of
  a house.
* **Labels are only usable where they are unambiguous.** The correspondence is by
  actor label, and a sub-level may hold many actors sharing one. Keeping the first
  of each — `setdefault`, which is the obvious thing to write — matched every
  map-side descriptor of that label against a single arbitrary position and fitted
  the transform to noise. One compound's sub-level "matched" all of its thousands
  of actors and was rejected for a median error of over ten metres, so the
  building the telemetry
  draws the clearest floor plan inside was missing from every export. A label
  occurring once on each side is a pair; anything else is worse than nothing.
* **Never reduce across the assembly.** A quadric simplifier collapses whichever
  edge adds least error, and across thousands of disconnected shells the cheapest
  collapses are the ones that bridge between separate panels — it keeps a few
  far-apart vertices and joins them. Squashing an assembled building to a triangle
  count produced buildings shot through with long thin spikes. The budget is spent
  per part instead, where the simplifier only ever sees one panel's own edges and
  the allowance is spent as a 15cm tolerance, so a wall comes back a wall. What
  bounds the building is then how many parts it can afford, and the parts it
  cannot afford are the smallest: fixtures, signage and door furniture.

**Rank by where the players went, not by how big things are.** Every other
decision in this exporter is a guess made by looking at the level — an actor's
size, its asset's name, how many actors a sub-level holds — and those guesses get
the important case wrong in a specific direction: a hillside prefab outranks a
town hall, because the hillside has more actors and a bigger bounding box.
It is also why the name lists in `classify.py` keep growing, and why they will do
less on a project that names things differently.

The telemetry knows the answer, and it is the same telemetry the viewer draws on
top of the export. `node tools/heat-grid.mjs` buckets it into 32m cells in Unreal
centimetres and records two numbers per cell: datapoints, and how many distinct
3m height bands were stood on — the second being what separates a building from a
field. The exporter reads that (see `heat3d/hotspots.py`) and uses it to choose
which sub-levels to open and which *copies* of each to stamp, because a house
placed fifty times is not fifty equally interesting houses.

On the test level that roughly tripled the building instances, and put geometry
under the busiest telemetry on the map: tens of thousands of triangles within 50m
of the densest cell, which had barely a hundred before any of this existed.

Two things it needs guarding against, both measured:

* **Traffic rewards small things in busy places.** Players walk *to* vehicles, so
  a parked-truck sub-level ranked among the busiest on the map and a flatbed truck
  was handed tens of thousands of triangles. Traffic decides what is worth
  exporting; a building's own footprint caps how much of it there is to draw
  (`BUILDING_TRIANGLES_PER_METRE`).
* **It is optional and must stay optional.** A map being exported for the first
  time has no telemetry yet, and with no grid the ranking falls back to size and
  says so in the log. The point of the grid is the *second* export.

**Water is exempt from the size *ceiling*, and the sea is a special case.** An
ocean is legitimately larger than the map — on the test level, kilometres wider
than the land — so `--max-size`, which exists to catch sky spheres and vista rings,
was eating it. Water now skips that check and is excluded from the content bounds
instead, then clipped back to them, so it cannot spread the ground grid over open
sea.

The sea also cannot be built from its spline, which is the natural assumption and
wrong: an `AWaterBodyOcean`'s spline marks the *island* — the hole the sea is not
in — so triangulating it produces a lake-shaped polygon in the middle of the
land. The first attempt did exactly that and reported plenty of lakes and no
sea. It is built as a plane over the map at its own surface height instead,
found by class.

**Scraps are stripped from foliage only.** `RemoveSmallComponents` is how you get
rid of the thousands of leaf cards that no reduction survives, and it is a trap
everywhere else: a wall panel is a flat shell with no volume, so a volume
threshold deletes half a modular building and leaves the other half hanging in
mid-air. Foliage gets it, by triangle count and area; nothing else does.

**The budget is objects, not detail.** This is the least obvious thing here and it
governs the whole design. A quadric simplifier collapses edges; it cannot merge
two objects that do not touch. So a mesh made of N separate objects has a floor of
roughly 24×N triangles no matter what target you ask for — 150,000 objects cannot
be 1.6M triangles, they are 3.6M at their absolute floor, and at that floor they
are unrecognisable blobs anyway.

Four attempts got this wrong before the arithmetic was believed. Asking for 1.6M
from several times that returned all of it, silently. Then a per-tile share
computed from what the tile *produced* rather than what it was *allotted* summed
to the map instead of to the budget. Then spending the budget on the largest objects by allowance kept the
few hundred biggest things on the map — everything hundreds of metres across —
and dropped every building.

The fourth is the subtlest and it survived two rounds of screenshots. How many
objects a budget holds was computed by dividing it by `MIN_OBJECT_TRIANGLES`, the
*allowance floor* — but an allowance is what an asset is offered and almost
nothing takes all of it, so the real cost was half that. Planning at the floor
therefore held back half the objects the file had room for, and since the ranking
is by size, the half it held back was every building under about twenty metres.
Their roof and floor slabs are large enough to survive on their own, so the export
came out as floor plans hanging over an empty map — which looks exactly like
simplification damage and is not. Planning at the *measured* cost,
`PLANNED_OBJECT_TRIANGLES`, moved the cut from over twenty metres to a few metres
and roughly quadrupled the instance count.

What works: decide how many objects the budget can hold, keep the largest of them,
give each a size-proportional share, and enforce a ceiling per tile *as geometry
arrives* rather than reducing afterwards. One actor can be a packed level holding a
whole district, so the ceiling has to be checked on every append; what gets refused
is the tail of a tile processed largest-first. Deliberately, the plan offers more
than the budget: the per-tile cap is what enforces it, and it drops from the bottom
of the same ranking, so over-offering is safe in a way that under-offering is not.
The report says how many instances did not fit.

So `--budget` trades **coverage against detail**: 4M triangles on the test level
is on the order of a hundred thousand instances, cut at a few metres, most of
them boxes. Double the budget for roughly
double the objects. The same floor applies to the viewer's own import decimation,
which is why exporting a hundred thousand tiny objects would not help even if it
fitted.

Two rankings have to agree, and once did not. Foliage actors are ranked by a
fraction of their bounds (`SCATTERED_RANK_WEIGHT`), because an
`InstancedFoliageActor`'s box spans every tree it plants and is therefore the size
of a district. Without that they sort to the front of everything and take the
budget with them. The per-tile ordering has to use the same weight, or a forest
jumps the queue inside a tile the plan had it at the back of. No single actor may
take more than `PER_ACTOR_TILE_SHARE` of a tile.

**The vegetation cap is per tile, and has to be.** It was one running total
across the map, and tiles are visited in spatial order — so the first tiles
reached spent the whole allowance and every tile after them got none. On the test
level that put all the trees in a band through the middle and the right-hand
side and left the rest bare, which reads as the exporter losing half the foliage
and is really the exporter spending it all in one place.

**Tiles may overspend, within reason.** Tile shares are proportional to what the
descriptor plan expected each tile to hold, and that expectation cannot know how
much of a tile is rock destined for the ground part or how much is one packed
actor. So a dense tile refused geometry while the map-wide budget went unspent —
thousands of instances turned away with most of the structure budget idle.
`TILE_SHARE_SLACK` lets a tile take up to twice its share; `begin_tile`
recomputes from what *remains*, so the total still holds.

**Allowances over-subscribe the budget on purpose.** The plan divides the budget
across every actor it kept, and a large fraction never spend it — routed to the
ground part, capped as foliage, or too small to be worth keeping. That left the
structures spending about a third of their allowance while the buildings that
were present had under a hundred triangles each. `PLAN_OVERSUBSCRIBE` offers each
object three times its strict share; the per-tile ceiling is what enforces the
budget, and it refuses from the bottom of the size ranking.

**Source models, not render LODs.** A render LOD is a rendering resource, and a
commandlet with `-nullrhi` has none — reading one asserts inside the engine
(`StaticMeshLODResourcesToDynamicMesh.cpp:52` first ensures, then it dies). The
alternative, `-AllowCommandletRendering`, compiles global shaders on startup and
fails on plenty of workspaces. So the exporter reads the authored source model and
does the reduction itself, which also means it does not depend on a project having
an LOD chain at all.

## What it does not do

- **Materials, textures and colour.** Geometry only. The viewer draws the level
  desaturated on purpose so the heat is the only saturated thing on screen.
- **Spline meshes** (roads and rivers built along splines) export as the
  undeformed mesh at the component's transform, so they sit in the right place
  but do not follow their spline.
- **Prebuilt HLOD proxies** are supported with `--hlod` where a project has them
  built, and are usually the cheapest possible source for this. They were not
  built in the project this was developed against, so that path is the least
  exercised.
- **Nanite-only geometry** with no source model would come out empty; nothing so
  far has hit this, since Nanite meshes keep their source.

## When something goes wrong

The `.json` written beside the `.glb` is the receipt: every filter's drop count,
per-step timings, triangle counts before and after each reduction, and the
settings that produced it. Start there — a kept count that is a small fraction of
the actor total is alarming until you read the breakdown underneath it.

**The editor exits immediately.** A plugin whose module has no compiled binary
aborts startup in `-unattended` mode, via a message dialog nobody sees.
`run-export.ps1` scans for those and disables them; the log names them.

**"Failed to compile global shader".** Something passed
`-AllowCommandletRendering`. The exporter does not need it.

**Nothing in the ground part.** The landscape's collision was not loaded, or the
level has no landscape. Check `steps.ground.hits` in the report; zero hits with a
non-zero ray count means the rays found nothing solid. A few hundred hits out of
half a million means the grid was stretched over the wrong area — that was the
symptom when the grid came from every actor's bounds instead of the landscape's,
with a sky sphere putting those bounds tens of kilometres apart.

**Most of the map is missing.** Read `steps.structures.instances_over_budget`.
The budget bought the largest objects and refused the rest; raise `--budget`.

**Buildings look like slabs floating in mid-air.** The same thing, one step
earlier: read `steps.collect.size_cut_m`. A modular building is many actors, and
if the cut lands between its roof slab and its wall panels you get the slab
without the walls. Anything above ~10m there means the budget is holding back
more than it should.

**Everything looks like crumpled foil rather than boxes.** Not the exporter, and
fixed on the viewer's side: turn *Sharp edges* back on under Level → Structures.
The `.glb` carries no normals, so the viewer derives them, and the derivation
averages across every face meeting at a vertex — one vertex where eight
triangles of a box corner meet — which rounds off every corner of every box.

Worth knowing because it is a trap for a reader of these screenshots: it makes
correct geometry look like simplification damage. The exporter does not need to
ship normals to fix it, and shipping them would be expensive — per-face normals
mean splitting every shared vertex, which would more than double the size of
the file. The face normal is recovered per fragment from the derivatives of
the world position instead, for nothing.

## A large map, end to end

For a large World Partition level, with a prebuilt editor:

```powershell
cd unreal
.\run-export.ps1 -Project C:\game\Game.uproject `
                 -Map /ExampleMap/Maps/L_Example `
                 -Out .out\L_Example.glb `
                 -Options '--min-size 5 --budget 4000000 --batch 1024'
```

Then look at it, which is not the same as reading the report. In the 3DHeat
viewer's repository:

```powershell
node tools\shot-level.mjs      # needs the dev server on :5273
```

It loads the newest `.glb` from `.out` with import decimation off, finds the
densest cluster of structures with a histogram rather than aiming at the middle
of the bounds, and writes six screenshots to `tools/.smoke/`: the whole map, then
three distances into the dense area opaque and unlit, then two in the look the
viewer actually ships. Every real problem with this exporter was found in those
pictures and none of them in the numbers — the run that produced floor plans
hanging in mid-air reported tens of thousands of instances and millions of
triangles, which looks like a healthy export.

Worth knowing about studio workspaces:

- A level that lives in a Game Feature plugin has the package path
  `/<PluginName>/Maps/<Level>`, not `/Game/...`.
- A project plugin with no compiled module for this editor aborts a headless
  start. The runner finds and disables such plugins automatically.
- Global shader compilation can fail in a workspace, so the runner does not use
  `-AllowCommandletRendering`. The exporter does not need it.
- The runner copies the plugin to `<project>\Plugins\Heat3DExporter` on every
  run, from `Heat3DExporter` beside it, so edit the copy in this repository.
  In a version-controlled project, keep that folder out of the depot.
- The first export of a map is slower than later ones: the derived data cache is
  cold and the engine builds each static mesh the first time it is read.

## Versions

Developed against **Unreal 5.2** and written to the 5.x
scripting API: `WorldPartitionBlueprintLibrary` (5.1+), Geometry Scripting, and
the Python plugin. Places where the API has moved between versions — reading a
hit result, asking whether an actor is editor-only, the simplification method
enum — try what exists and degrade rather than failing. Unreal 4 is not supported;
it has neither World Partition nor Geometry Scripting.

## Porting to another engine

The viewer's side of this is small and worth stating for the Unity, Godot and
Source exporters that come next:

- one `.glb`, self-contained, no external `.bin`;
- `POSITION` as float32 and `mode: 4` triangles — no `KHR_mesh_quantization`, no
  Draco, no interleaving requirement;
- node names `ground:<name>` and `structure:<name>`;
- world-space coordinates, Y-up, in whatever unit the telemetry uses.

Nothing else is read. Normals are optional (the viewer derives them from the
winding, so wind them correctly), and materials, UVs, skins and animation are
ignored.
