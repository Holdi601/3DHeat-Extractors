# Game-file exporter

Reads level data out of the archives a game already ships, for the 3DHeat viewer.

This is the third route to a level's geometry, and it exists because the other
two leave a gap:

| Route | Needs | Gives |
|---|---|---|
| `unreal/` | the project open in the editor | exact geometry |
| `gamecapture/` | a screen and a capture | an estimate |
| **this** | **the shipped build** | **exact geometry, no project** |

Engines read so far: **Unreal** (`.pak`; Unreal 5 IoStore `.utoc` + `.ucas`
with its packages, static meshes and textures), **Unity** (`SerializedFile`,
bundles, meshes), **Frostbite** (`.toc` + `.cas`), **Source 2** (`.vpk`),
**ForzaTech** (`.minizip`, `.modelbin`, terrain) and **BeamNG** (level archives,
`.cdae` shapes, `.ter` terrain). Courses export from Forza, BeamNG.drive and
Assetto Corsa Rally laps.

The case it answers is a developer who has the build but not the project — a
different studio's title, an old branch, a build from a partner. An engine export
is better whenever it is available.

## What it does

**It reads containers.** Parsing an archive format is ordinary file handling, and
it is what FModel, UModel and AssetRipper have done in the open for years.

**It ships no keys and recovers none.** An encrypted archive opens only if you
pass a key you already hold. There is no bundled key, nothing that searches an
executable for one, and nothing that reads another process's memory. Reading a
file and defeating an access control are different acts, and the second reaches
the tool itself rather than only what someone does with it — so the line is drawn
inside the tool, which is also how the established extractors are built.

**What you point it at is your responsibility.** Games carry their own licence
terms; no design choice here discharges them.

## Using it

```
python -m heat3d_gamefiles identify <file>              # what is this?
python -m heat3d_gamefiles list     <archive> [--filter text]
python -m heat3d_gamefiles extract  <archive> <path inside> [--out file]
python -m heat3d_gamefiles meshes   <unity .assets>
python -m heat3d_gamefiles course   <lap file | lap folder | course name> [-o course.glb]
python -m heat3d_gamefiles courses  [-o folder] [--only name ...]   # every course in the library
python -m heat3d_gamefiles terrain  <forza track folder> -o map.glb \
                                    (--box x0 z0 x1 z1 | --route lap.json)
```

Run from this folder with the packages in `requirements.txt` installed
(`pip install -r requirements.txt`).

`course` is the one to start with: give it a lap and it exports the track
around it - ground, road, kerbs, barriers, tyre walls, signs - as one textured
`.glb` for the viewer, with nothing else to supply. It reads which game the lap
is from out of the file (`heat3d-lap` files say; anything else is Forza's), finds
the installed game, the level or stage, and - for Assetto Corsa Rally - which
way round its axes are. See [ForzaTech](#forzatech--minizip-and-modelbin),
[BeamNG](#beamngdrive--level-archives-cdae-ter) and
[Assetto Corsa Rally](#assetto-corsa-rally--unreal-5-iostore) below.

```
python -m heat3d_gamefiles course <beamng or ac rally lap .json>   # all it needs
python -m heat3d_gamefiles course lap.json --level <name>         # name the level or stage
python -m heat3d_gamefiles course lap.json --install <game folder> # a game outside Steam
```

A lap from a game whose courses cannot be exported says why and what to do
instead.

`list` and `extract` work on whichever archive you point them at — the format is
read from the bytes, not from the extension or from a flag you have to pass.

`identify` reads the file rather than trusting its extension, so it is the right
first step on anything unfamiliar.

Most shipped Unreal content is Oodle-compressed. Oodle is proprietary and has no
redistributable build, so this finds a library instead of carrying one — any
installed game or engine that ships `oo2core_*.dll` has one, and
`HEAT3D_OODLE=<path>` points at it directly.

## How far each engine goes

Formats are undocumented, so "supported" has to mean *verified against real
files*, not *did not crash*. Each reader below is checked against something the
format itself guarantees.

### Unreal — `.pak`

**Complete: lists and extracts.** Versions 8–12, Zlib and Oodle, encrypted
archives with a supplied key.

Verified against three shipped archives across two games — 178,809 entries
including one of 27 GB with 167,412 of them — by the SHA-1 Unreal writes ahead of
every payload. That hash covers the bytes as stored, so matching it means the
index decode, the payload offset and every compressed block boundary were right.

IoStore (`.utoc`/`.ucas`), what Unreal 5 ships, is read too - its packages,
cooked static meshes, textures and virtual textures, as far as a course needs
them. See [Assetto Corsa Rally](#assetto-corsa-rally--unreal-5-iostore), the
game it was built and checked on.

### Unity — `SerializedFile`, bundles, meshes

**Complete: lists objects and decodes meshes.** Format 22 and later, which is
Unity 2020 through 6000.

Verified across 12 installed games and 12 million objects. The object table has
to tile the data section exactly — no overlaps, nothing but alignment padding
between, the last object ending on the final byte of the file — which a header
misread cannot produce.

Meshes decode to positions, normals, UVs and triangles, including vertices held
externally in `.resS`. Each decoded mesh is checked against the bounding box
Unity stored for it: a field read in the wrong order yields plausible floats, and
that box is what tells the two apart. 820 meshes across four games decode with
their bounds agreeing.

Not implemented: compressed meshes, which are a different encoding and are
refused with a message rather than guessed at; and pre-2020 formats, which moved
several header fields.

### Source 2 — `.vpk`

**Complete: lists and extracts, CRC-checked.** Versions 1 and 2, split archives
and directory-inline entries.

The kindest of these formats, and the only one its vendor documents. Every entry
carries a CRC of its own bytes, so extraction is checkable against the archive's
own claim rather than against a fixture: 200 of 200 entries sampled across
Counter-Strike 2's shipped maps come back byte-exact. All 43 map archives parse,
22,637 entries in total.

One detail is worth knowing: a map ships its **navigation mesh** alongside the
compiled geometry — `maps/de_dust2.nav` and so on. For a heatmap of player
positions that is the more useful of the two by a wide margin, because the nav
mesh *is* the walkable floor, already computed and a thousandth of the size.

Not implemented: decoding Source 2's compiled assets (`.vmdl_c`, `.vtex_c`), and
the nav mesh itself. The CS2 nav format is version 36 and differs from the
Source 1 one that is widely documented — its bytes are structured rather than
compressed (entropy 6.2, 31% zeroes) but only 3% of its words read as
coordinates, so the layout is not the obvious one and is not yet worked out.

### ForzaTech — `.minizip` and `.modelbin`

**Complete for a course: a lap in, the track around it out** - terrain with the
road, verge and grass told apart, the ground's own textures, and everything
placed on it (kerbs, guardrails, tyre walls, signs, buildings) in its own
textures. Everything else lists and extracts.

```
python -m heat3d_gamefiles course "Lakeside Circuit"              # from the lap library
python -m heat3d_gamefiles course path/to/lap.json -o course.glb  # or one lap file
python -m heat3d_gamefiles courses -o courses                     # the whole library
```

The lap can be a file written by `heat3d_capture`, a lap or course folder from
FH Companion's library (`%LOCALAPPDATA%/FHCompanion/laps`), or just a course's
name. The installed game and the track the lap belongs to are found on their own
(Steam libraries are searched; `--track` points at a track folder directly).
Options, none of them needed:

| Option | Default | What for |
|---|---|---|
| `--margin` | 80 m | ground kept either side of the driving |
| `--texture-size` | 2048 | ground texture per 512 m tile (25 cm a pixel) |
| `--season` | summer | which season's ground maps and model textures |
| `--detail-budget` | automatic | a ceiling on triangles for everything placed; see below |
| `--trees` | off | also place trees and bushes |
| `--no-placed` | | terrain only |
| `--no-textures` | | flat colours, a far smaller file |

What one circuit's export holds, from its 46 recorded laps: 225,000 terrain
triangles across two tiles (road 25%, verge 34%, grass 27%, markings 14%), two
2048-pixel ground textures, and 1.5 million triangles of 5,900 placed copies of
107 models in 91 textures - 102 kerbs, 2,278 guardrail and barrier pieces, 3,363
props, 129 signs, 11 buildings. About ten seconds, 79 MB.

**The cut is a corridor, not a box.** Everything within `--margin` of the
driven line is kept - terrain, textures, placed models - and nothing else: a
terrain tile the corridor does not touch is not even read. For a circuit that
is nearly the lap's bounding box; for a point-to-point race it is a tenth of
it. A 14.7 km sprint crosses a 6 by 3.6 km box, 22 square
kilometres and some twenty-five million terrain triangles; its corridor is
1.53 km2, 1.3 million triangles, 160 MB and a minute. Consecutive samples less
than 50 m apart are joined, so the band has no gaps; further apart they are
the seam between two lap files - one sprint's finish and the next one's start -
and are not.

`courses` runs `course` over every course folder in the library, one
`<name>.glb` each. A course already exported is skipped (`--force` redoes it),
a course that fails does not stop the rest, and the end of the run lists what
was written.

A Forza track is four `.minizip` archives — tens of gigabytes each for an
open-world map — holding hundreds of thousands of entries between them. The archives carry **no names at
all**: no name table, no per-entry header, nothing but an ordinal. The names are
in the `ChunkContentsMiniZip*.txt` beside them, one line per entry in entry
order, and that pairing is the only way to ask for a file.

Three things in the index are wrong in ways that still produce output:

| Detail | What a careless reader does | What it looks like |
|---|---|---|
| segment base is 64-bit | reads it as u32 plus padding | correct for the first 4 GB, then noise of the right length |
| base is 8-byte aligned | ignores the padding word | two of the four archives claim codecs that do not exist |
| entry length is the gap to the next, minus its padding | uses the gap | LZ4 refuses the block outright |

Verified by reading 14,541 entries sampled across all four archives: every one
unpacks to exactly the length its record declares, under all three codecs (LZ4,
raw deflate, stored).

Terrain is where this pays off. The manifest names it as a spatial index —
`autoterrain_x-8184_z13299_cluster012.i.modelbin` — and each mesh carries its own
world-space scale and bias, so a tile lands where the game puts it with no
instance transform at all. That means **a lap of telemetry is enough to ask for
a piece of map**, because the telemetry is in the same coordinates:

```
python -m heat3d_capture route -o lap            # drive; Ctrl+C
python -m heat3d_gamefiles terrain <track> --route lap.json -o map.glb
```

or, for a course somebody has already driven a few dozen times:

```
python -m heat3d_capture laps "Lakeside Circuit" -o races/lakeside
python -m heat3d_gamefiles terrain <track> --route races/lakeside.json -o races/lakeside.glb
```

The proof that the two halves meet is the driving itself: across that course's 46
recorded laps the car sits a median of 0.19 m above the extracted ground, sd
0.14 m, worst point 0.58 m. That is the car's ride height. An error of even a
couple of metres — let alone the tile-sized ones this format makes easy — would
not fit in it, and no amount of looking at the mesh would have shown it.

The check on the decode is the tiles' own placement: a tile has to land inside
the 512-metre square its name predicts. 600 of 600 sampled do. (The names step
by 1023 while the tile is 512 metres — take the name as metres and every lookup
returns perfectly good terrain from the wrong half of the map.)

A square kilometre of open-world map comes out as 1.16 million triangles in five seconds.

#### Road, verge and grass

Not from material names: around a circuit 94% of the ground carries a
road-shader material, because the terrain near a road is drawn with it and
asphalt or grass is decided per pixel. Road comes from the vertex layout - road
strips carry three UV sets and nothing else does - and within a strip, asphalt
and verge are split by the tile's own asphalt mask. Against the 46 laps, 99.2%
of the samples fall on road-strip triangles, and 94.3% on the asphalt part once
the mask has split the verge off.

#### Textures

A swatch (`.pb`) is the same chunk container as a model: a 96-byte header, then
block-compressed pixels. The format is the u32 at byte 72 - 9 is BC7, 0 is BC1,
13 is RGBA - and was identified by decoding real files every way the payload's
size allows (byte 47, which looked like the format, is 4 in all 3,000 sampled).
Each terrain tile ships maps that say where asphalt, grass and woodland are at 25
cm a pixel; those are baked into one colour image per tile, oriented north-up by
laying the recorded laps over it.

#### Everything placed

Terrain carries its own world position; nothing else does. Where each copy of a
model stands is in `.pgeo` placement files, one per category per cell, 80 bytes
a copy:

| Bytes | Field | How it was decided |
|---|---|---|
| 0-11 | position, **sign-magnitude** 16.16 fixed | consecutive guardrails exactly 4.000 m apart; two's complement puts negatives 32 km away |
| 12-23 | the model's x axis in the world | the end of each guardrail lands on the start of the next: 93.5% within 5 cm |
| 24-35 | its y axis; z is x cross y | as above |
| 36-47 | scale | |
| 60-63 | material variant | billboards 0, 2, 2, 3, 4; every guardrail 0 |

Only `_3D` names are records; the table before them lists variants by bare name,
which read as records land at infinity. Every `_3D` record of every category
sampled lands inside its own file's box. Around Lakeside, kerbs sit 3 cm (median)
above the extracted ground with their faces up, within 5 m of where the cars
drove; guardrails are never on the line.

A model's triangles each carry a level-of-detail mask. The finest level is what
is marked for it; triangles marked for every level are breakable pieces (a
guardrail's posts, a stack's single tyres) that overlap the whole model and are
used only by models with no marked level.

**Nothing in the corridor is left out by default.** Copies within 25 m of the
driving start at their finest level, those out to 60 m one coarser, the rest at
their coarsest; then the costliest models - copies times triangles - step down
a level at a time until the budget fits. The budget sizes itself: every copy at
its coarsest, plus a quarter for detail, and never under 1.5 million. On a
circuit that is the 1.5 million; on a city course, where the corridor holds
120,000 copies - air conditioners, bicycles, laundry, the walls along the
expressway - it is 15 million, and the file is about 660 MB. The viewer loads
that on both backends.

The files are written compact: normals as bytes (`KHR_mesh_quantization`), UVs
and colours as normalised integers, indices as shorts where a part is small
enough. A vertex is 20 bytes instead of 32 - a third off every file, and the
difference between a city course loading and not - for under half a degree of
normal and a fifteenth of a texel. `write_glb(..., quantize=False)` writes the
full-precision form.

Given a number, `--detail-budget` is a ceiling instead, for a smaller file:
where even the coarsest levels do not fit it, copies are left out - first
everything more than 25 m from the driving, and within that, what looks
smallest from the road (distance over size) first. Kerbs are never left out or
simplified. The summary says how many went.

**Painted layers are lifted.** Lane markings and tyre marks ship as their own
triangles lying exactly in the road, which the game draws with a depth bias.
Drawn as they are, road and paint fight over every pixel; the export gives them
vertices of their own, 5 cm above the surface.

**Buildings are drawn whole.** A city builds its buildings out of modular
pieces - around one city street circuit, 383,000 of them, forty million triangles
even at their coarsest - so leaving pieces out one at a time left lone wall
slabs standing. The game ships its own whole-building stand-ins
(`scene\hlod_tier_1\`, one mesh per building, in world coordinates like the
terrain), and wherever one covers a building it is used instead of the pieces
and kept or left out as a unit. Their shared facade atlas reserves a solid
(255, 64, 0) block for surfaces the building shader tints from other textures;
that block is painted in the atlas's own average colour, or the city comes out
orange.

Around a country circuit nothing is left out: 1.5 million triangles. A single
lap of the city street circuit - 1.3 by 1.5 km of streets - keeps the 3,400 barriers, 1,500
props and 700 signs along the street and 174 whole buildings, leaves out
120,000 more distant copies, and is 193 MB.

**Which texture** comes from the model's material, not its name. A material
lists its textures as hashes; `AssetManifest.xml` beside the archives maps each
hash to a swatch. Base colour is slot `0x88B483AA` in the standard shader (1,054
of 1,054 materials sampled); other shaders are read by the texture's own kind
(`_bclr_`, `_diff_`). `bar_rur_armco_crct_02_c` has no texture of its own name -
its material binds `02_a`'s.

**Variants.** Every submesh record opens with a table of `(summer, winter)`
materials, one per look a copy can wear: five prints for a billboard, six paint
colours for a tyre stack. A standard record holds one entry and no count; a
longer one is a count and that many entries, 238 + 8 * count - 4 bytes, which
is why every field after it moves back by exactly that much. The copy's variant
(bytes 60-63 above) picks the entry.

**See-through surfaces** - chain-link fences, nets - are alpha-tested quads. A
surface whose texture averages under half opaque is left out whole; drawn
solid, a fence is a wall across the view of the track. (Judged a triangle at a
time instead, half of every quad survived and fences came out as rows of teeth.)

Left out by default: trees and bushes (`--trees` adds them), grass and crowd
templates, festival sites that change with progression, and projected decals.
Not decoded: signage prints that are blended over their board through alpha,
which come out opaque.

### Frostbite — `.toc` and `.cas`

**Lists and extracts; does not yet decode meshes.**

Both table forms are read. The DbObject tree — which is plain after a 556-byte
signature, not encrypted, despite looking otherwise — is parsed with every
container's declared length checked against what was consumed. The binary form
current Battlefield ships is a section table whose offsets are both stated and
derivable, so the arithmetic closes or the file is refused.

Nothing about that format is documented, so every step was checked against the
files:

| Step | Check | Result |
|---|---|---|
| section chain | each section starts where the last ends | closes on all 136 tables |
| asset records | identifiers distinct | 74,175 of 74,175 in one level |
| asset locations | offset+size inside an installed `.cas` | 74,175 of 74,175, none past the end |
| payloads | block chain consumes the payload exactly | 218 of 299 sampled |

Across a whole Battlefield 6 install that catalogues 2.98 million assets. Payload
blocks are Oodle or stored, and decode to 63 MB of data carrying recognisable
Frostbite signatures.

The 81 sampled payloads that do not decode are high-entropy from their first
byte — a different encoding, or encrypted — and are refused rather than guessed
at. XOR-obfuscated tables are likewise detected and declined rather than
implemented against no sample to check.

Not implemented: the Frostbite mesh format. The tables say what a level contains,
where it lives, and hand you the bytes; turning those into geometry is a separate
and much larger problem, and this does not pretend to it.

### BeamNG.drive — level archives, `.cdae`, `.ter`

**Complete for courses: `course` exports a lap's level.**

BeamNG ships plain zip archives - the engine's own art, the shared assets,
one per level - and mounts them, with the user's mods on top, into one file
system that levels refer into. The reader builds the same file system and
reads a level the way the game does. Nothing about the formats was assumed;
each was checked on the shipped levels:

| Part | Read from | Checked against |
|---|---|---|
| terrain | `.ter`: heights, layer map, layer names | height is `position.z + h * maxHeight / 65535`, rows along +y: road nodes lie on it to 1-2 cm in the median on three levels; `h / 32` or the other axis order miss by tens to hundreds of metres |
| placement | scene files (one JSON object a line), TorqueScript and JSON prefabs, forest instances | the rows of `rotationMatrix` are an object's own axes: rail sections chained along a road run along row 0 (92%); `rotation = "x y z degrees"` turns right-handed (92% of chains, 19% the other way) |
| shapes | `.cdae`, the game's documented MessagePack cache of every Collada file | 6,000 of the 6,002 shipped shapes read (two are an older version); node quaternions are Torque's, transposed: the full models' world bounds match the source `.dae` to the millimetre, 38 cm out the textbook way |
| shapes without a cache | Collada `.dae`, as mods ship them | against the game's `.cdae` of the same files: 594 of 600 bounds within 1 cm, 583 the same triangle count, 94% of materials' UVs identical |
| ground texture | each terrain layer's base map with its detail and macro overlays; road decals in the game's order; modelled road surfaces seen from above | the base map's row 0 is north (its green follows the grass layers, correlation 0.72 and 0.71); decals draw by descending `renderPriority` (asphalt 12-16, markings 10, rubber 4 across the shipped levels) |

What a course is made of: the terrain in a band either side of the driving,
classed by its layers' ground models and by the road decals over it (`ASPHALT`
is road; dirt, gravel, sand, mud are gravel and dirt); circuits modelled as
meshes, whose up-facing triangles of a road material become ground baked into
the same tiles; placed models by category from the level's own annotations and
their names, with Forza's levels of detail by distance and budget; water. The
level is found from the lap - the share of its samples lying on a level's
terrain, or on its roads where a level runs on past its heightmap. Exported on
six shipped levels of every kind, 20 seconds to 2 minutes each.

Not drawn: ground cover (procedural grass) and scenery placed round the edge of
the world. A model whose texture repeats is drawn in that texture's average
colour, since the viewer packs textures into an atlas.

### Assetto Corsa Rally — Unreal 5 IoStore

**Complete for courses: `course` exports a lap's stage.**

Unreal Engine 5.6.1, a modified build, shipping IoStore containers, none of
them encrypted; Oodle decompression comes from a library already on the
machine (every Steam library is searched). A stage is a World Partition map
with no landscape: its ground is static meshes, the stage cut into tiles split
into road and terrain, and far tiles beyond. Every layer was checked on the
installed game:

| Layer | Checked against |
|---|---|
| IoStore tables of contents (version 8): chunks, compression blocks, directory | 54 containers, 47,246 package paths |
| packages: summary (5.3-5.6 layouts), names, imports, exports, dependency tables | every cell of every stage opens; an imported package's name carries a number (`_Terrain` numbered 1 is the file `_Terrain_0`) |
| unversioned properties, read with a schema of names and types derived from the Unreal Engine 5.6.1 headers | 32,310 static-mesh and 10,906 scene components across 16 levels decode, each ending at its class's fixed native tail; this build has two more slots in `UPrimitiveComponent` and one more in the virtual texture volume, found from the data, and a value in either is an error, not a guess |
| cooked static meshes (not Nanite) | 296 of 300 random meshes (the four others are skeletal); a road tile's decoded extent matches the bounds it stores |
| textures: cooked platform data, the package's bulk data map, BC1-BC7 | a 4096-pixel DXT5 satellite image whose thirteen levels are exactly the bulk entries' sizes |
| virtual textures: Morton-addressed tiles in chunks | the first chunk is 4 + 4,096 x 34,848 bytes, its tiles to the byte; the road tiles land on the road it shows |

Which way round the world is, the telemetry does not document, and it is not
assumed: of the eight ways Unreal's two horizontal axes can map onto the lap's,
the one that puts the most of the lap on the stage's road tiles is used, on the
stage where it scores best (or the one the lap names, checked first). Tested
with laps built in a random frame on every stage: the right stage and the right
frame every time. No lap recorded in the game has been through it yet.

The ground is textured from the stage's baked ground colour - the streaming
virtual texture its runtime virtual texture volume covers, 13 to 59 cm a pixel
- and the stage's satellite image around it, placed by fitting the far
terrain's UVs (linear to a few thousandths); one stage without either is drawn
in flat colours. Placed models come in flat colours by category, with Forza's
levels of detail.

Known gaps: instanced components' own properties are laid out differently from
the engine's headers and are not read; their mesh comes from the dependency table
(each names exactly one) and their instances from the native block after the
properties, and they are taken to sit at their actor's root, which is how the
foliage actors place them. Forests generated by PCG at run time are not stored
and cannot be exported. Spline meshes (invisible respawn walls), decal
components and placed models' own textures are not drawn.

## Tests

```
python -m pytest
```

Synthetic tests build an archive byte by byte and run anywhere. Tests against
installed games skip when the game is absent rather than failing. Both matter:
the synthetic ones pin the field layouts that were in fact wrong at some point,
and the real ones are the only thing that catches a layout that is right for a
small file and wrong for a large one.
