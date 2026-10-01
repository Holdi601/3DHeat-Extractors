# Games: recording and courses

How each supported game is recorded, what it sends, and which of their courses
can be exported from the game's own files.

The rules are the same for every game: a course comes out of files the user
owns, or is reconstructed from the user's own gameplay. Archives that are
encrypted open only with a key the user already holds - nothing here ships
keys or recovers them, and nothing injects into a game's process. Telemetry
comes only from the interfaces the games publish for it (UDP outputs, shared
memory, their own mod and plugin interfaces). What a user points the tools at
remains subject to that game's licence.

## Telemetry

Every game ends up as `heat3d-lap` files ([`lap-format.md`](lap-format.md)),
with canonical channel names and a `meta` block saying what its numbers mean -
how its slip is scaled, circuit or rally or arcade, the surface when known - so
a reader treats them all alike. All commands are `python -m heat3d_capture ...`
in `gamecapture/`.

| Game | How to record | Position | Tyres and slip | Drift angle | Status |
|---|---|---|---|---|---|
| Forza Horizon 4/5/6 | Data Out over UDP: `heat3d_capture telemetry` (relays to FH Companion) - or FH Companion's own `.tele.gz` beside each lap, loaded with the lap file | world XYZ | slip ratio, slip angle, combined slip per wheel, normalised; surface rumble | car-frame velocity | recorder tested; `.tele.gz` reader checked on real laps |
| Forza Motorsport (incl. 2023) | the same; told apart by packet length | world XYZ | the same | the same | recorder tested |
| Assetto Corsa, ACC, AC EVO | shared memory: `telemetry --game ac / acc / acevo` | mean of the tyre contact points | slip ratio and angle (physical, scaled by a tarmac peak); ABS and TC acting (ACC, EVO) | car-frame velocity | recorder tested on synthetic pages |
| Assetto Corsa Rally | `telemetry --game acrally` | the same | the same; wheel loads; temperatures converted from kelvin | car-frame velocity | recorder tested on synthetic pages |
| EA SPORTS WRC | UDP: `telemetry --game wrc --configure "<Documents>/My Games/WRC/telemetry/config.json"` switches the default `wrc` packet on (the file is backed up first) | world XYZ | contact patch speed per wheel; slip derived | from the car's axes and its velocity | recorder tested on packets built from the documented layout |
| DiRT Rally 2.0, DiRT Rally, DiRT 4 | UDP: set `<udp enabled="true" extradata="3" port="20777" delay="1" />` in `hardware_settings_config.xml`, then `telemetry --game dirtrally2` (or `dirtrally`, `dirt4`) | world XYZ | contact patch speed per wheel; slip derived. No handbrake channel | from the car's axes and its velocity | the same |
| Richard Burns Rally (RSF/NGP) | NGP's UDP telemetry (`udpTelemetry=1`): `telemetry --game rbr` | world, up axis found from the stage | spring deflection and force per wheel; no wheel speeds | the car's own velocities | the same; axes and speed units are undocumented and handled from the data |
| BeamNG.drive | the `heat3d` protocol mod: `telemetry --game beamng --install`, then Options > Other > Protocols > "others" | world, turned to Y up | wheel surface speed, rotation, load, ground contact; slip derived | from the car's axes and its velocity | recorder tested on packets built from the mod's struct; the mod is untested in the game |
| Trackmania (2020) | the Openplanet plugin `Heat3dRecorder`: `telemetry --game trackmania --install`, loaded in Openplanet's developer mode; one CSV per run in its storage folder | world XYZ | Trackmania's slip coefficient per wheel; loose ground from the wheels' materials; in the air | front and side speed | reader tested; the plugin is untested in the game |
| iRacing | its own `.ibt` files; no recorder needed | latitude/longitude/altitude, laid on a tangent plane | no slip (not logged); yaw rate and lateral acceleration | from the yaw rate against the path | reader tested on a synthetic file |

What none of the rally games sends - tyre slip - a reader works out: from
each wheel's contact patch or surface speed against the car's speed where a
game sends one (`(wheel - car) / car`), from its rotation and a rolling radius
measured on the lap itself where it sends rotation only. The drift angle comes
from the car's own velocity (Forza, the AC games, Trackmania, RBR), from the
world velocity projected onto the car's axes (EA WRC, DiRT, BeamNG), or from
the heading against the path. Whether the car is in the air comes from its
wheels' contact, loads or suspension.

Where a game's world is left-handed - Forza's is - the course comes out
mirrored in a right-handed viewer; the laps' own steering and suspension give
it away. Games whose handedness is undocumented
(RBR, BeamNG, Trackmania) are handled the same way: measured, not assumed.

Beyond the driving channels, the recorders keep what each game adds: the AC
games' ABS and traction control acting, brake bias, DRS, pit limiter, boost and
brake pad and disc life; iRacing's aid settings, brake bias, fuel, oil and
water temperatures, and every other channel it logs under its own name.

### Sources and what is not yet confirmed

- **Forza:** the official Data Out documentation (Forza Motorsport; Forza
  Horizon 6) for the fields, their units and the car frame (x right, y up,
  z forward - left-handed). What Forza normalises its slip against is not
  documented: measure where each car's grip actually falls away rather than
  assuming 1 is the limit.
- **Kunos:** the AC1 SDK, the ACC shared memory documentation v1.8.12,
  `albertowd/live-telemetry-evo` (EVO, measured; AC Rally uses ACC's layout,
  "confirmed via byte-by-byte probe"), `LuizZak/AssettoCorsaRallyTelemetryReader`.
  The side a car-frame x axis points to is documented both ways: use the
  drift angle's size, not its sign.
- **EA SPORTS WRC:** EA's UDP Telemetry Guide v1.3 and the channel list the
  game writes to `readme/channels.json`; offsets of the default `wrc` packet as
  decoded by nobonobo/obs-codemasters-telemetry. The stock default port is not
  documented; the recorder switches the packet on at 20777.
- **DiRT:** Codemasters' DiRT 4 UDP document, ErlerPhilipp/dr2_logger,
  soong-construction/dirt-rally-time-recorder. The side vector's sense and the
  suspension position's unit are documented both ways; neither decides a
  finding.
- **RBR:** NGP's `rbr.telemetry.data.TelemetryData.h` (via
  mika-n/RBRUDPTelemetryLogger, groybe/rbr-udp-telem). Position axes, attitude
  units and the speed unit are not documented: the recorder takes up from the
  stage and speed from the positions.
- **BeamNG:** documentation.beamng.com (protocols; refNodes: vehicle +X left,
  -Y forwards, Z up), and the game's own `lua/vehicle/protocols.lua` and
  `wheels.lua`. OutSim is a stub in current versions; OutGauge has no position
  and MotionSim no wheels, hence the mod.
- **Trackmania:** Openplanet's `CSceneVehicleVisState` and VehicleState plugin,
  the `CSmScriptPlayer` race time. Openplanet plugins cannot send UDP, so the
  plugin writes files. Unsigned plugins load only in developer mode, which
  limits online play; the plugin can be signed on openplanet.dev.

None of the recorders has yet been run against the live games here; the first
real session of each is the test that matters, and `gamecapture/tests/test_kunos.py`,
`test_rally.py` and `test_beamng.py` pin the layouts they were built
to. Each recorder checks its packets as they come - the velocity against the
speed, the struct's size - and says so when a layout does not match.

## Courses: what is exported, per game

A course adds the kerbs, walls and run-off that explain a line.
`python -m heat3d_gamefiles course <lap>` (in `gamefiles/`)
exports one from the game's own files where that has been built and checked on
an installed copy; where it has not, the reason is below, and the lap says so
when it is given to `course`.

| Game | Course from a lap | Why, or what it takes |
|---|---|---|
| Forza Horizon | yes | ForzaTech archives; see `gamefiles/README.md` |
| BeamNG.drive | yes | the level archives, read as the game mounts them |
| Assetto Corsa Rally | yes | Unreal 5 IoStore, not encrypted; the frame found from the lap |
| Assetto Corsa EVO | no | its content is one package obfuscated with a key; this tool recovers none |
| iRacing | no | track models ship in a protected format that is not a file-extraction target |
| Forza Motorsport | not yet | same engine as Horizon; needs an installed copy to map its track folders |
| Assetto Corsa | not yet | `.kn5` is described publicly; needs an installed copy to build and check against |
| Assetto Corsa Competizione | not yet | Unreal 4 `.pak` (the reader lists and extracts them); needs a UE4 static-mesh reader, checked on an installed copy |
| EA SPORTS WRC | not yet | Unreal; whether its archives are encrypted is unknown until checked on an installed copy |
| DiRT Rally 2.0 / 1 / DiRT 4, Richard Burns Rally | not yet | their own engines' formats; need installed copies |
| Trackmania | not yet | a map file lists blocks whose models live in the game's packs; needs an installed copy |

None of the "not yet" readers is written blind: a format reader that has not
been checked against the game's own files produces plausible rubbish as
readily as a course, so each waits for a copy to check it on. The gameplay
route below works for all of them meanwhile.

### BeamNG.drive

A level's terrain heightfield, its road decals (drawn on the terrain, as most
of BeamNG's roads are), circuits modelled as meshes, and every placed model -
static objects, prefabs, forest items - from the compiled `.cdae` shapes, or
from Collada for a mod that ships no cache. The ground is baked per 512 m tile
the way the game draws it: each terrain layer's base colour map with its detail
and macro overlays, the road decals over it in the game's order, the modelled
road surfaces from above. Positions come out in the frame the telemetry
recorder writes (`(x, z, -y)` of BeamNG's Z-up world), so a recorded lap sits
on the course; the level is found from the lap. Checked on six shipped levels;
how each convention was settled is in `gamefiles/README.md`.

### Assetto Corsa Rally

The stage's own ground - track tiles split into road and terrain, the far
terrain beyond - and the models placed along it, read from the game's IoStore
containers: package by package, with the properties that place a model decoded
by a schema derived from Unreal's engine headers and checked against every
component of every stage. The ground is textured from the stage's baked
virtual texture (13-59 cm a pixel) and its satellite image around it.

The telemetry's frame is not documented, so it is not assumed: of the eight
ways Unreal's horizontal axes can map onto the lap's, the one that puts the lap
on the stage's road tiles is used, and the export says which. Built and tested
with laps generated on all nine stages in random frames, each found right; a
lap recorded in the game is still to come.

### Assetto Corsa EVO

All of the game's content is one package of about 69 GB. Parts of it are
plain - texture blocks, readable as they are - but its structure and the
padding at its end are XOR-ed with a repeating eight-byte key: the end of the
file, where padding would be zeros, repeats one eight-byte pattern. Reading it
would mean recovering that key, which this project does not do. The gameplay
route is the way to a course.

### Forza Motorsport

The same engine as Forza Horizon (ForzaTech), the same archive formats
(`.minizip`, `.modelbin`, the texture and material chain in
`gamefiles/`). Data Out positions are the game's world frame, so the
route-to-archive search that `course` does for Horizon applies once the
Motorsport track folders are mapped - and FM2023's Data Out carries the track
id outright. Needs an installed copy.

### Assetto Corsa and Assetto Corsa Competizione

Assetto Corsa's tracks are folders under `content/tracks/<track>/<layout>`:
`.kn5` models (textures, materials and a node tree of meshes, described
publicly), `ai/fast_lane.ai` (the AI racing line as a world-space spline - a
reference line) and `data/surfaces.ini` (which surfaces are
road, kerb or grass). ACC is Unreal 4: the `.pak` reader here lists and
extracts its archives; turning a UE4 static mesh into geometry is the part
still to write, and the Unreal 5 reader built for AC Rally is most of the way
there. Both need an installed copy to build against.

### iRacing

Track models are shipped in a protected format and are not a file-extraction
target. The gameplay route, with poses from iRacing's own live telemetry
(latitude, longitude, altitude, yaw, pitch and roll at 60 Hz).

## The route that works for every game

`gamecapture/` reconstructs a level from gameplay video. Its hard part
is the camera path, and for Forza it is exact because the telemetry gives the
car's pose every frame. Every game above now has the same thing: position and
attitude, live, from a documented interface. Feeding those into
`pose/telemetry_pose.py` in place of Forza's is the one piece that makes the
gameplay route exact for all of them - the rest of that pipeline (depth,
fusion, texturing) does not care which game drew the frames.
