# 3DHeat Extractors

Free and open-source tools that get a game's courses, levels and lap telemetry
out into plain files: a level or course as a `.glb` (glTF), a lap as a JSON
document. They were written for the 3DHeat telemetry viewer, and anything that
reads glTF and JSON can use what they write.

MIT licensed - see [`LICENSE`](LICENSE), and [`LICENSES.md`](LICENSES.md) for
the third-party packages and models they use.

## What is here

| Folder | What it does | For |
|---|---|---|
| [`gamefiles/`](gamefiles/README.md) | Reads courses and levels out of the archives a game ships. `course <lap>` exports the track around a recorded lap - ground, road, kerbs, barriers, signs - as one textured `.glb` | Forza Horizon, BeamNG.drive, Assetto Corsa Rally; readers for Unreal `.pak` and IoStore, Unity, Frostbite and Source 2 archives |
| [`gamecapture/`](gamecapture/README.md) | Records lap telemetry from the interfaces games publish for it, and reconstructs a level from gameplay video | Forza Horizon and Motorsport, Assetto Corsa / ACC / EVO / Rally, EA SPORTS WRC, DiRT Rally 2.0 / 1 / DiRT 4, Richard Burns Rally, BeamNG.drive (protocol mod), Trackmania (Openplanet plugin) |
| [`unreal/`](unreal/README.md) | An Unreal Editor plugin that exports a level as a `.glb` | Developers with the project open |

Which game is recorded how, and which courses can be exported and why not:
[`docs/games.md`](docs/games.md). The lap file every recorder writes:
[`docs/lap-format.md`](docs/lap-format.md).

## Quick start

Python 3.13 is what these are tested on.

Record laps (here Assetto Corsa Competizione; `--game` takes `forza`, `ac`,
`acc`, `acevo`, `acrally`, `wrc`, `dirtrally2`, `dirtrally`, `dirt4`, `rbr`,
`beamng`, `trackmania`):

```bash
cd gamecapture
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m heat3d_capture telemetry --game acc -o laps
```

Export the course a lap was driven on, from the installed game:

```bash
cd gamefiles
pip install -r requirements.txt
python -m heat3d_gamefiles course ../gamecapture/laps/<lap>.json -o course.glb
```

`course` finds the game from the lap file, the installed game in any Steam
library, and the level or stage from where the lap was driven.

## The rules these are built under

- **Nothing injects into a game's process.** Telemetry comes only from what the
  games publish for it: UDP outputs, shared memory, their own mod and plugin
  interfaces. Video comes from Windows Graphics Capture, the API screen
  recorders use.
- **No keys are shipped and none are recovered.** An encrypted archive opens
  only with a key you already hold. Formats that are obfuscated with a key are
  declined, not broken.
- **Output is for your own analysis.** A course exported from a game is that
  game's content; it is not something to redistribute.
- **What you point the tools at is your responsibility.** Games carry their own
  licence terms, and nothing here discharges them.

## The `.glb` the exporters write

One file in the game's world coordinates, Y up, in the unit its telemetry uses -
metres for the racing games, Unreal's centimetres from the Unreal exporter
(`--scale 0.01` turns them into metres). Every part is a node whose name
says what it is: `ground:<kind>` (road, kerbs, verge, terrain, gravel and dirt,
markings ...), `structure:<kind>` (barriers, buildings, trees, props, signs ...)
or `water:<kind>`. A reader classifies by the node name, `^(ground|structure|water)\s*[:|]`;
everything else about the file is standard glTF 2.0, readable by Blender and
any other glTF tool. Textures, where an export has them, are embedded.

## Tests

Each folder has its own suite:

```bash
cd gamefiles && python -m pytest
cd gamecapture && .venv/Scripts/python -m pytest
```

Tests that read installed games find them in any Steam library on the machine
and skip when they are not there.
