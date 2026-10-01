# Game capture → level geometry

> **This is a separate project from the viewer and from the Unreal exporter**,
> with its own obligations. What a reconstruction may be used for depends on the
> depth model chosen: the default is Apache-2.0 and unrestricted, the larger and
> more accurate variants are CC-BY-NC-4.0 and non-commercial. Full detail in
> [`../LICENSES.md`](../LICENSES.md).
>
> The division is not administrative. A developer with the project open should use
> `unreal/` — it reads the real geometry and is exact. This is for
> people who have only what the screen shows.


Reconstructs a level from gameplay, so a map nobody exported can still be loaded
into the viewer as 3D context under the telemetry.

Walk a Battlefield level slowly, or drive a Forza track, and this turns what the
screen showed into a `.glb` the viewer reads with no special handling — the same
file contract `unreal/` writes.

## The rule this is built under

**Nothing is injected into the game and nothing is hooked.** Frames come from
Windows Graphics Capture, the public API screen recorders use, and telemetry
comes from Forza's own documented UDP feed. No part of this attaches to a game
process, reads its memory, or touches its graphics API.

That is a deliberate constraint and it costs real accuracy. The game knows the
depth of every pixel exactly, and a hooked depth buffer would make reconstruction
near-perfect and trivial. It would also mean injecting into processes protected
by anti-cheat, which risks the account of anyone who runs it. So depth is
*estimated* from the picture instead, and the rest of the design is about
recovering as much of that lost accuracy as possible from sources that are free
to use.

## How accuracy is recovered

Monocular depth is correct only up to an unknown scale — a photo of a room and a
photo of a doll's house are the same picture. The network is left to do what it
is good at (relative shape, edges, thin structure) and scale is taken from
somewhere that actually knows it:

| Game | Camera path | Scale |
|---|---|---|
| Forza Horizon / Motorsport | Exact, from the UDP telemetry feed | Exact — the baseline between frames is known in metres |
| Anything else | Estimated by visual odometry | Fixed once per capture from a known reference, not per frame |

This is why Forza reconstructs better than Battlefield, and why the split is
worth the extra code: telemetry turns the hardest half of the problem into a
lookup.

### Two ways to get the camera path

Estimating the path from the pictures alone can be done two ways, and which one
is right depends on what is being scanned.

**Incrementally** (`pose/odometry.py`) — match features between consecutive
frames, solve for the next pose, repeat. Live, cheap, and it works well in a
place with structure to hold on to: a building interior, a street, a map with
cover on it.

**All at once** (`pose/feedforward.py`) — hand a window of frames to a model
that predicts all their cameras together, so every frame constrains every other
one directly rather than through a chain. Slower, offline, and it is what a
racetrack needs.

The distinction is not a preference. A circuit is the worst case for incremental
tracking: the road is a smooth ribbon with little texture, the horizon barely
moves, and once a few poses drift there is nothing later that pulls them back.
The lap comes out as a curve that wanders instead of closing. Solving a window
jointly removes the chain, which is the actual failure.

A three-minute lap does not fit in one pass, so it is cut into overlapping
windows and stitched: each window returns in its own arbitrary frame *and its own
arbitrary scale*, and the frames two windows share determine the similarity
transform between them. The overlap defaults to a third of the window because a
thin overlap leaves the scale poorly determined, and scale error compounding
window over window is exactly what makes a lap spiral.

### What the frames look like, and a correction

Two things about a gameplay capture make a solve harder than it needs to be, and
both looked fixable before the model sees a pixel. One of them was, and one of
them was a mistake worth recording, because the mistake is the kind that is easy
to make again.

**Most of the picture is not the world.** In a cockpit view the dashboard, the
pillars, the wheel and the HUD are bolted to the camera, and they are exactly
what a feature detector likes: high contrast, corner-rich, perfectly still. On
the reference lap, **60–79% of every feature found sat on camera-attached
pixels**, so the median apparent motion between frames a second apart was 0–4 px.

`FlowMask` finds that region by asking which features persist *without
travelling*, and three things make its answer trustworthy:

- a candidate cell needs blocked neighbours, because a lone cell is noise and a
  dark frame produces a great deal of it;
- the region must reach an edge of the screen. Driving forwards, the focus of
  expansion has no flow, so the **road straight ahead** reads as stuck while
  being the geometry most wanted — and what is bolted to the camera always
  touches an edge, while a vanishing point does not;
- the limit on how much may be masked is applied to the region *after* it is
  grown. It was not, which is how a night lap came to have 70% of every frame
  discarded while the interface reported 31%.

**But masking it out makes the feed-forward solve worse.** The feature count
above is a true statement about an ORB detector and says nothing about a trained
network, which was the thing being fed. Measured on what actually matters — how
far the recovered path turns between consecutive steps, where a car is a few
degrees and a random direction is ninety:

| frames given to the solver | mean turn |
|---|---|
| raw | **31.5°** |
| local contrast lifted | 34.6° |
| cockpit masked | 45.0° |
| both | 85.4° |

The preprocessing that looked best by feature statistics was the worst by the
only measure that counts. So the solver gets raw frames. The mask is still
learned and still used by the incremental tracker in `odometry.py`, which does
match features and does benefit — and the interface still reports what it found.

The general lesson is cheap to state and was not cheap to learn: a measurement
justifies a change only if it measures the thing the change is supposed to help.

### Frames are spaced by motion, not by the clock

The remaining failure was subtler. At a fixed 1.2 s interval, median displacement
across one lap ran from **13 px where the car crawled to 149 px where it was
quick** — and the windows that came out wrong were the *fast* ones, where
consecutive frames no longer overlapped enough to be related to each other. A
clock gives the slow sections frames they do not need and starves the fast ones.

So the gap is chosen per step to hold the displacement near 5% of frame width.
On the same footage and the same frame budget that gives a median of 56 px with
an 11–123 px spread, instead of 124 px with a 13–149 px spread.

That is better and it is still not enough, which is the honest state of this
path. Solving 32-frame windows across the reference lap at a fixed 0.4 s gives a
coherent camera path on **7 of 15 sections** and a scribble on the rest — and no
image statistic tried (brightness, detail, feature count, inlier ratio,
displacement) predicts which is which. What distinguishes them is visible in the
footage: the failures are the tight corners, where the view swings hard between
frames.

Those sections are recoverable by sampling them harder. The worst corner goes
from 79° of mean turn to **5.9°** by moving from 0.4 s between frames to 0.1 s.
So the spacing cannot be planned in advance from the pictures; it has to be
chosen by solving, measuring the result, and re-solving denser where the path is
not one a car could have driven. `scan/offline.py` does the planning; the
adaptive retry is the work in progress.

The model is `facebook/VGGT-1B`, which is **CC-BY-NC-4.0 — non-commercial only,
and a reconstruction made with it inherits that.** `describe_licence()` says so,
and `LICENSES.md` has the detail. It needs PyTorch, which this project
deliberately does not pin; install it first, matched to your GPU.

## Runs on anyone's machine

Inference goes through ONNX Runtime, not PyTorch, and picks the best backend
present:

    TensorRT -> CUDA -> DirectML -> CPU
    (NVIDIA)    (NVIDIA) (any DX12)  (anything)

**DirectML is the default**, because it is the one that reaches AMD and Intel
GPUs as well as NVIDIA. NVIDIA is faster with its own providers and is welcome to
them — installing `onnxruntime-gpu` adds the CUDA path, and the app says so when
it notices an NVIDIA card running on DirectML — but NVIDIA is not allowed to be
the only option. CPU is last so a machine with no usable GPU gets a slow scan
rather than no scan.

Measured here, same file, same output, on an RTX 5080:

| Backend | Throughput |
|---|---|
| DirectML | 54 fps |
| CPU | 3.5 fps |

The app's environment is **349 MiB and contains no PyTorch** — against roughly
3.5 GB for torch with CUDA, which would also have been NVIDIA-only. Torch is
needed exactly once, to convert a checkpoint with `tools/export_onnx.py`, and
never at scan time.

## Telling the user what to expect

A scan is a physical act: someone walks a level for several minutes. The
expensive failure is not a crash, it is finishing the walk and discovering the
hardware could only manage a coarse result. So the window measures the actual
throughput of *this* machine on startup and states, before anything can be
started, what the scan will be like — "live scanning at full detail", "walk
slowly and turn gently", or "record first and reconstruct afterwards".

Measured rather than looked up in a table of GPUs. Thermal state, laptop power
profiles and driver versions all move it enough that a table would be wrong on
half the machines it was shown to, and being wrong about this costs a wasted walk.

While scanning, the things that are actionable *while still walking* are the
prominent ones: how close the next frame is to being kept (a walk-speed meter),
how much is being kept per second, and a plain warning when almost nothing is —
which usually means standing still or facing a blank wall.

## Knowing where to look again

The panel this was built for. Reconstruction quality is not uniform, and it fails
*quietly*: a wall you walked past once, at twenty metres, in a single glance comes
out as a surface that looks like geometry rather than as an obvious hole. Finding
those places afterwards in the mesh is hard. Finding them during the scan, while
you are still standing in the level, is easy.

Coverage is scored per half-metre cell on four things, and they are not the same
thing:

| | Why it counts separately |
|---|---|
| How many times seen | Once is a guess — a depth network's error on a single view has nothing to average against |
| **From how many directions** | The one people expect least. Twenty frames walking straight at a wall are twenty views from *one* direction with no parallax between them; two views thirty degrees apart are worth more than fifty from the same spot |
| From how far | Depth error grows roughly with the square of distance |
| How well the camera was placed | Geometry fused with a doubtful pose lands in the wrong place, which is worse than not fusing it |

Multiplied, not averaged. These are conditions that each have to hold: a place
seen two hundred times from one direction at sixty metres is not two-thirds well
observed, it is badly observed, and an average would hide that behind one
excellent term.

The result is a live plan view — green where the geometry can be believed, red
where it cannot, and a distinct dark colour for never seen at all, so an
unexplored corner does not read as a badly scanned one. Weak areas are merged by
connectivity and listed with an instruction: *"Only seen from one angle. Step
sideways and look at it again."* Numbered rings on the map match the list.

Connectivity matters more than it sounds. The first version grouped by a fixed
block grid and reported one neglected side room as eight separate numbered rings
in a row — and nobody reading that can tell eight places from one place counted
eight times.

## What is not the world

Two things in a game frame are not level geometry, and both wreck a
reconstruction if treated as though they were:

- **The HUD.** A crosshair is the most repeatable feature in any frame. ORB loves
  it, it matches perfectly between every pair of frames, and every match it
  contributes is evidence that the camera did not move — which compresses the
  whole reconstruction.
- **Things bolted to the camera.** A weapon, a bonnet, a cockpit frame. Real
  geometry in the wrong coordinate system: it travels with the camera, so fusing
  it smears a gun barrel along the entire path walked.

One signal covers both: they are *static in screen space while the world moves*.
The detector accumulates per-pixel change and compares each pixel against the
frame's own median, so the threshold follows how fast someone is walking instead
of assuming a speed. It learns only from frames where the camera actually moved —
a parked camera makes everything static — and refuses outright when more than
about half the screen reads as static, because that means a menu or a cutscene
rather than a HUD.

The mask is applied at **feature detection**, not only to the fused geometry.
Masking a crosshair out of the output while still letting it vote on the pose
would fix the visible symptom and keep the damaging one.

Measured on a rendered room with a synthetic HUD and weapon: 17% of the frame
correctly ignored, and the recovered walking distance closer to the truth with it
on than off.

## When tracking breaks

A break is reported rather than absorbed — but reporting it is not enough, and
the first version proved it. It simply started again from the next frame as a
fresh origin, which puts everything after the gap in its own coordinate frame,
stacked on the geometry from before it. A ground-truth run made it plain:
tracking was excellent, 35 frames of 36 placed to within a couple of centimetres,
and the path error still peaked at **8.7 metres** — entirely at the one break.

So a break is now followed by **relocalisation** against a database of past
keyframes. Until it succeeds, frames are dropped rather than fused; nothing is
ever placed at a guessed origin.

That database also gives **loop closure**: matching against keyframes from much
earlier detects a revisit and snaps the track back onto where it was, bounding
drift instead of letting it grow for as long as someone keeps walking. It is not
a global pose graph and does not retroactively move geometry already fused — it
stops the divergence compounding, and both recoveries and closures are counted on
screen rather than hidden, so a track that is not a straight line has an
explanation.

## State

Feature complete: capture, HUD rejection, keyframe selection, depth, pose with
relocalisation and loop closure, telemetry-driven poses, fusion, coverage,
colour projection, an in-game overlay, and a `.glb` the viewer loads with no
special handling.

| Piece | What it does | Tests |
|---|---|---|
| `runtime/preflight.py` | First-run dependency check that asks before installing | 27 |
| `runtime/device.py` | Backend ladder, and what to expect from this machine | 23 |
| `capture/sources.py` | Live window/monitor capture and video files | 15 |
| `depth/onnx_estimator.py` | The portable depth path the app uses | 13 |
| `depth/estimator.py` | The PyTorch path, for conversion and scale anchoring | 15 |
| `pose/odometry.py` | Where the camera was, with the scale chained frame to frame | 23 |
| `fusion/volume.py` | Depth maps into one surface, and a mesh out of it | 22 |
| `fusion/coverage.py` | What was seen well, and what to go back to | 31 |
| `scan/session.py` | A scan: settings, keyframe policy, progress, stop | 28 |
| `scan/reconstruct.py` | Keyframes into geometry and a coverage report | 19 |
| `telemetry/forza.py` | Decodes Forza's UDP feed, discovering its own layout | 14 |
| `telemetry/fulllap.py` | Every Forza channel, a lap file at a time, relayed on | 11 |
| `telemetry/kunos.py` | AC, ACC, AC EVO and AC Rally from shared memory | 9 |
| `telemetry/rally.py` | EA SPORTS WRC, DiRT Rally 2.0 / 1 / DiRT 4 and Richard Burns Rally over UDP | 5 |
| `telemetry/beamng.py` | BeamNG.drive through the `heat3d` protocol mod | 4 |
| `telemetry/lapfile.py` | The `heat3d-lap` document every recorder writes | through the three above |
| `geometry/glb.py` | Writes the viewer's GLB contract | 16, plus a round-trip through the viewer's own parser |
| `ui/preview.py` | Depth, coverage map and health, as pictures | 23 |
| `capture/screenmask.py` | Finds the HUD and anything bolted to the camera | 10 |
| `pose/telemetry_pose.py` | Exact poses from Forza, aligned to capture time | 17 |
| `fusion/texture.py` | Colour projected from the frames that saw best | 16 |
| `ui/overlay.py` | Click-through feedback drawn over the game | — |
| `tests/test_accuracy.py` | Ground truth: a known room, a known path | 10 |

### Accuracy, measured

`tests/test_accuracy.py` renders a room of known size from a known camera path
and reconstructs it, so the error is a number rather than an impression. A mesh
from real gameplay can only be eyeballed, and "looks about right" is exactly the
standard that lets a systematic error through.

On a 36-frame walk through a 12 x 8 m room:

| | |
|---|---|
| Frames placed | 36 of 36 |
| Mean path error | under 0.35 m |
| Walked distance | within 10% of truth |
| Surface on a real wall | over 75% within one voxel |

The depth fed to that test is analytic, which is deliberate: it isolates pose,
scale and fusion from the network, so a failure there is unambiguously in the
geometry code. How the network itself does on real gameplay is a separate
question no synthetic scene can answer.

### Known limits

- **Loop closure is not a pose graph.** It stops drift compounding at a revisit;
  it does not redistribute the correction over the frames in between, and
  geometry already fused stays where it was put.
- **Field of view is assumed, not measured.** It is the one number the scanner
  has to be told. Too narrow and the level reconstructs deeper than it is.
- **Depth is estimated, not read.** The ceiling of the whole approach, and the
  price of never touching the game process.
- **No texture atlas.** Colour is per-vertex (`COLOR_0`), which Blender and other
  glTF tools read, and which the viewer shows behind the *Captured colour*
  switch in its Level panel. Per-vertex rather than an atlas because the viewer's
  default look is a grey map under saturated heat — colour is there to find the
  road on a scanned circuit, where geometry alone does not separate it from the
  verge, and is switched off again to read the heat. An atlas would multiply the
  file size to serve a view that is deliberately not the default.
- **Feed-forward solving is not live.** It needs the whole window before it can
  answer, so it runs over a recording rather than while walking. The incremental
  tracker is what gives feedback during a capture.

### Measured here

A real scan of a 120-frame clip on an RTX 5080, through the shipped ONNX path:

| Preset | Views kept | Depth throughput |
|---|---|---|
| Draft | 17 of 120 | 49 /s |
| Balanced | 31 of 120 | 50 /s |
| Detailed | 58 of 120 | 42 /s |

Depth is comfortably faster than capture, so live scanning is not depth-bound —
which is the result that decides whether the feedback is useful while walking.

A full 150-frame run through the whole pipeline placed 144 of 144 keyframes,
fused them, and wrote a `.glb` that the viewer's own parser loads and classifies
from its labels.

## Running it

Double-click **`scan.bat`**.

The first run checks what is missing *before* anything needs it, says what each
piece is for in plain terms and what the download comes to, and asks. Nothing is
installed without a yes. The behaviour this replaces is a traceback:
`ModuleNotFoundError: No module named 'cv2'` is correct and useless — it names
one missing piece out of seven, says nothing about how to get it, and reads as
"this program is broken" rather than "this program is not finished installing".

Three rules it follows:

- **Everything is checked first.** Fixing one missing package only to hit the
  next is the worst version of this.
- **Silence is not consent.** A double-clicked shortcut has no terminal to answer
  at, and that is read as no, not as yes.
- **It installs into this tool's own folder**, never system-wide. The machine
  this was built on already had three conflicting copies of onnxruntime in its
  global Python from unrelated projects; adding a fourth would have been the
  wrong kind of helpful.

The viewer's own `start.bat` does the same for Node.js and its packages.

The depth model has to be converted once before anything can be scanned, and the
window offers to do it rather than naming a command. That matters more than it
sounds: the converter needs PyTorch, which the scanner deliberately does not ship
— a gigabyte, and NVIDIA-shaped, for a step that runs once. An instruction to run
`tools/export_onnx.py` would have failed at the next line with a missing import,
and looked like the user's fault.

By hand:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m heat3d_capture.ui        # offers to build the model
```

Verified here: Python 3.13, onnxruntime-directml 1.24.4, PySide6 6.11,
RTX 5080 — and the same environment with no NVIDIA-specific package in it.

## Forza telemetry

Turn it on in the game: **Settings → HUD and Gameplay → Data Out**, set to `ON`,
IP `127.0.0.1`, port `5300`. Forza Motorsport additionally offers a packet format
— it must be **Dash**, not Sled; Sled carries no position at all.

The decoder works out the packet layout itself rather than assuming it. The
published descriptions of the Horizon variant disagree about the padding between
the two halves of the packet, and the arithmetic does not close, so instead of
picking one the decoder tries the candidates and keeps whichever makes the Sled
block's velocity vector agree with the dash block's scalar speed. Those are the
same quantity stored in different sections, so only a correct layout reconciles
them.

This needs the car to be *moving*: while it is parked every candidate reads zero
and they all agree, which proves nothing. Detection waits rather than locking in
a coin flip, so drive a few metres before expecting output.

### Recording just the route

```bash
.venv/Scripts/python -m heat3d_capture route -o lap     # drive, then Ctrl+C
```

No depth model, no GPU, no capture — it listens on the telemetry port and writes
each position as it arrives, so the file is complete at every moment. That is the
point of it being separate: a scan holds the route in memory and writes it only
if the whole run succeeds, and a lap that is lost because a later stage failed
has to be driven again, with the game running and the right track loaded.

It writes `lap.npz` and `lap.json`, and the JSON is readable without this
project on purpose. The positions are the game's own world coordinates, which
makes them the handle for finding the same ground in the game's files:

```bash
cd ../gamefiles
python -m heat3d_gamefiles course ../gamecapture/lap.json -o map.glb
```

That exports the track the lap ran over straight out of the shipped archive —
exact geometry rather than an estimate of it: the ground with its road, verge
and grass textured, and the kerbs, guardrails, tyre walls and signs placed on
it. It finds the installed game and the right track by itself. See
`gamefiles/`.

### Laps somebody already drove

FH Companion keeps a folder per course under
`%LOCALAPPDATA%/FHCompanion/laps`, a file per lap, sampled about every five
metres. That is better than a fresh drive in the way that matters: the laps are
already repeated, and repetition is what turns a route into analysis.

```bash
.venv/Scripts/python -m heat3d_capture laps                       # what is there
.venv/Scripts/python -m heat3d_capture laps "Lakeside Circuit" -o races/lakeside
```

Two files come out. `lakeside.json` is the same route file as above, so the map
extractor takes it unchanged — or skip it and name the course, which reads the
same library directly:

```bash
cd ../gamefiles
python -m heat3d_gamefiles course "Lakeside Circuit" -o ../gamecapture/races/lakeside.glb
```

`lakeside.csv` is every sample of every lap — position,
speed, throttle, brake, steer, lateral and longitudinal g, gear, standing water,
distance along the lap and time since its start.

The column names are chosen so the viewer maps the whole thing without being
told: `lap` becomes the session, so two laps by one car are never joined into one
track; `car` becomes the thing that moves; `car_class` becomes the group to
colour by; `elapsed` becomes the time axis, which lines every lap up at its own
start rather than scattering them across the fortnight they were driven in.

**The check that matters** is whether the extracted ground is really under the
recorded car. Everything upstream can be wrong and still produce a handsome
mesh — of somewhere else. Across all 39 courses of one recorded library the car
sits a median of 0.28 to 0.47 m above the surface directly beneath it — ride
height — including under and over a city's expressway decks, and there is no
room in it for an error of even a couple of metres. `tests/test_course_ground.py`
pins it for any exported course:

```bash
HEAT3D_COURSE=.out/courses/lakeside_circuit .venv/Scripts/python -m pytest tests/test_course_ground.py
```

It passes on 38 of the 39. A cross-country course fails by design: the car is
up to 26 m in the air on its jumps, over ground that is there.

The course's shape *is* in the terrain — the corridor and its camber are there
in the mesh. Kerbs, guardrails, tyre walls and expressway decks are not: they
are separate models placed on it, which the `course` export adds, and the kerbs
among them sit a median of 3 cm above the terrain. So the height is not a layer
of track missing from the terrain; it is the car's own reference point above its
wheels.

### Every channel, for the Race tab

FH Companion keeps thirteen channels every five metres: enough to see a line
and a speed trace, not enough to say *why* a lap was slow. The game sends a
great deal more sixty times a second - every wheel's slip ratio, slip angle,
temperature, suspension travel and rotation speed, RPM, power, torque, yaw rate.
`telemetry` keeps all of it, a file per lap:

```bash
.venv/Scripts/python -m heat3d_capture telemetry -o laps                             # game -> 5300
.venv/Scripts/python -m heat3d_capture telemetry -o laps --forward 127.0.0.1:5301    # and FH Companion keeps working
```

The game sends Data Out to one address, so with `--forward` this listens on the
game's port and passes every packet on unchanged: set the game to 5300 and FH
Companion to 5301, and both record. A lap closes when the game's lap counter
moves on, with the game's own lap time; a point-to-point race is kept as a run.
Forza Horizon and Forza Motorsport (including the 2023 layout, with tyre wear
and the track id) are told apart by their packet length.

The Assetto Corsa games publish their telemetry as Windows shared memory, which
any app may read - nothing is injected into the game:

```bash
.venv/Scripts/python -m heat3d_capture telemetry --game acc -o laps       # ac, acc, acevo, acrally
```

All four share one physics layout for its first 800 bytes, including each
tyre's contact point in world coordinates, which is where the position comes
from. AC and ACC count laps; AC EVO and AC Rally are split where the car passes
the point recording began, and a stage closes when the car stops at its end.
Kunos's slip ratio and slip angle are physical values, so they are divided by a
typical peak (0.12, and 8 degrees) to read like Forza's "1 is the edge of grip";
the raw values are kept alongside, and the file says which scale it used. Where
grip actually ends, the viewer measures per car from the laps themselves. The
car-frame velocity (for the drift angle) and ABS and traction control acting
(ACC, AC EVO, AC Rally) are recorded too.

The rally games send UDP, each switched on in its own settings:

```bash
.venv/Scripts/python -m heat3d_capture telemetry --game wrc --configure "<Documents>/My Games/WRC/telemetry/config.json" -o laps
.venv/Scripts/python -m heat3d_capture telemetry --game dirtrally2 -o laps   # dirtrally, dirt4: extradata="3" in hardware_settings_config.xml
.venv/Scripts/python -m heat3d_capture telemetry --game rbr -o laps          # NGP's udpTelemetry=1, port 6776
```

`--configure` switches EA SPORTS WRC's default `wrc` packet on (the file is
backed up first). A stage closes when the car has covered the stage's length or
the game flags the finish; a restart abandons the partial run. Stages count as
gravel unless `--surface tarmac` (or `snow`, `ice`, `mixed`) says otherwise. None of them sends tyre slip: the lap file
keeps each wheel's contact patch speed, and the viewer derives slip and the
drift angle from it and the car's axes.

BeamNG.drive has no fixed telemetry packet with wheels in it, so a small
protocol mod sends one:

```bash
.venv/Scripts/python -m heat3d_capture telemetry --game beamng --install     # copies mods/beamng/heat3d into the game's mods
.venv/Scripts/python -m heat3d_capture telemetry --game beamng -o laps
```

then Options > Other > Protocols, "others", port 4460. Trackmania's is an
Openplanet plugin (`mods/trackmania/Heat3dRecorder`, `--game trackmania
--install`), loaded in Openplanet's developer mode; it writes a CSV per run into
its storage folder (`OpenplanetNext/PluginStorage/Heat3dRecorder`), which the
viewer reads as it is.

None of the rally, BeamNG or Trackmania recorders has been run against the live
game yet: they are built and tested on packets made from the documented layouts,
and each checks its packets as they arrive and says so when a layout does not
match. [`../docs/games.md`](../docs/games.md) has the sources and what is not
yet confirmed.

Every lap file is a `heat3d-lap` document: channels as columns under
canonical names, per wheel as `tireTemp.FL` and so on - see
[`../docs/lap-format.md`](../docs/lap-format.md). Drop the files on the
viewer's Race tab. iRacing needs no recorder: its
own `.ibt` files load directly.

## Tests

```bash
.venv/Scripts/python -m pytest tests/ -q     # 562 checks
```

`tests/test_launch.py` runs the real entry point as a subprocess and asserts the
QApplication invariant directly. It exists because the first build crashed on
launch, every time, and every other test passed: they all construct a
`QApplication` themselves and then call `build_window()`, which is convenient and
skips the one sequence a user performs. A component test that builds its own
environment cannot see a fault in how the environment is built.

The tests that need a converted model skip themselves until `tools/export_onnx.py`
has been run; the torch-based tests skip without CUDA. Everything else — the GLB
contract, the telemetry layout search, the keyframe policy, the backend ladder —
runs anywhere. The GLB contract is additionally
tested from the other side, in the viewer's own tests, which generate
a reference level through this exporter's own writer and parse it with the
viewer's real parser — so the file format stays under test on both sides of the
language boundary, with no game and no capture hardware.
