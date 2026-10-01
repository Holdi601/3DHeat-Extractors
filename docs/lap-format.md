# The `heat3d-lap` file

Every telemetry recorder in `gamecapture/` writes one file per lap or rally
stage, in the same shape whatever the game. The course export in `gamefiles/`
reads it back to find the game, the level and the driven line. Anything that
reads JSON can read it.

## The document

One JSON object. Channels are stored as columns: one array per channel, all
the same length, one entry per sample.

```json
{
  "format": "heat3d-lap",
  "version": 1,
  "game": "assetto-corsa-rally",
  "track": "stage name, when the game says",
  "recordedAt": "2026-09-21T12:48:59+02:00",
  "car": {},
  "lap": {"number": 3, "seconds": 52.103, "complete": true, "metres": 1908.9},
  "meta": {"slip": "scaled", "discipline": "rally", "surface": "loose"},
  "units": {"speed": "m/s", "tireTemp": "degC"},
  "channels": {
    "time": [0.0, 0.0167],
    "distance": [0.0, 0.41],
    "x": [], "y": [], "z": [],
    "speed": [],
    "tireTemp.FL": [], "tireTemp.FR": [], "tireTemp.RL": [], "tireTemp.RR": []
  }
}
```

| Field | Meaning |
|---|---|
| `format`, `version` | `heat3d-lap`, `1` |
| `game` | which game the lap is from; see the list below |
| `track` | the track, stage or map name, when the game sends one |
| `layout` | Forza only: which Data Out packet layout was decoded |
| `car` | whatever the game says about the car; free-form |
| `lap` | the lap's number, its time as the game counted it (or the samples' span), whether it was finished, its length |
| `meta` | how to read the numbers; see below |
| `units` | the unit of each channel present, for a reader that does not know the table below |
| `channels` | the samples |

`game` is one of `forza-horizon`, `forza-motorsport`, `forza` (packet layout
not told apart), `assetto-corsa`, `assetto-corsa-competizione`,
`assetto-corsa-evo`, `assetto-corsa-rally`, `ea-sports-wrc`, `dirt-rally-2`,
`dirt-rally`, `dirt-4`, `richard-burns-rally`, `beamng`, `trackmania`.

`meta` says what the numbers mean where games differ:

- `slip`: what the slip channels hold, where the game sends any - `peak`
  (normalised, so past 1 the tyre lets go: Forza), `scaled` (a physical slip
  divided by a typical tarmac peak: the Assetto Corsa games), `physical` (a
  fraction and radians), `index` (a 0..1 sliding measure: Trackmania);
- `discipline`: `circuit`, `rally` or `arcade`;
- `surface`: `tarmac`, `loose`, `snow`, `ice` or `mixed`, when known (the rally
  recorders take it from `--surface`).

## Channels

A channel a game does not send is absent, never filled with zeros. Per-wheel
channels carry the wheel after a dot: `.FL`, `.FR`, `.RL`, `.RR`.

Positions are the game's world, in metres, with `y` up. A game whose world is
Z-up is turned to Y-up by its recorder; nothing else is changed, so a
left-handed world (Forza's) stays left-handed.

| Channel | Unit | Meaning |
|---|---|---|
| `time` | s | since the lap's first sample |
| `distance` | m | along the driven path since the first sample |
| `x`, `y`, `z` | m | position; `y` is height |
| `speed` | m/s | |
| `throttle`, `brake`, `clutch`, `handbrake` | 0..1 | pedals and lever |
| `steer` | -1..1 | positive right |
| `gear` | | as the game counts it |
| `latAcc`, `longAcc`, `vertAcc` | m/s² | car frame |
| `yaw`, `pitch`, `roll` | rad | heading and attitude |
| `yawRate` | rad/s | |
| `latVel`, `longVel`, `vertVel` | m/s | velocity in the car's frame: right, forwards, up |
| `bodySlip` | rad | drift angle, where the game sends it |
| `loose` | 0/1 | on loose ground (Trackmania) |
| `airborne` | 0/1 | no wheel on the ground |
| `rpm` | rpm | |
| `power`, `torque` | W, N m | |
| `boost`, `fuel` | | as the game sends them |
| `waterTemp`, `oilTemp` | °C | |
| `absActive`, `tcActive`, `drs`, `pitLimiter` | 0/1 | aids acting, systems on |
| `absLevel`, `tcLevel` | | aid settings |
| `brakeBias` | 0..1 | front share |
| `tireTemp`, `tireTempInner`, `tireTempMiddle`, `tireTempOuter` | °C | per wheel |
| `tirePressure` | psi | per wheel |
| `tireWear` | 0..1 | per wheel |
| `slipRatio`, `slipAngle`, `combinedSlip` | see `meta.slip` | per wheel |
| `slipIndex` | 0..1 | per wheel sliding measure (Trackmania) |
| `camber` | rad | per wheel |
| `brakeTemp` | °C | per wheel |
| `padLife`, `discLife` | | per wheel |
| `suspensionM` | m | per wheel travel |
| `suspension` | 0..1 | per wheel travel, of the range this lap used |
| `wheelLoad` | N | per wheel |
| `wheelSpeed` | rad/s | per wheel rotation |
| `wheelSurfaceSpeed` | m/s | per wheel contact patch speed |
| `groundContact` | 0/1 | per wheel |
| `rumble`, `surfaceRumble` | | per wheel: on a rumble strip; the surface's rumble |
| `puddle` | | per wheel standing water depth |

A recorder may add channels of its own under other names; a reader keeps what
it does not know.

None of the rally games sends tyre slip. A reader can work it out from each
wheel's contact patch or surface speed against the car's speed,
`(wheel - car) / car`, or from its rotation and a rolling radius measured on the
lap itself where only rotation is sent. The drift angle comes from the car's own
velocity (`latVel`, `longVel`) where it is sent, or from the world velocity
projected onto the car's axes.

## The CSV form

The Trackmania plugin writes `heat3d-lap-csv` instead: a first line of `# `
and a JSON object (`format`, `version`, `game`, `track`, `complete`, `seconds`,
`meta`), a line of channel names, then one line per sample. Channel names are
the same as above.
