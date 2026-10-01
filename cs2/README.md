# CS2 demos and maps

Counter-Strike 2 demos into heatmap tables and match summaries, and CS2 maps
into level geometry, for the 3DHeat viewer - or anything that reads Parquet,
JSON and glTF.

```bash
pip install -r requirements.txt
python -m heat3d_cs2 demos <demo files or folders> -o out          # every match in them
python -m heat3d_cs2 map de_mirage -o out/de_mirage/de_mirage.glb  # the map from the installed game
python -m heat3d_cs2 scoreboard out/de_mirage/<match>.match.json   # a match's scoreboard as text
```

Per match, `demos` writes into a folder per map (`out/de_mirage/...`), so a
map's matches load together:

- `<match>.parquet` - every player eight times a second (`--every 8` ticks of
  64) and every event where it happened, one row each: the heatmap's data.
- `<match>.match.json` - teams, score, every round (winner, how, economy, the
  plant, every kill in order, clutches, survivors) and every player's
  scoreboard line: the match analysis' data.

A folder of a few hundred demos takes minutes (`--jobs`, half the cores by
default). Matches already written are skipped unless `--force`.

## The table

| Column | What |
|---|---|
| `match`, `map`, `round`, `roundId` | which match and round; `roundId` is `<match> #NN` |
| `phase` | `live` until the round is decided, `after` while the winners hunt and the losers save |
| `time`, `roundTime` | seconds since the match started; since this round's freeze time ended |
| `event` | `position`, `kill` (where the killer stood), `death` (where the victim fell), `hurt` (the victim), `shot`, `throw`, `smoke` / `flash` / `he` / `molotov` / `decoy` (where it went off), `blind` (the player flashed), `plant`, `defuse`, `explode` |
| `player`, `steamid`, `side`, `team` | who: `side` is `T` or `CT` this round, `team` the team's name |
| `x`, `y`, `z` | the game's own coordinates (inches, Z up) |
| `dirX`, `dirY`, `dirZ` | where a player looks, a unit vector in the same frame |
| `health`, `armor`, `money`, `equipment` | for positions |
| `weapon`, `place` | the weapon as the game names it (every knife skin as `Knife`); the map's own place name |
| `other`, `damage`, `duration` | the other player of a kill, hit or flash; health lost; seconds blinded |
| `headshot`, `opening`, `traded` | 0/1: a headshot; the round's first kill; for a kill, a trade - for a death, avenged within 5 s |
| `won`, `buy` | 0/1 whether this row's side won the round; its side's buy that round |

The file's Parquet footer carries reading instructions under the key
`heat3d`: which column is the position, the time, the player, the side, the
session and the look direction, and that the world is drawn with the
vertical as Y and the game's Y negated. CS2's world is right-handed with Z
up; drawn the way a left-handed Z-up world is, it comes out mirrored, A site
on the wrong side. The viewer reads the instructions and opens the file with
nothing to set.

## The conventions

Where the game does not define them:

- A death is **traded** when its killer is killed by one of the victim's
  teammates within 5 s; that kill is a **trade kill**.
- **KAST**: the share of rounds in which a player killed, assisted (a flash
  assist counts), survived, or was traded.
- **Damage** is health actually lost: a 137 damage headshot on a player with
  40 health is 40, as the scoreboard counts it. Enemies only.
- A **clutch** is being the last player alive on a side with enemies still
  standing; it is won when the side wins the round.
- **Buy**, per side, from the average equipment value when freeze time ends:
  the first round of each half is a pistol round, then under $1,500 an eco,
  under $4,000 a force buy, anything more a full buy.

## Demos as they come

Demos from the wild are not all one clean file per match, and each of these
was met in a collection of 177 tournament demos:

- **A match split across files.** Parts are put in order by the round count,
  which carries on across them, then by the server's clock. A part's
  suffix (`-p1`, `-p2`) says nothing: a `-p2` can be the earlier half.
- **A server restarted mid-match** to restore a round starts its clock
  again; the next part still follows on.
- **Two different matches under one name** - the same teams on the same map
  at two events - overlap in their round counts and are written apart
  (`<match>-b`).
- **A round restored from a backup** inside one file leaves the players'
  old bodies in the demo beside the new ones for some seconds; each player
  keeps the row that leads on into where they go next.
- **A demo that starts late** (round 2) or ends during the last round says
  so (`missing`, `incomplete` in the match summary) rather than guessing.
- **The clock.** Within a file, time is counted in ticks: the server's clock
  stands still through a pause while the ticks run on.
- **Coaches** on a team's side are never alive and are not players.

Checked over all of it: 165 matches, 3,565 rounds; in every match the score
follows the round winners, each side's kills are the other side's deaths,
and every round starts five against five.

## The map

`map` exports the world's collision mesh - what players stand on and hide
behind, without the decoration - from the map's `.vpk` in the installed
game, using Source2Viewer's command line
([ValveResourceFormat](https://github.com/ValveResourceFormat/ValveResourceFormat),
MIT licensed; download the CLI build, then put it on `PATH`, set
`HEAT3D_SOURCE2VIEWER` or pass `--source2viewer`). Clip brushes and the sky
are left out; each surface is split into ground (facing up, walkable) and
structure. It is written in the frame the tables are drawn in.

The check that the frame is right is the players standing on it: over a
match on each of the eight maps of the current pools (Mirage, Nuke, Ancient,
Inferno, Dust2, Train, Anubis, Overpass), 99.9-100% of player samples have
ground beneath them, a median of 0.03 to 2.1 units (under 6 cm) below their
feet. The same maps mirrored leave 36-64% over ground.

## What to point it at

Your own matches' demos, demos you downloaded from a tournament site by hand,
demos from an API you have access to. A demo is a recording of a match whose
content belongs to its publisher and organisers; use what comes out for your
own analysis, and do not redistribute demos or data derived from them -
Valve has had a dataset built from professional demos taken down. Sites that
offer demos often forbid automated downloading in their terms.

## Tests

```bash
python -m pytest
HEAT3D_CS2_DEMO=<a demo> HEAT3D_SOURCE2VIEWER=<cli> python -m pytest   # also the floor check on a real map
```
