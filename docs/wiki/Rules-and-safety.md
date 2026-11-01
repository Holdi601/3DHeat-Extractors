# Rules and safety

## The sources, and why only these

| Kind | What it is | Multiplayer games |
|---|---|---|
| telemetry | a UDP port or shared memory the game publishes for dashboards, motion rigs and wheel bases | yes - it is the game's own output |
| gsi | a game-state integration or local web API the game serves to the player's own machine | yes, within what the API shows the player |
| log | files the game writes while it runs: journals, logs, statistics, saves | yes |
| replay | demos and replays the game saves for its players, converted after the match | yes, unless encrypted |
| platform | what Steam and the graphics driver keep: statistics, achievements, timelines, highlights | yes |
| memory | the game's memory, read only - single-player games only, behind every check below | **never** |

Nothing is injected, hooked or changed in any game. A source a game's terms or
anti-cheat rules out is not implemented: the League of Legends and Valorant
client lockfile endpoints, encrypted replays (Fortnite, League of Legends,
Valorant, World of Tanks), Gran Turismo 7's encrypted telemetry (it would need a
key recovered from the game).

## What a recording is

A session folder per sitting (`%LOCALAPPDATA%\3DHeat\Watcher\sessions\<game>\...`):

- `events.jsonl`: one line per event, written as it happens, so nothing said is
  lost to a crash. Every line has `time` (seconds since the session began),
  `ms` (wall clock), `event`, and where known `x`, `y`, `z`, `player`, `team`,
  `session` (match, round, stage or flight), plus whatever else the game said.
- `heat.parquet`: the rows with a position, with reading instructions in the
  footer (key `heat3d`) so the 3DHeat viewer maps the columns and axes by itself.
  Events that bound a route - `spawn`, `death`, `respawn`, `teleport`,
  `session_start`, `finish` ... - are rows too, so the viewer's line heat draws
  each player's way from one to the next.
- `meta.json`: game, source, axes and units, event counts, statistics.

And `watcher.sqlite` (`sessions`, `events`, `stats`) with every session's numbers.

## NVIDIA highlights

Games that support NVIDIA Highlights tell the NVIDIA App only *that* a moment
happened - one of a fixed list of highlight types the game registered ("Kill",
"Headshot", "Victory" ...) - and when. There is no position, no player, no match
in it, and no way to get any from the driver. The watcher records each new
highlight or clip with its type, game, time and file, and where another source
recorded the player's position at that moment (a telemetry or game-state source
of the same game), takes the position from there, so a highlight lands on the
map. For games with no positional source, highlights stay a timeline of moments.

## Reading memory

For single-player games the watcher can read positions and events from the
game's memory - off by default, and fenced in:

1. Only games with a **profile** (`<data>\profiles\<game>.json`) for the exact
   build: the executable's size or SHA-256 must match a version the profile
   lists. A profile says whether the game can be played online at all.
2. **No anti-cheat:** none of the known anti-cheat files beside the game
   (EasyAntiCheat, BattlEye, Vanguard, EA Javelin, ACE, nProtect, XIGNCODE ...)
   and no anti-cheat driver or service running on the machine. Checked before
   the game is opened, because opening a protected game is itself what an
   anti-cheat notices.
3. **Not online:** a game that can be played online is read only while it has
   no connection to anything but this machine.
4. Then the game is opened for reading only (`PROCESS_VM_READ |
   PROCESS_QUERY_LIMITED_INFORMATION`), its modules are checked once more for an
   anti-cheat client, and values are read by byte pattern and pointer chain.
   There is no write, no allocation, no thread, no debug privilege, no
   elevation; a test fails the build if any such call appears in the code.

No profiles ship: offsets have to be verified on the build a player has, and a
profile that was right last patch reads garbage the next. The format is
described in `watcher/heat3d_watch/memory.py`. Games played online - and games
whose single-player mode runs under an anti-cheat, such as GTA V or Elden Ring
through its launcher - are never read.
