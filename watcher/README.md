# 3DHeat Watcher

A background tool that records what games publish about your play - where you
went, what happened where, and the numbers behind it - into files on your own
machine, for your own statistics and for the 3DHeat viewer's heatmaps.

It runs quietly, notices which game you started, and reads that game's own
outputs: a telemetry port, a game-state integration, the journal or log the
game writes, the replay it saves after a match, the statistics Steam keeps, the
highlights your graphics driver records. Nothing is sent anywhere.

```bash
cd watcher
pip install -r requirements.txt
python -m heat3d_watch games          # what each game gives, and from where
python -m heat3d_watch setup all      # what would have to be switched on (add --apply to do it)
python -m heat3d_watch run            # watch; leave it running
python -m heat3d_watch stats          # totals per game
python -m heat3d_watch stats --csv out   # sessions, events and stats as CSV files
```

Everything goes to `%LOCALAPPDATA%\3DHeat\Watcher` (set `HEAT3D_WATCH_HOME`, or
`python -m heat3d_watch config --set data=D:\somewhere`):

- `sessions/<game>/<date>_<source>/` - one folder per sitting:
  - `events.jsonl` - every event the moment it happens: time, event, position,
    player, team, match, and whatever else the game said
  - `heat.parquet` - the rows with a position, as a table the 3DHeat viewer
    opens directly (heatmaps, and the line mode that draws each player's routes
    between spawn and death)
  - `meta.json` - the game, the source, the axes and units, the counts and the
    statistics of the sitting
- `watcher.sqlite` - the statistics of every session, in three plain tables
  (`sessions`, `events`, `stats`) any SQL tool or spreadsheet reads
- `watcher.log` - what the watcher did

## What it may and may not do

- **Only what games publish.** Telemetry ports and shared memory the game
  creates for dashboards and motion rigs, game-state integrations and local APIs
  the game serves, files the game writes, replays the game saves, what Steam and
  the graphics driver keep. Nothing is injected into a game, nothing is hooked,
  nothing in a game is changed.
- **Multiplayer games: never their memory.** Reading a game's memory is allowed
  only for single-player games, only for builds a profile was checked on, only
  with no anti-cheat installed with the game or running on the machine, and only
  while the game has no connection out. It is off by default
  (`config --set memory=true`), it can only read, and no profiles ship with the
  tools. See [`docs/watcher.md`](../docs/watcher.md#reading-memory).
- **No encrypted formats.** Replays that need a key to read (Fortnite, League of
  Legends, Valorant, World of Tanks) are declined, as everywhere in these tools.
- **Switching a source on is your decision.** `setup` says which file it would
  change and how; only `setup --apply` changes it, and the original is kept as
  `<file>.heat3d-backup`.

## Updates

The watcher looks for a newer release of the tools on GitHub at most once an
hour while no game is running, installs it, and starts again - never in the
middle of a game. `config --set auto_update=false` turns that off;
`HEAT3D_NO_UPDATE=1` turns off the check as well. `python -m heat3d_watch update`
updates on request.

Which game is recorded how, and from where: [`docs/watcher.md`](../docs/watcher.md)
and the wiki.
