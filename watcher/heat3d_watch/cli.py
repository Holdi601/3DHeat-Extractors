"""
`python -m heat3d_watch <command>`

    run        watch in the background: record every game that is running
    games      what each game gives and from where (--markdown for the docs)
    setup      switch a game's own output on (--apply to change files)
    status     which games are running now and what would be recorded
    sessions   the recordings so far
    stats      totals per game; --game for one, --csv DIR to write the tables out
    table      rebuild a session's heat.parquet from its events.jsonl
    update     update the tools to the newest release (--check to only ask)
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from . import __version__, db
from .adapters import Context, broken, load
from .config import Settings, load as load_settings, save as save_settings
from .session import Axes, write_table


def _logging(settings: Settings, verbose: bool) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.FileHandler(settings.data_dir / "watcher.log", encoding="utf-8")]
    handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def cmd_run(args: argparse.Namespace, s: Settings) -> int:
    import os

    import psutil

    from .watcher import Watcher

    _logging(s, args.verbose)
    # One watcher per data folder: two would record every game twice, and the
    # second would take the first's open sessions for ones left by a crash.
    lock = s.data_dir / "watcher.lock"
    try:
        pid = int(lock.read_text(encoding="utf-8").strip())
        if pid != os.getpid() and psutil.pid_exists(pid) and "python" in psutil.Process(pid).name().lower():
            print(f"A watcher is already running (process {pid}) for {s.data_dir}.", file=sys.stderr)
            return 1
    except (OSError, ValueError, psutil.Error):
        pass
    lock.write_text(str(os.getpid()), encoding="utf-8")
    try:
        Watcher(s).run(once=args.once)
    finally:
        try:
            if lock.read_text(encoding="utf-8").strip() == str(os.getpid()):
                lock.unlink()
        except OSError:
            pass
    return 0


def cmd_games(args: argparse.Namespace, s: Settings) -> int:
    adapters = sorted(load().values(), key=lambda a: (a.game.lower(), a.id))
    if args.markdown:
        print("| Game | What is recorded | From where | To switch it on | Online play | Checked |")
        print("|---|---|---|---|---|---|")
        for a in adapters:
            records = "<br>".join(a.records)
            online = "allowed (reads only what the game publishes)" if a.multiplayer and a.kind != "memory" else (
                "never read while online" if a.kind == "memory" else "single-player"
            )
            print(f"| {a.game} | {records} | {a.source} | {a.enable or '-'} | {online} | {a.verified or '-'} |")
        return 0
    for a in adapters:
        missing = a.missing()
        state = f"  [unavailable: {missing}]" if missing else ""
        print(f"{a.game}  ({a.id}, {a.kind}){state}")
        print(f"    from: {a.source}")
        for r in a.records:
            print(f"    - {r}")
        if a.enable:
            print(f"    switch on: {a.enable}")
    for module, why in broken.items():
        print(f"(adapter module {module} not loaded: {why})")
    return 0


def cmd_setup(args: argparse.Namespace, s: Settings) -> int:
    adapters = load()
    targets = list(adapters.values()) if args.adapter == "all" else [adapters[args.adapter]] if args.adapter in adapters else []
    if not targets:
        print(f"no adapter {args.adapter!r}; `games` lists them", file=sys.stderr)
        return 2
    ctx = Context(s, s.data_dir)
    for a in targets:
        changes = a.setup(ctx, apply=args.apply)
        for c in changes:
            print(f"{a.id}: {'changed' if args.apply else 'would change'}: {c}")
    if not args.apply:
        print("Nothing was changed. Run again with --apply to make these changes.")
    return 0


def cmd_status(args: argparse.Namespace, s: Settings) -> int:
    from .watcher import Watcher, running_processes

    w = Watcher(s)
    want = w.wanted(running_processes())
    print(f"3DHeat Watcher {__version__}; data in {s.data_dir}")
    games = [w.adapters[a] for a in want if not w.adapters[a].always]
    print("Games running: " + (", ".join(f"{a.game} ({a.id})" for a in games) if games else "none"))
    print("Always on: " + ", ".join(a for a in want if w.adapters[a].always))
    return 0


def cmd_sessions(args: argparse.Namespace, s: Settings) -> int:
    con = db.connect(s.data_dir)
    q = "SELECT game, adapter, datetime(started,'unixepoch','localtime'), ROUND(seconds/60.0,1), events, positions, folder FROM sessions"
    params: tuple = ()
    if args.game:
        q += " WHERE game = ?"
        params = (args.game,)
    q += " ORDER BY started DESC LIMIT ?"
    for row in con.execute(q, (*params, args.limit)):
        game, adapter, start, minutes, events, positions, folder = row
        print(f"{start}  {game:28.28}  {minutes:7} min  {events:7} events  {positions:7} positions  {folder}")
    return 0


def cmd_stats(args: argparse.Namespace, s: Settings) -> int:
    con = db.connect(s.data_dir)
    if args.csv:
        out = Path(args.csv)
        out.mkdir(parents=True, exist_ok=True)
        for table in ("sessions", "events", "stats"):
            cols, rows = db.rows(con, table)
            with (out / f"{table}.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(cols)
                w.writerows(rows)
        print(f"wrote sessions.csv, events.csv and stats.csv to {out}")
        return 0
    games = [args.game] if args.game else [g[0] for g in db.per_game(con)]
    if not args.game:
        print(f"{'game':30} {'sessions':>8} {'hours':>8} {'events':>10} {'positions':>10}  last played")
        for g, n, hours, events, positions, last in db.per_game(con):
            print(f"{g:30.30} {n:8} {hours:8} {events or 0:10} {positions or 0:10}  {last}")
        print()
    for g in games:
        print(f"== {g}")
        for event, count in db.event_totals(con, g)[:25]:
            print(f"   {event:32} {count:10}")
        stats = db.stat_totals(con, g)
        if stats:
            print(f"   {'stat':32} {'sessions':>8} {'total':>12} {'best':>12} {'average':>12}")
            for key, n, total, best, avg in stats:
                print(f"   {key:32} {n:8} {total:12.6g} {best:12.6g} {avg:12.6g}")
    return 0


def cmd_table(args: argparse.Namespace, s: Settings) -> int:
    import json

    folder = Path(args.folder)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    a = meta.get("axes", {})
    axes = Axes(**{k: v for k, v in a.items() if k in Axes.__dataclass_fields__})
    out = write_table(folder, axes, meta.get("game", ""))
    print(out or "no positions in this session")
    return 0


def cmd_update(args: argparse.Namespace, s: Settings) -> int:
    import heat3d_update.__main__ as up

    return up.main(["--check"] if args.check else [])


def cmd_config(args: argparse.Namespace, s: Settings) -> int:
    if args.set:
        for item in args.set:
            key, _, value = item.partition("=")
            if key not in Settings.__dataclass_fields__:
                print(f"no setting {key!r}", file=sys.stderr)
                return 2
            current = getattr(s, key)
            if isinstance(current, bool):
                setattr(s, key, value.lower() in ("1", "true", "yes", "on", "ja"))
            elif isinstance(current, float):
                setattr(s, key, float(value))
            elif isinstance(current, list):
                setattr(s, key, [v for v in value.split(",") if v])
            else:
                setattr(s, key, value)
        print(f"saved {save_settings(s)}")
    for k in Settings.__dataclass_fields__:
        print(f"{k} = {getattr(s, k)!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="heat3d_watch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="watch in the background")
    r.add_argument("--once", action="store_true", help="one look at the running games, then stop")
    r.add_argument("-v", "--verbose", action="store_true")
    g = sub.add_parser("games", help="what each game gives and from where")
    g.add_argument("--markdown", action="store_true")
    st = sub.add_parser("setup", help="switch a game's own output on")
    st.add_argument("adapter", help="an adapter id from `games`, or all")
    st.add_argument("--apply", action="store_true", help="make the changes (backups are kept)")
    sub.add_parser("status", help="which games are running now")
    se = sub.add_parser("sessions", help="the recordings so far")
    se.add_argument("--game")
    se.add_argument("--limit", type=int, default=40)
    sa = sub.add_parser("stats", help="totals per game")
    sa.add_argument("--game")
    sa.add_argument("--csv", metavar="DIR", help="write sessions, events and stats as CSV files")
    t = sub.add_parser("table", help="rebuild a session's heat.parquet")
    t.add_argument("folder")
    u = sub.add_parser("update", help="update to the newest release")
    u.add_argument("--check", action="store_true")
    c = sub.add_parser("config", help="show or change the settings")
    c.add_argument("--set", nargs="*", metavar="KEY=VALUE")
    args = p.parse_args(argv)
    # Game names come in every script; a console or pipe in the Windows code page
    # would stop at the first one it cannot spell.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    settings = load_settings()
    if args.command not in ("update",):
        try:
            import heat3d_update

            heat3d_update.startup("heat3d_watch", ask=None if args.command != "run" else False)
        except ImportError:
            pass
    handler = {
        "run": cmd_run,
        "games": cmd_games,
        "setup": cmd_setup,
        "status": cmd_status,
        "sessions": cmd_sessions,
        "stats": cmd_stats,
        "table": cmd_table,
        "update": cmd_update,
        "config": cmd_config,
    }[args.command]
    return handler(args, settings)
