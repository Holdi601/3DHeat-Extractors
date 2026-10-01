"""
Command line.

    python -m heat3d_cs2 demos <demo files or folders> -o <folder> [--every 8] [--jobs N]
    python -m heat3d_cs2 map <de_mirage | path to its .vpk> -o mirage.glb
    python -m heat3d_cs2 scoreboard <match.json>

`demos` groups a match's parts by name (`...-p1.dem`, `...-p2.dem` are one
match) and writes `<map>/<match>.parquet` and `<map>/<match>.match.json` for
each, skipping what is already written unless `--force`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PART = re.compile(r"[-_]p(\d+)$", re.IGNORECASE)


def matches_in(paths: list[str]) -> dict[str, list[Path]]:
    """Match name -> its demo files, from files and folders (recursively)."""
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(p.rglob("*.dem"))
        elif p.suffix.lower() == ".dem":
            files.append(p)
    out: dict[str, list[Path]] = {}
    for f in files:
        out.setdefault(PART.sub("", f.stem), []).append(f)
    return out


def convert(name: str, files: list[Path], folder: Path, every: int) -> str:
    from .demo import read_part
    from .match import build_match, chains
    from .output import write

    lines = []
    # Files named as one match can be two: each chain of parts is its own.
    groups = chains([read_part(f, every) for f in files])
    if not groups:
        return f"{name}: no round in {len(files)} file(s) - nothing written"
    for k, raws in enumerate(groups):
        label = name if k == 0 else f"{name}-{chr(ord('a') + k)}"
        match = build_match(raws, label)
        if not match.rounds:
            lines.append(f"{label}: no complete round in {len(raws)} file(s) - nothing written")
            continue
        # A folder per map: a heatmap is of one map, so loading one folder is loading one map.
        write(match, folder / (match.map or "unknown"))
        score = " ".join(f"{t} {s}" for t, s in match.rounds[-1].score.items()) if match.rounds else "no rounds"
        cut = f", {len(match.incomplete)} incomplete round(s) left out" if match.incomplete else ""
        late = f", recorded from round {match.rounds[0].number}" if match.rounds and match.rounds[0].number > 1 else ""
        lines.append(f"{label}: {match.map}, {len(match.rounds)} rounds ({score}), {len(match.rows):,} rows{cut}{late}")
    return "\n".join(lines)


def _demos(args) -> int:
    folder = Path(args.out)
    written = {p.stem for p in folder.rglob("*.parquet")} if folder.exists() else set()
    todo = {n: f for n, f in matches_in(args.paths).items() if args.force or n not in written}
    if args.map:
        todo = {n: f for n, f in todo.items() if args.map.lower() in n.lower()}
    if not todo:
        print("nothing to do: no demos found, or every match already written (--force redoes them)")
        return 0
    jobs = max(1, min(args.jobs or max(1, (os.cpu_count() or 2) // 2), len(todo)))
    print(f"{len(todo)} match(es), {jobs} at a time")
    failed = 0
    with ProcessPoolExecutor(jobs) as pool:
        futures = {pool.submit(convert, n, f, folder, args.every): n for n, f in todo.items()}
        for done in as_completed(futures):
            try:
                print(done.result(), flush=True)
            except Exception as e:  # one bad demo does not stop the rest
                failed += 1
                print(f"{futures[done]}: failed - {e}", flush=True)
    return 1 if failed else 0


def _map(args) -> int:
    from .mapexport import export_map

    out = export_map(args.name, args.out, game=args.game, source2viewer=args.source2viewer)
    print(out)
    return 0


def _scoreboard(args) -> int:
    from .stats import scoreboard_text

    print(scoreboard_text(json.loads(Path(args.match).read_text(encoding="utf-8"))))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="heat3d_cs2", description="CS2 demos and maps for the 3DHeat viewer")
    sub = parser.add_subparsers(dest="verb", required=True)
    d = sub.add_parser("demos", help="demos -> one heatmap table and one match summary per match")
    d.add_argument("paths", nargs="+", help="demo files or folders")
    d.add_argument("-o", "--out", required=True, help="folder to write into")
    d.add_argument("--every", type=int, default=8, help="sample players every N ticks (64 a second; default 8: 8 a second)")
    d.add_argument("--jobs", type=int, default=0, help="matches converted at once (default: half the cores)")
    d.add_argument("--map", help="only matches whose name contains this, e.g. mirage")
    d.add_argument("--force", action="store_true", help="rewrite matches already written")
    d.set_defaults(run=_demos)
    m = sub.add_parser("map", help="a map's collision geometry -> .glb in the demos' frame")
    m.add_argument("name", help="map name (de_mirage) or the path to its .vpk")
    m.add_argument("-o", "--out", required=True, help=".glb to write")
    m.add_argument("--game", help="the CS2 install folder, when Steam does not know it")
    m.add_argument("--source2viewer", help="Source2Viewer-CLI executable, when it is not on PATH")
    m.set_defaults(run=_map)
    s = sub.add_parser("scoreboard", help="print a match summary's scoreboard")
    s.add_argument("match", help="<match>.match.json")
    s.set_defaults(run=_scoreboard)
    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
