"""
Command line for the capture exporter.

The interface (`python -m heat3d_capture.ui`) is the way in for scanning live —
it has the overlay, the coverage map and the preview. This is for the other case:
a recording that is already on disk, reconstructed offline with the camera path
solved all at once.

    python -m heat3d_capture reconstruct lap.mp4 -o lap.glb

That path exists because live tracking loses a racetrack, and it is worth being
plain that it is slow: it hands overlapping windows of frames to a model that
solves all their cameras together, which is minutes per window on a desktop GPU,
not frames per second. Progress is printed as it goes so a long run is legible
rather than silent.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _reconstruct(args: argparse.Namespace) -> int:
    from .pose.feedforward import DEFAULT_MODEL, describe_licence
    from .scan.offline import OfflineProgress, reconstruct_video

    print(describe_licence(args.model))
    print()

    started = time.monotonic()
    last = {"stage": ""}

    def show(p: OfflineProgress) -> None:
        elapsed = time.monotonic() - started
        if p.stage != last["stage"]:
            print(f"[{elapsed:6.0f}s] {p.stage}")
            last["stage"] = p.stage
        if p.total:
            print(f"[{elapsed:6.0f}s]   {p.done + 1}/{p.total}  {p.note}", flush=True)

    try:
        result = reconstruct_video(
            args.video,
            args.out,
            solve_budget=args.solve_budget,
            fuse_interval=args.fuse_interval,
            on_progress=show,
        )
    except (OSError, ValueError) as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    print()
    print(result.summary())
    # A run that produced no file is a failure even though nothing raised, and
    # the exit code has to say so or a script around this will not notice.
    return 0 if result.export else 1


def _route(args: argparse.Namespace) -> int:
    """
    Write the driven route as it arrives, rather than at the end.

    The positions are the cheapest thing in the pipeline and the hardest to get
    again — they need the game running, the right track loaded, and someone to
    drive it. A lap held in memory and lost to a later failure is exactly what
    this is here to stop, so the file is complete at every moment.
    """
    import time

    from .telemetry.record import Recorder

    print(f"listening on port {args.port}. Drive, then press Ctrl+C when done.")
    started = time.monotonic()
    last_report = 0.0
    with Recorder(port=args.port) as recorder:
        try:
            while True:
                recorder.poll()
                elapsed = time.monotonic() - started
                if elapsed - last_report >= 1.0:
                    last_report = elapsed
                    kept = len(recorder.positions)
                    if recorder.packets == 0:
                        print(
                            f"\r[{elapsed:5.0f}s] nothing on port {args.port} yet - "
                            "is Data Out on?",
                            end="",
                            flush=True,
                        )
                    else:
                        where = recorder.positions[-1] if kept else None
                        place = (
                            f"  at {where[0]:8.0f} {where[1]:7.0f} {where[2]:8.0f}"
                            if where is not None
                            else ""
                        )
                        print(
                            f"\r[{elapsed:5.0f}s] {recorder.packets} packets, "
                            f"{kept} positions{place}   ",
                            end="",
                            flush=True,
                        )
                if args.seconds and elapsed >= args.seconds:
                    break
        except KeyboardInterrupt:
            pass
        route = recorder.finish()

    print()
    if not len(route):
        print(
            "No positions recorded.\n"
            "  In Forza: Settings -> HUD and Gameplay -> Data Out = ON,\n"
            f"  IP 127.0.0.1, port {args.port}. The car has to be moving."
        )
        return 1

    written = route.save(args.out)
    print(route.summary())
    print(f"\nwrote {written} and {written.with_suffix('.json')}")
    return 0


def _telemetry(args: argparse.Namespace) -> int:
    """
    Record every channel the game sends, one file per lap, for the Race tab.

    Relays each packet on to `--forward` unchanged, so an app already reading
    the game's telemetry (FH Companion) keeps working: point the game at this
    port, and that app at the forward one.
    """
    if args.game == "trackmania":
        # Trackmania is recorded by an Openplanet plugin that writes the files itself.
        import shutil

        source = Path(__file__).resolve().parents[1] / "mods" / "trackmania" / "Heat3dRecorder"
        target = Path.home() / "OpenplanetNext" / "Plugins" / "Heat3dRecorder"
        if args.install:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
            print(f"plugin copied to {target}. In Openplanet: Developer > Load plugin, then drive.")
        print(f"runs are written to {Path.home() / 'OpenplanetNext' / 'PluginStorage' / 'Heat3dRecorder'} - load those CSVs in the Race tab.")
        return 0
    if args.game == "beamng":
        from .telemetry import beamng

        if args.install:
            print(f"protocol mod copied to {beamng.install()}. Switch on Options > Other > Protocols > others.")
        written = beamng.run(args.out, port=args.port if args.port != 5300 else beamng.PORT)
        print(f"\n{len(written)} laps written to {args.out}")
        return 0
    if args.game in ("wrc", "dirtrally2", "dirtrally", "dirt4", "rbr"):
        from .telemetry import rally

        port = args.port if args.port != 5300 else rally.DEFAULT_PORTS[args.game]
        if args.game == "wrc" and args.configure:
            config = Path(args.configure)
            print(f"{config}: {rally.configure_wrc(config, port=port)}")
        written = rally.run(args.out, game=args.game, port=port, surface=args.surface)
        print(f"\n{len(written)} stages written to {args.out}")
        return 0
    if args.game != "forza":
        from .telemetry import kunos

        print(f"reading {args.game}'s shared memory. Laps go to {args.out}. Ctrl+C to stop.")
        written = kunos.run(args.out, game=args.game)
        print(f"\n{len(written)} laps written to {args.out}")
        return 0

    from .telemetry.fulllap import parse_forward, run

    forward = parse_forward(args.forward)
    relay = f", passing everything on to {forward[0]}:{forward[1]}" if forward else ""
    print(f"listening on port {args.port}{relay}. Laps go to {args.out}. Ctrl+C to stop.")
    written = run(args.out, port=args.port, forward=forward)
    print(f"\n{len(written)} laps written to {args.out}")
    return 0


def _laps(args: argparse.Namespace) -> int:
    """
    Turn a recorded course into the two files everything downstream wants.

    A route, which says which piece of map to cut out of the game's archive, and
    a table of every sample of every lap, which is the analysis. They are written
    together because separating them is how they end up out of step: a heatmap
    over the wrong stretch of ground looks plausible and is worthless.
    """
    from .telemetry.fhcompanion import (
        NoLapLibrary,
        courses,
        find_course,
        read_course,
    )

    try:
        if args.course is None:
            found = courses(args.library)
            print(f"{len(found)} courses")
            for folder in found:
                name = folder.name.split(" (")[0]
                laps = sum(
                    1 for p in folder.rglob("*.json") if p.name != "course.json"
                )
                print(f"  {laps:4} laps  {name}")
            print("\nName one to write it out:  -o <file> <course>")
            return 0
        folder = find_course(args.course, args.library)
    except (NoLapLibrary, KeyError) as exc:
        print(exc, file=sys.stderr)
        return 1

    course = read_course(folder)
    if not len(course):
        print(f"{course.name} has no readable laps", file=sys.stderr)
        return 1
    print(course.summary())

    out = args.out or Path(course.name.lower().replace(" ", "_"))
    route = course.save_route(out, margin=args.margin)
    samples = course.save_samples(out)
    low, high = course.bounds
    print(
        f"\nwrote {route} and {samples}"
        f"\n\nThe map under it:"
        f"\n  python -m heat3d_gamefiles terrain <track folder> "
        f"--route {route.name} -o {Path(out).stem}.glb"
        f"\n  (or --box {low[0] - args.margin:.0f} {low[2] - args.margin:.0f} "
        f"{high[0] + args.margin:.0f} {high[2] + args.margin:.0f})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="heat3d_capture",
        description="Reconstruct a level from a recording.",
        epilog=(
            "For live scanning with the overlay and coverage map, run "
            "python -m heat3d_capture.ui instead."
        ),
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    p = sub.add_parser(
        "route",
        help="record a lap's positions from the game's telemetry, and nothing else",
    )
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--port", type=int, default=5300)
    p.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="stop after this long; 0 means run until interrupted",
    )
    p.set_defaults(run=_route)

    p = sub.add_parser(
        "telemetry",
        help="record every telemetry channel, one file per lap, for the viewer's Race tab",
    )
    p.add_argument("-o", "--out", type=Path, default=Path("laps"), help="folder for the lap files")
    p.add_argument("--port", type=int, default=5300, help="the port the game sends to")
    p.add_argument(
        "--forward",
        default=None,
        help="host:port to pass every packet on to, e.g. 127.0.0.1:5301 for FH Companion",
    )
    p.add_argument(
        "--game",
        choices=["forza", "ac", "acc", "acevo", "acrally", "wrc", "dirtrally2", "dirtrally", "dirt4", "rbr", "beamng", "trackmania"],
        default="forza",
        help=(
            "forza (Data Out over UDP, the default); an Assetto Corsa game from its shared memory; "
            "wrc, dirtrally2, dirtrally, dirt4 or rbr from their UDP telemetry; beamng from the heat3d "
            "protocol mod; trackmania from the Openplanet plugin's files (see docs/games.md)"
        ),
    )
    p.add_argument("--install", action="store_true", help="beamng / trackmania: copy the mod or plugin into the game's folder first")
    p.add_argument("--configure", default=None, help="wrc: the game's telemetry config.json to switch the packet on in (backed up first)")
    p.add_argument(
        "--surface",
        choices=["tarmac", "loose", "snow", "ice", "mixed"],
        default=None,
        help="rally games: the stage's surface, when it is not gravel",
    )
    p.set_defaults(run=_telemetry)

    p = sub.add_parser(
        "laps", help="read a recorded lap library, and write out one course"
    )
    p.add_argument(
        "course",
        nargs="?",
        default=None,
        help="course name, loosely matched; omit to list what is there",
    )
    p.add_argument("-o", "--out", type=Path, default=None)
    p.add_argument(
        "--library",
        type=Path,
        default=None,
        help="where the laps live (default: %%LOCALAPPDATA%%/FHCompanion/laps)",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=80.0,
        help="metres of map to keep around the course (default 80)",
    )
    p.set_defaults(run=_laps)

    p = sub.add_parser("reconstruct", help="solve a recording offline and write a .glb")
    p.add_argument("video", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument(
        "--solve-budget",
        type=int,
        default=260,
        help=(
            "how many frames the solver may use (default 260). They are spaced by "
            "how far the picture moves rather than by the clock, so a fast section "
            "gets more of them. More is more accurate and much slower: cost is "
            "minutes per window of 16."
        ),
    )
    p.add_argument(
        "--fuse-interval",
        type=float,
        default=0.4,
        help="seconds between frames fused into geometry (default 0.4)",
    )
    p.add_argument("--model", default="vggt-1b", help="which weights to solve with")
    p.set_defaults(run=_reconstruct)

    args = parser.parse_args(argv)
    # The verbs that take no video: the existence check below is about the
    # recording a reconstruction reads, and these do not read one.
    if args.verb in ("route", "laps", "telemetry"):
        return args.run(args)
    if not args.video.exists():
        print(f"no such file: {args.video}", file=sys.stderr)
        return 2
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
