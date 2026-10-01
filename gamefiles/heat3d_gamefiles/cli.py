"""
Command line for the game-file exporter.

Verbs in the order the questions come up: what engine is this, what is inside,
and give me that. Everything else — which mesh is the level, how to decimate it
— belongs to the viewer's own import, not here.

    python -m heat3d_gamefiles identify <path>
    python -m heat3d_gamefiles list     <archive> [--filter text] [--limit n]
    python -m heat3d_gamefiles extract  <archive> <path inside> [--out file]
    python -m heat3d_gamefiles meshes   <unity .assets> [--limit n]
    python -m heat3d_gamefiles course   <lap> [--out map.glb]
    python -m heat3d_gamefiles terrain  <forza track folder> --out map.glb
                                        (--box x0 z0 x1 z1 | --route lap.json)

The archive is whatever the game ships: an Unreal `.pak`, a Unity `.assets` or
bundle, a Frostbite `.toc`. `identify` does not require knowing which.

The last two are the odd ones out, and deliberately. They do not take an archive
but a whole Forza track folder, because a piece of map is spread across four of
them and their manifests, and it is asked for by *place* rather than by name.

`course` is the one to reach for: give it a lap and it exports the ground around
it. It works out which game is installed, which of its tracks the lap was driven
on, and where to cut, so the only thing anyone has to know is which lap they
mean. `terrain` is the same cut with all of that spelled out by hand, for when
the answer is a box rather than a drive.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _identify(path: Path) -> str:
    """Name the container format, by reading it rather than by extension."""
    from .frostbite import (
        UnsupportedFrostbite,
        parse_toc,
        read_header,
        read_sections,
        superbundles,
    )
    from .pak import UnsupportedPak, read_footer
    from .unity import UnsupportedUnity, read_serialized

    lines = [f"{path.name}  ({path.stat().st_size:,} bytes)"]

    try:
        info = read_footer(path)
        lines.append(f"  Unreal pak, version {info.version}")
        lines.append(f"    index      {info.index_size:,} bytes at {info.index_offset:,}")
        lines.append(f"    encrypted  {'yes' if info.encrypted_index else 'no'}")
        lines.append(f"    methods    {', '.join(info.methods) or 'none'}")
        return "\n".join(lines)
    except UnsupportedPak:
        pass

    try:
        head = path.open("rb").read(8)
        if head.startswith(b"UnityFS"):
            from .unity import UnityBundle

            bundle = UnityBundle(path)
            lines.append(f"  Unity bundle, engine {bundle.engine_version}")
            lines.append(f"    {len(bundle)} entries")
            return "\n".join(lines)
        container = read_serialized(path)
        lines.append(f"  Unity SerializedFile, format {container.version}")
        lines.append(f"    engine     {container.unity_version}")
        lines.append(f"    objects    {len(container):,}")
        top = list(container.counts().items())[:6]
        lines.append("    holds      " + ", ".join(f"{n}x{k}" for k, n in top))
        return "\n".join(lines)
    except UnsupportedUnity:
        pass

    try:
        data = path.read_bytes()[:64]
        info = read_header(data, path.name)
        lines.append(f"  Frostbite table of contents, {info.kind}")
        lines.append(f"    tree begins at {info.payload_offset}")
        whole = path.read_bytes()
        try:
            sections = read_sections(whole, info, path.name)
            lines.append(f"    assets     {sections.asset_count:,}")
        except UnsupportedFrostbite:
            names = superbundles(parse_toc(path)[1])
            lines.append(f"    superbundles {len(names)}")
        return "\n".join(lines)
    except UnsupportedFrostbite:
        pass

    try:
        from .vpk import UnsupportedVpk, VpkArchive

        archive = VpkArchive(path)
        lines.append(f"  Source 2 VPK, version {archive.version}")
        lines.append(f"    entries    {len(archive):,}")
        navs = [n for n in archive.list() if n.endswith(".nav")]
        if navs:
            lines.append(f"    navmesh    {navs[0]}")
        return "\n".join(lines)
    except UnsupportedVpk:
        pass

    try:
        from .forzatech import UnsupportedForza, read_model

        model = read_model(path)
        lines.append(f"  ForzaTech model, version {model.version}")
        lines.append(f"    chunks     {len(model)}")
        lines.append(
            "    holds      "
            + ", ".join(f"{n}x{t}" for t, n in list(model.counts().items())[:6])
        )
        lines.append("    note       chunk contents are not decoded")
        return "\n".join(lines)
    except UnsupportedForza as exc:
        lines.append("  not a format this tool reads")
        lines.append(f"    {exc}")
    return "\n".join(lines)


def _open_archive(path: Path, key: str | None):
    """
    Open whichever kind of archive this is.

    Dispatching on the bytes rather than on the extension, and rather than on a
    flag the user has to pass: `list` and `extract` mean the same thing for an
    Unreal pak and a Source 2 VPK, and making someone say which they have is
    asking them to know something the tool can see.
    """
    from .pak import PakArchive, UnsupportedPak
    from .vpk import UnsupportedVpk, VpkArchive

    aes = None
    if key:
        text = key[2:] if key.lower().startswith("0x") else key
        aes = bytes.fromhex(text)

    try:
        return PakArchive(path, aes_key=aes)
    except UnsupportedPak as pak_error:
        try:
            return VpkArchive(path)
        except UnsupportedVpk:
            # Report the Unreal reason, not the VPK one: a file that is neither
            # is far more often a pak variant than a malformed VPK, and that
            # message names the likelier cause.
            raise pak_error from None



def _write_course(out, track, target, args, low, high, route=None, corridor=None) -> Path:
    """
    Bake, place, plan and write: the part of an export both verbs share.

    Textures are baked per tile from the track's own maps unless turned off,
    and a tile whose maps are missing falls back to flat colour rather than
    failing the export - detail is optional, geometry is not. The same goes for
    placed models: kerbs, barriers and the rest are added from the track's
    placement files, each with its own texture where the game gives it one,
    and with the detail going to what stands nearest `route`, the driving.
    """
    import numpy as np

    from .courseexport import bake_tiles, lift_decals, plan_parts, planar_uvs
    from .glb import Part, compute_normals, write_glb

    images = {}
    if not args.no_textures and out.tile_keys:
        images = bake_tiles(
            track, out.tile_keys, size=args.texture_size, season=args.season
        )
        missing = len(out.tile_keys) - len(images)
        note = f", {missing} without maps" if missing else ""
        print(
            f"\n  baked {len(images)} ground textures at {args.texture_size} px "
            f"({512 / args.texture_size * 100:.0f} cm a pixel){note}"
        )

    plan = plan_parts(out, images)
    total = sum(plan.counts.values())
    print("\n  what the ground is made of:")
    for label, many in sorted(plan.counts.items(), key=lambda kv: -kv[1]):
        print(f"    {many:9,} triangles  {many / total:5.1%}  {label}")

    ground_normals = out.normals if out.normals is not None else compute_normals(out.positions, out.faces)
    parts, decal_positions, decal_normals = lift_decals(plan.parts, out.positions, ground_normals)
    positions = [out.positions, decal_positions]
    normals = [ground_normals, decal_normals]
    uvs = [
        planar_uvs(out) if plan.textured else np.zeros((len(out.positions), 2), np.float32),
        np.zeros((len(decal_positions), 2), np.float32),
    ]
    textured = plan.textured
    if not args.no_placed:
        from .forzaplacement import DEFAULT_CATEGORIES, VEGETATION, place_in_track

        categories = DEFAULT_CATEGORIES + (VEGETATION if args.trees else ())
        placed = place_in_track(
            track,
            low,
            high,
            categories=categories,
            budget=args.detail_budget,
            season=args.season,
            textures=not args.no_textures,
            route=route,
            corridor=corridor,
        )
        if placed.parts:
            print(
                f"\n  placed on it ({placed.triangles():,} triangles of a "
                f"{placed.budget:,} budget, {len(placed.textures)} textures):"
            )
            print(placed.summary())
        offset = len(out.positions) + len(decal_positions)
        for name, mesh in sorted(placed.parts.items()):
            positions.append(mesh.positions)
            normals.append(mesh.normals)
            uvs.append(mesh.uvs if mesh.uvs is not None else np.zeros((len(mesh.positions), 2), np.float32))
            texture = placed.textures.get(mesh.texture) if mesh.texture else None
            textured = textured or texture is not None
            parts.append(
                Part(name=name, faces=mesh.faces + offset, colour=None if texture is not None else mesh.colour, texture=texture)
            )
            offset += len(mesh.positions)

    written = write_glb(
        target,
        np.concatenate(positions),
        parts=parts,
        normals=np.concatenate(normals),
        uvs=np.concatenate(uvs).astype(np.float32) if textured else None,
    )
    print(f"\nwrote {written} ({written.stat().st_size:,} bytes)")
    return written


def _course(args) -> int:
    """
    A lap in, the ground around it out.

    Everything between those two is derivable and so is derived: which game is
    installed, which of its tracks these coordinates belong to, and what to
    cut. The alternative is a command line carrying a forty-character path to a
    folder inside a game someone already has installed, which nobody should have
    to look up to export a corner of a racetrack.
    """
    from .lapinput import UnreadableLap, read

    try:
        driven = read(args.lap)
    except UnreadableLap as exc:
        print(exc, file=sys.stderr)
        return 1
    target = args.out or Path(_slug(driven.name) + ".glb")
    game = (getattr(args, "game", None) or driven.game or "forza").lower()
    if game == "beamng":
        from .beamngcourse import export

        if len(driven) < 2:
            print(f"{driven.name} holds no driving", file=sys.stderr)
            return 1
        print(f"{driven.name}: {len(driven):,} samples, {driven.metres:,.0f} m driven (BeamNG.drive)")
        return export(driven, target, args, install=getattr(args, "install", None))
    if game in ("assetto-corsa-rally", "acrally"):
        from .acrallycourse import export as export_rally

        if len(driven) < 2:
            print(f"{driven.name} holds no driving", file=sys.stderr)
            return 1
        print(f"{driven.name}: {len(driven):,} samples, {driven.metres:,.0f} m driven (Assetto Corsa Rally)")
        return export_rally(driven, target, args, paks=getattr(args, "install", None))
    if not game.startswith("forza"):
        print(_no_course(game), file=sys.stderr)
        return 1
    return export_course(driven, target, args)


#: Why a game's lap cannot be turned into a course from its files yet. Said
#: plainly, with the route that does work for every game.
_WHY_NOT = {
    "assetto-corsa-evo": (
        "Assetto Corsa EVO keeps all its content in one package obfuscated "
        "with a key, and this tool recovers none"
    ),
    "iracing": "iRacing's tracks ship in a protected format that is not a file-extraction target",
}


def _no_course(game: str) -> str:
    reason = _WHY_NOT.get(game, f"there is no course reader for {game} yet")
    return (
        f"no course export for this lap: {reason}.\n"
        "  Courses can be exported from Forza, BeamNG.drive and Assetto Corsa Rally laps. For the "
        "rest, reconstruct the course from gameplay\n"
        "  (gamecapture), or analyse the laps without one - the "
        "Race tab draws a road from them."
    )


def _slug(name: str) -> str:
    """A course name as a file name: `Lakeside Circuit` -> `lakeside_circuit`."""
    import re

    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "course"


def export_course(driven, target: Path, args) -> int:
    """
    Export the ground and everything on it along one course's driving.

    The cut is a corridor `args.margin` metres either side of the driven line,
    not the driving's bounding box: the same thing for a circuit, and a tenth
    of it for a point-to-point sprint, whose box is mostly hillside nobody
    drives past.
    """
    from .corridor import Corridor
    from .forzainstall import find_track
    from .forzaterrain import extract_from_track

    if len(driven) < 2:
        print(f"{driven.name} holds no driving", file=sys.stderr)
        return 1

    corridor = Corridor(driven.positions, args.margin)
    low, high = driven.box(args.margin)
    laps = f"{driven.laps} laps, " if driven.laps > 1 else ""
    print(
        f"{driven.name}: {laps}{len(driven):,} samples, {driven.metres:,.0f} m driven"
    )

    track = args.track
    if track is None:
        try:
            track = find_track(low, high)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"  on {track.name}, in {track.parents[2].name}")

    print(
        f"  cutting {args.margin:,.0f} m either side of the driving: "
        f"{corridor.area / 1e6:,.2f} km2 of a {high[0] - low[0]:,.0f} by "
        f"{high[1] - low[1]:,.0f} m box"
    )
    out = extract_from_track(track, low, high, corridor=corridor)
    if not len(out.faces):
        print(
            f"\nno terrain there. {track.name} has tiles nearby but none that "
            "cover the driving, which usually means the lap and the track are "
            "from different games.",
            file=sys.stderr,
        )
        return 1

    print()
    print(out.summary())
    _write_course(out, track, target, args, low, high, route=driven.positions, corridor=corridor)
    return 0


def _courses(args) -> int:
    """
    Every course in a recorded lap library, one file each.

    Carries on past a course that fails, and says at the end which did; a
    course already exported is skipped unless `--force`, so a run that was
    interrupted picks up where it stopped.
    """
    import time

    from .lapinput import UnreadableLap, default_library, read

    try:
        library = Path(args.library) if args.library else default_library()
    except UnreadableLap as exc:
        print(exc, file=sys.stderr)
        return 1
    if not library.is_dir():
        print("no recorded lap library found; name one with --library", file=sys.stderr)
        return 1
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    courses = sorted(
        p for p in Path(library).iterdir() if p.is_dir() and " (course_" in p.name
    )
    if args.only:
        wanted = [w.lower() for w in args.only]
        courses = [p for p in courses if any(w in p.name.lower() for w in wanted)]
    results = []
    for number, course in enumerate(courses, 1):
        name = course.name.split(" (course_")[0]
        target = folder / (_slug(name) + ".glb")
        if target.exists() and not args.force:
            results.append((name, "kept", target.stat().st_size, 0.0))
            continue
        print(f"\n=== {number}/{len(courses)}: {name} ===")
        started = time.time()
        try:
            driven = read(course)
            status = export_course(driven, target, args)
        except (UnreadableLap, OSError, ValueError, MemoryError) as exc:
            print(f"  failed: {exc}", file=sys.stderr)
            status = 1
        took = time.time() - started
        size = target.stat().st_size if target.exists() and status == 0 else 0
        results.append((name, "written" if status == 0 else "FAILED", size, took))

    print("\n" + "-" * 72)
    for name, status, size, took in results:
        extra = f"{size / 1e6:8.1f} MB" if size else " " * 11
        timing = f"{took:6.0f} s" if took else ""
        print(f"  {status:8} {extra}  {timing:>8}  {name}")
    failed = [r for r in results if r[1] == "FAILED"]
    print(f"\n{len(results) - len(failed)} of {len(results)} courses in {folder}")
    return 1 if failed else 0


def _route_box(path: Path, margin: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    The ground a recorded lap covers, widened a little.

    The margin is not cosmetic. A lap is a line, and a line's bounding box in the
    plane is the road and nothing either side of it — no verge, no barrier, no
    hill the road runs along. A hundred metres is enough to see what the driver
    saw.
    """
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    box = data.get("bounds")
    if not box:
        raise SystemExit(f"{path} has no bounds; it is not a recorded route")
    low, high = box["min"], box["max"]
    return (
        (low[0] - margin, low[2] - margin),
        (high[0] + margin, high[2] + margin),
    )


def _terrain(args) -> int:
    from .forzaterrain import extract_from_track
    from .glb import write_glb

    if bool(args.box) == bool(args.route):
        print("give exactly one of --box and --route", file=sys.stderr)
        return 2
    route = corridor = None
    if args.route:
        from .corridor import Corridor
        from .lapinput import UnreadableLap, read_file

        low, high = _route_box(args.route, args.margin)
        try:
            route = read_file(args.route)
        except UnreadableLap:
            route = None
        # A route file that only carries its bounds is a box, not a line.
        if route is not None and len(route) >= 10:
            corridor = Corridor(route, args.margin)
    else:
        x0, z0, x1, z1 = args.box
        low, high = (min(x0, x1), min(z0, z1)), (max(x0, x1), max(z0, z1))

    span = (high[0] - low[0], high[1] - low[1])
    print(
        f"cutting {span[0]:,.0f} by {span[1]:,.0f} metres "
        f"at x {low[0]:,.0f}..{high[0]:,.0f}, z {low[1]:,.0f}..{high[1]:,.0f}"
    )
    out = extract_from_track(args.path, low, high, corridor=corridor)
    if not len(out.faces):
        print(
            "no terrain there.\n"
            "  The coordinates are the game's own metres, as its telemetry reports\n"
            "  them. A box outside the map, or in the sea, is empty rather than an\n"
            "  error.",
            file=sys.stderr,
        )
        return 1

    print(out.summary())
    _write_course(out, args.path, args.out, args, low, high, route=route, corridor=corridor)
    return 0


def _export_options(p) -> None:
    """The options `course` and `terrain` share: what goes into the file."""
    from .forzaplacement import DEFAULT_BUDGET

    p.add_argument(
        "--texture-size",
        type=int,
        choices=(2048, 1024, 512),
        default=2048,
        help="ground texture per 512 m tile, in pixels (2048 = 25 cm a pixel)",
    )
    p.add_argument(
        "--no-textures",
        action="store_true",
        help="flat colours only: a far smaller file, and no ground detail",
    )
    p.add_argument(
        "--season",
        choices=("summer", "autumn", "winter", "spring"),
        default="summer",
        help="which season's ground maps and model textures to use",
    )
    p.add_argument(
        "--no-placed",
        action="store_true",
        help="terrain only: leave out kerbs, barriers, tyre walls, signs and buildings",
    )
    p.add_argument(
        "--trees",
        action="store_true",
        help="also place trees and bushes (many triangles, and they hide the road)",
    )
    p.add_argument(
        "--detail-budget",
        type=int,
        default=None,
        help=(
            "a ceiling on triangles for everything placed; the costliest models drop "
            "to coarser levels of detail until it fits, and past that the furthest "
            "copies are left out (default: enough to keep every copy, and at least "
            f"{DEFAULT_BUDGET:,}; kerbs always stay at full detail)"
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="heat3d_gamefiles",
        description="Read level data out of shipped game archives.",
        epilog=(
            "Encrypted archives need a key you already hold; this tool has none "
            "and recovers none. What you point it at is subject to that game's "
            "own licence."
        ),
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    p = sub.add_parser("identify", help="say what a file is")
    p.add_argument("path", type=Path)

    p = sub.add_parser("list", help="list what is inside an archive")
    p.add_argument("path", type=Path)
    p.add_argument("--filter", default=None, help="only paths containing this text")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--key", default=None, help="AES key in hex, if you hold one")

    p = sub.add_parser("extract", help="write one entry to disk")
    p.add_argument("path", type=Path)
    p.add_argument("entry")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--key", default=None)

    p = sub.add_parser("meshes", help="list the meshes in a Unity container")
    p.add_argument("path", type=Path)
    p.add_argument("--limit", type=int, default=25)

    p = sub.add_parser(
        "course", help="export the ground around a lap, working out the rest"
    )
    _export_options(p)
    p.add_argument(
        "lap",
        help=(
            "a lap or course: a file written by heat3d_capture, a lap or course "
            "folder from a recorded library, or just a course's name"
        ),
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="where to write the .glb (default: the course's name)",
    )
    p.add_argument(
        "--track",
        type=Path,
        default=None,
        help="the track folder, if the installed game should not be searched for",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=80.0,
        help="metres of ground to keep either side of the driving (default 80)",
    )
    p.add_argument(
        "--game",
        choices=("forza", "beamng", "acrally"),
        default=None,
        help="which game the lap is from (default: what the lap file says, else Forza)",
    )
    p.add_argument(
        "--level",
        default=None,
        help="BeamNG, AC Rally: the level or stage, if it should not be found from the lap",
    )
    p.add_argument(
        "--install",
        type=Path,
        default=None,
        help="BeamNG, AC Rally: the game's folder, if it is not in a Steam library",
    )
    p.set_defaults(run=_course)

    p = sub.add_parser(
        "courses", help="export every course in a recorded lap library, one file each"
    )
    _export_options(p)
    p.add_argument("-o", "--out", default="courses", help="folder to write into (default ./courses)")
    p.add_argument(
        "--library",
        default=None,
        help="the lap library (default: FH Companion's, under %%LOCALAPPDATA%%)",
    )
    p.add_argument("--only", nargs="+", default=None, help="just the courses whose names contain these")
    p.add_argument("--force", action="store_true", help="export again what is already there")
    p.add_argument("--track", type=Path, default=None, help="the track folder, if not the installed game's")
    p.add_argument("--margin", type=float, default=80.0, help="metres either side of the driving (default 80)")
    p.set_defaults(run=_courses)

    p = sub.add_parser(
        "terrain", help="cut a piece of Forza terrain out, by world coordinates"
    )
    _export_options(p)
    p.add_argument("path", type=Path, help="a track folder under media/Tracks")
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument(
        "--box",
        type=float,
        nargs=4,
        metavar=("X0", "Z0", "X1", "Z1"),
        default=None,
        help="the region to cut, in the game's metres",
    )
    p.add_argument(
        "--route",
        type=Path,
        default=None,
        help="a lap recorded by heat3d_capture, whose extent is the region",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=100.0,
        help="metres to add around a route, so the verge comes with the road",
    )

    args = parser.parse_args(argv)

    if args.verb == "course":
        return _course(args)
    if args.verb == "courses":
        return _courses(args)

    if not args.path.exists():
        print(f"no such file: {args.path}", file=sys.stderr)
        return 2

    if args.verb == "identify":
        print(_identify(args.path))
        return 0

    if args.verb == "list":
        archive = _open_archive(args.path, args.key)
        names = archive.list(args.filter)
        print(f"{len(names):,} entries" + (f" matching {args.filter!r}" if args.filter else ""))
        for name in names[: args.limit]:
            entry = archive.entries[name]
            # An Unreal entry names its compression method; a VPK entry does not
            # compress at all. Asking the entry rather than the archive keeps
            # one printing path for both.
            how = (
                archive.info.method_name(entry.method)
                if hasattr(archive, "info")
                else "stored"
            )
            size = getattr(entry, "uncompressed_size", None)
            if size is None:
                size = entry.size
            print(f"  {size:>12,}  {how:<6} {name}")
        if len(names) > args.limit:
            print(f"  ... {len(names) - args.limit:,} more")
        return 0

    if args.verb == "extract":
        archive = _open_archive(args.path, args.key)
        if args.entry not in archive:
            print(f"not in the archive: {args.entry}", file=sys.stderr)
            return 1
        data = archive.read(args.entry)
        out = args.out or Path(args.entry.rsplit("/", 1)[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        print(f"{len(data):,} bytes -> {out}")
        return 0

    if args.verb == "terrain":
        return _terrain(args)

    if args.verb == "meshes":
        from .unity import read_serialized
        from .unity_mesh import read_mesh

        container = read_serialized(args.path)
        meshes = container.of_class("Mesh")
        print(f"{len(meshes)} meshes in {args.path.name}")
        shown = 0
        for obj in meshes:
            if shown >= args.limit:
                break
            try:
                mesh = read_mesh(container, obj)
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                print(f"  [{obj.path_id}] could not be read: {exc}")
                shown += 1
                continue
            check = "ok" if mesh.aabb_matches() else "BOUNDS DISAGREE"
            print(
                f"  {mesh.name[:40]:<40} {len(mesh.vertices):>8,} verts "
                f"{len(mesh.triangles):>8,} tris  {check}"
            )
            shown += 1
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
