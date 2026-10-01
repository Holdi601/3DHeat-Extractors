"""
A CS2 map as level geometry: the world's collision mesh, in the frame the
demo tables are drawn in.

The collision mesh is what the players stand on and hide behind - walls,
floors, boxes, without the decoration - which is what a heatmap wants under
it. It is read from the map's own `.vpk` in the installed game by
Source2Viewer's command line (ValveResourceFormat, MIT licensed,
https://github.com/ValveResourceFormat/ValveResourceFormat), which decodes
Source 2's compiled physics; nothing here ships or reads anything else.

Source2Viewer writes glTF in metres with Y up: (Y, Z, X) of the game's
inches, scaled by 0.0254. That is undone, and the result written the way the
viewer draws the demo tables (see output.HINTS): game units, (X, Z, -Y). Clip
brushes - the invisible walls that stop players and grenades - and the sky
are left out. Each surface is split into ground (facing up, walkable) and
structure, by the triangle's normal.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import glb

GAME_FOLDER = "Counter-Strike Global Offensive"
STEAM_DEFAULTS = (r"C:\Program Files (x86)\Steam", r"C:\Program Files\Steam", "~/.steam/steam", "~/.local/share/Steam")
LIBRARY_PATH = re.compile(r'"path"\s*"([^"]+)"')
#: glTF metres of Source2Viewer's export -> game inches.
INCH = 0.0254
#: A face whose normal is at least this far up is ground: CS2's players walk slopes up to about 45 degrees.
GROUND = 0.7
SOURCE2VIEWER_URL = "https://github.com/ValveResourceFormat/ValveResourceFormat/releases"


def steam_libraries() -> list[Path]:
    roots: list[Path] = []
    for base in STEAM_DEFAULTS:
        base = Path(base).expanduser()
        config = base / "config" / "libraryfolders.vdf"
        if base.is_dir():
            roots.append(base)
        if config.exists():
            for found in LIBRARY_PATH.finditer(config.read_text(encoding="utf-8", errors="replace")):
                roots.append(Path(found.group(1).replace("\\\\", "\\")))
    seen, out = set(), []
    for r in roots:
        key = str(r).lower()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def find_vpk(name: str, game: str | None = None) -> Path:
    """The map's .vpk: a path as given, or by name in the CS2 install."""
    given = Path(name)
    if given.suffix.lower() == ".vpk" and given.exists():
        return given
    folders = [Path(game)] if game else [lib / "steamapps" / "common" / GAME_FOLDER for lib in steam_libraries()]
    for folder in folders:
        candidate = folder / "game" / "csgo" / "maps" / f"{given.stem}.vpk"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no {given.stem}.vpk found in an installed CS2 - pass --game <CS2 folder> or the .vpk path")


def find_source2viewer(given: str | None = None) -> Path:
    for candidate in (given, os.environ.get("HEAT3D_SOURCE2VIEWER"), shutil.which("Source2Viewer-CLI"), shutil.which("Source2Viewer-CLI.exe")):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(
        "Source2Viewer-CLI not found: download the command line build for your system from "
        f"{SOURCE2VIEWER_URL}, then put it on PATH, set HEAT3D_SOURCE2VIEWER, or pass --source2viewer"
    )


def to_game(gl: np.ndarray) -> np.ndarray:
    """Source2Viewer's glTF metres (Y up) -> the game's inches (Z up)."""
    return np.stack([gl[:, 2], gl[:, 0], gl[:, 1]], axis=1) / INCH


def to_viewer(game: np.ndarray) -> np.ndarray:
    """The game's frame -> the frame the demo tables are drawn in: (X, Z, -Y)."""
    return np.stack([game[:, 0], game[:, 2], -game[:, 1]], axis=1)


def surface_of(node: str) -> str | None:
    """The surface a physics group is made of; None for clip brushes and the sky."""
    if "clip" in node or "sky" in node:
        return None
    surface = re.sub(r"^physics(_group|_passbullets)?_?", "", node)
    return surface or "world"


def parts_from(meshes: list[tuple[str, np.ndarray, np.ndarray]]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Ground and structure parts by surface, in the viewer's frame."""
    grouped: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for node, gl, triangles in meshes:
        surface = surface_of(node)
        if surface is None or not len(triangles):
            continue
        game = to_game(gl)
        a, b, c = (game[triangles[:, k]] for k in range(3))
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal, axis=1)
        up = np.divide(normal[:, 2], length, out=np.zeros(len(length)), where=length > 0)
        for kind, keep in (("ground", up >= GROUND), ("structure", up < GROUND)):
            if keep.any():
                grouped.setdefault(f"{kind}:{surface}", []).append((to_viewer(game), triangles[keep]))
    parts = []
    for name, pieces in sorted(grouped.items()):
        offset, points, tris = 0, [], []
        for p, t in pieces:
            points.append(p)
            tris.append(t + offset)
            offset += len(p)
        parts.append((name, np.concatenate(points), np.concatenate(tris)))
    return parts


def export_map(name: str, out: str | Path, game: str | None = None, source2viewer: str | None = None) -> str:
    vpk = find_vpk(name, game)
    tool = find_source2viewer(source2viewer)
    stem = vpk.stem
    with tempfile.TemporaryDirectory() as tmp:
        run = subprocess.run(
            [str(tool), "-i", str(vpk), "-d", "-f", f"maps/{stem}/world_physics.vmdl_c", "--gltf_export_format", "glb", "-o", tmp],
            capture_output=True,
            text=True,
        )
        # It writes the model (empty: the world has no render mesh of its own)
        # and its physics beside it; the physics is the big one.
        found = sorted(Path(tmp).rglob("*.glb"), key=lambda f: f.stat().st_size, reverse=True)
        if run.returncode != 0 or not found:
            raise RuntimeError(f"Source2Viewer did not export {stem}'s collision mesh:\n{run.stdout[-800:]}{run.stderr[-800:]}")
        doc, blob = glb.read(found[0])
    parts = parts_from(glb.meshes(doc, blob))
    if not parts:
        raise RuntimeError(f"{stem}'s collision mesh has nothing but clip brushes")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    glb.write(out, parts)
    triangles = sum(len(t) for _, _, t in parts)
    ground = sum(len(t) for n, _, t in parts if n.startswith("ground:"))
    return f"{out}: {stem}, {triangles:,} triangles ({ground:,} ground) in {len(parts)} parts"
