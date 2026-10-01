"""
A room with known geometry, rendered from a known camera path.

Built because the accuracy of a reconstruction cannot be judged by looking at it.
A mesh from real gameplay can only be eyeballed — it looks about right, or it
does not — and "about right" is exactly the standard that lets a systematic
error through. Here the answer is known: the room is 12 by 8 metres, the camera
was at a stated place, and the difference between that and what came out is a
number in centimetres.

Ray-cast rather than rasterised, which makes the depth *exact* rather than
quantised by a depth buffer, and makes the ground truth trustworthy to the
precision the test claims. Fully vectorised over pixels, so a frame is one numpy
expression rather than a loop.

What this does and does not prove
---------------------------------
It exercises pose recovery, fusion, coverage and export against known values, and
those are the parts where an error is silent and systematic. It does *not* test
the depth network, because the depth here is analytic — real gameplay brings
motion blur, HUD elements, textureless walls and lighting changes that no
synthetic scene reproduces. The two kinds of test answer different questions and
neither substitutes for the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from heat3d_capture.pose.odometry import Intrinsics, Pose

#: Room extent in metres: x, y (height), z.
ROOM = np.array([12.0, 4.0, 8.0])


@dataclass
class Frame:
    """One rendered view, with the truth that produced it."""

    image: np.ndarray  # (H,W,3) uint8 RGB
    depth: np.ndarray  # (H,W) float32, metres, exact
    pose: Pose


def _texture(u: np.ndarray, v: np.ndarray, seed: int) -> np.ndarray:
    """
    A procedural surface pattern with corners in it.

    Corners, specifically: ORB needs repeatable corner features, and a smooth
    gradient or a plain colour gives it nothing to match between frames. Tiles
    with offset sub-blocks produce plenty at several scales.
    """
    rng = np.random.default_rng(seed)
    palette = rng.integers(40, 235, size=(16, 3)).astype(np.float32)

    tile_u = np.floor(u * 6.0).astype(int)
    tile_v = np.floor(v * 4.0).astype(int)
    index = (tile_u * 7 + tile_v * 13 + seed * 3) % 16
    colour = palette[np.clip(index, 0, 15)]

    # A second, finer pattern inside each tile, so there are features at more
    # than one scale and matching does not fail the moment the camera closes in.
    fine_u = np.floor(u * 24.0).astype(int)
    fine_v = np.floor(v * 16.0).astype(int)
    speckle = ((fine_u * 5 + fine_v * 11) % 5 == 0).astype(np.float32)[..., None]
    return np.clip(colour * (0.78 + 0.3 * speckle), 0, 255).astype(np.uint8)


def render(pose: Pose, k: Intrinsics) -> Frame:
    """
    Render one view of the room by ray-casting against its six walls.

    The camera is inside the box, so every ray hits exactly one wall: the nearest
    positive intersection along each axis-aligned slab.
    """
    ys, xs = np.mgrid[0 : k.height, 0 : k.width]
    # Pixel rays in camera space, then into world space.
    directions = np.stack(
        [(xs - k.cx) / k.fx, (ys - k.cy) / k.fy, np.ones_like(xs, dtype=np.float64)], axis=-1
    )
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    world_dirs = directions @ pose.rotation.T
    origin = pose.translation

    low = np.zeros(3)
    high = ROOM
    best = np.full(xs.shape, np.inf)
    hit_axis = np.zeros(xs.shape, dtype=np.int8)
    hit_sign = np.zeros(xs.shape, dtype=np.int8)

    for axis in range(3):
        d = world_dirs[..., axis]
        safe = np.where(np.abs(d) < 1e-12, np.nan, d)
        for sign, plane in ((-1, low[axis]), (1, high[axis])):
            t = (plane - origin[axis]) / safe
            # Only forward hits, and only where the other two axes land inside
            # the face — otherwise the ray leaves through a different wall.
            valid = np.isfinite(t) & (t > 1e-6)
            point = origin + world_dirs * t[..., None]
            for other in range(3):
                if other == axis:
                    continue
                valid &= (point[..., other] >= low[other] - 1e-9) & (
                    point[..., other] <= high[other] + 1e-9
                )
            closer = valid & (t < best)
            best = np.where(closer, t, best)
            hit_axis = np.where(closer, axis, hit_axis)
            hit_sign = np.where(closer, sign, hit_sign)

    point = origin + world_dirs * best[..., None]
    image = np.zeros((k.height, k.width, 3), dtype=np.uint8)
    for axis in range(3):
        u_axis, v_axis = [a for a in range(3) if a != axis]
        for sign in (-1, 1):
            face = (hit_axis == axis) & (hit_sign == sign)
            if not face.any():
                continue
            u = point[..., u_axis] / ROOM[u_axis]
            v = point[..., v_axis] / ROOM[v_axis]
            colours = _texture(u, v, seed=axis * 2 + (sign > 0))
            image[face] = colours[face]

    # Depth along the view axis, which is what a depth map means — not the
    # distance along the ray, which is larger toward the edges of the frame.
    forward = pose.rotation[:, 2]
    depth = ((point - origin) @ forward).astype(np.float32)
    depth[~np.isfinite(best)] = np.inf
    return Frame(image=image, depth=depth, pose=pose)


def look_at(position: np.ndarray, target: np.ndarray) -> Pose:
    """A pose at `position` looking toward `target`, with no roll."""
    forward = target - position
    forward = forward / max(np.linalg.norm(forward), 1e-9)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, forward)
    norm = np.linalg.norm(right)
    if norm < 1e-6:
        # Looking straight up or down: any roll is as good as any other.
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / norm
    up = np.cross(forward, right)
    return Pose(rotation=np.stack([right, up, forward], axis=1), translation=position.copy())


def walk(
    k: Intrinsics,
    *,
    steps: int = 40,
    height: float = 1.7,
    sidestep: float = 1.6,
) -> list[Frame]:
    """
    A walk down the room, weaving, looking ahead.

    Weaving on purpose. A camera moving straight at a wall gives every point one
    viewing direction and no parallax, and a reconstruction from it is
    unconstrained however many frames it has — which is the precise failure the
    coverage grid exists to warn about. The weave is what makes this scene
    solvable at all.
    """
    frames = []
    for i in range(steps):
        t = i / max(steps - 1, 1)
        x = 1.5 + t * (ROOM[0] - 3.0)
        z = ROOM[2] / 2 + np.sin(t * np.pi * 2.0) * sidestep
        position = np.array([x, height, z])
        target = np.array([x + 2.5, height * 0.92, ROOM[2] / 2])
        frames.append(render(look_at(position, target), k))
    return frames


def survey(k: Intrinsics, *, steps: int = 40, height: float = 1.7) -> list[Frame]:
    """
    The same room, scanned the way the tool tells people to scan.

    Walking forward while looking forward is the natural thing to do and the
    worst thing to do: every surface ahead is seen from one direction with no
    parallax, however many frames are spent on it. A survey moves sideways and
    sweeps the view across what it is passing, which is what actually
    triangulates.

    Kept next to `walk` so the coverage metric can be checked for the property
    that matters — that it can tell these two apart. A metric that called both of
    them good would be worse than no metric, because it would send someone home
    with an unreconstructable capture and a green map.
    """
    frames = []
    for i in range(steps):
        t = i / max(steps - 1, 1)
        x = 1.5 + t * (ROOM[0] - 3.0)
        # A wide weave, so each wall point is seen from genuinely separated
        # positions rather than from a line.
        z = ROOM[2] / 2 + np.sin(t * np.pi * 4.0) * (ROOM[2] / 2 - 1.2)
        position = np.array([x, height, z])
        # And the view sweeps across the walls instead of down the room.
        sweep = np.sin(t * np.pi * 4.0 + np.pi / 2)
        target = np.array([x + 1.0, height * 0.95, ROOM[2] / 2 + sweep * ROOM[2]])
        frames.append(render(look_at(position, target), k))
    return frames
