"""
A racetrack of exactly known width, for asking whether width survives.

Telemetry gives the line the car drove, not the road it drove on — no width, no
verges, no kerbs. So telemetry alone cannot be the answer, and it was never meant
to be: it replaces the *pose*, and the depth model supplies everything around the
camera. This scene exists to check that the second half of that claim is true.

The geometry is a ground plane with three bands running along a curving
centreline: road, kerb, verge. The road is `ROAD_WIDTH` metres across and nothing
about it is inferred — it is drawn, so the number that comes back out can be
compared with the number that went in.

Depth here is analytic rather than predicted, exactly as in `scene.py`, and for
the same reason: it isolates the geometry from the network. How well a real depth
model does on real gameplay is a separate question that no synthetic scene can
answer. What this can answer is whether exact poses plus per-pixel depth
reconstruct a road of the right width — which is the part the design rests on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from heat3d_capture.pose.odometry import Intrinsics, Pose

from .scene import Frame, _texture

#: Metres across, kerb to kerb. The number the tests check for.
ROAD_WIDTH = 12.0
#: A kerb each side, then open verge.
KERB_WIDTH = 1.5
#: How far from the centreline the ground is modelled at all.
GROUND_HALF = 40.0

#: The lap: a closed oval, so a driven line can differ from the centreline the
#: way a real one does.
LAP_RADIUS = 60.0
LAP_SQUASH = 0.55


def centreline(s: np.ndarray) -> np.ndarray:
    """Points along the track's centreline, `s` in turns."""
    angle = 2.0 * np.pi * np.asarray(s)
    return np.stack(
        [
            LAP_RADIUS * np.cos(angle),
            np.zeros_like(angle),
            LAP_RADIUS * LAP_SQUASH * np.sin(angle),
        ],
        axis=-1,
    )


def tangent(s: np.ndarray) -> np.ndarray:
    angle = 2.0 * np.pi * np.asarray(s)
    out = np.stack(
        [
            -LAP_RADIUS * np.sin(angle),
            np.zeros_like(angle),
            LAP_RADIUS * LAP_SQUASH * np.cos(angle),
        ],
        axis=-1,
    )
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def _nearest_on_lap(points: np.ndarray, samples: int = 720):
    """
    Signed distance from the centreline, and distance along it.

    By brute force against a fine sampling of the lap, because the closed form
    for an ellipse is a quartic and this is a test fixture, not a renderer.
    """
    s = np.linspace(0.0, 1.0, samples, endpoint=False)
    line = centreline(s)
    flat = points.reshape(-1, 3)
    # (N, samples) distances in the ground plane.
    delta = flat[:, None, [0, 2]] - line[None, :, [0, 2]]
    dist = np.einsum("nsi,nsi->ns", delta, delta)
    nearest = np.argmin(dist, axis=1)

    along = s[nearest]
    to_point = flat[:, [0, 2]] - line[nearest][:, [0, 2]]
    direction = tangent(along)[:, [0, 2]]
    # Left is positive: rotate the tangent ninety degrees in the ground plane.
    normal = np.stack([-direction[:, 1], direction[:, 0]], axis=-1)
    offset = np.einsum("ni,ni->n", to_point, normal)
    return offset.reshape(points.shape[:-1]), along.reshape(points.shape[:-1])


def render(pose: Pose, k: Intrinsics) -> Frame:
    """
    Render one view by intersecting every pixel ray with the ground plane.

    The world is flat, so each ray hits y = 0 once or never — which is all a
    road needs, and keeps the fixture small enough to read.
    """
    ys, xs = np.mgrid[0 : k.height, 0 : k.width]
    directions = np.stack(
        [(xs - k.cx) / k.fx, (ys - k.cy) / k.fy, np.ones_like(xs, dtype=np.float64)],
        axis=-1,
    )
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    world = directions @ pose.rotation.T
    origin = pose.translation

    with np.errstate(divide="ignore", invalid="ignore"):
        t = -origin[1] / world[..., 1]
    hit = np.isfinite(t) & (t > 1e-6)
    t = np.where(hit, t, np.inf)
    point = origin + world * np.where(np.isfinite(t), t, 0.0)[..., None]

    offset, along = _nearest_on_lap(point)
    beyond = np.abs(offset) > GROUND_HALF
    hit &= ~beyond

    half = ROAD_WIDTH / 2.0
    on_road = np.abs(offset) <= half
    on_kerb = (~on_road) & (np.abs(offset) <= half + KERB_WIDTH)

    # Texture coordinates that run along and across, so features move with the
    # camera the way markings and grass do rather than sliding.
    u = along * 40.0
    v = offset / 4.0
    image = np.zeros((k.height, k.width, 3), dtype=np.uint8)
    image[on_road] = _texture(u, v, seed=1)[on_road]
    image[on_kerb] = _texture(u * 3.0, v * 3.0, seed=5)[on_kerb]
    verge = hit & ~on_road & ~on_kerb
    image[verge] = _texture(u * 0.5, v * 0.5, seed=9)[verge]

    forward = pose.rotation[:, 2]
    depth = ((point - origin) @ forward).astype(np.float32)
    depth[~hit] = np.inf
    return Frame(image=image, depth=depth, pose=pose)


def drive(
    k: Intrinsics,
    *,
    steps: int = 48,
    height: float = 1.2,
    line_offset: float = 3.0,
    laps: float = 1.0,
) -> list[Frame]:
    """
    One lap, driven off-centre on purpose.

    `line_offset` puts the camera three metres from the centreline, because that
    is the whole point of the question this fixture answers: the driven line is
    not the road, and the road still has to come out the right width.
    """
    frames = []
    for s in np.linspace(0.0, laps, steps, endpoint=False):
        base = centreline(np.array(s))
        direction = tangent(np.array(s))
        left = np.array([-direction[2], 0.0, direction[0]])
        position = base + left * line_offset + np.array([0.0, height, 0.0])

        forward = direction
        right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        rotation = np.stack([right, up, forward], axis=1)
        frames.append(render(Pose(rotation=rotation, translation=position), k))
    return frames


@dataclass
class ExactPoses:
    """
    Stands in for a telemetry feed: the true pose at any moment.

    The interface `Reconstructor` expects of a pose source is one method, so a
    test can supply truth where a game would supply its own measurements.
    """

    frames: list[Frame]

    def at(self, t: float):
        index = int(round(t))
        if 0 <= index < len(self.frames):
            return self.frames[index].pose
        return None
