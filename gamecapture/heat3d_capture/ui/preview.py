"""
Turning scan state into pictures.

Kept out of the window so it can be tested without a display, and because these
are the parts where a *working* scan can be made to look broken. A preview that
misleads is worse than no preview: someone watching a good scan render as a mess
will stop and start again, and someone watching a bad scan render as fine will
keep walking.
"""

from __future__ import annotations

import numpy as np

#: Colour for places never observed at all. Deliberately not black and not a ramp
#: colour: "no data" has to be visibly a different *kind* of thing from "data
#: that is bad", or an unexplored corner looks identical to a badly-scanned one.
UNSEEN = (26, 28, 33)

#: The coverage ramp, worst to best. Red/amber/green because this is an
#: instruction — go back / maybe go back / leave it alone — not a measurement to
#: be read off precisely.
COVERAGE_STOPS = np.array(
    [
        [201, 42, 42],  # bad
        [214, 132, 25],  # thin
        [212, 196, 40],  # acceptable
        [63, 185, 80],  # good
    ],
    dtype=np.float32,
)


def depth_to_preview(depth: np.ndarray) -> np.ndarray:
    """
    Colour a depth map for display.

    Normalised per frame against percentiles rather than min and max: a single
    sky pixel at infinity, or one speck of noise, would otherwise flatten the
    entire visible range into the bottom of the ramp and make a working depth map
    look broken.
    """
    import cv2

    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    low, high = np.percentile(finite, (2, 98))
    if high - low < 1e-9:
        high = low + 1e-9
    normalised = np.clip((depth - low) / (high - low), 0, 1)
    eight_bit = (normalised * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(eight_bit, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def ramp(values: np.ndarray) -> np.ndarray:
    """Map 0..1 onto the coverage ramp, linearly between stops."""
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    scaled = values * (len(COVERAGE_STOPS) - 1)
    low = np.floor(scaled).astype(int)
    high = np.minimum(low + 1, len(COVERAGE_STOPS) - 1)
    t = (scaled - low)[..., None]
    return (COVERAGE_STOPS[low] * (1 - t) + COVERAGE_STOPS[high] * t).astype(np.uint8)


def coverage_map(
    grid: np.ndarray,
    *,
    trajectory: np.ndarray | None = None,
    extent: tuple[float, float, float, float] | None = None,
    size: int = 320,
) -> np.ndarray:
    """
    Draw the plan view: where has been seen, and how well.

    `grid` comes from `CoverageGrid.top_down()`, where negative means never
    observed. The walked path is drawn over it so the user can relate the map to
    what they remember doing — a coverage map with no path on it is an abstract
    blob, and the same map with the route drawn is immediately legible.
    """
    import cv2

    if grid.size == 0:
        return np.full((size, size, 3), UNSEEN, dtype=np.uint8)

    seen = grid >= 0
    image = np.full((*grid.shape, 3), UNSEEN, dtype=np.uint8)
    if seen.any():
        image[seen] = ramp(grid[seen])

    # Thickened before scaling. A wall is one voxel deep, so in plan view it is a
    # single-pixel line that nearest-neighbour scaling turns into a dotted trail
    # with more gap than line — legible as "sparse noise" rather than as "a wall
    # I have scanned". Dilation is a display choice only; nothing downstream sees
    # it, and it never invents coverage where a column was genuinely unobserved
    # beyond one cell of bleed.
    if seen.any():
        thick = cv2.dilate(image, np.ones((2, 2), np.uint8))
        image = np.where(cv2.dilate(seen.astype(np.uint8), np.ones((2, 2), np.uint8))[..., None] > 0, thick, image)

    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)

    if trajectory is not None and len(trajectory) >= 2 and extent is not None:
        min_x, max_x, min_z, max_z = extent
        span_x = max(max_x - min_x, 1e-6)
        span_z = max(max_z - min_z, 1e-6)
        xs = np.clip((trajectory[:, 0] - min_x) / span_x, 0, 1) * (size - 1)
        zs = np.clip((trajectory[:, 2] - min_z) / span_z, 0, 1) * (size - 1)
        points = np.stack([xs, zs], axis=1).astype(np.int32)
        cv2.polylines(image, [points], False, (235, 240, 245), 1, cv2.LINE_AA)
        # Where the camera is now, which is what someone looks for first.
        cv2.circle(image, tuple(points[-1]), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, tuple(points[-1]), 6, (40, 44, 52), 1, cv2.LINE_AA)
    return image


def mark_weak_spots(
    image: np.ndarray,
    spots,
    extent: tuple[float, float, float, float] | None,
) -> np.ndarray:
    """
    Ring the places worth revisiting, numbered to match the list beside them.

    Numbered rather than only circled because the list has to say *why* each one
    is weak, and "the third circle" is not something anyone can match up at a
    glance while walking.
    """
    import cv2

    if not spots or extent is None:
        return image
    size = image.shape[0]
    min_x, max_x, min_z, max_z = extent
    span_x = max(max_x - min_x, 1e-6)
    span_z = max(max_z - min_z, 1e-6)
    out = image.copy()
    for i, spot in enumerate(spots, start=1):
        x = int(np.clip((spot.position[0] - min_x) / span_x, 0, 1) * (size - 1))
        z = int(np.clip((spot.position[2] - min_z) / span_z, 0, 1) * (size - 1))
        cv2.circle(out, (x, z), 9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            out, str(i), (x + 11, z + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
            cv2.LINE_AA,
        )
    return out


def health_text(state) -> tuple[str, str]:
    """
    A one-line verdict on the reconstruction, and a colour for it.

    Tracking health is the number that actually predicts whether a scan will be
    any good, and it is not obvious from the picture — frames are being captured
    and depth looks fine right up until the geometry turns out to be in pieces.
    """
    if state is None or state.frames == 0:
        return "Waiting for the first frame.", "#8b949e"
    health = state.tracking_health
    if health >= 0.9:
        return f"Tracking well — {state.tracked} of {state.frames} frames placed.", "#3fb950"
    if health >= 0.6:
        return (
            f"Tracking is patchy — {state.lost} of {state.frames} frames could not be placed. "
            "Turn more slowly.",
            "#d29922",
        )
    return (
        f"Losing track — only {state.tracked} of {state.frames} frames placed. "
        "Move slowly, and avoid blank walls and sky.",
        "#f85149",
    )
