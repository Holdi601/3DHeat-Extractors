"""
What has been seen well, and what has not.

This is the part that answers "where should I go back and look again", and it is
worth more than it first appears. Reconstruction quality is not uniform: a wall
you walked straight past once, at twenty metres, in a single glance, reconstructs
badly — and it reconstructs badly *quietly*, producing a surface that looks like
geometry rather than an obvious hole. Finding those places after the scan, in the
mesh, is hard. Finding them during the scan, while the user is still standing in
the level, is easy and worth doing.

What makes an observation good
------------------------------
Four things, and they are not the same thing:

- **How many times** a place was seen. Once is a guess; a depth network's error
  on a single view is unbounded and has nothing to average against.
- **From how many directions.** This is the one people expect least and it
  matters most. Twenty frames taken while walking straight at a wall are twenty
  observations from *one* direction, and they cannot resolve its depth any better
  than one can — there is no parallax between them. Two views thirty degrees
  apart are worth more than fifty views from the same spot.
- **From how far.** Depth error grows with distance roughly as the square, so a
  surface seen only from across a courtyard is far less certain than the same
  surface seen from five metres, however many times it was seen.
- **How well the camera was located.** Geometry fused with a badly tracked pose
  lands in the wrong place, and that is worse than not fusing it at all.

Held in a sparse voxel grid keyed by position, so memory follows the size of what
was actually walked rather than the size of the level's bounding box. An outdoor
map is mostly empty space and a dense grid over it would be almost entirely
zeros.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .voxelindex import VoxelIndex, grow, pack

#: Directions are accumulated as a summed unit vector. The length of that sum,
#: divided by the count, is 1.0 when every view came from an identical direction
#: and falls toward 0 as they spread out — a standard directional-spread measure
#: that costs three floats per voxel instead of a histogram.
#:
#: Below this, the views are considered to have enough spread to triangulate.
GOOD_SPREAD = 0.93

#: Observations at or beyond this distance (world units) carry little weight,
#: since monocular depth error grows roughly with the square of distance.
FAR_DISTANCE = 40.0

#: Fewer views than this and a place has not really been measured.
MIN_VIEWS = 3


@dataclass
class WeakSpot:
    """A place worth going back to, and why."""

    #: Centre in world units.
    position: np.ndarray
    #: How many voxels of this weakness are clustered here.
    extent: int
    #: 0..1, where 0 is unobserved and 1 is thoroughly measured.
    quality: float
    reason: str

    @property
    def advice(self) -> str:
        return {
            "few views": "Seen only once or twice — walk past it again.",
            "one direction": "Only seen from one angle. Step sideways and look at it again.",
            "distant": "Only seen from far away. Get closer to it.",
            "poor tracking": "The camera was not well located here. Approach it more slowly.",
        }.get(self.reason, "Worth another look.")


class CoverageGrid:
    """
    A sparse voxel record of how well each part of the level was observed.

    Deliberately separate from the surface reconstruction. It answers a different
    question — not "what shape is this" but "how much should I believe the shape
    I have" — and it has to be cheap enough to update on every keyframe while a
    scan is running, which a mesh is not.
    """

    def __init__(self, *, voxel: float = 0.5, capacity: int = 1 << 16):
        if voxel <= 0:
            raise ValueError("voxel size must be positive")
        self.voxel = float(voxel)
        self._voxels = VoxelIndex()
        self._count = np.zeros(capacity, dtype=np.int32)
        # Summed unit view directions, for the spread measure.
        self._direction = np.zeros((capacity, 3), dtype=np.float64)
        self._nearest = np.full(capacity, np.inf, dtype=np.float32)
        self._confidence = np.zeros(capacity, dtype=np.float32)
        self._used = 0

    def __len__(self) -> int:
        return self._used

    @property
    def voxel_count(self) -> int:
        return self._used

    def _grow(self, needed: int) -> None:
        if needed <= len(self._count):
            return
        size = max(needed, len(self._count) * 2)
        self._count = grow(self._count, size)
        self._direction = grow(self._direction, size)
        self._nearest = grow(self._nearest, size, fill=np.inf)
        self._confidence = grow(self._confidence, size)

    def observe(
        self,
        points: np.ndarray,
        camera_position: np.ndarray,
        *,
        confidence: float = 1.0,
    ) -> int:
        """
        Record that a camera at `camera_position` saw these world points.

        Returns the number of voxels touched. Points that are not finite are
        dropped rather than rejected — a depth map legitimately contains
        infinities where it predicted sky, and those are not an error.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        camera_position = np.asarray(camera_position, dtype=np.float64).reshape(3)
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if points.size == 0:
            return 0

        offsets = points - camera_position
        distances = np.linalg.norm(offsets, axis=1)
        usable = distances > 1e-6
        points, offsets, distances = points[usable], offsets[usable], distances[usable]
        if points.size == 0:
            return 0
        directions = offsets / distances[:, None]

        keys = np.floor(points / self.voxel).astype(np.int64)
        # Group the frame's points by voxel first, so a voxel hit by ten thousand
        # pixels of one surface counts as one observation from one direction
        # rather than ten thousand — otherwise a close-up wall would swamp the
        # spread measure with a single viewpoint.
        # Packed to int64 and made unique in one dimension, which is the whole
        # performance story here. `np.unique(keys, axis=0)` looks like the
        # natural call and builds a structured view to sort rows: measured on a
        # real keyframe's 129,600 points it takes **114 ms**, against **8 ms**
        # for the same answer via a packed scalar key. It appeared in both fusion
        # grids, so it alone was most of the 212 ms a keyframe spent fusing —
        # and that is what held the scan to four keyframes a second and, at
        # driving speed, pulled consecutive keyframes far enough apart that
        # feature matching failed. A throughput bug wearing a tracking bug's
        # clothes.
        unique, inverse = np.unique(pack(keys), return_inverse=True)
        count = len(unique)
        # `bincount` per column rather than `np.add.at`, which is unbuffered and
        # four times slower for the same sum.
        mean_direction = np.stack(
            [np.bincount(inverse, weights=directions[:, c], minlength=count) for c in range(3)],
            axis=1,
        )
        per_voxel = np.bincount(inverse, minlength=count)
        mean_direction /= per_voxel[:, None]
        lengths = np.linalg.norm(mean_direction, axis=1)
        mean_direction = np.divide(
            mean_direction, lengths[:, None], out=np.zeros_like(mean_direction), where=lengths[:, None] > 1e-9
        )
        nearest = np.full(count, np.inf)
        np.minimum.at(nearest, inverse, distances)

        # One vectorised lookup, not a Python loop over tens of thousands of
        # voxels. The loop version cost 117 ms per keyframe against 38 ms for
        # the depth network, and starved the scan badly enough to break tracking.
        rows = self._voxels.add(unique)
        self._grow(len(self._voxels))
        self._used = len(self._voxels)

        self._count[rows] += 1
        self._direction[rows] += mean_direction
        self._nearest[rows] = np.minimum(self._nearest[rows], nearest.astype(np.float32))
        # Kept as a running maximum: one good look at a place is what makes its
        # geometry trustworthy, and later sloppy glances should not erase it.
        self._confidence[rows] = np.maximum(self._confidence[rows], np.float32(confidence))
        return len(unique)

    # -- reading it back -------------------------------------------------

    def _snapshot(self) -> int:
        """
        How many voxels every reader in this call should agree to see.

        The scan runs on a worker thread and the interface reads these arrays
        from the GUI thread, without a lock. That is safe only because rows are
        *appended* and never moved or removed, so any prefix is a consistent
        past state — but only if each reader picks one length and slices
        everything to it.

        Reading the index and the per-voxel arrays separately is what broke:
        `observe` grows the key array and sets `_used` a line later, so an
        interface refresh landing between the two saw N+k positions and N
        qualities, and the coverage map died with "array is not broadcastable to
        correct shape" mid-scan. The overlay then showed nothing for the rest of
        the run, which is precisely when someone needs it.
        """
        return min(self._used, len(self._voxels))

    def positions(self) -> np.ndarray:
        """(N,3) centres of every observed voxel, in row order."""
        from .voxelindex import unpack

        used = self._snapshot()
        if used == 0:
            return np.zeros((0, 3))
        return (unpack(self._voxels.keys[:used]) + 0.5) * self.voxel

    def spread(self) -> np.ndarray:
        """
        0..1 per voxel: how concentrated the viewing directions were.

        1.0 means every view came from the same direction and there is no
        parallax to work with; lower is better.
        """
        used = self._snapshot()
        count = np.maximum(self._count[:used], 1)
        return (np.linalg.norm(self._direction[:used], axis=1) / count).astype(np.float32)

    def quality(self) -> np.ndarray:
        """
        0..1 per voxel, combining every term. Higher is better observed.

        Multiplied rather than averaged, because these are not independent
        opinions to be blended — they are conditions that each have to hold. A
        place seen two hundred times from one direction at sixty metres is not
        two-thirds well observed; it is badly observed, and an average would hide
        that behind one excellent term.
        """
        used = self._snapshot()
        if used == 0:
            return np.zeros(0, dtype=np.float32)
        views = np.clip(self._count[:used] / MIN_VIEWS, 0, 1)
        # `spread` takes its own snapshot, which may be one keyframe newer.
        # Trimmed here so the terms multiply element-wise whatever happened in
        # between.
        angles = np.clip((1.0 - self.spread()[:used]) / (1.0 - GOOD_SPREAD), 0, 1)
        nearness = np.clip(1.0 - self._nearest[:used] / FAR_DISTANCE, 0, 1)
        tracking = self._confidence[:used]
        shortest = min(len(views), len(angles), len(nearness), len(tracking))
        return (
            views[:shortest]
            * angles[:shortest]
            * nearness[:shortest]
            * tracking[:shortest]
        ).astype(np.float32)

    def reconstructable(self) -> tuple[bool, str]:
        """
        Whether the scan is accumulating usable evidence, and what is wrong.

        Deliberately *not* phrased as "this footage cannot be reconstructed",
        which is what it said first and was wrong. Low measured parallax has two
        very different causes and this cannot tell them apart:

        - the capture genuinely has none, as when a camera faces the way it
          travels and never looks aside; or
        - the poses are drifting, so the same surface lands in a different voxel
          every frame and no voxel ever accumulates a second viewpoint.

        The second is what was actually happening on the reference recording.
        Instrument-panel features were dominating the pose solve, every pose came
        back saying the camera had barely moved, and this metric dutifully
        reported no parallax — of a clip whose optical flow shows a textbook
        focus-of-expansion field. The measurement was a symptom being read as a
        diagnosis, and it sent a day of work in the wrong direction.

        So it reports a problem and names both possibilities, rather than
        declaring the footage unusable.
        """
        if self._used < 500:
            return True, ""
        parallax = float((self.spread() < GOOD_SPREAD).mean())
        near = float(np.median(self._nearest[: self._used]))
        if parallax < 0.02:
            return False, (
                f"Almost no parallax ({parallax * 100:.1f}%). Either the camera never "
                "looks aside from the way it travels, or the tracking is drifting so "
                "the same surface never lands twice in the same place. Check the "
                "placed-frame count: if that is high, it is drift."
            )
        if near > 25.0:
            return False, (
                f"Everything was seen from {near:.0f} m away or further. Depth error "
                "grows with the square of distance, so this will reconstruct as noise. "
                "Get closer to what you want."
            )
        if parallax < 0.06:
            return True, (
                f"Thin parallax ({parallax * 100:.1f}%) — the geometry will be rough. "
                "Moving sideways relative to what you are looking at helps most."
            )
        return True, ""

    def summary(self) -> dict:
        """Headline numbers for the interface."""
        quality = self.quality()
        if quality.size == 0:
            return {"voxels": 0, "well_observed": 0.0, "mean_quality": 0.0, "volume": 0.0}
        return {
            "voxels": int(self._used),
            "well_observed": float((quality >= 0.5).mean()),
            "mean_quality": float(quality.mean()),
            "volume": float(self._used * self.voxel**3),
        }

    def weak_spots(
        self, *, threshold: float = 0.35, min_cluster: int = 8, limit: int = 8
    ) -> list[WeakSpot]:
        """
        Places worth revisiting, largest first.

        Grouped by connectivity, over the occupied voxels only.

        The first version rasterised the whole bounding box and ran
        `scipy.ndimage.label` over it, which is the obvious way and is wrong for
        this data. A walked route touches a tiny, stringy fraction of the box it
        spans, and driving makes that ratio absurd: measured here, 600 m of
        driving left 196k occupied voxels inside a 6.8M-cell box, and the label
        pass took **13.5 seconds**. It ran on the interface thread every 900 ms.
        The window did not respond to anything, including its own stop button.

        Union-find over a hash of the occupied voxels is O(weak voxels x 26) and
        does not know how far apart they are. Same answer, 30 m or 600 m: at the
        scale that took 13.5 seconds it now takes tens of milliseconds.
        """
        quality = self.quality()
        if quality.size == 0:
            return []
        weak = quality < threshold
        if not weak.any():
            return []

        positions = self.positions()[weak]
        reasons = self._reasons()[weak]
        weak_quality = quality[weak]

        keys = np.floor(positions / self.voxel).astype(np.int64)
        # A sorted packed-key array instead of a dict of tuples: the neighbour
        # search below is then 26 vectorised lookups rather than 26 dict probes
        # per voxel.
        packed = pack(keys)
        order = np.argsort(packed, kind="stable")
        sorted_keys = packed[order]

        # Union-find with path compression. Iterative rather than recursive: a
        # long thin cluster — which is exactly what a driven route produces — can
        # be tens of thousands of voxels deep and would blow the stack.
        parent = np.arange(len(keys))

        def find(i: int) -> int:
            root = i
            while parent[root] != root:
                root = parent[root]
            while parent[i] != root:
                parent[i], i = root, parent[i]
            return root

        # Half the 26-neighbourhood: every pair is considered once, from the
        # lower index, so the other half is redundant work on identical answers.
        offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if (dx, dy, dz) > (0, 0, 0)
        ]
        for dx, dy, dz in offsets:
            neighbour = pack(keys + np.array([dx, dy, dz], dtype=np.int64))
            position = np.searchsorted(sorted_keys, neighbour)
            position = np.minimum(position, sorted_keys.size - 1)
            hit = sorted_keys[position] == neighbour
            for i, j in zip(np.flatnonzero(hit), order[position[hit]]):
                a, b = find(int(i)), find(int(j))
                if a != b:
                    parent[a] = b

        groups: dict[int, list[int]] = {}
        for i in range(len(keys)):
            groups.setdefault(find(i), []).append(i)

        spots: list[WeakSpot] = []
        for members in groups.values():
            if len(members) < min_cluster:
                continue
            index = np.array(members)
            values, counts = np.unique(reasons[index], return_counts=True)
            spots.append(
                WeakSpot(
                    position=positions[index].mean(axis=0),
                    extent=len(members),
                    quality=float(weak_quality[index].mean()),
                    reason=str(values[counts.argmax()]),
                )
            )
        spots.sort(key=lambda s: (-s.extent, s.quality))
        return spots[:limit]

    def _reasons(self) -> np.ndarray:
        """The single largest problem per voxel, as a label."""
        used = self._snapshot()
        views = np.clip(self._count[:used] / MIN_VIEWS, 0, 1)
        angles = np.clip((1.0 - self.spread()) / (1.0 - GOOD_SPREAD), 0, 1)
        nearness = np.clip(1.0 - self._nearest[:used] / FAR_DISTANCE, 0, 1)
        tracking = self._confidence[:used]
        terms = np.stack([views, angles, nearness, tracking])
        labels = np.array(["few views", "one direction", "distant", "poor tracking"])
        return labels[terms.argmin(axis=0)]

    def top_down(self, *, resolution: int = 220) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        """
        A plan view of observation quality, for the interface to draw.

        Collapsed on the vertical axis by taking the *best* quality in each
        column rather than the mean. Someone standing in a room has poor coverage
        of the ceiling above them and excellent coverage of the floor; averaging
        those paints the room amber and sends them back to a place that is
        already fine.

        Returns the map and its (min_x, max_x, min_z, max_z) extent.
        """
        positions = self.positions()
        quality = self.quality()
        # Belt as well as braces. Both took a snapshot, but they took two, and a
        # keyframe can land between them — so pair them explicitly rather than
        # letting a mismatch reach `np.maximum.at`, which reports it as an
        # unhelpful broadcasting error from deep inside NumPy.
        usable = min(len(positions), len(quality))
        positions, quality = positions[:usable], quality[:usable]
        if usable == 0:
            return np.zeros((resolution, resolution), dtype=np.float32), (0.0, 1.0, 0.0, 1.0)

        x, z = positions[:, 0], positions[:, 2]
        min_x, max_x = float(x.min()), float(x.max())
        min_z, max_z = float(z.min()), float(z.max())
        span_x = max(max_x - min_x, 1e-6)
        span_z = max(max_z - min_z, 1e-6)

        col = np.clip(((x - min_x) / span_x * (resolution - 1)).astype(int), 0, resolution - 1)
        row = np.clip(((z - min_z) / span_z * (resolution - 1)).astype(int), 0, resolution - 1)
        grid = np.full((resolution, resolution), -1.0, dtype=np.float32)
        np.maximum.at(grid, (row, col), quality)
        return grid, (min_x, max_x, min_z, max_z)
