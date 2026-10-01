"""
Finding what is bolted to the camera, by watching what goes nowhere.

The previous detector asked "which pixels stay the same?" and that premise is
wrong in a way that took a real recording to expose. A cockpit is not static: the
speedometer needle sweeps, the rev counter twitches, the minimap *rotates*. Those
pixels change constantly, so a change detector calls them world — and they are
the worst possible features to hand a pose solver. High contrast, corner-rich,
and trackable for hundreds of frames without ever going anywhere.

Measured on a Forza cockpit lap, tracking features for 120 frames:

    features on the road and verges   median displacement 127 px
    features on dials and minimap     median displacement   4 px

Two orders of magnitude apart, on the one quantity that matters. So the question
becomes "which features persist without travelling?", which is exactly what
belongs to the camera rather than to the world, whether it is painted on, modelled
in, or animated.

This is also why the first diagnosis of that recording was wrong. Dashboard
features dominated the pose solve, every pose came back saying the camera had
barely moved, points never registered to the same place twice, and the coverage
metric duly reported no parallax. The parallax was there the whole time — a
textbook focus-of-expansion flow field — behind a mask that was letting the
instrument panel vote.
"""

from __future__ import annotations

import numpy as np

#: Frames a feature must survive before its travel means anything.
MIN_TRACK = 20

#: Pixels of net travel, as a fraction of frame width, below which a persistent
#: feature is considered attached to the camera. World features on a moving
#: camera cross a substantial part of the frame; instruments do not.
STUCK_FRACTION = 0.02

#: Grid the verdict is accumulated on. Coarse on purpose: the decision is about
#: *regions* — a dial, a minimap, a windscreen pillar — and a per-pixel answer
#: from sparse features would be mostly holes.
CELLS = 24

#: A cell needs this many stuck features before it is masked out, so one drifting
#: track cannot delete a piece of road.
MIN_STUCK_PER_CELL = 2

#: And this many blocked neighbours out of eight, so a lone cell is noise rather
#: than an instrument. Three keeps a patchy panel whole while dropping the
#: speckle that dark footage produces in quantity.
MIN_NEIGHBOURS = 3

#: Frames per measurement window. Short on purpose: tracked over a long run almost
#: every feature eventually dies, so the evidence thins out to nothing. Measured
#: on real footage, one 400-frame pass masked **0%** of the frame while four
#: 60-frame passes over the same footage masked 22% — the same cockpit, seen or
#: missed depending only on how the looking was divided up.
WINDOW_FRAMES = 30

#: And refuse to mask more than this much, which would mean the premise failed.
MAX_COVERAGE = 0.6


class FlowMask:
    """
    Learns which screen regions hold features that never travel.

    Run over a short warm-up window rather than continuously: what is bolted to
    the camera does not move house, and re-deriving it every frame would cost
    more than it could possibly discover.
    """

    def __init__(self, *, cells: int = CELLS):
        self.cells = cells
        self._stuck = np.zeros((cells, cells), dtype=np.int32)
        self._moving = np.zeros((cells, cells), dtype=np.int32)
        self.tracked = 0
        self._shape: tuple[int, int] | None = None

    @property
    def ready(self) -> bool:
        return self.tracked > 0

    def observe_sequence(self, greys: list[np.ndarray]) -> None:
        """
        Measure travel over short overlapping windows and accumulate the verdict.

        Chunked rather than run end to end. A feature on the world leaves the
        frame within a second or two of driving, so a single long pass ends with
        almost nothing alive and almost no evidence either way — which is how a
        400-frame warm-up came to mask nothing at all while shorter passes over
        the same footage found the cockpit reliably.

        Each window is fresh: seed, track, record who travelled. The counts add
        up across windows, so more footage means a firmer answer rather than a
        thinner one.
        """
        step = max(1, WINDOW_FRAMES // 2)
        for start in range(0, max(1, len(greys) - MIN_TRACK + 1), step):
            chunk = greys[start : start + WINDOW_FRAMES]
            if len(chunk) >= MIN_TRACK:
                self._observe_window(chunk)

    def _observe_window(self, greys: list[np.ndarray]) -> None:
        """
        One window: seed features, follow them, record how far each travelled.

        Lucas-Kanade with a backward check, because a tracker that silently
        drifts onto a neighbouring pattern would manufacture exactly the
        "persistent but stationary" signature this is looking for.
        """
        import cv2

        if len(greys) < MIN_TRACK:
            return
        self._shape = greys[0].shape[:2]
        height, width = self._shape

        seeds = cv2.goodFeaturesToTrack(
            greys[0], maxCorners=800, qualityLevel=0.01, minDistance=8
        )
        if seeds is None or len(seeds) == 0:
            return

        start = seeds.reshape(-1, 2).copy()
        current = seeds
        alive = np.ones(len(start), dtype=bool)
        lk = dict(
            winSize=(21, 21),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        for i in range(1, len(greys)):
            forward, status, _ = cv2.calcOpticalFlowPyrLK(
                greys[i - 1], greys[i], current, None, **lk
            )
            back, _, _ = cv2.calcOpticalFlowPyrLK(greys[i], greys[i - 1], forward, None, **lk)
            consistent = (
                np.abs(current - back).reshape(-1, 2).max(axis=1) < 1.0
            ) & status.ravel().astype(bool)
            alive &= consistent
            current = forward

        finished = current.reshape(-1, 2)
        travelled = np.linalg.norm(finished - start, axis=1)
        stuck_limit = width * STUCK_FRACTION

        # Binned by where the feature *started*, since that is the screen region
        # the verdict is about.
        col = np.clip((start[:, 0] / width * self.cells).astype(int), 0, self.cells - 1)
        row = np.clip((start[:, 1] / height * self.cells).astype(int), 0, self.cells - 1)
        stuck = alive & (travelled < stuck_limit)
        moving = alive & (travelled >= stuck_limit)

        np.add.at(self._stuck, (row[stuck], col[stuck]), 1)
        np.add.at(self._moving, (row[moving], col[moving]), 1)
        self.tracked += int(alive.sum())

    def _blocked_cells(self) -> np.ndarray:
        """
        Cells that belong to the camera rather than to the world.

        Three steps, and each one removes a different way of being wrong.

        **Stuck features.** A cell is a candidate when its persistent features do
        not travel. Requiring a majority rather than any presence keeps a stray
        drifting track from deleting a piece of road.

        **Company.** A candidate cell has to have several blocked neighbours.
        Isolated cells are noise, and on dark footage there is a great deal of it
        — half a night frame carries no trackable feature at all, and a cell
        holding two pieces of noise with nothing travelling looks exactly like a
        dial.

        Counting neighbours rather than eroding, which was the first attempt: a
        real panel's evidence is *patchy*, because features land where the
        texture is, and erosion requires a cell's whole neighbourhood to be
        blocked. On a test panel spanning the bottom third of the frame, erosion
        cut it from 18% of cells to a 3% core that no longer reached an edge —
        and the next step below then discarded it entirely.

        **Connected to the frame edge.** This is the one that is not obvious. A
        cockpit, a pillar, a HUD corner — everything bolted to the camera reaches
        an edge of the screen. What sits in the middle and does not travel is the
        *vanishing point*: driving forwards, the focus of expansion has no flow,
        so the road ahead reads as stuck while genuinely being the geometry most
        wanted. Keeping only border-connected regions distinguishes the two by
        where they are rather than by how still they are.

        Only then is the region grown by a cell, because the edge of an
        instrument panel is a gradient and features sitting on it belong to
        neither side. The growth is large — every cell gains a ring — so it
        happens here, before anything measures the coverage, rather than inside
        `mask()` where the limit and the user's summary could not see it.
        """
        import cv2

        blocked = (self._stuck >= MIN_STUCK_PER_CELL) & (self._stuck > self._moving)
        if not blocked.any():
            return blocked

        kernel = np.ones((3, 3), np.uint8)
        candidate = blocked.astype(np.uint8)
        neighbours = cv2.filter2D(
            candidate, -1, kernel.astype(np.float32), borderType=cv2.BORDER_CONSTANT
        ) - candidate
        cleaned = ((candidate > 0) & (neighbours >= MIN_NEIGHBOURS)).astype(np.uint8)
        if not cleaned.any():
            return cleaned.astype(bool)

        count, labels = cv2.connectedComponents(cleaned, connectivity=4)
        touching = set(labels[0, :]) | set(labels[-1, :])
        touching |= set(labels[:, 0]) | set(labels[:, -1])
        touching.discard(0)
        if not touching:
            return np.zeros_like(blocked)

        edge_bound = np.isin(labels, list(touching)).astype(np.uint8)
        return cv2.dilate(edge_bound, kernel).astype(bool)

    def coverage(self) -> float:
        """Fraction of the frame the mask would actually discard."""
        return float(self._blocked_cells().mean()) if self.ready else 0.0

    def mask(self, shape: tuple[int, int] | None = None) -> np.ndarray | None:
        """
        255 where the pixel is world, 0 where it belongs to the camera.

        None while there is not enough evidence, which callers must read as "use
        everything" rather than "use nothing".
        """
        import cv2

        if not self.ready:
            return None
        blocked = self._blocked_cells()
        # Past this much, the premise has failed: whatever was measured, it is
        # not an instrument panel. Night footage is how this gets reached — most
        # of a dark frame carries no trackable feature at all, and a cell with
        # two pieces of noise in it and nothing travelling looks exactly like a
        # dial. Masking most of the level is worse than masking none of it.
        if blocked.mean() > MAX_COVERAGE:
            return None
        target = shape or self._shape
        if target is None:
            return None
        if not blocked.any():
            return np.full(target, 255, dtype=np.uint8)

        small = np.where(blocked, 0, 255).astype(np.uint8)
        return cv2.resize(small, (target[1], target[0]), interpolation=cv2.INTER_NEAREST)

    def describe(self) -> str:
        if not self.ready:
            return "watching what moves"
        share = self.coverage()
        if share <= 0.001:
            return "nothing attached to the camera"
        if share > MAX_COVERAGE:
            return (
                f"{share * 100:.0f}% of the frame looks camera-mounted, which is too "
                "much to believe — masking nothing. Dark footage does this: most of "
                "the frame holds no trackable feature, which reads the same as an "
                "instrument panel."
            )
        return f"ignoring {share * 100:.0f}% of the frame — it never travels"
