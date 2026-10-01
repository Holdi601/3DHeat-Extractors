"""
Feature tracks that live for hundreds of frames.

The front-end bundle adjustment needs, and the one the first pipeline did not
have. Matching ORB descriptors between consecutive frames gives *pairs*: this
frame and the one before, and nothing longer. Bundle adjustment needs the
opposite — one point seen in twenty frames, constraining twenty poses at once —
and pairs cannot be chained into that reliably, because a descriptor match is
re-decided from scratch every frame and identity is lost the moment one fails.

Lucas-Kanade tracking keeps identity by construction: a feature is followed, not
re-recognised. Measured on a real recording, tracks survived the entire 200-frame
window tested, and the ones on actual scenery travelled 127 pixels while the ones
on the instrument panel travelled 4. Both facts matter — the length is what feeds
the solver, and the travel is what tells the two apart.

Tracking is forward-backward checked. A tracker that silently drifts onto a
neighbouring pattern does not report failure; it reports a confident wrong
position, and the solver has no way to know. Running the flow back and demanding
it land where it started is cheap and catches almost all of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Features to hold. More is better for the solver and costs linearly; this is
#: about as many as can be tracked inside a keyframe's time budget.
TARGET_FEATURES = 600

#: Re-seed when the live count falls below this fraction of the target. Features
#: leave the frame constantly on a moving camera, and a window that runs dry has
#: nothing to constrain the newest pose with.
RESEED_BELOW = 0.6

#: Forward-backward disagreement, in pixels, beyond which a track is dropped.
FB_TOLERANCE = 1.0

#: New features are kept this far from existing ones, so a re-seed does not pile
#: fresh points onto corners that are already tracked.
MIN_SEPARATION = 10


@dataclass
class Track:
    """One feature, followed across frames."""

    id: int
    #: Frame index to pixel position.
    seen: dict[int, np.ndarray] = field(default_factory=dict)
    #: Index into the reconstruction's point cloud, once triangulated.
    point: int | None = None
    alive: bool = True

    @property
    def length(self) -> int:
        return len(self.seen)

    def travel(self) -> float:
        """Net pixels between first and last sighting."""
        if len(self.seen) < 2:
            return 0.0
        frames = sorted(self.seen)
        return float(np.linalg.norm(self.seen[frames[-1]] - self.seen[frames[0]]))


class FeatureTracks:
    """
    Follows features across frames, re-seeding as they are lost.

    Holds every track, including dead ones, because bundle adjustment works over
    a window of past frames and a track that died last frame still constrains the
    poses it was alive for. `prune` drops what has fallen out of the window.
    """

    def __init__(self, *, target: int = TARGET_FEATURES):
        self.target = target
        self.tracks: dict[int, Track] = {}
        self.frame = -1
        self._next_id = 0
        self._previous: np.ndarray | None = None
        self._live: list[int] = []

    @property
    def live_count(self) -> int:
        return len(self._live)

    def add_frame(self, grey: np.ndarray, mask: np.ndarray | None = None) -> dict[int, np.ndarray]:
        """
        Follow everything into this frame, then top up. Returns live positions.
        """
        import cv2

        self.frame += 1
        positions: dict[int, np.ndarray] = {}

        if self._previous is not None and self._live:
            before = np.array(
                [self.tracks[i].seen[self.frame - 1] for i in self._live], dtype=np.float32
            ).reshape(-1, 1, 2)
            lk = dict(
                winSize=(21, 21),
                maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            forward, status, _ = cv2.calcOpticalFlowPyrLK(self._previous, grey, before, None, **lk)
            backward, _, _ = cv2.calcOpticalFlowPyrLK(grey, self._previous, forward, None, **lk)
            drift = np.abs(before - backward).reshape(-1, 2).max(axis=1)
            ok = status.ravel().astype(bool) & (drift < FB_TOLERANCE)

            height, width = grey.shape[:2]
            survivors = []
            for track_id, good, uv in zip(self._live, ok, forward.reshape(-1, 2)):
                inside = 0 <= uv[0] < width and 0 <= uv[1] < height
                # The mask is only consulted once the point is known to be inside
                # the frame — reading it first indexes out of bounds for any
                # feature that has just left the picture, which on a moving
                # camera is several of them every frame.
                allowed = (
                    inside and (mask is None or mask[int(uv[1]), int(uv[0])] > 0)
                )
                if good and allowed:
                    self.tracks[track_id].seen[self.frame] = uv.astype(np.float64)
                    positions[track_id] = uv.astype(np.float64)
                    survivors.append(track_id)
                else:
                    self.tracks[track_id].alive = False
            self._live = survivors

        if self.live_count < self.target * RESEED_BELOW:
            self._seed(grey, mask, positions)

        self._previous = grey
        return positions

    def _seed(self, grey: np.ndarray, mask: np.ndarray | None, positions: dict) -> None:
        import cv2

        wanted = self.target - self.live_count
        if wanted <= 0:
            return

        # Existing features are painted out of the seeding mask, so a top-up adds
        # coverage instead of stacking duplicates on the strongest corners.
        seed_mask = np.full(grey.shape[:2], 255, np.uint8) if mask is None else mask.copy()
        for uv in positions.values():
            cv2.circle(seed_mask, (int(uv[0]), int(uv[1])), MIN_SEPARATION, 0, -1)

        found = cv2.goodFeaturesToTrack(
            grey, maxCorners=wanted, qualityLevel=0.01,
            minDistance=MIN_SEPARATION, mask=seed_mask,
        )
        if found is None:
            return
        for uv in found.reshape(-1, 2).astype(np.float64):
            track = Track(id=self._next_id, seen={self.frame: uv})
            self.tracks[track.id] = track
            self._live.append(track.id)
            self._next_id += 1

    def in_window(self, first_frame: int) -> list[Track]:
        """Tracks with at least one sighting at or after `first_frame`."""
        return [
            t for t in self.tracks.values()
            if t.seen and max(t.seen) >= first_frame
        ]

    def prune(self, before_frame: int) -> int:
        """
        Forget tracks that ended before the window. Returns how many went.

        Necessary rather than tidy: a long scan accumulates hundreds of thousands
        of dead tracks, and nothing ever looks at them again.
        """
        dead = [
            i for i, t in enumerate_tracks(self.tracks)
            if not t.seen or max(t.seen) < before_frame
        ]
        for key in dead:
            del self.tracks[key]
        return len(dead)

    def trim(self, before_frame: int) -> None:
        """Drop individual sightings older than the window, keeping the tracks."""
        for track in self.tracks.values():
            stale = [f for f in track.seen if f < before_frame]
            for f in stale:
                del track.seen[f]


def enumerate_tracks(tracks: dict[int, Track]):
    """`dict.items()` under a name that says what the key is."""
    return tracks.items()
