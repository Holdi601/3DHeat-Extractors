"""
Tests for the travel-based camera mask.

The detector this replaces asked which pixels stay the same, and a cockpit does
not: needles sweep, minimaps rotate. Those pixels change constantly, so a change
detector calls them world — and they are the worst features a pose solver can be
given, because they are trackable for hundreds of frames without going anywhere.

Measured on a real Forza cockpit lap over 120 frames:

    features on road and verges    median travel 127 px
    features on dials and minimap  median travel   4 px

Two orders of magnitude apart, on exactly the quantity that separates world from
camera.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_capture.capture.flowmask import FlowMask


def scene(frames: int = 40, width: int = 320, height: int = 200):
    """A world that slides past, with a fixed 'instrument panel' painted on."""
    import cv2

    rng = np.random.default_rng(3)
    wide = np.full((height, width * 3), 30, np.uint8)
    for _ in range(300):
        x, y = int(rng.integers(0, width * 3)), int(rng.integers(0, height))
        cv2.rectangle(wide, (x, y), (x + 14, y + 14), int(rng.integers(90, 250)), -1)

    out = []
    for i in range(frames):
        frame = np.ascontiguousarray(wide[:, i * 5 : i * 5 + width]).copy()
        # A panel across the bottom third that does not move with the world.
        # Textured, because that is the point: a featureless region produces no
        # features to poison anything and needs no masking. A dashboard has
        # edges, numbers and dial rims, and those are what must be caught.
        frame[int(height * 0.68) :, :] = 40
        panel_rng = np.random.default_rng(17)
        for _ in range(90):
            px = int(panel_rng.integers(0, width - 10))
            py = int(panel_rng.integers(int(height * 0.70), height - 10))
            cv2.rectangle(frame, (px, py), (px + 8, py + 8),
                          int(panel_rng.integers(120, 255)), -1)
        angle = i * 0.12
        centre = (60, int(height * 0.85))
        tip = (int(centre[0] + 34 * np.cos(angle)), int(centre[1] - 34 * np.sin(angle)))
        cv2.circle(frame, centre, 36, 200, 2)
        cv2.line(frame, centre, tip, 240, 3)
        out.append(frame)
    return out


class TestFlowMask:
    def test_says_nothing_before_it_has_looked(self):
        mask = FlowMask()
        assert not mask.ready
        assert mask.mask((100, 100)) is None
        assert "watching" in mask.describe()

    def test_too_few_frames_teaches_it_nothing(self):
        mask = FlowMask()
        mask.observe_sequence(scene(frames=5))
        assert not mask.ready

    def test_it_finds_the_panel_and_keeps_the_world(self):
        frames = scene()
        mask = FlowMask()
        mask.observe_sequence(frames)
        assert mask.ready
        full = mask.mask(frames[0].shape[:2])
        assert full is not None
        height, width = frames[0].shape[:2]
        # A flat part of the panel, away from the needle. This is what a change
        # detector keeps and should not: it is bolted to the camera.
        assert full[int(height * 0.75), int(width * 0.75)] == 0, "the panel was not masked"
        # And the sliding world above must survive.
        assert full[int(height * 0.25), 160] == 255, "real scenery was masked"

    def test_a_large_sweeping_needle_is_a_known_limitation(self):
        # Worth stating rather than hiding. Features on a needle that sweeps
        # through a wide arc genuinely travel, so this detector can keep them.
        # On the reference recording it did not matter — real dials and minimaps
        # oscillate rather than traverse, and measured at 4 px of net travel
        # against 127 px for scenery — but a large animated gauge could fool it.
        frames = scene()
        mask = FlowMask()
        mask.observe_sequence(frames)
        assert mask.ready

    def test_a_world_with_no_panel_is_left_alone(self):
        import cv2

        rng = np.random.default_rng(9)
        wide = np.full((200, 960), 30, np.uint8)
        for _ in range(400):
            x, y = int(rng.integers(0, 960)), int(rng.integers(0, 200))
            cv2.rectangle(wide, (x, y), (x + 12, y + 12), int(rng.integers(90, 250)), -1)
        frames = [np.ascontiguousarray(wide[:, i * 5 : i * 5 + 320]) for i in range(40)]
        mask = FlowMask()
        mask.observe_sequence(frames)
        full = mask.mask(frames[0].shape[:2])
        # Either nothing masked, or a mask that keeps essentially everything.
        assert full is None or (full == 0).mean() < 0.15

    def test_it_refuses_when_nothing_travels(self):
        # A parked camera, or a menu. Masking most of the screen would stop the
        # scan rather than improve it.
        still = [np.full((120, 200), 80, np.uint8) for _ in range(40)]
        mask = FlowMask()
        mask.observe_sequence(still)
        assert mask.mask((120, 200)) is None or (mask.mask((120, 200)) == 0).mean() < 0.6

    def test_it_describes_what_it_did(self):
        mask = FlowMask()
        mask.observe_sequence(scene())
        assert "never travels" in mask.describe() or "nothing attached" in mask.describe()


class TestWhatItSaysMatchesWhatItDoes:
    """
    The reported coverage and the applied mask must be the same thing.

    They were not. `mask()` grew the blocked region by a cell in every direction
    — deliberately, because the edge of an instrument panel is a gradient — while
    the safety limit and the sentence shown to the user both measured the region
    *before* that growth. On a 24-cell grid the growth is not a rounding detail:
    a scattered 30% becomes 70%. So the scanner discarded seven tenths of every
    frame while reporting three, and the "never mask more than 60%" guard could
    not fire because it was looking at the wrong number.

    Found on night footage, where the failure it was supposed to prevent is the
    likely one: most of a dark frame holds no trackable feature at all, and a
    cell with a little noise and nothing travelling in it looks exactly like a
    dial.
    """

    def test_reported_coverage_is_the_coverage_applied(self):
        mask = FlowMask()
        mask.observe_sequence(scene())
        applied = mask.mask((200, 320))
        if applied is None:
            # Refusing is a valid outcome; then coverage must be why.
            assert mask.coverage() > 0.6 or not mask.ready
            return

        assert mask.coverage() == pytest.approx((applied == 0).mean(), abs=0.02)

    def test_the_description_quotes_the_same_number(self):
        mask = FlowMask()
        mask.observe_sequence(scene())
        if not mask.ready or mask.coverage() <= 0.001:
            return
        assert f"{mask.coverage() * 100:.0f}%" in mask.describe()

    def test_the_limit_is_applied_after_the_region_is_grown(self):
        """
        A grid under the limit before growing and over it after must be refused.
        That is the exact case the bug let through.

        Vertical bars two cells wide with two-cell gaps: half the grid as
        measured, and the whole of it once every bar gains a ring. Bars rather
        than a scatter because a scatter is now discarded as noise before it gets
        this far, which is a different guard doing a different job.
        """
        mask = FlowMask()
        mask.tracked = 1
        mask._shape = (200, 320)
        for start in range(0, mask.cells, 4):
            mask._stuck[:, start : start + 2] = 5

        assert mask.coverage() > 0.6
        assert mask.mask((200, 320)) is None
        assert "too much to believe" in mask.describe()

    def test_the_vanishing_point_is_not_mistaken_for_an_instrument(self):
        """
        Driving forwards, the focus of expansion has no flow.

        So the road straight ahead reads as "persistent but never travels" —
        exactly the signature of a dial — while being the geometry most wanted.
        What separates them is position: anything bolted to the camera reaches an
        edge of the screen, and the vanishing point does not.
        """
        mask = FlowMask()
        mask.tracked = 1
        mask._shape = (200, 320)
        middle = mask.cells // 2
        mask._stuck[middle - 2 : middle + 2, middle - 2 : middle + 2] = 9

        blocked = mask._blocked_cells()

        assert not blocked.any(), "a stuck patch in mid-frame was taken for an instrument"

    def test_a_panel_reaching_the_edge_is_still_caught(self):
        """The counterpart: the same patch, but touching the bottom edge."""
        mask = FlowMask()
        mask.tracked = 1
        mask._shape = (200, 320)
        mask._stuck[-4:, 8:16] = 9

        blocked = mask._blocked_cells()

        assert blocked.any()
        assert blocked[-1, 12]
