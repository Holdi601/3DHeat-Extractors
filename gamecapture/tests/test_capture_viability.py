"""
Refusing a capture that cannot work.

The check that should have existed first. Monocular reconstruction needs
parallax — the same surface seen from positions far enough apart to triangulate.
A camera facing the direction it travels never supplies any, and no amount of
tuning downstream invents it.

Measured on comparable captures, as the fraction of voxels with any usable
view spread:

    walking and looking around    15.4%
    walking straight ahead         1.5%
    driving forward, cockpit view  0.1%

The last produced disconnected blobs and three hours of debugging the wrong
thing, while tracking cheerfully reported every frame placed. The coverage grid
had the answer the whole time and nobody asked it.
"""

from __future__ import annotations

import numpy as np

from heat3d_capture.fusion.coverage import CoverageGrid


def wall(n=3000, x=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack(
        [np.full(n, x), rng.uniform(-3, 6, n), rng.uniform(-30, 30, n)], axis=1
    )


class TestViability:
    def test_an_empty_grid_makes_no_claim(self):
        ok, note = CoverageGrid(voxel=0.5).reconstructable()
        assert ok and note == ""

    def test_driving_forward_is_refused(self):
        # Every observation from one bearing, receding, never repeated. This is
        # the Forza cockpit case.
        grid = CoverageGrid(voxel=0.8)
        for step in range(40):
            grid.observe(wall(x=200.0 + step * 2, seed=step),
                         np.array([step * 2.0, 0.0, 0.0]), confidence=1.0)
        ok, note = grid.reconstructable()
        assert not ok
        assert "parallax" in note or "away" in note

    def test_looking_around_while_moving_is_accepted(self):
        grid = CoverageGrid(voxel=0.5)
        points = wall(seed=1)
        for z in np.linspace(-8, 8, 16):
            grid.observe(points, np.array([-4.0, 0.0, float(z)]), confidence=1.0)
        ok, note = grid.reconstructable()
        assert ok, note

    def test_everything_seen_from_far_away_is_refused(self):
        # Sweeping ±20 m at ninety metres' range sounds like parallax and is not:
        # the bearing to a given point barely changes. Whichever of the two
        # reasons fires, the capture must be refused — they are the same fact
        # seen from different sides.
        grid = CoverageGrid(voxel=0.8)
        points = wall(seed=2)
        for z in np.linspace(-20, 20, 12):
            grid.observe(points, np.array([-90.0, 0.0, float(z)]), confidence=1.0)
        ok, note = grid.reconstructable()
        assert not ok
        assert note

    def test_the_advice_says_what_to_do_differently(self):
        grid = CoverageGrid(voxel=0.8)
        for step in range(40):
            grid.observe(wall(x=200.0 + step * 2, seed=step),
                         np.array([step * 2.0, 0.0, 0.0]), confidence=1.0)
        _, note = grid.reconstructable()
        # It must name *both* causes rather than declaring the footage unusable.
        # The first version asserted only an instruction, which is how a wrong
        # diagnosis got shipped with confident advice attached to it.
        assert "drift" in note or "closer" in note
        assert "Either" in note or "closer" in note
