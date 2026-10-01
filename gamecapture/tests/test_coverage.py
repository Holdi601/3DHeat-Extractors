"""
Tests for coverage tracking.

This is the module the user reads during a scan to decide where to walk next, so
the tests are written around the situations that decision depends on — above all
the one that is easy to get wrong: many views from one place must *not* count as
good coverage, because they carry no parallax and cannot resolve depth.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_capture.fusion.coverage import (
    FAR_DISTANCE,
    CoverageGrid,
    WeakSpot,
)


def wall(n: int = 400, x: float = 0.0, spread: float = 4.0, seed: int = 0) -> np.ndarray:
    """A patch of surface in the YZ plane at a given X."""
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            np.full(n, x),
            rng.uniform(-spread, spread, n),
            rng.uniform(-spread, spread, n),
        ],
        axis=1,
    )


class TestObserving:
    def test_records_voxels(self):
        grid = CoverageGrid(voxel=0.5)
        touched = grid.observe(wall(), camera_position=np.array([-5.0, 0, 0]))
        assert touched > 0
        assert len(grid) == touched

    def test_the_same_surface_twice_does_not_double_the_voxels(self):
        grid = CoverageGrid(voxel=0.5)
        points = wall()
        first = grid.observe(points, np.array([-5.0, 0, 0]))
        grid.observe(points, np.array([-5.0, 0, 0]))
        assert len(grid) == first

    def test_drops_infinite_points_rather_than_failing(self):
        # A depth map legitimately contains infinities where it predicted sky.
        grid = CoverageGrid(voxel=0.5)
        points = wall(50)
        points[0] = [np.inf, 0, 0]
        points[1] = [np.nan, 0, 0]
        assert grid.observe(points, np.array([-5.0, 0, 0])) > 0

    def test_an_empty_observation_is_not_an_error(self):
        grid = CoverageGrid(voxel=0.5)
        assert grid.observe(np.zeros((0, 3)), np.array([0.0, 0, 0])) == 0
        assert grid.summary()["voxels"] == 0

    def test_a_point_at_the_camera_is_ignored(self):
        # Zero distance has no viewing direction to record.
        grid = CoverageGrid(voxel=0.5)
        assert grid.observe(np.zeros((1, 3)), np.array([0.0, 0.0, 0.0])) == 0

    def test_grows_past_its_initial_capacity(self):
        grid = CoverageGrid(voxel=0.2, capacity=16)
        grid.observe(wall(3000, spread=8.0), np.array([-5.0, 0, 0]))
        assert len(grid) > 16
        assert np.isfinite(grid.quality()).all()

    def test_rejects_a_nonsense_voxel_size(self):
        with pytest.raises(ValueError, match="positive"):
            CoverageGrid(voxel=0)


class TestSpread:
    def test_many_views_from_one_place_have_no_spread(self):
        # The heart of it. Twenty frames walking straight at a wall are twenty
        # observations from one direction and cannot resolve its depth any
        # better than one can.
        grid = CoverageGrid(voxel=0.5)
        points = wall()
        for _ in range(20):
            grid.observe(points, np.array([-10.0, 0, 0]))
        assert grid.spread().mean() > 0.99

    def test_views_from_different_sides_do_have_spread(self):
        grid = CoverageGrid(voxel=0.5)
        points = wall()
        for camera in ([-10.0, 0, 0], [-8.0, 0, 6.0], [-8.0, 0, -6.0], [-7.0, 5.0, 0]):
            grid.observe(points, np.array(camera))
        assert grid.spread().mean() < 0.95

    def test_repetition_alone_does_not_make_coverage_good(self):
        # The failure this module exists to prevent: standing still and being
        # told the level is thoroughly scanned.
        stationary = CoverageGrid(voxel=0.5)
        moving = CoverageGrid(voxel=0.5)
        points = wall()
        for _ in range(30):
            stationary.observe(points, np.array([-6.0, 0, 0]))
        for angle in np.linspace(-1.0, 1.0, 6):
            moving.observe(points, np.array([-6.0, 0, 6.0 * angle]))
        assert moving.quality().mean() > stationary.quality().mean()


class TestQuality:
    def test_a_single_glance_scores_low(self):
        grid = CoverageGrid(voxel=0.5)
        grid.observe(wall(), np.array([-6.0, 0, 0]))
        assert grid.quality().mean() < 0.5

    def test_a_close_thorough_look_scores_high(self):
        grid = CoverageGrid(voxel=0.5)
        points = wall()
        for z in (-4.0, -2.0, 0.0, 2.0, 4.0):
            grid.observe(points, np.array([-3.0, 0.0, z]), confidence=1.0)
        assert grid.quality().mean() > 0.5

    def test_distance_costs_quality(self):
        near, far = CoverageGrid(voxel=0.5), CoverageGrid(voxel=0.5)
        points = wall()
        for z in (-4.0, 0.0, 4.0):
            near.observe(points, np.array([-4.0, 0.0, z]))
            far.observe(points, np.array([-(FAR_DISTANCE * 0.9), 0.0, z]))
        assert near.quality().mean() > far.quality().mean()

    def test_bad_tracking_costs_quality(self):
        good, bad = CoverageGrid(voxel=0.5), CoverageGrid(voxel=0.5)
        points = wall()
        for z in (-4.0, 0.0, 4.0):
            good.observe(points, np.array([-4.0, 0.0, z]), confidence=1.0)
            bad.observe(points, np.array([-4.0, 0.0, z]), confidence=0.2)
        assert good.quality().mean() > bad.quality().mean()

    def test_terms_multiply_so_one_failing_condition_dominates(self):
        # Seen hundreds of times, from one direction, from far away. An average
        # would call that acceptable; it is not.
        grid = CoverageGrid(voxel=0.5)
        points = wall()
        for _ in range(200):
            grid.observe(points, np.array([-(FAR_DISTANCE * 0.95), 0, 0]))
        assert grid.quality().mean() < 0.15

    def test_quality_is_bounded(self):
        grid = CoverageGrid(voxel=0.5)
        points = wall()
        for z in np.linspace(-8, 8, 40):
            grid.observe(points, np.array([-2.0, 0.0, float(z)]), confidence=1.0)
        q = grid.quality()
        assert q.min() >= 0.0 and q.max() <= 1.0


class TestWeakSpots:
    def test_nothing_observed_means_nothing_to_report(self):
        assert CoverageGrid(voxel=0.5).weak_spots() == []

    def test_a_thoroughly_scanned_area_reports_nothing(self):
        grid = CoverageGrid(voxel=0.5)
        points = wall()
        for z in np.linspace(-6, 6, 12):
            grid.observe(points, np.array([-2.5, 0.0, float(z)]), confidence=1.0)
        assert grid.weak_spots() == []

    def test_finds_the_place_that_was_glanced_at(self):
        grid = CoverageGrid(voxel=0.5)
        # A well-scanned wall...
        good = wall(600, x=0.0, seed=1)
        for z in np.linspace(-5, 5, 10):
            grid.observe(good, np.array([-2.5, 0.0, float(z)]), confidence=1.0)
        # ...and one glimpsed once, far away, off to the side.
        neglected = wall(600, x=30.0, seed=2)
        grid.observe(neglected, np.array([-2.0, 0.0, 0.0]), confidence=1.0)

        spots = grid.weak_spots()
        assert spots, "the neglected wall should have been reported"
        assert spots[0].position[0] > 20.0, "it should point at the far wall, not the near one"

    def test_explains_itself_in_words_someone_can_act_on(self):
        grid = CoverageGrid(voxel=0.5)
        points = wall(800)
        for _ in range(6):
            grid.observe(points, np.array([-6.0, 0, 0]), confidence=1.0)
        spots = grid.weak_spots()
        assert spots
        assert spots[0].reason == "one direction"
        assert "sideways" in spots[0].advice

    def test_small_scattered_noise_is_not_reported(self):
        # Four hundred individual bad voxels is not an actionable list.
        grid = CoverageGrid(voxel=0.5)
        grid.observe(wall(4), np.array([-30.0, 0, 0]))
        assert grid.weak_spots(min_cluster=8) == []

    def test_is_limited_and_ordered_by_size(self):
        grid = CoverageGrid(voxel=0.5)
        for i in range(12):
            grid.observe(wall(500, x=float(i * 12), seed=i), np.array([-2.0, 0.0, 0.0]))
        spots = grid.weak_spots(limit=4)
        assert len(spots) <= 4
        assert [s.extent for s in spots] == sorted((s.extent for s in spots), reverse=True)


class TestAdvice:
    @pytest.mark.parametrize(
        "reason,word",
        [
            ("few views", "again"),
            ("one direction", "sideways"),
            ("distant", "closer"),
            ("poor tracking", "slowly"),
        ],
    )
    def test_every_reason_has_an_instruction(self, reason, word):
        spot = WeakSpot(np.zeros(3), 10, 0.1, reason)
        assert word in spot.advice.lower()

    def test_an_unknown_reason_still_says_something(self):
        assert WeakSpot(np.zeros(3), 1, 0.0, "mystery").advice


class TestTopDown:
    def test_empty_grid_gives_an_empty_map(self):
        grid, extent = CoverageGrid(voxel=0.5).top_down()
        assert grid.shape == (220, 220)
        assert extent == (0.0, 1.0, 0.0, 1.0)

    def test_unobserved_columns_are_negative(self):
        # Distinguishable from "observed and bad", which the colouring relies on.
        grid = CoverageGrid(voxel=0.5)
        grid.observe(wall(60, spread=0.5), np.array([-4.0, 0, 0]))
        plan, _ = grid.top_down(resolution=64)
        assert (plan < 0).any()

    def test_reports_the_extent_it_covers(self):
        grid = CoverageGrid(voxel=0.5)
        grid.observe(wall(400, x=0.0, spread=6.0), np.array([-4.0, 0, 0]))
        _, (min_x, max_x, min_z, max_z) = grid.top_down()
        assert max_z - min_z > 8.0
        assert max_x >= min_x

    def test_collapses_by_best_not_by_mean(self):
        # Someone standing in a room has poor coverage of the ceiling and good
        # coverage of the floor. Averaging paints the room amber and sends them
        # back somewhere that is already fine.
        grid = CoverageGrid(voxel=0.5)
        floor = np.stack(
            [np.zeros(300), np.full(300, -2.0), np.linspace(-3, 3, 300)], axis=1
        )
        ceiling = floor + np.array([0.0, 6.0, 0.0])
        for z in np.linspace(-3, 3, 8):
            grid.observe(floor, np.array([-2.0, -1.0, float(z)]), confidence=1.0)
        grid.observe(ceiling, np.array([-2.0, -1.0, 0.0]), confidence=1.0)
        plan, _ = grid.top_down(resolution=64)
        observed = plan[plan >= 0]
        assert observed.max() > 0.4, "the well-scanned floor must survive the collapse"
