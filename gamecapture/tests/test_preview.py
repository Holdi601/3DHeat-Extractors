"""
Tests for the preview rendering.

These matter more than they look. A preview that misleads is worse than none: a
good scan rendered as a mess makes someone stop and start again, and a bad scan
rendered as fine makes them keep walking.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_capture.fusion.coverage import CoverageGrid, WeakSpot
from heat3d_capture.ui.preview import (
    UNSEEN,
    coverage_map,
    depth_to_preview,
    health_text,
    mark_weak_spots,
    ramp,
)


class TestDepthPreview:
    def test_produces_a_displayable_rgb_image(self):
        depth = np.linspace(1, 10, 64 * 64).reshape(64, 64).astype(np.float32)
        out = depth_to_preview(depth)
        assert out.shape == (64, 64, 3) and out.dtype == np.uint8

    def test_one_infinite_pixel_does_not_flatten_the_rest(self):
        # A single sky pixel would take min/max normalisation with it and squash
        # every real surface into one colour. The scan is fine; the preview lies.
        depth = np.linspace(1, 5, 64 * 64).reshape(64, 64).astype(np.float32)
        depth[0, 0] = np.inf
        assert len(np.unique(depth_to_preview(depth).reshape(-1, 3), axis=0)) > 20

    def test_an_outlier_does_not_flatten_the_rest(self):
        depth = np.linspace(1, 5, 64 * 64).reshape(64, 64).astype(np.float32)
        depth[5, 5] = 1e9
        assert len(np.unique(depth_to_preview(depth).reshape(-1, 3), axis=0)) > 20

    def test_a_flat_map_does_not_divide_by_zero(self):
        out = depth_to_preview(np.full((32, 32), 3.0, dtype=np.float32))
        assert np.isfinite(out).all()

    def test_pointing_at_the_sky_is_not_a_crash(self):
        out = depth_to_preview(np.full((16, 16), np.inf, dtype=np.float32))
        assert out.shape == (16, 16, 3) and (out == 0).all()

    def test_near_and_far_differ(self):
        depth = np.zeros((32, 32), dtype=np.float32)
        depth[16:] = 9.0
        depth[:16] = 1.0
        out = depth_to_preview(depth)
        assert not np.array_equal(out[0, 0], out[31, 0])


class TestRamp:
    def test_bad_is_red_and_good_is_green(self):
        bad, good = ramp(np.array([0.0])), ramp(np.array([1.0]))
        assert bad[0][0] > bad[0][1], "worst should be red-dominant"
        assert good[0][1] > good[0][0], "best should be green-dominant"

    def test_is_continuous_between_stops(self):
        values = ramp(np.linspace(0, 1, 64))
        steps = np.abs(np.diff(values.astype(int), axis=0)).max()
        assert steps < 90, "the ramp should not jump between stops"

    def test_clamps_out_of_range_values(self):
        assert np.array_equal(ramp(np.array([-3.0])), ramp(np.array([0.0])))
        assert np.array_equal(ramp(np.array([9.0])), ramp(np.array([1.0])))


class TestCoverageMap:
    def test_an_empty_grid_is_all_unseen(self):
        out = coverage_map(np.zeros((0, 0), dtype=np.float32), size=64)
        assert out.shape == (64, 64, 3)
        assert (out == np.array(UNSEEN, dtype=np.uint8)).all()

    def test_unobserved_cells_are_visibly_not_a_ramp_colour(self):
        # "No data" has to look like a different kind of thing from "data that is
        # bad", or an unexplored corner reads as a badly scanned one.
        grid = np.full((16, 16), -1.0, dtype=np.float32)
        grid[8, 8] = 0.0
        out = coverage_map(grid, size=64)
        colours = {tuple(c) for c in out.reshape(-1, 3)}
        assert UNSEEN in colours
        assert len(colours) > 1

    def test_good_and_bad_coverage_look_different(self):
        good = coverage_map(np.full((8, 8), 1.0, dtype=np.float32), size=32)
        bad = coverage_map(np.full((8, 8), 0.0, dtype=np.float32), size=32)
        assert not np.array_equal(good, bad)

    def test_draws_the_walked_path(self):
        grid = np.full((16, 16), 0.5, dtype=np.float32)
        extent = (0.0, 10.0, 0.0, 10.0)
        plain = coverage_map(grid, size=64)
        path = np.array([[1.0, 0, 1.0], [5.0, 0, 5.0], [9.0, 0, 9.0]])
        drawn = coverage_map(grid, trajectory=path, extent=extent, size=64)
        assert not np.array_equal(plain, drawn)

    def test_a_single_point_path_is_not_drawn(self):
        grid = np.full((16, 16), 0.5, dtype=np.float32)
        extent = (0.0, 10.0, 0.0, 10.0)
        one = coverage_map(grid, trajectory=np.array([[1.0, 0, 1.0]]), extent=extent, size=64)
        assert np.array_equal(one, coverage_map(grid, size=64))

    def test_works_end_to_end_from_a_real_grid(self):
        coverage = CoverageGrid(voxel=0.5)
        rng = np.random.default_rng(0)
        points = np.stack(
            [np.zeros(500), rng.uniform(-3, 3, 500), rng.uniform(-3, 3, 500)], axis=1
        )
        for z in (-2.0, 0.0, 2.0):
            coverage.observe(points, np.array([-4.0, 0.0, z]), confidence=1.0)
        grid, extent = coverage.top_down(resolution=48)
        out = coverage_map(grid, extent=extent, size=96)
        assert out.shape == (96, 96, 3)
        assert len({tuple(c) for c in out.reshape(-1, 3)}) > 1


class TestWeakSpotMarks:
    def _image(self):
        return np.full((64, 64, 3), 30, dtype=np.uint8)

    def test_nothing_to_mark_leaves_the_image_alone(self):
        image = self._image()
        assert np.array_equal(mark_weak_spots(image, [], (0, 1, 0, 1)), image)
        assert np.array_equal(mark_weak_spots(image, [WeakSpot(np.zeros(3), 1, 0, "x")], None), image)

    def test_marks_are_drawn(self):
        image = self._image()
        spot = WeakSpot(np.array([5.0, 0.0, 5.0]), 20, 0.1, "few views")
        out = mark_weak_spots(image, [spot], (0.0, 10.0, 0.0, 10.0))
        assert not np.array_equal(out, image)

    def test_does_not_modify_the_image_it_was_given(self):
        image = self._image()
        before = image.copy()
        mark_weak_spots(image, [WeakSpot(np.array([5.0, 0, 5.0]), 20, 0.1, "few views")], (0, 10, 0, 10))
        assert np.array_equal(image, before)

    def test_a_spot_outside_the_extent_is_clamped_not_dropped(self):
        # Better to point at the edge of the map than to silently omit a place
        # that needs revisiting.
        image = self._image()
        spot = WeakSpot(np.array([500.0, 0.0, -500.0]), 20, 0.1, "distant")
        assert not np.array_equal(mark_weak_spots(image, [spot], (0.0, 10.0, 0.0, 10.0)), image)


class TestHealthText:
    class State:
        def __init__(self, frames, tracked):
            self.frames = frames
            self.tracked = tracked
            self.lost = frames - tracked

        @property
        def tracking_health(self):
            return self.tracked / self.frames if self.frames else 0.0

    def test_nothing_yet(self):
        text, _ = health_text(None)
        assert "Waiting" in text
        assert "Waiting" in health_text(self.State(0, 0))[0]

    def test_good_tracking_says_so(self):
        text, colour = health_text(self.State(100, 98))
        assert "well" in text and colour == "#3fb950"

    def test_patchy_tracking_gives_an_instruction(self):
        text, colour = health_text(self.State(100, 70))
        assert "slowly" in text and colour == "#d29922"

    def test_bad_tracking_is_loud_and_actionable(self):
        text, colour = health_text(self.State(100, 30))
        assert colour == "#f85149"
        assert "blank walls" in text
