"""
Nothing on the interface thread may scale with the scan.

This exists because of the worst bug the program had. Weak-coverage clustering
rasterised the whole bounding box of the scan and ran connected components over
it — fine while walking a room, catastrophic while driving, where a route spans
hundreds of metres and leaves a stringy 3% of the box occupied. Measured: 196k
occupied voxels inside a 6.8M-cell box, **13.5 seconds** per call, on a 900 ms
timer, on the thread that paints the window and handles clicks.

What the user saw was not a slow map. It was a dead program: the stop button did
nothing, no progress was ever drawn, and the health line sat on "waiting for the
first frame" because the queued updates were never processed. The scan underneath
was working perfectly and had no way to say so.

No unit test could have caught it, because every part was individually correct.
The fault was in *where* correct work ran and *how it grew*. So these tests
assert the two properties that actually matter: the clustering stays cheap as the
scan grows, and the interface hands the work away rather than doing it.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from heat3d_capture.fusion.coverage import CoverageGrid

#: A single call may not cost more than this. Well under the refresh interval,
#: because the budget is not "fast enough to finish" but "fast enough that
#: nothing queues behind it".
BUDGET_SECONDS = 2.0


def route(span: float, points: int, *, seed: int = 0) -> CoverageGrid:
    """A coverage grid shaped like a journey: long, thin, and mostly empty."""
    rng = np.random.default_rng(seed)
    grid = CoverageGrid(voxel=0.5)
    steps = 24
    for i in range(steps):
        position = np.array([i / (steps - 1) * span, 0.0, 0.0])
        cloud = np.stack(
            [
                rng.normal(position[0], span / 12, points // steps),
                rng.uniform(-3, 8, points // steps),
                rng.uniform(-40, 40, points // steps),
            ],
            axis=1,
        )
        grid.observe(cloud, position, confidence=1.0)
    return grid


class TestClusteringCost:
    @pytest.mark.parametrize(
        "span,points,label",
        [(30.0, 40_000, "a walked room"), (600.0, 200_000, "a driven route")],
    )
    def test_stays_within_budget(self, span, points, label):
        grid = route(span, points)
        started = time.perf_counter()
        grid.weak_spots()
        elapsed = time.perf_counter() - started
        assert elapsed < BUDGET_SECONDS, (
            f"{label} took {elapsed:.1f}s to cluster — the window would be frozen"
        )

    def test_cost_follows_the_data_not_the_bounding_box(self):
        # The heart of it. Twenty times the span with only five times the data
        # must not be twenty times the work: the old implementation was
        # proportional to the box, which is what made driving fatal.
        small = route(30.0, 40_000)
        large = route(600.0, 200_000)

        def timed(grid):
            started = time.perf_counter()
            grid.weak_spots()
            return time.perf_counter() - started

        # Warm anything cacheable first, so the comparison is of the algorithm.
        timed(small)
        timed(large)
        ratio = timed(large) / max(timed(small), 1e-6)
        assert ratio < 12.0, (
            f"cost grew {ratio:.0f}x for 5x the data — it is still following the bounding box"
        )

    def test_the_answer_is_the_same_shape_at_both_scales(self):
        # Cheap is worthless if it stopped being right.
        for grid in (route(30.0, 40_000), route(600.0, 200_000)):
            spots = grid.weak_spots()
            assert spots
            assert all(s.extent >= 8 for s in spots)
            assert all(s.advice for s in spots)
            assert [s.extent for s in spots] == sorted((s.extent for s in spots), reverse=True)


class TestTheInterfaceHandsWorkAway:
    def test_the_refresh_returns_immediately(self):
        """
        `_refresh_map` must start work, not do it.

        Driven through the real window, with a reconstructor holding a
        driving-scale grid behind it — the exact situation that locked up.
        """
        from PySide6 import QtWidgets

        from heat3d_capture.pose.odometry import Intrinsics
        from heat3d_capture.scan.reconstruct import Reconstructor
        from heat3d_capture.ui import app as app_module

        app_module.application()
        window = app_module.build_window()()
        try:
            reconstructor = Reconstructor(Intrinsics.from_fov(320, 180, 90.0))
            reconstructor.coverage = route(600.0, 200_000)

            class Session:
                # Enough of the interface for the window to treat it as a live
                # scan, including the teardown it performs on close.
                is_running = staticmethod(lambda: True)
                stop = staticmethod(lambda **_: None)
                export = staticmethod(lambda *a, **k: None)

            session = Session()
            session.reconstructor = reconstructor
            window.session = session

            started = time.perf_counter()
            window._refresh_map()
            elapsed = time.perf_counter() - started
            # It is allowed to spawn a thread and return. It is not allowed to
            # cluster two hundred thousand voxels before giving control back.
            assert elapsed < 0.25, (
                f"the refresh blocked the interface for {elapsed * 1000:.0f} ms"
            )
        finally:
            window.close()

    def test_a_second_refresh_does_not_pile_work_up(self):
        from heat3d_capture.pose.odometry import Intrinsics
        from heat3d_capture.scan.reconstruct import Reconstructor
        from heat3d_capture.ui import app as app_module

        app_module.application()
        window = app_module.build_window()()
        try:
            reconstructor = Reconstructor(Intrinsics.from_fov(320, 180, 90.0))
            reconstructor.coverage = route(600.0, 200_000)

            class Session:
                # Enough of the interface for the window to treat it as a live
                # scan, including the teardown it performs on close.
                is_running = staticmethod(lambda: True)
                stop = staticmethod(lambda **_: None)
                export = staticmethod(lambda *a, **k: None)

            session = Session()
            session.reconstructor = reconstructor
            window.session = session

            window._refresh_map()
            assert window._mapping is True
            # The timer fires again long before a large map finishes. A second
            # thread on the same data would double the cost and race the first.
            started = time.perf_counter()
            window._refresh_map()
            assert time.perf_counter() - started < 0.05
        finally:
            window.close()
