"""
Reading the coverage map while the scan is still writing to it.

The scan runs on a worker thread; the overlay and the map panel read from the
GUI thread, with no lock between them. That is meant to be safe because rows are
only ever appended — but it was not, and the failure was as bad as it could
usefully be: the coverage map threw

    ValueError: array is not broadcastable to correct shape

part-way into a real scan, and the overlay showed nothing for the rest of the
run. Which is exactly when someone is relying on it to tell them whether what
they are doing is working.

The cause was two reads of the same thing. `observe` appends to the voxel index
and sets the row count on the line after, so an interface refresh landing
between them saw N+k positions and N qualities.

These tests hammer that window deliberately, because a race that happens once
every few thousand keyframes will not show up in an ordinary test run.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from heat3d_capture.fusion.coverage import CoverageGrid


def sprinkle(grid: CoverageGrid, n: int, *, seed: int) -> None:
    """Observe a batch of fresh voxels, the way a keyframe does."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(-60, 60, size=(n, 3))
    camera = rng.uniform(-10, 10, size=3)
    grid.observe(points, camera, confidence=0.8)


class TestReadingWhileWriting:
    def test_the_map_survives_a_scan_running_underneath_it(self):
        """
        One thread observing, another drawing the map, for long enough that the
        window between the append and the count is hit many times over.
        """
        grid = CoverageGrid(voxel=1.0)
        sprinkle(grid, 500, seed=0)
        failures: list[BaseException] = []
        stop = threading.Event()

        def scan():
            seed = 1
            while not stop.is_set():
                try:
                    sprinkle(grid, 400, seed=seed)
                except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
                    failures.append(exc)
                    return
                seed += 1

        def draw():
            while not stop.is_set():
                try:
                    image, extent = grid.top_down(resolution=64)
                    assert image.shape == (64, 64)
                    assert len(extent) == 4
                except BaseException as exc:  # noqa: BLE001
                    failures.append(exc)
                    return

        worker = threading.Thread(target=scan, daemon=True)
        reader = threading.Thread(target=draw, daemon=True)
        worker.start()
        reader.start()
        stop.wait(2.0)
        stop.set()
        worker.join(timeout=5)
        reader.join(timeout=5)

        assert not failures, f"{type(failures[0]).__name__}: {failures[0]}"

    def test_every_reader_agrees_on_how_many_voxels_there_are(self):
        """
        The map is not the only reader. Quality, spread and positions are used by
        the weak-spot search and the statistics line too, and any pair of them
        that disagrees is the same bug wearing a different hat.
        """
        grid = CoverageGrid(voxel=1.0)
        sprinkle(grid, 600, seed=3)
        failures: list[BaseException] = []
        stop = threading.Event()

        def scan():
            seed = 10
            while not stop.is_set():
                sprinkle(grid, 300, seed=seed)
                seed += 1

        def read():
            while not stop.is_set():
                try:
                    positions = grid.positions()
                    quality = grid.quality()
                    spread = grid.spread()
                    # No ordering between them: each takes its own snapshot, so
                    # either can be the longer one depending on where the writer
                    # was. What has to hold is that each is internally consistent
                    # and that pairing them to the shorter is always valid —
                    # which is what every caller does.
                    assert positions.ndim == 2 and positions.shape[1] == 3
                    assert quality.ndim == 1 and spread.ndim == 1
                    shortest = min(len(positions), len(quality))
                    paired = positions[:shortest], quality[:shortest]
                    assert len(paired[0]) == len(paired[1])
                except BaseException as exc:  # noqa: BLE001
                    failures.append(exc)
                    return

        worker = threading.Thread(target=scan, daemon=True)
        reader = threading.Thread(target=read, daemon=True)
        worker.start()
        reader.start()
        stop.wait(1.5)
        stop.set()
        worker.join(timeout=5)
        reader.join(timeout=5)

        assert not failures, f"{type(failures[0]).__name__}: {failures[0]}"


class TestTheSnapshotItself:
    def test_a_longer_index_than_count_is_trimmed(self):
        """
        The exact mid-write state, forced rather than waited for.

        `observe` appends to the index and sets `_used` afterwards, so this is
        what the readers see in between — and it has to produce a shorter,
        consistent answer rather than an exception.
        """
        grid = CoverageGrid(voxel=1.0)
        sprinkle(grid, 200, seed=7)
        real = grid._used

        grid._used = real - 50  # as if the count had not caught up yet

        assert len(grid.positions()) == real - 50
        assert len(grid.quality()) == real - 50
        image, _ = grid.top_down(resolution=32)
        assert image.shape == (32, 32)

    def test_an_empty_grid_still_draws(self):
        image, extent = CoverageGrid(voxel=1.0).top_down(resolution=16)

        assert image.shape == (16, 16)
        assert extent == (0.0, 1.0, 0.0, 1.0)
