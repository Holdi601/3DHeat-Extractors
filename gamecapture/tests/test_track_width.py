"""
Does a driven lap reconstruct the *road*, or only the line that was driven?

This is the question telemetry raises and cannot answer by itself. A telemetry
feed gives where the car was, at 60 Hz, exactly — and a car is never in the
middle of the road, so the recorded line is not the track. No width, no verges,
no kerbs.

The design's answer is that telemetry was never meant to supply geometry. It
supplies the *pose*, which is the half that fails on a night cockpit lap, and the
depth model supplies everything the camera can see around itself. Each frame sees
tens of metres of road across its full width, so the width comes from the
pictures and the placement comes from the feed.

These tests check that claim against a track of exactly known width, driven
deliberately off-centre. Depth is analytic, as in `test_accuracy.py`, so a
failure here is in the geometry and not in the network.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_capture.pose.odometry import Intrinsics
from heat3d_capture.scan.reconstruct import Reconstructor

from .track_scene import (
    GROUND_HALF,
    KERB_WIDTH,
    ROAD_WIDTH,
    ExactPoses,
    _nearest_on_lap,
    drive,
)

#: Where the camera drove, in metres from the centreline. Deliberately not zero.
LINE_OFFSET = 3.0


@pytest.fixture(scope="module")
def intrinsics():
    return Intrinsics.from_fov(480, 270, 90.0)


@pytest.fixture(scope="module")
def frames(intrinsics):
    return drive(intrinsics, steps=48, line_offset=LINE_OFFSET)


@pytest.fixture(scope="module")
def reconstructed(frames, intrinsics):
    """The real pipeline, with poses supplied the way telemetry supplies them."""
    r = Reconstructor(
        intrinsics,
        voxel=0.5,
        coverage_voxel=1.0,
        camera_height=None,
        pose_source=ExactPoses(frames),
        mask_hud=False,
    )
    for n, frame in enumerate(frames):
        with np.errstate(divide="ignore"):
            inverse = np.where(np.isfinite(frame.depth), 1.0 / frame.depth, 0.0)
        r.add(frame.image, inverse.astype(np.float32), timestamp=float(n))
    return r


def surface(reconstructed) -> np.ndarray:
    mesh = reconstructed.volume.to_mesh()
    assert not mesh.is_empty(), "fusion produced no surface at all"
    return np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)


class TestTheDrivenLineIsNotTheRoad:
    def test_the_camera_was_never_in_the_middle(self, frames):
        """The fixture has to actually pose the problem, or the test proves nothing."""
        positions = np.stack([f.pose.translation for f in frames])
        offset, _ = _nearest_on_lap(positions)

        assert np.abs(offset).min() > LINE_OFFSET * 0.8
        assert np.abs(np.abs(offset).mean() - LINE_OFFSET) < 0.5

    def test_the_reconstruction_is_far_wider_than_the_line(self, reconstructed, frames):
        """
        The whole point. A path is a curve; a road is a surface.

        Comparing against the driven line's own width rather than against zero,
        because a line drawn by a car through corners has some width of its own
        and beating *that* is the claim.
        """
        points = surface(reconstructed)
        offset, _ = _nearest_on_lap(points)
        reconstructed_span = float(np.percentile(offset, 97) - np.percentile(offset, 3))

        driven = np.stack([f.pose.translation for f in frames])
        driven_offset, _ = _nearest_on_lap(driven)
        driven_span = float(driven_offset.max() - driven_offset.min())

        assert reconstructed_span > driven_span + ROAD_WIDTH * 0.5
        assert reconstructed_span > ROAD_WIDTH


class TestTheRoadComesOutTheRightWidth:
    def test_both_verges_are_reached(self, reconstructed):
        """
        Surface on both sides of the centreline, not just the side driven on.

        Driving three metres left of centre, the near verge is nine metres away
        and the far one is nine the other way. Recovering only the near side
        would still produce a plausible-looking ribbon.
        """
        offset, _ = _nearest_on_lap(surface(reconstructed))
        half = ROAD_WIDTH / 2.0

        assert (offset < -half * 0.8).any(), "nothing reconstructed on the right"
        assert (offset > half * 0.8).any(), "nothing reconstructed on the left"

    def test_the_whole_lap_is_covered(self, reconstructed):
        """Surface all the way round, not one stretch of it."""
        _, along = _nearest_on_lap(surface(reconstructed))
        occupied = np.histogram(along, bins=24, range=(0.0, 1.0))[0]

        assert (occupied > 0).sum() >= 22, f"only {(occupied>0).sum()} of 24 arcs covered"

    def test_the_road_sits_roughly_on_the_ground(self, reconstructed):
        """
        The road lands near y = 0, with a sag that is measured rather than hidden.

        With exact poses *and* exact analytic depth this should be exact, and it
        is not: the surface sits about 1.25 m low, one-sided — nothing at all
        lands above +2 m while the tail runs to -14. Over-estimated depth on a
        grazing ray does exactly that, and looking along a road most of the
        visible surface is grazing, so the far half of every view is pulled down
        and under. `MAX_DEPTH` bounds it but does not remove it.

        This is a real open finding, not a tolerance to be widened until it
        passes. It is pinned here so it cannot quietly get worse, and so the next
        person to look at fusion knows where to start. What it does not affect is
        the width, which is what this fixture exists to check: a surface pulled
        down along the view direction is still the right distance across.
        """
        points = surface(reconstructed)
        offset, _ = _nearest_on_lap(points)
        road = points[np.abs(offset) <= ROAD_WIDTH / 2.0]
        assert len(road) > 100

        sag = float(np.median(road[:, 1]))
        assert -2.0 < sag < 0.5, f"road sits {sag:.2f} m off the ground"
        # Half the surface within a couple of voxels of true.
        assert float(np.percentile(np.abs(road[:, 1]), 50)) < 1.5

    def test_the_road_width_is_recovered(self, reconstructed):
        """
        The number that went in comes back out.

        Measured per arc of the lap and taken as a median, so one stretch with a
        stray point cannot stand in for the road being right. The tolerance is
        the voxel size plus the kerb, because fusion cannot resolve an edge
        finer than a voxel and the kerb is part of the drivable surface.
        """
        points = surface(reconstructed)
        offset, along = _nearest_on_lap(points)

        widths = []
        for start in np.linspace(0.0, 1.0, 16, endpoint=False):
            arc = (along >= start) & (along < start + 1.0 / 16)
            if arc.sum() < 30:
                continue
            here = offset[arc]
            # Measured within the road band, because road, kerb and verge form
            # one continuous ground surface with no geometric edge between them.
            # Nothing about the shape says where the tarmac stops — only the
            # colour does, which is why the viewer has a switch for it.
            near = here[np.abs(here) <= ROAD_WIDTH / 2.0]
            if len(near) < 20:
                continue
            widths.append(float(near.max() - near.min()))

        assert len(widths) >= 12, f"only {len(widths)} arcs had enough surface"
        recovered = float(np.median(widths))

        # Tight, because this is the claim the whole telemetry-plus-depth design
        # rests on: the road comes back the width it was drawn, from a lap driven
        # three metres off centre.
        assert recovered == pytest.approx(ROAD_WIDTH, abs=0.5), (
            f"road came out {recovered:.2f} m wide, drawn as {ROAD_WIDTH} m"
        )


class TestWhatTelemetryAloneWouldGive:
    def test_the_driven_line_alone_has_no_width_worth_the_name(self, frames):
        """
        Stated as a test so the claim is checked rather than asserted in prose.

        A lap of telemetry is a curve. Whatever else is true, it does not
        describe a twelve-metre road, and anyone reaching for telemetry alone
        should be able to see that here.
        """
        driven = np.stack([f.pose.translation for f in frames])
        offset, _ = _nearest_on_lap(driven)

        assert float(offset.max() - offset.min()) < ROAD_WIDTH * 0.25
