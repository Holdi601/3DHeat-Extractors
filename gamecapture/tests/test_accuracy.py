"""
Accuracy against known truth.

Everything else in this suite checks that a component behaves; this checks that
the whole thing is *right*. The room is 12 by 8 metres, the camera was at stated
places, and the difference between that and what comes out is a number.

The depth fed in here is analytic, not predicted. That is deliberate and it is
the point: it isolates pose recovery, scale propagation and fusion from the depth
network, so a failure here is unambiguously in the geometry code. How well the
network does on real gameplay is a separate question that no synthetic scene can
answer.

The tolerances are loose in absolute terms and tight in the terms that matter.
Half a metre of drift over a twelve-metre walk would be useless; five centimetres
is fine, because the surface is being voxelised at twenty-five anyway.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_capture.pose.odometry import Intrinsics
from heat3d_capture.scan.reconstruct import Reconstructor

from .scene import ROOM, survey, walk


@pytest.fixture(scope="module")
def frames():
    # 480x270 keeps the whole suite quick while leaving ORB plenty of corners.
    return walk(Intrinsics.from_fov(480, 270, 90.0), steps=36)


@pytest.fixture(scope="module")
def reconstructed(frames):
    """One run of the real pipeline over the whole walk."""
    k = Intrinsics.from_fov(480, 270, 90.0)
    r = Reconstructor(k, voxel=0.2, coverage_voxel=0.4, camera_height=None)
    for frame in frames:
        # Inverse depth, as the network would produce — so the scale-chaining
        # path is exercised rather than bypassed.
        with np.errstate(divide="ignore"):
            inverse = np.where(np.isfinite(frame.depth), 1.0 / frame.depth, 0.0)
        r.add(frame.image, inverse.astype(np.float32))
    return r


class TestTracking:
    def test_it_places_essentially_every_frame(self, reconstructed, frames):
        state = reconstructed.state
        assert state.frames == len(frames)
        assert state.tracking_health > 0.9, f"only placed {state.tracked} of {state.frames}"

    def test_the_recovered_path_matches_the_real_one(self, reconstructed, frames):
        # The headline number. The first frame defines the origin, so the
        # comparison is of the path's *shape*, not its absolute placement.
        truth = np.array([f.pose.translation for f in frames])
        got = reconstructed.trajectory.positions
        assert len(got) >= len(truth) - 2

        truth = truth[: len(got)]
        truth_rel = truth - truth[0]
        # The recovered path is in the first camera's frame; rotate it back.
        got_world = got @ frames[0].pose.rotation.T
        error = np.linalg.norm(got_world - truth_rel, axis=1)
        assert error.mean() < 0.35, f"mean path error {error.mean():.2f} m"
        assert error.max() < 1.0, f"worst path error {error.max():.2f} m"

    def test_the_walked_distance_is_right(self, reconstructed, frames):
        truth = np.array([f.pose.translation for f in frames])
        walked = float(np.linalg.norm(np.diff(truth, axis=0), axis=1).sum())
        got = reconstructed.trajectory.length
        # Within a tenth. A systematic scale error would show here first, and it
        # is the error that a monocular pipeline is most prone to.
        assert got == pytest.approx(walked, rel=0.1), f"walked {walked:.1f} m, recovered {got:.1f} m"


class TestGeometry:
    def test_the_room_comes_out_the_right_size(self, reconstructed):
        mesh = reconstructed.volume.to_mesh()
        assert not mesh.is_empty()
        # Only the walked part of the room is seen, so the length is not
        # expected to be the full 12 m. The width is crossed by the weave and is.
        span = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
        assert span[2] == pytest.approx(ROOM[2], abs=1.5), f"room width came out {span[2]:.1f} m"
        assert span[1] < ROOM[1] + 1.5, f"room height came out {span[1]:.1f} m"

    def test_the_surface_is_where_the_walls_are(self, reconstructed, frames):
        # Every reconstructed point should lie near one of the six planes. A
        # point floating in the middle of the room is geometry that is not there.
        mesh = reconstructed.volume.to_mesh()
        origin = frames[0].pose
        world = mesh.vertices.astype(np.float64) @ origin.rotation.T + origin.translation

        distances = np.minimum(
            np.abs(world - 0.0), np.abs(world - ROOM[None, :])
        ).min(axis=1)
        # Three quarters within a voxel of a wall. Not all of it: the corners of
        # the room and the edges of the frustum are genuinely less certain.
        assert (distances < 0.45).mean() > 0.75, (
            f"only {(distances < 0.45).mean():.0%} of the surface landed on a wall"
        )

    def test_it_exports_something_the_viewer_can_read(self, reconstructed, tmp_path):
        out = reconstructed.export(tmp_path / "room.glb", name="room")
        assert out is not None and out.exists()
        assert out.stat().st_size > 2000


class TestCoverage:
    def test_it_measures_something(self, reconstructed):
        summary = reconstructed.coverage.summary()
        assert summary["voxels"] > 500
        assert 0.0 <= summary["well_observed"] <= 1.0

    def test_it_can_tell_a_good_scan_from_a_lazy_one(self):
        # The property that makes the coverage panel worth having. Walking
        # forward while looking forward is the natural thing to do and the worst
        # thing to do — every surface ahead is seen from one direction, with no
        # parallax, however many frames are spent on it.
        #
        # A metric that scored both of these the same would be worse than none,
        # because it would send someone home with a green map and an
        # unreconstructable capture.
        k = Intrinsics.from_fov(480, 270, 90.0)
        scores = {}
        for name, frames in (("forward", walk(k, steps=30)), ("survey", survey(k, steps=30))):
            r = Reconstructor(k, voxel=0.2, coverage_voxel=0.4, camera_height=None)
            for frame in frames:
                with np.errstate(divide="ignore"):
                    inverse = np.where(np.isfinite(frame.depth), 1.0 / frame.depth, 0.0)
                r.add(frame.image, inverse.astype(np.float32))
            scores[name] = r.coverage.summary()["well_observed"]
        assert scores["survey"] > scores["forward"] * 1.5, (
            f"forward walk scored {scores['forward']:.0%}, survey {scores['survey']:.0%} — "
            "the metric cannot tell them apart"
        )

    def test_the_far_end_of_the_room_is_flagged(self, reconstructed):
        # The walk stops short of the far wall, which is therefore only ever seen
        # from a distance and from one direction. That is exactly what the panel
        # is for, and it must say so.
        spots = reconstructed.refresh_weak_spots()
        assert spots, "a partly-walked room should have something worth revisiting"
        assert all(s.advice for s in spots)


class TestAgainstNoStructure:
    def test_a_blank_scene_is_refused_rather_than_invented(self):
        # The counterpart to everything above: given nothing to track, the
        # pipeline must produce nothing rather than a confident wrong answer.
        k = Intrinsics.from_fov(320, 180, 90.0)
        r = Reconstructor(k, voxel=0.25, camera_height=None)
        blank = np.full((180, 320, 3), 120, dtype=np.uint8)
        depth = np.full((180, 320), 0.25, dtype=np.float32)
        for _ in range(8):
            r.add(blank, depth)
        assert r.state.tracked <= 1, "a featureless scene must not track"
        assert r.state.lost >= 6
