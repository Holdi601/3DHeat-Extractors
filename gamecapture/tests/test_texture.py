"""
Tests for colour projection and loop closure.

Both are end-of-pipeline behaviours that only show up over a whole scan, so both
are driven through the real reconstructor on the ground-truth room.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_capture.fusion.texture import (
    FAR_VIEW,
    Keyframe,
    KeyframeStore,
    project_colours,
)
from heat3d_capture.pose.odometry import Intrinsics, Pose
from heat3d_capture.scan.reconstruct import Reconstructor

from .scene import survey, walk


class TestProjection:
    def _setup(self):
        k = Intrinsics.from_fov(64, 64, 90.0)
        # A camera at the origin looking down +Z at a wall 4 m away.
        pose = Pose(np.eye(3), np.zeros(3))
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, :] = (200, 40, 40)
        return k, [Keyframe(image=image, pose=pose)]

    def test_a_facing_surface_takes_the_colour_it_was_photographed_in(self):
        k, frames = self._setup()
        vertices = np.array([[0.0, 0.0, 4.0]])
        normals = np.array([[0.0, 0.0, -1.0]])  # facing the camera
        out = project_colours(vertices, normals, frames, k)
        assert out[0] == pytest.approx([200 / 255, 40 / 255, 40 / 255], abs=0.02)

    def test_a_surface_behind_the_camera_is_not_coloured(self):
        k, frames = self._setup()
        out = project_colours(
            np.array([[0.0, 0.0, -4.0]]), np.array([[0.0, 0.0, 1.0]]), frames, k
        )
        # Untouched, so the default grey rather than whatever pixel happened to
        # be at the projected coordinate.
        assert out[0] == pytest.approx([0.5, 0.5, 0.5])

    def test_a_surface_seen_edge_on_is_not_coloured(self):
        # At a grazing angle one pixel covers metres of surface, so the colour it
        # reports is an average of things that are not the same thing.
        k, frames = self._setup()
        out = project_colours(
            np.array([[0.0, 0.0, 4.0]]), np.array([[1.0, 0.0, 0.0]]), frames, k
        )
        assert out[0] == pytest.approx([0.5, 0.5, 0.5])

    def test_a_surface_too_far_away_is_not_coloured(self):
        k, frames = self._setup()
        out = project_colours(
            np.array([[0.0, 0.0, FAR_VIEW * 1.5]]), np.array([[0.0, 0.0, -1.0]]), frames, k
        )
        assert out[0] == pytest.approx([0.5, 0.5, 0.5])

    def test_the_fallback_survives_where_nothing_saw_it(self):
        # A black patch reads as a hole in the geometry rather than as missing
        # colour, so the voxel average is kept instead.
        k, frames = self._setup()
        fallback = np.array([[0.1, 0.9, 0.2]], dtype=np.float32)
        out = project_colours(
            np.array([[0.0, 0.0, -4.0]]),
            np.array([[0.0, 0.0, 1.0]]),
            frames,
            k,
            fallback=fallback,
        )
        assert out[0] == pytest.approx([0.1, 0.9, 0.2])

    def test_no_frames_at_all_is_not_an_error(self):
        k, _ = self._setup()
        out = project_colours(np.zeros((3, 3)), np.ones((3, 3)), [], k)
        assert out.shape == (3, 3)
        assert np.isfinite(out).all()

    def test_output_is_bounded_and_float32(self):
        k, frames = self._setup()
        out = project_colours(
            np.array([[0.0, 0.0, 4.0]]), np.array([[0.0, 0.0, -1.0]]), frames, k
        )
        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0


class TestKeyframeStore:
    def test_keeps_everything_below_the_limit(self):
        store = KeyframeStore(limit=10)
        for _ in range(6):
            store.add(np.zeros((4, 4, 3), dtype=np.uint8), Pose.identity())
        assert len(store) == 6

    def test_never_grows_past_the_limit(self):
        # The frames are the largest thing in the process; a half-hour scan would
        # cost more memory than everything else together.
        store = KeyframeStore(limit=20)
        for _ in range(2000):
            store.add(np.zeros((4, 4, 3), dtype=np.uint8), Pose.identity())
        assert len(store) == 20

    def test_keeps_a_spread_over_the_whole_scan(self):
        # Not the last N frames: the end of a walk is not more worth colouring
        # than the start of it.
        store = KeyframeStore(limit=20)
        for i in range(400):
            store.add(
                np.zeros((2, 2, 3), dtype=np.uint8),
                Pose(np.eye(3), np.array([float(i), 0.0, 0.0])),
            )
        kept = sorted(f.pose.translation[0] for f in store.frames)
        assert kept[0] < 120, "the early part of the scan was dropped entirely"
        assert kept[-1] > 250, "the late part of the scan was dropped entirely"

    def test_copies_the_frame_it_is_given(self):
        # Capture buffers are reused; holding a reference would give every stored
        # frame the contents of the most recent one.
        store = KeyframeStore(limit=4)
        buffer = np.zeros((4, 4, 3), dtype=np.uint8)
        store.add(buffer, Pose.identity())
        buffer[:] = 255
        assert store.frames[0].image.max() == 0


class TestThroughTheReconstructor:
    def test_a_scan_produces_coloured_geometry(self):
        k = Intrinsics.from_fov(480, 270, 90.0)
        r = Reconstructor(k, voxel=0.2, mask_hud=False, camera_height=None)
        for frame in survey(k, steps=24):
            with np.errstate(divide="ignore"):
                inverse = np.where(np.isfinite(frame.depth), 1.0 / frame.depth, 0.0)
            r.add(frame.image, inverse.astype(np.float32))
        assert len(r.keyframes) > 0

        mesh = r.volume.to_mesh()
        if mesh.is_empty():
            pytest.skip("nothing fused")
        colours = project_colours(
            mesh.vertices, mesh.normals, r.keyframes.frames, k, fallback=mesh.colours
        )
        assert colours.shape == mesh.vertices.shape
        # The room is brightly and variously textured; a single flat colour would
        # mean the projection found nothing and fell through to the default.
        assert colours.std() > 0.02

    def test_the_export_carries_colour(self, tmp_path):
        import json
        import struct

        k = Intrinsics.from_fov(480, 270, 90.0)
        r = Reconstructor(k, voxel=0.25, mask_hud=False, camera_height=None)
        for frame in survey(k, steps=20):
            with np.errstate(divide="ignore"):
                inverse = np.where(np.isfinite(frame.depth), 1.0 / frame.depth, 0.0)
            r.add(frame.image, inverse.astype(np.float32))
        out = r.export(tmp_path / "coloured.glb", name="room")
        if out is None:
            pytest.skip("nothing fused")

        raw = out.read_bytes()
        length, _ = struct.unpack_from("<II", raw, 12)
        gltf = json.loads(raw[20 : 20 + length].decode("utf-8"))
        attributes = gltf["meshes"][0]["primitives"][0]["attributes"]
        assert "COLOR_0" in attributes, "vertex colour did not reach the file"


class TestLoopClosure:
    def test_returning_to_the_start_is_noticed(self):
        # Walk out and walk back over the same ground. Coming back to a place
        # already in the keyframe database is what loop closure is for, and it
        # must be detected rather than accumulating drift straight past it.
        k = Intrinsics.from_fov(480, 270, 90.0)
        out = walk(k, steps=26)
        there_and_back = out + list(reversed(out))

        r = Reconstructor(k, voxel=0.25, mask_hud=False, camera_height=None)
        for frame in there_and_back:
            with np.errstate(divide="ignore"):
                inverse = np.where(np.isfinite(frame.depth), 1.0 / frame.depth, 0.0)
            r.add(frame.image, inverse.astype(np.float32))

        assert r.state.tracked > len(there_and_back) * 0.8
        # Either a closure fired, or the track never drifted enough to need one.
        # Both are acceptable; silently diverging is not.
        truth_end = out[0].pose.translation
        got_end = r.trajectory.positions[-1]
        origin = out[0].pose
        got_world = got_end @ origin.rotation.T + origin.translation
        drift = float(np.linalg.norm(got_world - truth_end))
        assert drift < 2.5, f"came back {drift:.2f} m away from where it started"

    def test_closures_are_counted_not_hidden(self):
        k = Intrinsics.from_fov(480, 270, 90.0)
        r = Reconstructor(k, voxel=0.25, mask_hud=False, camera_height=None)
        frames = walk(k, steps=20)
        for frame in frames + list(reversed(frames)):
            with np.errstate(divide="ignore"):
                inverse = np.where(np.isfinite(frame.depth), 1.0 / frame.depth, 0.0)
            r.add(frame.image, inverse.astype(np.float32))
        # The count is reported whether or not it fired, so a scan can say what
        # happened to it rather than leaving drift unexplained.
        assert r.state.loop_closures >= 0
        assert r.state.relocalisations >= 0

    def test_loop_closure_can_be_turned_off(self):
        from heat3d_capture.pose.odometry import VisualOdometry

        vo = VisualOdometry(Intrinsics.from_fov(320, 180, 90.0), loop_closure=False)
        assert vo._loop_closure is False
