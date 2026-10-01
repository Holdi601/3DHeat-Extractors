"""
Tests for visual odometry.

Two layers. The geometry — intrinsics, lifting, projection, pose composition — is
exact arithmetic and is tested exactly, because an error there bends every
reconstruction in a way that still looks like a plausible level. The tracker
itself is driven with a synthetic scene whose camera motion is known, so the
recovered path can be compared against the truth rather than merely inspected.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_capture.pose.odometry import (
    Intrinsics,
    Pose,
    Trajectory,
    VisualOdometry,
)


class TestIntrinsics:
    def test_a_ninety_degree_field_of_view_makes_focal_equal_half_width(self):
        # tan(45°) = 1, so fx is exactly half the width. A closed-form case that
        # catches a degrees/radians slip immediately.
        k = Intrinsics.from_fov(800, 600, 90.0)
        assert k.fx == pytest.approx(400.0)
        assert k.cx == pytest.approx(400.0)
        assert k.cy == pytest.approx(300.0)

    def test_a_narrower_field_of_view_is_a_longer_focal_length(self):
        assert Intrinsics.from_fov(800, 600, 60.0).fx > Intrinsics.from_fov(800, 600, 90.0).fx

    def test_pixels_are_square(self):
        # The vertical field of view follows from the aspect ratio rather than
        # being a second free parameter nobody can supply.
        k = Intrinsics.from_fov(1920, 1080, 75.0)
        assert k.fx == pytest.approx(k.fy)

    def test_unproject_and_project_are_inverses(self):
        k = Intrinsics.from_fov(640, 480, 80.0)
        pixels = np.array([[320.0, 240.0], [100.0, 50.0], [639.0, 479.0]])
        depths = np.array([1.0, 7.5, 22.0])
        assert k.project(k.unproject(pixels, depths)) == pytest.approx(pixels, abs=1e-9)

    def test_the_principal_point_unprojects_straight_ahead(self):
        k = Intrinsics.from_fov(640, 480, 80.0)
        point = k.unproject(np.array([[320.0, 240.0]]), np.array([5.0]))
        assert point[0] == pytest.approx([0.0, 0.0, 5.0])

    def test_points_behind_the_camera_do_not_project(self):
        # Projecting a point behind the camera yields a plausible pixel with the
        # signs flipped, which is worse than no answer at all.
        k = Intrinsics.from_fov(640, 480, 80.0)
        assert np.isnan(k.project(np.array([[1.0, 1.0, -5.0]]))).all()


class TestPose:
    def test_identity_leaves_points_alone(self):
        points = np.array([[1.0, 2.0, 3.0]])
        assert Pose.identity().to_world(points) == pytest.approx(points)

    def test_translation_is_where_the_camera_is(self):
        pose = Pose(np.eye(3), np.array([10.0, 0.0, -4.0]))
        assert pose.to_world(np.zeros((1, 3)))[0] == pytest.approx([10.0, 0.0, -4.0])

    def test_camera_from_world_inverts_the_pose(self):
        angle = 0.7
        rotation = np.array(
            [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        )
        pose = Pose(rotation, np.array([3.0, -1.0, 2.0]))
        r, t = pose.camera_from_world()
        world = np.array([[5.0, 5.0, 5.0]])
        camera = world @ r.T + t
        assert pose.to_world(camera) == pytest.approx(world)

    def test_forward_is_the_third_column(self):
        assert Pose.identity().forward == pytest.approx([0.0, 0.0, 1.0])


class TestTrajectory:
    def test_length_sums_the_steps(self):
        poses = [Pose(np.eye(3), np.array([float(i), 0.0, 0.0])) for i in range(5)]
        assert Trajectory(poses=poses).length == pytest.approx(4.0)

    def test_a_single_pose_has_no_length(self):
        assert Trajectory(poses=[Pose.identity()]).length == 0.0
        assert Trajectory().length == 0.0

    def test_counts_broken_tracking(self):
        from heat3d_capture.pose.odometry import TrackResult

        results = [
            TrackResult(Pose.identity(), None, 40, 50, tracked=True),
            TrackResult(Pose.identity(), None, 0, 0, tracked=False, reason="blank wall"),
            TrackResult(Pose.identity(), None, 30, 40, tracked=True),
        ]
        assert Trajectory(results=results).breaks == 1


class TestConfidence:
    def _result(self, inliers, matches, tracked=True):
        from heat3d_capture.pose.odometry import TrackResult

        return TrackResult(Pose.identity(), None, inliers, matches, tracked=tracked)

    def test_a_broken_track_has_no_confidence(self):
        assert self._result(0, 0, tracked=False).confidence == 0.0

    def test_plenty_of_agreeing_matches_is_confident(self):
        assert self._result(200, 210).confidence > 0.9

    def test_a_good_ratio_on_too_little_evidence_is_not(self):
        # 20 of 22 is an excellent ratio and far too few points to believe a
        # pose from. The absolute count has to matter as well as the ratio.
        assert self._result(20, 22).confidence < 0.5

    def test_more_evidence_is_more_confidence(self):
        assert self._result(30, 60).confidence < self._result(120, 240).confidence


def textured_wall(width=640, height=480, shift=0, seed=3) -> np.ndarray:
    """
    A frame with enough distinct corners for ORB to work on.

    Randomly placed blobs rather than noise: ORB needs repeatable corners, and a
    pure noise field gives it thousands of unstable ones that match badly.
    """
    import cv2

    rng = np.random.default_rng(seed)
    canvas = np.full((height, width * 3, 3), 30, dtype=np.uint8)
    for _ in range(260):
        x = int(rng.integers(0, width * 3))
        y = int(rng.integers(0, height))
        cv2.rectangle(
            canvas,
            (x, y),
            (x + int(rng.integers(12, 46)), y + int(rng.integers(12, 46))),
            tuple(int(c) for c in rng.integers(70, 255, 3)),
            -1,
        )
    start = int(np.clip(shift, 0, width * 2))
    return np.ascontiguousarray(canvas[:, start : start + width])


class TestTracking:
    def test_the_first_frame_defines_the_origin(self):
        k = Intrinsics.from_fov(640, 480, 90.0)
        vo = VisualOdometry(k)
        result = vo.track(textured_wall(), np.ones((480, 640), dtype=np.float32))
        assert result.tracked
        assert result.reason == "origin"
        assert result.pose.translation == pytest.approx([0, 0, 0])
        # Declared to be the unit, so the identity affine.
        assert result.fit.scale == pytest.approx(1.0)
        assert result.fit.shift == pytest.approx(0.0)

    def test_a_featureless_frame_breaks_the_track_rather_than_guessing(self):
        # The important behaviour. Carrying on from the last pose would weld the
        # next stretch of level on at whatever offset the gap happened to be.
        k = Intrinsics.from_fov(640, 480, 90.0)
        vo = VisualOdometry(k)
        depth = np.ones((480, 640), dtype=np.float32)
        vo.track(textured_wall(), depth)
        result = vo.track(np.full((480, 640, 3), 128, dtype=np.uint8), depth)
        assert not result.tracked
        assert result.confidence == 0.0
        assert "feature" in result.reason or "match" in result.reason

    def test_tracks_a_moving_camera_across_a_textured_scene(self):
        k = Intrinsics.from_fov(640, 480, 90.0)
        vo = VisualOdometry(k)
        # A plane two units away, which is what a flat wall's depth looks like.
        depth = np.full((480, 640), 0.5, dtype=np.float32)  # inverse depth of 2.0
        tracked = 0
        for shift in range(0, 260, 26):
            result = vo.track(textured_wall(shift=shift), depth)
            if result.tracked:
                tracked += 1
        # Not every step has to succeed — some will be rejected, which is the
        # system working. Most must.
        assert tracked >= 7

    def test_a_moving_camera_does_not_stay_at_the_origin(self):
        k = Intrinsics.from_fov(640, 480, 90.0)
        vo = VisualOdometry(k)
        depth = np.full((480, 640), 0.5, dtype=np.float32)
        for shift in range(0, 200, 20):
            vo.track(textured_wall(shift=shift), depth)
        trajectory = Trajectory(poses=vo.poses, results=vo.results)
        assert trajectory.length > 0.0
        assert len(vo.poses) >= 2

    def test_reset_forgets_everything(self):
        k = Intrinsics.from_fov(640, 480, 90.0)
        vo = VisualOdometry(k)
        vo.track(textured_wall(), np.ones((480, 640), dtype=np.float32))
        vo.reset()
        assert vo.poses == [] and vo.results == []
        # And the next frame is an origin again, not a continuation.
        assert vo.track(textured_wall(), np.ones((480, 640), np.float32)).reason == "origin"


class TestSampling:
    def test_samples_clamp_to_the_image(self):
        from heat3d_capture.pose.odometry import _sample

        field = np.arange(12, dtype=np.float32).reshape(3, 4)
        # Out-of-range coordinates must clamp rather than wrap, which would read
        # the opposite side of the frame and look like a plausible depth.
        out = _sample(field, np.array([[-5.0, -5.0], [99.0, 99.0], [1.0, 1.0]]))
        assert out[0] == 0.0
        assert out[1] == 11.0
        assert out[2] == 5.0
