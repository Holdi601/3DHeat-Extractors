"""
Tests for HUD and camera-mounted geometry detection.

The synthetic room is given a HUD and a weapon model and then reconstructed
twice, with the detector on and off. That comparison is the only honest way to
test this: a mask can be measured as "correct" against a hand-drawn answer and
still not help, and what matters is whether the reconstruction is better for it.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_capture.capture.screenmask import MIN_FRAMES, ScreenMask
from heat3d_capture.pose.odometry import Intrinsics
from heat3d_capture.scan.reconstruct import Reconstructor

from .scene import ROOM, survey


def add_hud(image: np.ndarray, *, weapon: bool = True) -> np.ndarray:
    """
    Paint a plausible HUD and a weapon onto a frame.

    Deliberately high-contrast and full of corners, because that is what makes a
    real HUD dangerous: it is the most repeatable feature in the frame and it
    never moves, so every match it contributes is evidence the camera is still.
    """
    import cv2

    out = image.copy()
    h, w = out.shape[:2]

    # Crosshair, dead centre — the single worst offender.
    cv2.line(out, (w // 2 - 12, h // 2), (w // 2 + 12, h // 2), (255, 255, 255), 2)
    cv2.line(out, (w // 2, h // 2 - 12), (w // 2, h // 2 + 12), (255, 255, 255), 2)
    # Health bar and ammo counter.
    cv2.rectangle(out, (14, h - 34), (14 + w // 5, h - 18), (30, 220, 60), -1)
    cv2.rectangle(out, (14, h - 34), (14 + w // 5, h - 18), (255, 255, 255), 2)
    cv2.putText(out, "128 / 240", (w - 130, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    # Minimap, with internal detail so it offers plenty of corners.
    cv2.rectangle(out, (w - 108, 12), (w - 12, 108), (20, 24, 30), -1)
    cv2.rectangle(out, (w - 108, 12), (w - 12, 108), (200, 200, 210), 2)
    for i in range(4):
        cv2.line(out, (w - 108, 12 + i * 24), (w - 12, 12 + i * 24), (90, 96, 110), 1)

    if weapon:
        # A gun barrel in the lower right, as a first-person game draws it. Real
        # geometry, and in the wrong coordinate system — it travels with the
        # camera, so fusing it smears a barrel along the whole path walked.
        corners = np.array(
            [[w - 10, h], [int(w * 0.52), h], [int(w * 0.64), int(h * 0.58)],
             [int(w * 0.80), int(h * 0.52)]], dtype=np.int32
        )
        cv2.fillPoly(out, [corners], (52, 48, 44))
        cv2.polylines(out, [corners], True, (120, 112, 100), 2)
    return out


class TestLearning:
    def test_says_nothing_before_it_has_seen_enough(self):
        # Masking on a handful of frames throws away real geometry on the basis
        # of statistics that are mostly noise.
        mask = ScreenMask()
        rng = np.random.default_rng(0)
        for _ in range(MIN_FRAMES - 2):
            mask.update((rng.random((120, 200, 3)) * 255).astype(np.uint8))
        assert not mask.ready
        assert mask.mask((120, 200)) is None
        assert "learning" in mask.describe()

    def test_a_parked_camera_teaches_it_nothing(self):
        # Everything is static when nobody moves. A detector that learned from
        # those frames would decide the whole level was a HUD.
        mask = ScreenMask()
        frame = (np.random.default_rng(1).random((120, 200, 3)) * 255).astype(np.uint8)
        for _ in range(40):
            mask.update(frame, moving=False)
        assert not mask.ready

    def test_a_moving_world_with_no_hud_masks_nothing(self):
        # The other failure: inventing a HUD where there is none and quietly
        # discarding a slice of every frame.
        k = Intrinsics.from_fov(320, 180, 90.0)
        mask = ScreenMask()
        for frame in survey(k, steps=24):
            mask.update(frame.image)
        assert mask.ready
        assert mask.static_fraction() < 0.1

    def test_finds_a_hud_bolted_to_a_moving_world(self):
        k = Intrinsics.from_fov(320, 180, 90.0)
        mask = ScreenMask()
        for frame in survey(k, steps=24):
            mask.update(add_hud(frame.image))
        assert mask.ready
        assert mask.static_fraction() > 0.02, "the HUD should have been noticed"
        full = mask.mask((180, 320))
        assert full is not None and full.shape == (180, 320)

    def test_the_mask_covers_the_crosshair_and_the_weapon(self):
        k = Intrinsics.from_fov(320, 180, 90.0)
        mask = ScreenMask()
        for frame in survey(k, steps=28):
            mask.update(add_hud(frame.image))
        full = mask.mask((180, 320))
        assert full is not None
        # Centre — the crosshair.
        assert full[90, 160] == 0, "the crosshair was not masked"
        # Lower right — the weapon.
        assert full[170, 300] == 0, "the weapon model was not masked"
        # And a patch of plain wall, which must survive.
        assert full[40, 40] == 255, "real geometry was masked"


class TestRefusals:
    def test_refuses_when_almost_everything_is_static(self):
        # A menu, a cutscene, a loading screen. The premise has failed, and
        # masking most of the screen would stop the scan rather than degrade it.
        mask = ScreenMask()
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        frame[:, :100] = 200
        rng = np.random.default_rng(3)
        for _ in range(40):
            noisy = frame.copy()
            noisy[:, 190:] = (rng.random((120, 10, 3)) * 255).astype(np.uint8)
            mask.update(noisy)
        assert mask.mask((120, 200)) is None

    def test_a_changed_frame_size_starts_again(self):
        mask = ScreenMask()
        rng = np.random.default_rng(4)
        for _ in range(20):
            mask.update((rng.random((120, 200, 3)) * 255).astype(np.uint8))
        mask.update((rng.random((240, 400, 3)) * 255).astype(np.uint8))
        assert mask.frames == 0


class TestAgainstReconstruction:
    """The comparison that decides whether any of this was worth doing."""

    @staticmethod
    def _run(frames, *, mask_hud: bool):
        k = Intrinsics.from_fov(480, 270, 90.0)
        r = Reconstructor(k, voxel=0.2, coverage_voxel=0.4, mask_hud=mask_hud, camera_height=None)
        for frame in frames:
            with np.errstate(divide="ignore"):
                inverse = np.where(np.isfinite(frame.depth), 1.0 / frame.depth, 0.0)
            r.add(add_hud(frame.image), inverse.astype(np.float32))
        return r

    @pytest.fixture(scope="class")
    def frames(self):
        return survey(Intrinsics.from_fov(480, 270, 90.0), steps=30)

    def test_masking_keeps_the_walked_distance_honest(self, frames):
        # The damage a HUD does is not cosmetic. A crosshair matches perfectly
        # between every pair of frames and says the camera did not move, which
        # compresses the recovered path.
        truth = np.array([f.pose.translation for f in frames])
        walked = float(np.linalg.norm(np.diff(truth, axis=0), axis=1).sum())

        with_mask = self._run(frames, mask_hud=True).trajectory.length
        without = self._run(frames, mask_hud=False).trajectory.length

        assert abs(with_mask - walked) <= abs(without - walked) + 0.35, (
            f"truth {walked:.2f} m, masked {with_mask:.2f} m, unmasked {without:.2f} m"
        )

    def test_the_weapon_is_not_fused_into_the_level(self, frames):
        # A camera-mounted object fused as world geometry smears along the whole
        # path. It shows up as surface far outside the room.
        masked = self._run(frames, mask_hud=True)
        mesh = masked.volume.to_mesh()
        if mesh.is_empty():
            pytest.skip("nothing fused")
        origin = frames[0].pose
        world = mesh.vertices.astype(np.float64) @ origin.rotation.T + origin.translation
        outside = (
            (world < -1.5).any(axis=1) | (world > ROOM[None, :] + 1.5).any(axis=1)
        ).mean()
        assert outside < 0.25, f"{outside:.0%} of the surface landed outside the room"

    def test_it_reports_what_it_is_ignoring(self, frames):
        state = self._run(frames, mask_hud=True).state
        assert state.mask_note
        assert 0.0 <= state.masked_fraction < 0.5
