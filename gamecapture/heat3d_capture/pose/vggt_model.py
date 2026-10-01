"""
Running VGGT over a capture.

Kept apart from `feedforward.py` so that the stitching maths — which is where the
mistakes hide and where the tests live — stays pure NumPy and needs neither torch
nor a GPU to exercise. This module is the part that cannot be tested without both,
and it is deliberately thin.

Frames, not files
-----------------
VGGT's own helper reads image files from disk. A capture here is a video or a
window, so frames arrive as arrays and the preprocessing is done directly: resize
so the long side is 518 with both sides a multiple of 14 — the patch size — and
normalise. Writing every frame to a temporary JPEG first would cost more than the
inference for short windows and would quantise the input for no reason.
"""

from __future__ import annotations

import numpy as np

from .feedforward import DEFAULT_MODEL, MODELS, Window

#: The model's patch size. Both image sides must be a multiple of it.
PATCH = 14

#: Long side after resizing, as VGGT was trained.
LONG_SIDE = 518


def prepare(frames: list[np.ndarray]) -> "np.ndarray":
    """
    Resize and normalise frames into the tensor VGGT expects.

    Takes RGB uint8 (H, W, 3) and returns float32 (N, 3, H', W') with both sides
    a multiple of the patch size. Aspect ratio is preserved: a 16:9 capture
    becomes 518x294, not a square, because squashing it would misstate the
    camera's field of view and bend the recovered path.
    """
    import cv2

    if not frames:
        raise ValueError("no frames to prepare")

    height, width = frames[0].shape[:2]
    if width >= height:
        new_width = LONG_SIDE
        new_height = max(PATCH, round(height * LONG_SIDE / width / PATCH) * PATCH)
    else:
        new_height = LONG_SIDE
        new_width = max(PATCH, round(width * LONG_SIDE / height / PATCH) * PATCH)

    out = np.empty((len(frames), 3, new_height, new_width), dtype=np.float32)
    for i, frame in enumerate(frames):
        if frame.shape[:2] != (height, width):
            raise ValueError(
                f"frame {i} is {frame.shape[:2]}, but the first is {(height, width)}; "
                "a window must be uniform"
            )
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        out[i] = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    return out


class VggtRunner:
    """
    Loads VGGT once and solves windows with it.

    The model is several gigabytes and takes a noticeable time to load, so a run
    over a whole capture keeps one instance rather than paying that per window.
    """

    def __init__(self, model: str = DEFAULT_MODEL, *, device: str | None = None):
        if model not in MODELS:
            raise KeyError(f"unknown model {model!r}; choose from {sorted(MODELS)}")
        self.model_id = MODELS[model]
        self._device = device
        self._model = None

    @property
    def device(self) -> str:
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _load(self):
        if self._model is None:
            from vggt.models.vggt import VGGT

            self._model = VGGT.from_pretrained(self.model_id).to(self.device).eval()
        return self._model

    def solve(
        self, frames: list[np.ndarray], indices: list[int], *, keep_points: bool = False
    ) -> Window:
        """
        Solve one window.

        `keep_points` is off by default because the point map is the largest
        thing the model produces — a 16-frame window at 518x294 is about 30 MB of
        float32 — and the camera path alone answers the question of whether the
        shape came out right.
        """
        import torch

        model = self._load()
        batch = torch.from_numpy(prepare(frames)).to(self.device)

        # bfloat16 rather than float16: the point maps carry world coordinates
        # that can be far from the origin, and half precision runs out of
        # exponent there while bfloat16 does not.
        with torch.no_grad(), torch.amp.autocast(self.device, dtype=torch.bfloat16):
            predictions = model(batch[None])

        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        extrinsic, _ = pose_encoding_to_extri_intri(
            predictions["pose_enc"], batch.shape[-2:]
        )
        extrinsic = extrinsic[0].float().cpu().numpy()

        # The model returns world-from-camera as a 3x4; the centre is where the
        # camera sits, which is -R^T t, not the translation itself.
        rotations = np.stack([e[:3, :3].T for e in extrinsic])
        centres = np.stack([-e[:3, :3].T @ e[:3, 3] for e in extrinsic])

        points = confidence = None
        if keep_points and "world_points" in predictions:
            points = predictions["world_points"][0].float().cpu().numpy()
            if "world_points_conf" in predictions:
                confidence = predictions["world_points_conf"][0].float().cpu().numpy()

        del predictions, batch
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return Window(
            frames=list(indices),
            centres=centres,
            rotations=rotations,
            points=points,
            confidence=confidence,
        )
