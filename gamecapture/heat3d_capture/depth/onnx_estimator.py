"""
Depth estimation through ONNX Runtime, so it runs on anyone's hardware.

The torch path (`estimator.py`) is kept for converting checkpoints and for
research, but it is not what the scan app uses. Torch with CUDA is several
gigabytes and runs on one vendor's cards; this is one 95 MiB file and an
onnxruntime install, and it reaches AMD and Intel GPUs through DirectML. Measured
on an RTX 5080: 54 fps on DirectML against 3.5 fps on the processor, from the
same file, with identical output.

Letterboxing, not stretching
----------------------------
The exported graph takes a fixed square input, and gameplay is 16:9. Squashing a
widescreen frame into a square would change the apparent shape of everything in
it, and since the whole output is a *geometric* estimate, that distortion would
be fused into the level as real wrong geometry rather than showing up as a
stretched picture. So frames are padded to square, and the padding is cropped
back off the depth map. It costs some working resolution and keeps the angles
honest.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..runtime.device import Backend, select_backend
from .estimator import DepthMap

#: Neutral grey for the letterbox margin. Black or white both read as a strong
#: edge to the network and bleed a false depth discontinuity into the real
#: content next to them; mid-grey is the least eventful thing to pad with.
_PAD_VALUE = 114


@dataclass
class _Letterbox:
    """How one frame was fitted into the square, so it can be undone."""

    top: int
    left: int
    height: int
    width: int
    source: tuple[int, int]


class OnnxDepthEstimator:
    """
    Runs the exported depth model on the best available backend.

    The session is built lazily and once. Building it is where a provider
    actually compiles kernels — on DirectML that is a noticeable pause — so it
    happens at an explicit `warm_up()` the UI can show progress for, rather than
    silently inside the first frame of a scan.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        backend: Backend | None = None,
        prefer: str | None = None,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"no depth model at {self.model_path}. Run tools/export_onnx.py to make one."
            )
        self.backend = backend or select_backend(prefer)
        self._session = None
        self._side: int | None = None

    @property
    def side(self) -> int:
        """The square input size the model was exported at."""
        self._build()
        assert self._side is not None
        return self._side

    def _build(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # The provider ladder already fell back if this one is missing, but a
        # provider can also fail at session creation — a driver too old for
        # DirectML, say. CPU is appended so that is a slow scan, not a crash.
        providers = [self.backend.provider]
        if self.backend.provider != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")
        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=providers
        )
        shape = self._session.get_inputs()[0].shape
        # (batch, 3, side, side); batch is dynamic, the spatial size is not.
        self._side = int(shape[2])

    def warm_up(self) -> float:
        """
        Build the session and run one frame. Returns seconds taken.

        Worth doing explicitly: the first inference on a freshly compiled graph
        is several times slower than the steady state, and attributing that to
        "scanning is slow" would be wrong.
        """
        started = time.perf_counter()
        self._build()
        blank = np.zeros((self.side, self.side, 3), dtype=np.uint8)
        self.predict(blank)
        return time.perf_counter() - started

    def measure_throughput(self, *, frames: int = 8) -> float:
        """
        Frames per second on this machine, measured rather than assumed.

        This is what the interface quotes before a scan. A table of expected
        speeds per GPU would be wrong on half the machines it was shown to —
        thermal state, laptop power profiles and driver versions all move it —
        and being wrong about this costs the user a wasted walk through a level.
        """
        self.warm_up()
        blank = np.zeros((self.side, self.side, 3), dtype=np.uint8)
        started = time.perf_counter()
        for _ in range(frames):
            self.predict(blank)
        elapsed = time.perf_counter() - started
        return frames / elapsed if elapsed > 0 else 0.0

    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, _Letterbox]:
        import cv2

        side = self.side
        h, w = image.shape[:2]
        scale = min(side / h, side / w)
        new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
        resized = cv2.resize(
            image,
            (new_w, new_h),
            # Area down, cubic up: the model is sensitive to the ringing that
            # a sharpening filter leaves on upscaled frames.
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        top = (side - new_h) // 2
        left = (side - new_w) // 2
        canvas = np.full((side, side, 3), _PAD_VALUE, dtype=np.uint8)
        canvas[top : top + new_h, left : left + new_w] = resized
        return canvas, _Letterbox(top, left, new_h, new_w, (h, w))

    def predict(self, images: np.ndarray | Sequence[np.ndarray]) -> list[DepthMap]:
        """Predict inverse depth for one frame or a batch, at source resolution."""
        import cv2

        self._build()
        batch = [images] if isinstance(images, np.ndarray) and images.ndim == 3 else list(images)
        if not batch:
            return []

        boxes: list[_Letterbox] = []
        tensor = np.empty((len(batch), 3, self.side, self.side), dtype=np.float32)
        for i, image in enumerate(batch):
            canvas, box = self._letterbox(image)
            boxes.append(box)
            # HWC uint8 to CHW float. Normalisation lives inside the graph, so
            # there is exactly one copy of the ImageNet constants and it is not
            # this one.
            tensor[i] = canvas.transpose(2, 0, 1).astype(np.float32)

        assert self._session is not None
        predicted = self._session.run(None, {"pixels": tensor})[0]

        out: list[DepthMap] = []
        for i, box in enumerate(boxes):
            field = predicted[i]
            if field.ndim == 3:
                field = field[0]
            # Crop the padding away before resizing back, so the margin never
            # bleeds into real pixels.
            cropped = field[box.top : box.top + box.height, box.left : box.left + box.width]
            restored = cv2.resize(
                cropped.astype(np.float32),
                (box.source[1], box.source[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            out.append(DepthMap(values=restored, metric=False))
        return out
