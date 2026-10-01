"""
Monocular depth.

The one thing external capture cannot give us is geometry. The game knows the
depth of every pixel exactly and it is sitting in a buffer we have deliberately
chosen not to touch, so depth has to be inferred from the picture instead. That
is the accuracy ceiling of this whole approach and it is worth being plain about:
a hooked depth buffer would be exact, and this is an estimate.

Relative, then anchored
-----------------------
Depth Anything V2 predicts *relative inverse* depth — a disparity-like field that
is correct up to an unknown affine transform, ``metric_disparity = a * predicted
+ b``. It does not know how big anything is, and it cannot: a photograph of a
room and a photograph of a doll's house are the same picture.

That is not a problem to work around, it is the right division of labour. The
network is very good at the part that is hard from one image (relative shape,
edges, thin structure) and silent about the part it cannot know (scale). Scale
comes from somewhere that actually knows it:

- Forza, from the telemetry feed — exact positions, so the baseline between two
  frames is known in metres and `a` and `b` fall out of it.
- Anywhere else, from the camera path recovered by visual odometry, which fixes
  the geometry up to one global factor for the whole capture. That last factor is
  pinned once from something of known size rather than per frame.

Fitting an affine transform per frame, against whatever anchors are available,
is what `fit_scale()` does. Keeping that separate from the network means a better
depth model can be swapped in without touching how scale is resolved, and a
better source of scale can be swapped in without touching the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

#: Relative models predict inverse depth up to an affine transform. The metric
#: variants predict metres directly and need no anchors, at some cost in edge
#: quality and a strong bias toward the indoor or outdoor scenes they were tuned
#: on — which is why the relative model plus real anchors is the default.
#:
#: **The sizes are not licensed alike**, and the difference decides whether the
#: output may be used at work:
#:
#:     Small   Apache-2.0      commercial use permitted
#:     Base    CC-BY-NC-4.0    non-commercial only
#:     Large   CC-BY-NC-4.0    non-commercial only
#:
#: Base was the default here until that was checked, which would have quietly
#: made every scan a licence problem for anyone using this for work. Small is
#: the default now, and the larger two are opt-in with the restriction stated.
RELATIVE_MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}

#: Which of the above may be used commercially. Checked against the model cards
#: rather than assumed from the family name.
COMMERCIAL_MODELS = {"small"}
METRIC_MODELS = {
    "outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
}

#: Apache-2.0, so a scan made with it carries no licence restriction.
DEFAULT_MODEL = RELATIVE_MODELS["small"]

#: Predicted inverse depth at or below this is sky, whatever the affine fit
#: would make of it. Not zero, because a network asked about a bright overcast
#: sky returns a small positive number rather than an exact zero — and the
#: anchoring then turns that into a confident distance.
SKY_DISPARITY = 1e-4


@dataclass
class DepthMap:
    """One frame's depth prediction, at the source frame's resolution."""

    #: (H, W) float32. Inverse depth for a relative model, metres for a metric one.
    values: np.ndarray
    #: True when `values` are already metres and need no anchoring.
    metric: bool

    @property
    def shape(self) -> tuple[int, int]:
        return (self.values.shape[0], self.values.shape[1])


@dataclass
class AffineFit:
    """`metric_disparity = scale * predicted + shift`, and how well it held."""

    scale: float
    shift: float
    #: Median absolute residual in disparity units, over the anchors used.
    residual: float
    #: How many anchors survived the robust fit.
    inliers: int

    def apply(self, disparity: np.ndarray) -> np.ndarray:
        """
        Turn predicted inverse depth into metric depth, in metres.

        Disparity at or below zero is a prediction of something at or beyond
        infinity — sky, mostly. Those become +inf rather than a negative or
        enormous depth, so the fusion stage can discard them by a finite check
        instead of by an arbitrary distance threshold.

        The check is on the *predicted* disparity as well as on the anchored one,
        and that second test is the one that matters. The anchored value is
        `scale · predicted + shift`, so a sky pixel predicting zero comes out at
        `shift` — a small positive number, not zero — and inverts to a large but
        perfectly finite distance. Every sky pixel in the frame then lands on a
        shell at that distance and fuses into a surface that was never there.

        Measured on a synthetic drive where half of every frame is sky: it put
        35,000 of 92,000 mesh vertices into a phantom dome.
        """
        disparity = np.asarray(disparity, dtype=np.float32)
        metric_disparity = self.scale * disparity + self.shift
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(metric_disparity > 1e-6, 1.0 / metric_disparity, np.inf)
        # Nothing the network called sky is rescued by an offset.
        depth = np.where(disparity > SKY_DISPARITY, depth, np.inf)
        return depth.astype(np.float32)


def fit_scale(
    predicted: Sequence[float] | np.ndarray,
    metric_depth: Sequence[float] | np.ndarray,
    *,
    iterations: int = 12,
) -> AffineFit:
    """
    Solve the affine transform from anchors of known depth.

    Least squares in *disparity* rather than in depth, because the network's
    error is roughly uniform in disparity: fitting in depth lets a handful of
    distant anchors, where a pixel of disparity error is tens of metres, dominate
    every nearby one and tip the whole frame.

    Iteratively reweighted, because anchors are not clean. They come from feature
    matches and from telemetry, and both produce occasional gross outliers —
    a match on a moving car, a tracking glitch. One such anchor in an ordinary
    least-squares fit drags the entire frame's scale with it, and the result is a
    frame that fuses into the volume in the wrong place and corrupts geometry
    that was previously correct.
    """
    predicted = np.asarray(predicted, dtype=np.float64).ravel()
    metric_depth = np.asarray(metric_depth, dtype=np.float64).ravel()
    if predicted.size != metric_depth.size:
        raise ValueError(
            f"{predicted.size} predictions against {metric_depth.size} depths"
        )

    usable = np.isfinite(predicted) & np.isfinite(metric_depth) & (metric_depth > 1e-6)
    predicted = predicted[usable]
    target = 1.0 / metric_depth[usable]
    # Two points define an affine transform; fewer cannot be fitted at all, and
    # exactly two is fitted but believed only as far as its residual says.
    if predicted.size < 2:
        raise ValueError(f"need at least 2 usable anchors, got {predicted.size}")

    weights = np.ones_like(predicted)
    scale, shift = 1.0, 0.0
    for _ in range(iterations):
        design = np.stack([predicted, np.ones_like(predicted)], axis=1)
        weighted = design * weights[:, None]
        solution, *_ = np.linalg.lstsq(weighted, target * weights, rcond=None)
        scale, shift = float(solution[0]), float(solution[1])

        residuals = np.abs(design @ solution - target)
        # Median absolute deviation as the scale of "normal" error, so the
        # threshold adapts to the frame instead of being a constant that is too
        # tight close up and too loose at distance.
        mad = float(np.median(residuals)) or 1e-9
        weights = 1.0 / (1.0 + (residuals / (3.0 * mad)) ** 2)

    residuals = np.abs(scale * predicted + shift - target)
    mad = float(np.median(residuals))
    return AffineFit(
        scale=scale,
        shift=shift,
        residual=mad,
        inliers=int(np.count_nonzero(residuals <= 3.0 * (mad or 1e-9))),
    )


class DepthEstimator:
    """
    Depth Anything V2 on the GPU.

    Loaded lazily so that importing this module — which the tests and the CLI
    both do — costs nothing until depth is actually wanted. A few hundred
    megabytes of weights and a CUDA context is not something to pay for on
    `--help`.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        half: bool = True,
    ):
        self.model_id = model_id
        self.metric = "metric" in model_id.lower()
        self._device = device
        self._half = half
        self._model = None
        self._processor = None

    @property
    def device(self) -> str:
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        # Half precision only on the GPU: CPU fp16 is emulated and slower than
        # the fp32 path it replaces.
        dtype = torch.float16 if (self._half and self.device == "cuda") else torch.float32
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = (
            AutoModelForDepthEstimation.from_pretrained(self.model_id, dtype=dtype)
            .to(self.device)
            .eval()
        )

    def predict(self, images: np.ndarray | Sequence[np.ndarray]) -> list[DepthMap]:
        """
        Predict depth for one frame or a batch, each at its own input size.

        Batched because the GPU is idle between frames otherwise: the model is
        small enough that a single 960-pixel frame does not fill a 5080, and the
        per-call overhead dominates.
        """
        import torch

        self._load()
        batch = [images] if isinstance(images, np.ndarray) and images.ndim == 3 else list(images)
        if not batch:
            return []

        # Grouped by frame size before batching. The processor preserves aspect
        # ratio, so two different shapes become two different tensor shapes and
        # cannot be stacked — a real capture is uniform and forms one group, but
        # a mixed list must not be a crash.
        groups: dict[tuple[int, int], list[int]] = {}
        for i, im in enumerate(batch):
            groups.setdefault((im.shape[0], im.shape[1]), []).append(i)

        out: list[DepthMap | None] = [None] * len(batch)
        with torch.inference_mode():
            for (h, w), positions in groups.items():
                inputs = self._processor(
                    images=[batch[i] for i in positions], return_tensors="pt"
                ).to(self.device)
                if self._half and self.device == "cuda":
                    inputs = inputs.to(torch.float16)
                predicted = self._model(**inputs).predicted_depth

                # The model works at its own fixed resolution; bring each frame
                # back to the size it came in at, so depth and pixels correspond
                # one to one and reprojection is not skewed.
                resized = torch.nn.functional.interpolate(
                    predicted.unsqueeze(1).float(),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                )
                values = resized.squeeze(1).cpu().numpy().astype(np.float32)
                for slot, position in enumerate(positions):
                    out[position] = DepthMap(values=values[slot], metric=self.metric)

        return [d for d in out if d is not None]
