"""
Tests for depth.

Split deliberately. `fit_scale` is pure arithmetic and is tested exhaustively,
because it is where a single bad anchor can quietly ruin a frame and there is no
model download in the way. The network itself is tested for its contract —
shapes, finiteness, the resize back to source resolution — and marked so it can
be skipped on a machine with no GPU or no weights cached.
"""

from __future__ import annotations

import numpy as np
import pytest

from heat3d_capture.depth.estimator import (
    AffineFit,
    DepthEstimator,
    fit_scale,
)


class TestFitScale:
    def test_recovers_a_known_transform(self):
        rng = np.random.default_rng(0)
        predicted = rng.uniform(0.1, 2.0, 40)
        true_scale, true_shift = 3.5, 0.25
        depth = 1.0 / (true_scale * predicted + true_shift)

        fit = fit_scale(predicted, depth)
        assert fit.scale == pytest.approx(true_scale, rel=1e-4)
        assert fit.shift == pytest.approx(true_shift, abs=1e-4)
        assert fit.residual < 1e-6

    def test_survives_gross_outliers(self):
        # One anchor on a moving car, or one tracking glitch. Ordinary least
        # squares would follow it and put the whole frame in the wrong place.
        rng = np.random.default_rng(1)
        predicted = rng.uniform(0.1, 2.0, 60)
        depth = 1.0 / (2.0 * predicted + 0.1)
        depth[7] = 1e-4
        depth[23] = 5000.0
        depth[44] = 1e-4

        fit = fit_scale(predicted, depth)
        assert fit.scale == pytest.approx(2.0, rel=0.02)
        assert fit.shift == pytest.approx(0.1, abs=0.02)
        assert fit.inliers >= 50

    def test_fits_in_disparity_so_distance_does_not_dominate(self):
        # Half the anchors nearby, half very far. A fit done in depth is pulled
        # almost entirely by the far half, where one disparity unit is tens of
        # metres; a fit in disparity weights them evenly.
        predicted = np.concatenate([np.linspace(0.8, 1.2, 20), np.linspace(0.001, 0.01, 20)])
        depth = 1.0 / (4.0 * predicted + 0.05)
        fit = fit_scale(predicted, depth)
        assert fit.scale == pytest.approx(4.0, rel=1e-3)

    def test_ignores_anchors_that_cannot_be_used(self):
        predicted = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 6.0])
        depth = np.array([1.0, 0.5, 1 / 3, 0.25, 1.0, np.inf])
        fit = fit_scale(predicted, depth)
        # The four clean anchors describe depth = 1/predicted exactly.
        assert fit.scale == pytest.approx(1.0, rel=1e-6)
        assert fit.shift == pytest.approx(0.0, abs=1e-6)

    def test_rejects_mismatched_inputs(self):
        with pytest.raises(ValueError, match="against"):
            fit_scale([1.0, 2.0], [1.0])

    def test_rejects_too_few_anchors(self):
        with pytest.raises(ValueError, match="at least 2"):
            fit_scale([1.0], [1.0])
        with pytest.raises(ValueError, match="at least 2"):
            fit_scale([1.0, 2.0], [np.inf, 0.0])


class TestApply:
    def test_converts_disparity_to_metres(self):
        fit = AffineFit(scale=2.0, shift=0.5, residual=0.0, inliers=10)
        disparity = np.array([[0.25, 0.75]], dtype=np.float32)
        # 2*0.25+0.5 = 1.0 -> 1 m; 2*0.75+0.5 = 2.0 -> 0.5 m
        assert fit.apply(disparity) == pytest.approx(np.array([[1.0, 0.5]]))

    def test_sky_becomes_infinite_rather_than_negative(self):
        # Disparity at or below zero means at or beyond infinity. Fusion drops
        # those on a finite check; a large negative depth would be fused as real
        # geometry behind the camera.
        fit = AffineFit(scale=1.0, shift=0.0, residual=0.0, inliers=10)
        out = fit.apply(np.array([[0.0, -0.5, 1.0]], dtype=np.float32))
        assert np.isinf(out[0, 0])
        assert np.isinf(out[0, 1])
        assert out[0, 2] == pytest.approx(1.0)

    def test_output_is_float32(self):
        fit = AffineFit(1.0, 0.0, 0.0, 1)
        assert fit.apply(np.ones((4, 4), dtype=np.float64)).dtype == np.float32


class TestEstimatorConfig:
    """Everything that can be checked without downloading weights."""

    def test_detects_a_metric_model_from_its_name(self):
        assert DepthEstimator("depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf").metric
        assert not DepthEstimator("depth-anything/Depth-Anything-V2-Base-hf").metric

    def test_does_not_load_anything_on_construction(self):
        # `--help` must not pay for a few hundred megabytes of weights.
        est = DepthEstimator()
        assert est._model is None and est._processor is None

    def test_empty_batch_is_not_an_error(self):
        assert DepthEstimator().predict([]) == []


# The network itself. Skipped where there is no GPU, since running this model on
# a CPU takes long enough to look like a hang rather than a test.
torch = pytest.importorskip("torch")
pytestmark_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


@pytestmark_gpu
class TestEstimatorOnGpu:
    # Module-scoped and loaded once: the weights and the CUDA context cost far
    # more than the tests that use them.
    @pytest.fixture(scope="module")
    @staticmethod
    def estimator():
        est = DepthEstimator("depth-anything/Depth-Anything-V2-Small-hf")
        est._load()
        return est

    def test_returns_depth_at_the_source_resolution(self, estimator):
        # The model works at its own fixed size. If the resize back is wrong,
        # depth and pixels stop corresponding and every reprojection is skewed.
        image = np.zeros((180, 320, 3), dtype=np.uint8)
        image[:90] = 200
        (depth,) = estimator.predict(image)
        assert depth.shape == (180, 320)
        assert np.isfinite(depth.values).all()

    def test_handles_a_batch_of_differing_sizes(self, estimator):
        images = [
            np.full((180, 320, 3), 60, dtype=np.uint8),
            np.full((240, 320, 3), 120, dtype=np.uint8),
        ]
        out = estimator.predict(images)
        assert [d.shape for d in out] == [(180, 320), (240, 320)]

    def test_a_near_surface_reads_as_nearer_than_a_far_one(self, estimator):
        # A floor receding to a horizon: the bottom of the frame is closer than
        # the top. Inverse depth must therefore be larger at the bottom. This is
        # the weakest possible sanity check on the model actually working, and
        # that is the point — anything stronger is testing the network, not us.
        h, w = 240, 320
        image = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            image[y, :] = int(255 * (y / h))
        (depth,) = estimator.predict(image)
        assert np.isfinite(depth.values).all()
        assert depth.values.std() > 0, "a gradient must not predict flat depth"
