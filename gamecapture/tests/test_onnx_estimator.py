"""
Tests for the portable depth estimator.

Skipped whole when no exported model is present, since the `.onnx` is built by
`tools/export_onnx.py` rather than committed. What is tested without one is the
letterbox arithmetic, which is where a geometric error would silently distort
every reconstruction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from heat3d_capture.depth.onnx_estimator import OnnxDepthEstimator

MODELS = Path(__file__).resolve().parents[1] / "models"
FOUND = sorted(MODELS.glob("depth-*.onnx"))
needs_model = pytest.mark.skipif(not FOUND, reason="no exported model; see tools/export_onnx.py")


def test_missing_model_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="export_onnx"):
        OnnxDepthEstimator(tmp_path / "absent.onnx")


class TestLetterbox:
    """Geometry, checked without loading a model."""

    @pytest.fixture
    def est(self):
        est = OnnxDepthEstimator.__new__(OnnxDepthEstimator)
        est._side = 518
        est._session = object()  # so `side` does not try to build
        return est

    def test_widescreen_is_padded_not_stretched(self, est):
        # The one that matters. Squashing 16:9 into a square changes the apparent
        # shape of everything, and because the output is a geometric estimate,
        # that distortion would be fused into the level as real wrong geometry.
        image = np.zeros((540, 960, 3), dtype=np.uint8)
        canvas, box = est._letterbox(image)
        assert canvas.shape == (518, 518, 3)
        # 960x540 scaled by 518/960 gives 518x291, centred vertically.
        assert (box.width, box.height) == (518, 291)
        assert box.left == 0
        assert box.top == (518 - 291) // 2
        # Aspect ratio preserved to within a pixel of rounding.
        assert box.width / box.height == pytest.approx(960 / 540, rel=0.01)

    def test_padding_is_neutral_grey_not_black(self, est):
        # Black or white pad reads as a hard edge and bleeds a false depth
        # discontinuity into the content beside it.
        canvas, box = est._letterbox(np.zeros((540, 960, 3), dtype=np.uint8))
        assert canvas[0, 0].tolist() == [114, 114, 114]
        assert box.top > 0

    def test_a_tall_frame_pads_on_the_sides(self, est):
        canvas, box = est._letterbox(np.zeros((800, 400, 3), dtype=np.uint8))
        assert box.top == 0
        assert box.left > 0
        assert box.height == 518

    def test_a_square_frame_needs_no_padding(self, est):
        _, box = est._letterbox(np.zeros((300, 300, 3), dtype=np.uint8))
        assert (box.top, box.left) == (0, 0)
        assert (box.height, box.width) == (518, 518)

    def test_records_the_source_size_for_the_trip_back(self, est):
        _, box = est._letterbox(np.zeros((540, 960, 3), dtype=np.uint8))
        assert box.source == (540, 960)


@needs_model
class TestAgainstTheRealModel:
    @pytest.fixture(scope="module")
    @staticmethod
    def est():
        estimator = OnnxDepthEstimator(FOUND[0])
        estimator.warm_up()
        return estimator

    def test_runs_on_whatever_this_machine_has(self, est):
        assert est.backend.provider.endswith("ExecutionProvider")
        assert est.side % 14 == 0, "input must divide by the ViT patch size"

    def test_returns_depth_at_the_source_resolution(self, est):
        # If the un-letterbox is wrong, depth and pixels stop corresponding and
        # every reprojection is skewed — while still looking like a depth map.
        image = np.zeros((540, 960, 3), dtype=np.uint8)
        image[270:] = 200
        (depth,) = est.predict(image)
        assert depth.shape == (540, 960)
        assert np.isfinite(depth.values).all()

    def test_handles_an_odd_size(self, est):
        (depth,) = est.predict(np.zeros((237, 411, 3), dtype=np.uint8))
        assert depth.shape == (237, 411)

    def test_batches(self, est):
        images = [np.full((540, 960, 3), v, dtype=np.uint8) for v in (40, 90, 160)]
        out = est.predict(images)
        assert len(out) == 3
        assert all(d.shape == (540, 960) for d in out)

    def test_a_structured_frame_does_not_predict_flat_depth(self, est):
        h, w = 540, 960
        image = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            image[y, :] = int(255 * (y / h))
        (depth,) = est.predict(image)
        assert depth.values.std() > 1e-4

    def test_measures_a_throughput(self, est):
        fps = est.measure_throughput(frames=4)
        assert fps > 0
        assert np.isfinite(fps)

    def test_empty_batch_is_not_an_error(self, est):
        assert est.predict([]) == []
