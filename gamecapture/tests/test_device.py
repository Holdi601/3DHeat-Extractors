"""
Tests for backend selection.

The thing worth testing here is the *policy*, not the hardware: that the ladder
prefers the fast path, that it never falls off the end, and above all that a
machine without NVIDIA hardware still gets a GPU. Real availability differs on
every machine, so providers are injected rather than probed.
"""

from __future__ import annotations

import pytest

from heat3d_capture.runtime import device as dev
from heat3d_capture.runtime.device import (
    Backend,
    available_backends,
    describe_expectations,
    select_backend,
)


@pytest.fixture
def fake(monkeypatch):
    """Pretend a given set of providers and adapters is present."""

    def apply(providers, adapters=()):
        import sys
        import types

        module = types.ModuleType("onnxruntime")
        module.get_available_providers = lambda: list(providers)
        monkeypatch.setitem(sys.modules, "onnxruntime", module)
        monkeypatch.setattr(dev, "list_adapters", lambda: list(adapters))

    return apply


DML = "DmlExecutionProvider"
CUDA = "CUDAExecutionProvider"
TRT = "TensorrtExecutionProvider"
CPU = "CPUExecutionProvider"


class TestLadder:
    def test_prefers_tensorrt_then_cuda_then_directml(self, fake):
        fake([CPU, DML, CUDA, TRT], ["NVIDIA GeForce RTX 5080"])
        assert [b.api for b in available_backends()] == ["TensorRT", "CUDA", "DirectML", "CPU"]
        assert select_backend().api == "TensorRT"

    def test_an_amd_machine_still_gets_a_gpu(self, fake):
        # The requirement this module exists for. No CUDA anywhere, and the
        # selected backend must still not be the processor.
        fake([CPU, DML], ["AMD Radeon RX 7900 XTX"])
        chosen = select_backend()
        assert chosen.gpu is True
        assert chosen.api == "DirectML"
        assert chosen.vendor == "amd"

    def test_intel_integrated_graphics_is_a_gpu_too(self, fake):
        fake([CPU, DML], ["Intel(R) Arc(TM) A770 Graphics"])
        chosen = select_backend()
        assert chosen.vendor == "intel"
        assert chosen.gpu is True

    def test_falls_back_to_cpu_rather_than_failing(self, fake):
        fake([CPU], [])
        chosen = select_backend()
        assert chosen.gpu is False
        assert chosen.provider == CPU

    def test_never_returns_nothing_even_with_no_providers(self, fake):
        fake([], [])
        assert select_backend().provider == CPU

    def test_survives_onnxruntime_being_absent(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def no_ort(name, *args, **kwargs):
            if name == "onnxruntime":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_ort)
        monkeypatch.setattr(dev, "list_adapters", list)
        assert select_backend().provider == CPU


class TestPreference:
    def test_honours_a_named_provider(self, fake):
        fake([CPU, DML, CUDA], ["NVIDIA GeForce RTX 5080"])
        assert select_backend(prefer="DmlExecutionProvider").api == "DirectML"

    def test_honours_a_named_api_case_insensitively(self, fake):
        fake([CPU, DML, CUDA], ["NVIDIA GeForce RTX 5080"])
        assert select_backend(prefer="directml").api == "DirectML"
        assert select_backend(prefer="CPU").gpu is False

    def test_ignores_a_preference_that_is_not_available(self, fake):
        # Asking for CUDA on an AMD machine must not produce a backend that
        # cannot run; the best available one is the useful answer.
        fake([CPU, DML], ["AMD Radeon RX 7900 XTX"])
        assert select_backend(prefer="cuda").api == "DirectML"


class TestAdapterNaming:
    def test_matches_a_vendor_locked_provider_to_its_adapter(self, fake):
        # Laptops enumerate the integrated adapter first; CUDA will not be
        # running on it, so naming it would be a lie in the UI.
        fake([CPU, DML, CUDA], ["Intel(R) UHD Graphics 630", "NVIDIA GeForce RTX 4060"])
        by_api = {b.api: b for b in available_backends()}
        assert by_api["CUDA"].device == "NVIDIA GeForce RTX 4060"
        assert by_api["DirectML"].device == "Intel(R) UHD Graphics 630"

    def test_labels_read_as_prose(self, fake):
        fake([CPU, DML], ["AMD Radeon RX 7900 XTX"])
        assert select_backend().label == "AMD Radeon RX 7900 XTX via DirectML"

    def test_copes_with_no_adapter_names(self, fake):
        fake([CPU, DML], [])
        assert select_backend().device == "GPU"


class TestExpectations:
    def _backend(self, **over) -> Backend:
        base = dict(
            provider=DML, api="DirectML", device="AMD Radeon RX 7900 XTX", vendor="amd", gpu=True
        )
        return Backend(**{**base, **over})

    @pytest.mark.parametrize(
        "fps,tier",
        [(60.0, "realtime"), (25.0, "realtime"), (15.0, "workable"), (5.0, "slow"), (0.5, "impractical")],
    )
    def test_tiers_follow_throughput(self, fps, tier):
        assert describe_expectations(self._backend(), fps).tier == tier

    def test_tells_a_cpu_user_how_to_get_a_gpu(self, fake):
        backend = self._backend(provider=CPU, api="CPU", device="16 cores", vendor="cpu", gpu=False)
        notes = " ".join(describe_expectations(backend, 3.0).notes)
        assert "directml" in notes.lower()

    def test_tells_an_nvidia_user_on_directml_about_cuda(self):
        # Not an error — it works — but leaving the faster path undiscovered
        # would be a silent loss.
        backend = self._backend(device="NVIDIA GeForce RTX 5080", vendor="nvidia")
        notes = " ".join(describe_expectations(backend, 40.0).notes)
        assert "onnxruntime-gpu" in notes

    def test_does_not_nag_an_amd_user_about_cuda(self):
        notes = " ".join(describe_expectations(self._backend(), 40.0).notes)
        assert "cuda" not in notes.lower()

    def test_a_slow_machine_is_told_to_record_first(self):
        notes = " ".join(describe_expectations(self._backend(), 4.0).notes)
        assert "record" in notes.lower()

    def test_every_tier_produces_a_headline_and_a_note(self):
        for fps in (100.0, 20.0, 6.0, 0.1):
            e = describe_expectations(self._backend(), fps)
            assert e.headline and e.notes


class TestRealMachine:
    """A smoke check against whatever this machine actually has."""

    def test_selection_works_here(self):
        chosen = select_backend()
        assert chosen.provider.endswith("ExecutionProvider")
        assert chosen.label
        expectation = describe_expectations(chosen, 30.0)
        assert expectation.tier in {"realtime", "workable", "slow", "impractical"}
