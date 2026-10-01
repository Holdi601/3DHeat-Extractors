"""
Which hardware this machine can actually scan on, and what to expect from it.

Two jobs, and the second matters as much as the first.

**Pick a backend that works.** ONNX Runtime reaches the GPU through execution
providers, and which ones exist depends on how onnxruntime was installed, not on
what hardware is present. So availability is discovered rather than assumed, and
the ladder falls through to something that always works:

    TensorRT  -> CUDA  -> DirectML  -> CPU
    (NVIDIA)     (NVIDIA)  (any DX12)   (anything)

DirectML is the one that makes this vendor-neutral. It runs on any Direct3D 12
device — AMD, Intel, NVIDIA alike — which is why it is the default install rather
than CUDA. NVIDIA is faster with its own providers and is welcome to them; it is
not allowed to be the only option. Measured on an RTX 5080 here, DirectML reached
54 fps against 3.5 fps on CPU with identical output, so the fallback is a real
fallback and not a token one.

**Say what the machine will do before it is asked to do it.** A scan is a slow,
physical thing — someone walks a level for several minutes — and discovering
afterwards that their hardware could only manage a coarse result is the worst
possible moment. `describe_expectations()` turns a measured throughput into plain
statements about resolution, scan length and what the output will look like, so
the decision is made before the walk rather than after it.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import dataclass, field
from typing import Literal

Vendor = Literal["nvidia", "amd", "intel", "apple", "cpu", "unknown"]

#: Best first. Each entry is (provider name, short label, whether it is a GPU).
_LADDER: list[tuple[str, str, bool]] = [
    ("TensorrtExecutionProvider", "TensorRT", True),
    ("CUDAExecutionProvider", "CUDA", True),
    ("DmlExecutionProvider", "DirectML", True),
    ("ROCMExecutionProvider", "ROCm", True),
    ("CoreMLExecutionProvider", "CoreML", True),
    ("CPUExecutionProvider", "CPU", False),
]

_VENDOR_PATTERNS: list[tuple[str, Vendor]] = [
    (r"nvidia|geforce|rtx|gtx|quadro|tesla", "nvidia"),
    (r"\bamd\b|radeon|\brx\s*\d|vega|firepro", "amd"),
    (r"\bintel\b|\barc\b|iris|uhd graphics|hd graphics", "intel"),
    (r"apple m\d", "apple"),
]


def _vendor_of(name: str) -> Vendor:
    lowered = name.lower()
    for pattern, vendor in _VENDOR_PATTERNS:
        if re.search(pattern, lowered):
            return vendor
    return "unknown"


@dataclass(frozen=True)
class Backend:
    """One way of running the model, and what it is."""

    provider: str
    #: Short name of the execution path, e.g. "DirectML".
    api: str
    #: The adapter this will run on, as the system reports it.
    device: str
    vendor: Vendor
    gpu: bool

    @property
    def label(self) -> str:
        return f"{self.device} via {self.api}" if self.gpu else f"{self.device} (CPU)"


def list_adapters() -> list[str]:
    """
    Display adapters, by name, without assuming a vendor's tooling is present.

    `nvidia-smi` would answer on one third of machines and be absent on the rest,
    which is precisely the bias this module exists to remove. On Windows the
    display-adapter class key lists every adapter regardless of who made it, so
    that is read instead. Failure is not an error — the name is for telling the
    user what they are running on, and an unnamed backend still works.
    """
    if platform.system() != "Windows":
        return []
    try:
        out = subprocess.run(
            [
                "reg",
                "query",
                r"HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}",
                "/s",
                "/v",
                "DriverDesc",
            ],
            capture_output=True,
            text=True,
            timeout=6,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []

    names: list[str] = []
    for match in re.finditer(r"DriverDesc\s+REG_SZ\s+(.+)", out):
        name = match.group(1).strip()
        # The class key also holds non-rendering entries — remote-desktop mirror
        # drivers and the Basic Render Driver — which are not what anyone means
        # by "my graphics card".
        if name and name not in names and "basic render" not in name.lower():
            names.append(name)
    return names


def available_backends() -> list[Backend]:
    """Every backend this installation can actually use, best first."""
    try:
        import onnxruntime as ort

        present = set(ort.get_available_providers())
    except ImportError:
        present = {"CPUExecutionProvider"}

    adapters = list_adapters()
    cpu_name = platform.processor() or "CPU"

    backends: list[Backend] = []
    for provider, api, is_gpu in _LADDER:
        if provider not in present:
            continue
        if not is_gpu:
            backends.append(
                Backend(provider, api, _tidy_cpu(cpu_name), "cpu", gpu=False)
            )
            continue
        device = _adapter_for(api, adapters)
        backends.append(Backend(provider, api, device, _vendor_of(device), gpu=True))
    return backends


def _tidy_cpu(name: str) -> str:
    # `platform.processor()` on Windows returns the raw family/model/stepping
    # string, which is not a name anyone recognises.
    if re.match(r"^\w+ Family \d+", name):
        return f"{os.cpu_count() or '?'} cores"
    return name


def _adapter_for(api: str, adapters: list[str]) -> str:
    """
    Guess which adapter a GPU provider will land on.

    Vendor-locked providers name the matching adapter; DirectML takes the system
    default, which is the first enumerated one. This is for display only — if it
    is wrong the model still runs on whatever the provider actually chose, and a
    measured throughput is reported alongside it regardless.
    """
    wanted: Vendor | None = {"CUDA": "nvidia", "TensorRT": "nvidia", "ROCm": "amd"}.get(api)
    if wanted:
        for name in adapters:
            if _vendor_of(name) == wanted:
                return name
    return adapters[0] if adapters else "GPU"


def select_backend(prefer: str | None = None) -> Backend:
    """
    The backend to use. `prefer` names a provider or an api, case-insensitively.

    Always returns something: CPU is in the ladder precisely so that this cannot
    fail on a machine with no usable GPU, which would otherwise turn "slow" into
    "does not run at all".
    """
    backends = available_backends()
    if not backends:
        return Backend("CPUExecutionProvider", "CPU", _tidy_cpu(platform.processor()), "cpu", False)
    if prefer:
        needle = prefer.lower()
        for backend in backends:
            if needle in (backend.provider.lower(), backend.api.lower()):
                return backend
    return backends[0]


@dataclass
class Expectation:
    """What a scan on this machine will be like, in plain terms."""

    backend: Backend
    #: Measured depth-model throughput, frames per second.
    fps: float
    tier: Literal["realtime", "workable", "slow", "impractical"]
    headline: str
    notes: list[str] = field(default_factory=list)


#: Frames per second of depth estimation, and what that means for a scan. The
#: boundaries are about the *experience*, not the number: above roughly fifteen
#: the overlay keeps up with someone walking, and below about two a scan takes
#: longer to process than it took to record.
_TIERS: list[tuple[float, str, str]] = [
    (25.0, "realtime", "Live scanning at full detail."),
    (12.0, "workable", "Live scanning, at reduced capture resolution."),
    (2.0, "slow", "Record first, reconstruct afterwards."),
    (0.0, "impractical", "This machine is too slow to scan usefully."),
]


def describe_expectations(backend: Backend, fps: float) -> Expectation:
    """
    Turn a measured throughput into advice.

    Deliberately said before a scan rather than discovered during one. Someone is
    about to spend several minutes walking a level; finding out afterwards that
    their hardware could only manage a coarse result is the worst possible moment
    to learn it.
    """
    tier, headline = next(
        (name, text) for threshold, name, text in _TIERS if fps >= threshold
    )
    notes: list[str] = []

    if not backend.gpu:
        notes.append(
            "Running on the processor because no supported GPU was found. "
            "Installing onnxruntime-directml usually fixes this on Windows, on "
            "any graphics card from the last decade."
        )
    elif backend.vendor == "nvidia" and backend.api == "DirectML":
        notes.append(
            "This NVIDIA card can go faster with onnxruntime-gpu, which adds the "
            "CUDA provider. DirectML works fine; it is just not the quickest "
            "option here."
        )

    if tier == "realtime":
        notes.append("Walk at a normal pace; the overlay will keep up.")
    elif tier == "workable":
        notes.append("Walk slowly and turn gently — fast turns will leave gaps.")
    elif tier == "slow":
        notes.append(
            "Record a video of the walk and reconstruct from the file afterwards. "
            "The result is identical; only the feedback is not live."
        )
    else:
        notes.append(
            "Scanning would take many times longer than the recording. A shorter "
            "route, or a smaller model, is the practical option."
        )
    return Expectation(backend=backend, fps=fps, tier=tier, headline=headline, notes=notes)
