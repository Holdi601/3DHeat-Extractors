"""
Convert a Depth Anything V2 checkpoint to ONNX, once, so the app never needs torch.

Run on a machine that has PyTorch; ship the `.onnx` it produces. The scan app
itself depends only on onnxruntime, which is the whole point: torch plus CUDA is
a multi-gigabyte install that works on one vendor's hardware, and onnxruntime
with DirectML is a small one that works on all of them.

    python tools/export_onnx.py --size base --out models/depth-base.onnx

Fixed spatial size rather than dynamic axes. DirectML and TensorRT both compile
per shape, so a dynamic height and width means recompiling on the first frame of
every new resolution — which in a live capture is a visible stall right when the
user is watching the overlay for feedback. Frames are letterboxed to this size
instead, which costs a little resolution and nothing in latency.

Exported through the TorchScript path (`dynamo=False`) rather than the newer
dynamo exporter, for two reasons found by trying the other one:

- The dynamo graph **does not run on DirectML at all**. It fails partway through
  with a runtime exception on a Reshape node, which would have left AMD and Intel
  users with no GPU path — the exact thing this conversion exists to prevent.
- It writes weights as a separate `.onnx.data` file. Two files that must travel
  together is a distribution footgun for no benefit at this size; the TorchScript
  path produces one self-contained file.

The tracer warns about baking shapes in as constants. That is correct and wanted
here: the spatial size is fixed by design, so there is nothing to generalise to.
"""

from __future__ import annotations

import argparse
from pathlib import Path

#: Depth Anything's backbone is a ViT with a patch size of 14, so the input must
#: divide by 14. 518 = 14 x 37 is the resolution the models were trained at.
PATCH = 14
DEFAULT_SIDE = 518

#: Licences differ by size: Small is Apache-2.0, Base and Large are CC-BY-NC-4.0.
MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def export(model_id: str, out: Path, side: int, opset: int) -> Path:
    import torch
    from transformers import AutoModelForDepthEstimation

    if side % PATCH:
        raise SystemExit(f"--side must be a multiple of {PATCH}; {side} is not")

    model = AutoModelForDepthEstimation.from_pretrained(model_id).eval()

    class Wrapped(torch.nn.Module):
        """
        Normalisation folded in, so the app does not have to reproduce it.

        The ImageNet mean and standard deviation are part of the model as far as
        anyone using it is concerned, and a preprocessing constant duplicated on
        the other side of a file format is a classic silent drift: change the
        checkpoint, forget the constants, and the depth is subtly wrong forever
        with nothing to point at. Taking uint8-ranged RGB in and doing it here
        means there is only one copy.
        """

        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            normalised = (pixels / 255.0 - self.mean) / self.std
            return self.inner(pixel_values=normalised).predicted_depth

    wrapped = Wrapped(model).eval()
    dummy = torch.zeros(1, 3, side, side, dtype=torch.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapped,
            (dummy,),
            str(out),
            input_names=["pixels"],
            output_names=["inverse_depth"],
            # Batch stays dynamic — batching is free throughput and does not
            # trigger the per-shape recompilation that varying H and W does.
            dynamic_axes={"pixels": {0: "batch"}, "inverse_depth": {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Small by default: it is the only size under Apache-2.0. Base and Large
    # are CC-BY-NC-4.0, so a model converted from them may not be used
    # commercially, and that restriction travels with every scan made.
    parser.add_argument("--size", choices=sorted(MODELS), default="small")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--side", type=int, default=DEFAULT_SIDE)
    # 17 is widely supported by both DirectML and CUDA execution providers.
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    out = args.out or Path("models") / f"depth-{args.size}-{args.side}.onnx"
    written = export(MODELS[args.size], out, args.side, args.opset)
    print(f"{written}  ({written.stat().st_size / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
