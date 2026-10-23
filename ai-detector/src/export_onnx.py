"""Export a checkpoint to ONNX (dynamic batch) and check it against PyTorch.

Class names and input size are stored in the ONNX metadata so a deployment
only needs the .onnx file. The exported graph outputs logits; apply softmax
downstream.

  python -m src.export_onnx --checkpoint models/best.pt --out models/detector.onnx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .dataset import IMAGENET_MEAN, IMAGENET_STD
from .model import load_checkpoint


def export(checkpoint: str, out: str, opset: int = 17, atol: float = 1e-3) -> str:
    model, _ = load_checkpoint(checkpoint, "cpu")
    size = model.cfg.image_size
    dummy = torch.randn(2, 3, size, size)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        out,
        input_names=["image"],
        output_names=["logits"],
        dynamic_shapes=({0: torch.export.Dim("batch")},),
        opset_version=opset,
        dynamo=True,
        external_data=False,
    )

    proto = onnx.load(out)
    meta = {
        "class_names": json.dumps(list(model.cfg.class_names)),
        "image_size": str(size),
        "mean": json.dumps(IMAGENET_MEAN),
        "std": json.dumps(IMAGENET_STD),
        "backbone": model.cfg.backbone,
    }
    for k, v in meta.items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.checker.check_model(proto)
    onnx.save(proto, out)

    # Parity check on a different batch size than the export dummy.
    x = torch.randn(3, 3, size, size)
    with torch.no_grad():
        expected = model(x).numpy()
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"image": x.numpy()})[0]
    max_diff = float(np.abs(expected - got).max())
    if max_diff > atol:
        raise RuntimeError(f"ONNX output differs from PyTorch by {max_diff:.2e} (> {atol})")
    print(f"exported {out}  (max |torch - onnx| = {max_diff:.2e})")
    return out


def _main() -> None:
    parser = argparse.ArgumentParser(description="Export a checkpoint to ONNX.")
    parser.add_argument("--checkpoint", default="models/best.pt")
    parser.add_argument("--out", default="models/detector.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export(args.checkpoint, args.out, args.opset)


if __name__ == "__main__":
    _main()
