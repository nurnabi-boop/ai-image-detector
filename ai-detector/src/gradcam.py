"""Grad-CAM heatmaps over the backbone's last feature map.

Shows which regions push the model towards a class (by default the
predicted one). The heatmap is computed on the letterboxed model input and
then mapped back onto the original image.

  python -m src.gradcam --checkpoint models/best.pt --image path/to/img.jpg
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from .dataset import build_transforms
from .model import AIDetector, load_checkpoint


@dataclass
class Explanation:
    probs: dict[str, float]
    predicted: str
    target: str
    heatmap: np.ndarray  # (H, W) in [0, 1], same size as the original image
    overlay: Image.Image


def gradcam(model: AIDetector, x: torch.Tensor, target: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """Grad-CAM for a single preprocessed image tensor (1, 3, S, S).

    Returns (probs [C], cam [h, w] in [0, 1] at feature-map resolution).
    """
    model.eval()
    with torch.enable_grad():
        feats = model.features(x)
        if feats.ndim != 4:
            raise ValueError(
                f"Grad-CAM needs a (B, C, H, W) feature map, backbone gave {tuple(feats.shape)}"
            )
        feats.retain_grad()
        logits = model.classify(feats)
        probs = torch.softmax(logits, dim=1)[0]
        if target is None:
            target = int(probs.argmax())
        model.zero_grad(set_to_none=True)
        logits[0, target].backward()

    weights = feats.grad[0].mean(dim=(1, 2))  # (C,)
    cam = torch.relu((weights[:, None, None] * feats[0]).sum(dim=0))
    cam = cam.detach().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return probs.detach().cpu().numpy(), cam


def _unletterbox(cam: np.ndarray, image_size: int, orig_w: int, orig_h: int) -> np.ndarray:
    """Upsample the CAM to the model input, crop away the letterbox padding
    added by the eval transform (centered), resize to the original image."""
    cam = cv2.resize(cam, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    scale = image_size / max(orig_w, orig_h)
    new_w, new_h = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
    top, left = (image_size - new_h) // 2, (image_size - new_w) // 2
    cam = cam[top : top + new_h, left : left + new_w]
    cam = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(cam, 0, 1)


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    colored = cv2.applyColorMap(np.uint8(heatmap * 255), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32)
    blended = (1 - alpha) * rgb + alpha * colored
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def explain(
    model: AIDetector,
    image: Image.Image,
    target: Optional[int] = None,
    device: str | torch.device = "cpu",
) -> Explanation:
    image = image.convert("RGB")
    size = model.cfg.image_size
    transform = build_transforms(size, train=False)
    x = transform(image=np.asarray(image))["image"].unsqueeze(0).to(device)
    probs, cam = gradcam(model, x, target)
    names = model.cfg.class_names
    pred = int(probs.argmax())
    tgt = pred if target is None else target
    heatmap = _unletterbox(cam, size, image.width, image.height)
    return Explanation(
        probs={names[i]: float(p) for i, p in enumerate(probs)},
        predicted=names[pred],
        target=names[tgt],
        heatmap=heatmap,
        overlay=overlay_heatmap(image, heatmap),
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Save a Grad-CAM overlay for an image.")
    parser.add_argument("--checkpoint", default="models/best.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--target", type=int, default=None, help="Class index; default = predicted.")
    parser.add_argument("--out", default=None, help="Default: <image>_gradcam.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_checkpoint(args.checkpoint, device)
    result = explain(model, Image.open(args.image), args.target, device)
    out = args.out or str(Path(args.image).with_name(Path(args.image).stem + "_gradcam.png"))
    result.overlay.save(out)
    for name, p in result.probs.items():
        print(f"{name:>14s}: {p:.4f}")
    print(f"prediction: {result.predicted}  (heatmap for: {result.target}) -> {out}")


if __name__ == "__main__":
    _main()
