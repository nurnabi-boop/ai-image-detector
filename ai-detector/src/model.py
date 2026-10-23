"""timm backbone + classification head for real vs AI-generated images.

The backbone is created headless (num_classes=0) and the forward pass is
split into `features()` (spatial feature map) and `classify()` (pool + head)
so Grad-CAM can hook the last conv feature map without per-architecture
layer lookups. Works with any timm CNN whose forward_features returns a
(B, C, H, W) map (EfficientNet, ConvNeXt, ResNet, ...).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import timm
import torch
from torch import nn

from .dataset import CLASS_NAMES

# Native input resolution for the backbones the project targets.
DEFAULT_IMAGE_SIZES = {
    "efficientnet_b3": 300,
    "convnext_tiny": 224,
}


@dataclass
class ModelConfig:
    backbone: str = "efficientnet_b3"
    num_classes: int = 2
    image_size: int = 300
    drop_rate: float = 0.3
    class_names: tuple[str, ...] = field(default_factory=lambda: CLASS_NAMES)


class AIDetector(nn.Module):
    def __init__(self, cfg: ModelConfig, pretrained: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=pretrained, num_classes=0
        )
        self.head = nn.Sequential(
            nn.Dropout(cfg.drop_rate),
            nn.Linear(self.backbone.num_features, cfg.num_classes),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def classify(self, feats: torch.Tensor) -> torch.Tensor:
        pooled = self.backbone.forward_head(feats, pre_logits=True)
        return self.head(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.features(x))

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        # Keep BatchNorm running stats fixed while the backbone is frozen.
        self.backbone.eval()

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    @property
    def backbone_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.backbone.parameters())

    def train(self, mode: bool = True) -> "AIDetector":
        super().train(mode)
        if mode and self.backbone_frozen:
            self.backbone.eval()
        return self


def build_model(
    backbone: str = "efficientnet_b3",
    num_classes: int = 2,
    image_size: Optional[int] = None,
    class_names: tuple[str, ...] = CLASS_NAMES,
    drop_rate: float = 0.3,
    pretrained: bool = True,
) -> AIDetector:
    if image_size is None:
        image_size = DEFAULT_IMAGE_SIZES.get(backbone, 224)
    cfg = ModelConfig(
        backbone=backbone,
        num_classes=num_classes,
        image_size=image_size,
        drop_rate=drop_rate,
        class_names=tuple(class_names),
    )
    return AIDetector(cfg, pretrained=pretrained)


def save_checkpoint(model: AIDetector, path: str, **extra) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = asdict(model.cfg)
    cfg["class_names"] = list(cfg["class_names"])
    torch.save({"state_dict": model.state_dict(), "config": cfg, **extra}, out)
    return str(out)


def load_checkpoint(
    path: str, device: str | torch.device = "cpu"
) -> tuple[AIDetector, dict]:
    """Rebuild the model from a checkpoint (no pretrained download needed)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt["config"])
    cfg_dict["class_names"] = tuple(cfg_dict["class_names"])
    model = AIDetector(ModelConfig(**cfg_dict), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def ai_class_index(class_names: tuple[str, ...]) -> int:
    """Index of the AI-generated class, used as the positive class for
    ROC/AUC. Datasets disagree on label order, so look for "real" and take
    the other class; fall back to the project convention (1 = AI)."""
    lowered = [n.lower() for n in class_names]
    real = [i for i, n in enumerate(lowered) if "real" in n]
    if len(class_names) == 2 and len(real) == 1:
        return 1 - real[0]
    return 1
