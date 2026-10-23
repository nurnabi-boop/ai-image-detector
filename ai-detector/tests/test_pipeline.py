"""Fast CPU tests: tiny non-pretrained backbone + synthetic data, no downloads."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from src.dataset import make_synthetic_dataset
from src.model import ai_class_index, build_model, load_checkpoint, save_checkpoint

BACKBONE = "efficientnet_b0"
SIZE = 64


@pytest.fixture
def model():
    torch.manual_seed(0)
    return build_model(BACKBONE, image_size=SIZE, pretrained=False)


def test_forward_shape(model):
    out = model(torch.randn(3, 3, SIZE, SIZE))
    assert out.shape == (3, 2)


def test_freeze_unfreeze(model):
    model.freeze_backbone()
    assert model.backbone_frozen
    assert all(p.requires_grad for p in model.head.parameters())
    model.train()
    assert not model.backbone.training  # BN stats stay fixed while frozen
    model.unfreeze_backbone()
    model.train()
    assert not model.backbone_frozen and model.backbone.training


def test_checkpoint_roundtrip(model, tmp_path):
    model.eval()
    x = torch.randn(2, 3, SIZE, SIZE)
    path = save_checkpoint(model, str(tmp_path / "m.pt"))
    loaded, ckpt = load_checkpoint(path)
    assert loaded.cfg == model.cfg
    torch.testing.assert_close(loaded(x), model(x))


@pytest.mark.parametrize(
    "names, expected",
    [
        (("Real", "AI-Generated"), 1),
        (("AiArtData", "RealArt"), 0),
        (("fake", "real"), 0),
        (("a", "b"), 1),
    ],
)
def test_ai_class_index(names, expected):
    assert ai_class_index(names) == expected


def test_gradcam(model):
    from src.gradcam import explain

    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (50, 80, 3), dtype=np.uint8))
    result = explain(model, img)
    assert result.heatmap.shape == (50, 80)
    assert 0.0 <= result.heatmap.min() and result.heatmap.max() <= 1.0
    assert result.overlay.size == img.size
    assert abs(sum(result.probs.values()) - 1.0) < 1e-5
    assert result.predicted in model.cfg.class_names


def test_train_end_to_end_and_export(tmp_path):
    """Synthetic data -> both training phases -> best.pt -> test metrics -> ONNX."""
    pytest.importorskip("onnxruntime")
    from src.export_onnx import export
    from src.train import main

    models_dir, reports_dir = tmp_path / "models", tmp_path / "reports"
    result = main(
        [
            "--synthetic", "80",
            "--backbone", BACKBONE,
            "--image-size", str(SIZE),
            "--no-pretrained",
            "--batch-size", "8",
            "--num-workers", "0",
            "--epochs-frozen", "1",
            "--epochs-unfrozen", "2",
            "--lr-backbone", "1e-3",
            "--out-dir", str(models_dir),
            "--reports-dir", str(reports_dir),
        ]
    )
    assert (models_dir / "best.pt").exists()
    assert [r["phase"] for r in result["history"]] == ["frozen", "unfrozen", "unfrozen"]
    metrics = json.loads((reports_dir / "metrics.json").read_text())
    assert metrics["n"] == 8 and 0.0 <= metrics["accuracy"] <= 1.0
    assert (reports_dir / "confusion_matrix.png").exists()

    onnx_path = export(str(models_dir / "best.pt"), str(tmp_path / "m.onnx"))
    import onnx

    meta = {p.key: p.value for p in onnx.load(onnx_path).metadata_props}
    assert json.loads(meta["class_names"]) == ["Real", "AI-Generated"]
    assert meta["image_size"] == str(SIZE)


def test_synthetic_dataset_balanced():
    ds = make_synthetic_dataset(10, size=32)
    assert len(ds) == 10 and sorted(set(ds["label"])) == [0, 1]
