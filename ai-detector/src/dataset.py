"""Real vs AI-generated image dataset.

Wraps a Hugging Face image dataset (defaulting to a GenImage-style HF mirror)
in a torch.utils.data.Dataset with albumentations transforms. Provides
helpers for class balancing, train/val/test splitting, and batch
visualization.

Label convention used downstream: 0 = Real, 1 = AI-Generated.
"""
from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# A HF-hosted real-vs-AI image dataset. Override with --dataset to point at
# a GenImage / AIGCDetect mirror once you have one available locally.
DEFAULT_HF_DATASET = "Hemg/AI-Generated-vs-Real-Images-Datasets"

CLASS_NAMES = ("Real", "AI-Generated")


@dataclass
class DataConfig:
    hf_name: str = DEFAULT_HF_DATASET
    image_col: str = "image"
    label_col: str = "label"
    image_size: int = 300  # EfficientNet-B3 native input
    cache_dir: Optional[str] = None
    seed: int = 42


def build_transforms(image_size: int, train: bool) -> A.Compose:
    """Albumentations pipeline. Augmentations target artifacts that often
    distinguish AI vs real images (compression, blur, color shifts)."""
    if train:
        pad = int(image_size * 1.15)
        return A.Compose(
            [
                A.LongestMaxSize(max_size=pad),
                A.PadIfNeeded(min_height=pad, min_width=pad, border_mode=0, value=0),
                A.RandomCrop(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7)),
                        A.MotionBlur(blur_limit=5),
                        A.MedianBlur(blur_limit=5),
                    ],
                    p=0.3,
                ),
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5
                ),
                A.CoarseDropout(
                    max_holes=4,
                    max_height=image_size // 8,
                    max_width=image_size // 8,
                    p=0.3,
                ),
                A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size, min_width=image_size, border_mode=0, value=0
            ),
            A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def _detect_columns(hf_ds: HFDataset) -> tuple[str, str]:
    cols = hf_ds.column_names
    image_col = next((c for c in ("image", "img", "picture") if c in cols), None)
    label_col = next(
        (c for c in ("label", "labels", "class", "target") if c in cols), None
    )
    if image_col is None or label_col is None:
        raise ValueError(
            f"Could not auto-detect image/label columns in {cols}. "
            "Pass image_col / label_col explicitly via DataConfig."
        )
    return image_col, label_col


def _resolve_class_names(hf_ds: HFDataset, label_col: str) -> tuple[str, ...]:
    feat = hf_ds.features.get(label_col)
    names = getattr(feat, "names", None)
    if not names:
        return CLASS_NAMES
    # Heuristic mapping to our (Real, AI) convention. If the dataset's first
    # class looks like "real", we keep ordering; otherwise we still surface
    # the dataset's own names so visualization stays accurate.
    return tuple(names)


def balance_classes(hf_ds: HFDataset, label_col: str, seed: int = 42) -> HFDataset:
    labels = np.array(hf_ds[label_col])
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        return hf_ds
    target = int(counts.min())
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for c in classes:
        idx = np.where(labels == c)[0]
        chosen = rng.choice(idx, size=target, replace=False)
        keep.extend(int(i) for i in chosen)
    keep.sort()
    return hf_ds.select(keep)


class AIDetectorDataset(Dataset):
    def __init__(
        self,
        hf_dataset: HFDataset,
        transform: A.Compose,
        image_col: str = "image",
        label_col: str = "label",
    ) -> None:
        self.ds = hf_dataset
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.ds)

    def _to_pil(self, img) -> Image.Image:
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, dict) and "bytes" in img and img["bytes"] is not None:
            return Image.open(io.BytesIO(img["bytes"]))
        if isinstance(img, dict) and img.get("path"):
            return Image.open(img["path"])
        return Image.fromarray(np.asarray(img))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.ds[int(idx)]
        img = self._to_pil(item[self.image_col]).convert("RGB")
        label = item[self.label_col]
        if not isinstance(label, int):
            label = int(label)
        arr = np.array(img)
        out = self.transform(image=arr)["image"]
        return out, torch.tensor(label, dtype=torch.long)


def load_splits(
    cfg: DataConfig,
    max_samples: Optional[int] = None,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    balance: bool = True,
) -> tuple[HFDataset, HFDataset, HFDataset, tuple[str, ...]]:
    """Load a HF dataset, optionally subsample/balance, and split."""
    raw = load_dataset(cfg.hf_name, cache_dir=cfg.cache_dir)
    parts = [raw[s] for s in raw.keys()]
    full = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    return split_dataset(
        full,
        cfg,
        max_samples=max_samples,
        val_frac=val_frac,
        test_frac=test_frac,
        balance=balance,
    )


def split_dataset(
    full: HFDataset,
    cfg: DataConfig,
    max_samples: Optional[int] = None,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    balance: bool = True,
) -> tuple[HFDataset, HFDataset, HFDataset, tuple[str, ...]]:
    """Subsample/balance an already-loaded HF dataset and split it
    deterministically (same seed -> same test split for evaluate.py)."""
    image_col, label_col = _detect_columns(full)
    cfg.image_col = image_col
    cfg.label_col = label_col
    class_names = _resolve_class_names(full, label_col)

    if max_samples is not None and max_samples < len(full):
        full = full.shuffle(seed=cfg.seed).select(range(max_samples))

    if balance:
        full = balance_classes(full, label_col, seed=cfg.seed)

    full = full.shuffle(seed=cfg.seed)
    n = len(full)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(f"Not enough samples after splitting: n={n}")

    train_ds = full.select(range(n_train))
    val_ds = full.select(range(n_train, n_train + n_val))
    test_ds = full.select(range(n_train + n_val, n))
    return train_ds, val_ds, test_ds, class_names


def make_dataloaders(
    cfg: DataConfig,
    batch_size: int = 32,
    num_workers: int = 2,
    max_samples: Optional[int] = None,
    balance: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, tuple[str, ...]]:
    splits = load_splits(cfg, max_samples=max_samples, balance=balance)
    return make_loaders_from_splits(
        cfg, splits, batch_size=batch_size, num_workers=num_workers
    )


def make_loaders_from_splits(
    cfg: DataConfig,
    splits: tuple[HFDataset, HFDataset, HFDataset, tuple[str, ...]],
    batch_size: int = 32,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader, tuple[str, ...]]:
    train_hf, val_hf, test_hf, class_names = splits
    train_ds = AIDetectorDataset(
        train_hf,
        build_transforms(cfg.image_size, train=True),
        cfg.image_col,
        cfg.label_col,
    )
    val_ds = AIDetectorDataset(
        val_hf,
        build_transforms(cfg.image_size, train=False),
        cfg.image_col,
        cfg.label_col,
    )
    test_ds = AIDetectorDataset(
        test_hf,
        build_transforms(cfg.image_size, train=False),
        cfg.image_col,
        cfg.label_col,
    )
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader, class_names


def make_synthetic_dataset(n: int = 64, size: int = 96, seed: int = 0) -> HFDataset:
    """Tiny offline dataset for smoke tests / CI: "real" images are smooth
    noise, "AI" images carry a periodic grid artifact. Learnable in a few
    steps, needs no download."""
    from datasets import ClassLabel, Features
    from datasets import Image as HFImage

    rng = np.random.default_rng(seed)
    images, labels = [], []
    yy, xx = np.mgrid[0:size, 0:size]
    grid = ((np.sin(xx * 1.3) + np.sin(yy * 1.3)) * 40).astype(np.float32)
    for i in range(n):
        label = i % 2
        base = rng.normal(128, 30, size=(size, size, 3)).astype(np.float32)
        if label == 1:
            base += grid[..., None]
        images.append(Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)))
        labels.append(label)
    features = Features(
        {"image": HFImage(), "label": ClassLabel(names=list(CLASS_NAMES))}
    )
    return HFDataset.from_dict({"image": images, "label": labels}, features=features)


def denormalize(images: np.ndarray) -> np.ndarray:
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    images = images * std + mean
    return np.clip(images, 0, 1).transpose(0, 2, 3, 1)


def visualize_batch(
    loader: DataLoader,
    out_path: str,
    n: int = 16,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> str:
    import matplotlib.pyplot as plt

    images, labels = next(iter(loader))
    images = images[:n].numpy()
    labels = labels[:n].numpy()
    images = denormalize(images)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    for i, ax in enumerate(np.array(axes).flat):
        if i >= n:
            ax.axis("off")
            continue
        ax.imshow(images[i])
        label_idx = int(labels[i])
        name = class_names[label_idx] if label_idx < len(class_names) else str(label_idx)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the dataset pipeline.")
    parser.add_argument("--dataset", default=DEFAULT_HF_DATASET, help="HF dataset name")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Cap on samples loaded before splitting (keeps the smoke test fast).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="0 is safest on Windows; bump up on Linux.",
    )
    parser.add_argument("--out", default="batch_preview.png")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-balance", action="store_true")
    args = parser.parse_args()

    cfg = DataConfig(
        hf_name=args.dataset, image_size=args.image_size, cache_dir=args.cache_dir
    )
    train_loader, val_loader, test_loader, class_names = make_dataloaders(
        cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        balance=not args.no_balance,
    )

    print(f"dataset:      {args.dataset}")
    print(f"image_col:    {cfg.image_col}")
    print(f"label_col:    {cfg.label_col}")
    print(f"class_names:  {class_names}")
    print(
        f"sizes:        train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}"
    )

    images, labels = next(iter(train_loader))
    print(f"batch shape:  images={tuple(images.shape)} labels={tuple(labels.shape)}")
    unique, counts = np.unique(labels.numpy(), return_counts=True)
    print(f"batch labels: {dict(zip(unique.tolist(), counts.tolist()))}")

    out = visualize_batch(train_loader, args.out, class_names=class_names)
    print(f"saved batch preview -> {out}")


if __name__ == "__main__":
    _main()
