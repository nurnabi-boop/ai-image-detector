"""Evaluate a trained checkpoint: accuracy / precision / recall / F1 / ROC-AUC,
confusion matrix and ROC curve plots, metrics.json.

Rebuilds the test split with the same dataset, max-samples and seed used in
training, so the numbers are on held-out data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (
    DEFAULT_HF_DATASET,
    DataConfig,
    make_dataloaders,
    make_loaders_from_splits,
    make_synthetic_dataset,
    split_dataset,
)
from .model import AIDetector, ai_class_index, load_checkpoint


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device,
    desc: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (softmax probs [N, C], labels [N], mean CE loss)."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    probs, labels, total_loss = [], [], 0.0
    for images, targets in tqdm(loader, desc=desc, leave=False, disable=desc is None):
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        total_loss += criterion(logits, targets).item()
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(targets.cpu().numpy())
    n = sum(len(l) for l in labels)
    return np.concatenate(probs), np.concatenate(labels), total_loss / max(n, 1)


def compute_metrics(probs: np.ndarray, labels: np.ndarray, positive: int = 1) -> dict:
    preds = probs.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, pos_label=positive, zero_division=0)),
        "recall": float(recall_score(labels, preds, pos_label=positive, zero_division=0)),
        "f1": float(f1_score(labels, preds, pos_label=positive, zero_division=0)),
    }
    # AUC is undefined when only one class is present (tiny val splits).
    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels == positive, probs[:, positive]))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def plot_confusion_matrix(
    probs: np.ndarray, labels: np.ndarray, class_names: tuple[str, ...], out_path: str
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(labels, probs.argmax(axis=1), labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_roc(probs: np.ndarray, labels: np.ndarray, positive: int, out_path: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(labels == positive, probs[:, positive])
    auc = roc_auc_score(labels == positive, probs[:, positive])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC (positive = AI-generated)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def evaluate_loader(
    model: AIDetector,
    loader: DataLoader,
    device: str | torch.device,
    out_dir: str,
) -> dict:
    class_names = model.cfg.class_names
    positive = ai_class_index(class_names)
    probs, labels, loss = collect_predictions(model, loader, device, desc="eval")
    metrics = compute_metrics(probs, labels, positive)
    metrics["loss"] = loss
    metrics["n"] = int(len(labels))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(probs, labels, class_names, str(out / "confusion_matrix.png"))
    if len(np.unique(labels)) > 1:
        plot_roc(probs, labels, positive, str(out / "roc_curve.png"))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the test split.")
    parser.add_argument("--checkpoint", default="models/best.pt")
    parser.add_argument("--dataset", default=None, help="Defaults to the one used in training.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(args.checkpoint, device)
    train_args = ckpt.get("train_args", {})
    max_samples = args.max_samples or train_args.get("max_samples")
    cfg = DataConfig(
        hf_name=args.dataset or train_args.get("dataset", DEFAULT_HF_DATASET),
        image_size=model.cfg.image_size,
        seed=train_args.get("seed", 42),
    )
    balance = not train_args.get("no_balance", False)
    if train_args.get("synthetic") and not args.dataset:
        splits = split_dataset(
            make_synthetic_dataset(train_args["synthetic"], seed=cfg.seed), cfg, balance=balance
        )
        _, _, test_loader, _ = make_loaders_from_splits(
            cfg, splits, batch_size=args.batch_size, num_workers=args.num_workers
        )
    else:
        _, _, test_loader, _ = make_dataloaders(
            cfg,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=max_samples,
            balance=balance,
        )
    metrics = evaluate_loader(model, test_loader, device, args.out_dir)
    print(json.dumps(metrics, indent=2))
    print(f"saved plots + metrics -> {args.out_dir}/")


if __name__ == "__main__":
    _main()
