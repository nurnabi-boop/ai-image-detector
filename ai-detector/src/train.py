"""Train the real vs AI-generated classifier.

Two-phase schedule:
  1. frozen  — backbone frozen, only the head trains (fast, stable start).
  2. unfrozen — whole network fine-tunes with a lower backbone LR, cosine
     decay, and early stopping on validation ROC-AUC.

The best checkpoint (by val AUC) is written to <out-dir>/best.pt and then
evaluated on the held-out test split.

Examples:
  python -m src.train --max-samples 2000 --epochs-frozen 1 --epochs-unfrozen 3
  python -m src.train --synthetic 128 --backbone efficientnet_b0 --image-size 96 --no-pretrained
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (
    DEFAULT_HF_DATASET,
    DataConfig,
    load_splits,
    make_loaders_from_splits,
    make_synthetic_dataset,
    split_dataset,
)
from .evaluate import collect_predictions, compute_metrics, evaluate_loader
from .model import AIDetector, ai_class_index, build_model, load_checkpoint, save_checkpoint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: AIDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    desc: str = "train",
) -> float:
    model.train()
    total, n = 0.0, 0
    use_amp = scaler is not None
    for images, targets in tqdm(loader, desc=desc, leave=False):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = criterion(model(images), targets)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * len(targets)
        n += len(targets)
    return total / max(n, 1)


def _cosine_with_warmup(total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return fn


def fit(
    model: AIDetector,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: str = "models",
    epochs_frozen: int = 2,
    epochs_unfrozen: int = 8,
    lr_head: float = 1e-3,
    lr_backbone: float = 1e-4,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    patience: int = 3,
    amp: bool = True,
    log_fn: Optional[Callable[[dict], None]] = None,
    extra_ckpt: Optional[dict] = None,
) -> dict:
    """Run both phases; returns history and the path of the best checkpoint."""
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda") if amp and device.type == "cuda" else None
    positive = ai_class_index(model.cfg.class_names)
    best_path = str(Path(out_dir) / "best.pt")
    best_auc, best_epoch, bad_epochs = -1.0, -1, 0
    history: list[dict] = []
    epoch = 0

    def run_validation(phase: str, train_loss: float, lr: float) -> bool:
        """Logs, checkpoints on improvement; returns True if improved."""
        nonlocal best_auc, best_epoch
        probs, labels, val_loss = collect_predictions(model, val_loader, device)
        m = compute_metrics(probs, labels, positive)
        # Fall back to accuracy when AUC is undefined (single-class val).
        score = m["roc_auc"] if not math.isnan(m["roc_auc"]) else m["accuracy"]
        row = {
            "epoch": epoch,
            "phase": phase,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in m.items()},
        }
        history.append(row)
        print(
            f"[{phase:8s}] epoch {epoch:02d}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={m['accuracy']:.4f}  val_auc={m['roc_auc']:.4f}"
        )
        if log_fn:
            log_fn(row)
        if score > best_auc:
            best_auc, best_epoch = score, epoch
            save_checkpoint(model, best_path, epoch=epoch, val_metrics=m, **(extra_ckpt or {}))
            return True
        return False

    # Phase 1: frozen backbone, head only.
    if epochs_frozen > 0:
        model.freeze_backbone()
        opt = torch.optim.AdamW(model.head.parameters(), lr=lr_head, weight_decay=weight_decay)
        for _ in range(epochs_frozen):
            epoch += 1
            loss = train_one_epoch(
                model, train_loader, opt, criterion, device, scaler, desc=f"frozen {epoch}"
            )
            run_validation("frozen", loss, lr_head)

    # Phase 2: full fine-tune, discriminative LRs, cosine decay, early stop.
    if epochs_unfrozen > 0:
        model.unfreeze_backbone()
        opt = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": lr_backbone},
                {"params": model.head.parameters(), "lr": lr_head * 0.1},
            ],
            weight_decay=weight_decay,
        )
        total_steps = epochs_unfrozen * len(train_loader)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, _cosine_with_warmup(total_steps, warmup_steps=len(train_loader) // 2)
        )
        for _ in range(epochs_unfrozen):
            epoch += 1
            loss = train_one_epoch(
                model, train_loader, opt, criterion, device, scaler, sched, desc=f"unfrozen {epoch}"
            )
            if run_validation("unfrozen", loss, opt.param_groups[0]["lr"]):
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"early stopping: no val improvement for {patience} epochs")
                    break

    print(f"best val score {best_auc:.4f} at epoch {best_epoch} -> {best_path}")
    return {"history": history, "best_path": best_path, "best_score": best_auc, "best_epoch": best_epoch}


def _init_wandb(args: argparse.Namespace) -> Optional[Callable[[dict], None]]:
    if not args.wandb:
        return None
    import wandb

    run = wandb.init(project=args.wandb_project, config=vars(args))
    return lambda row: run.log(row, step=row["epoch"])


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the AI image detector.")
    p.add_argument("--dataset", default=DEFAULT_HF_DATASET, help="HF dataset name")
    p.add_argument(
        "--synthetic",
        type=int,
        default=0,
        metavar="N",
        help="Use N generated images instead of downloading (smoke test).",
    )
    p.add_argument("--backbone", default="efficientnet_b3", help="Any timm CNN, e.g. convnext_tiny")
    p.add_argument("--image-size", type=int, default=None, help="Defaults to backbone native size.")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--no-balance", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs-frozen", type=int, default=2)
    p.add_argument("--epochs-unfrozen", type=int, default=8)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="models")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    p.add_argument("--wandb-project", default="ai-image-detector")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> dict:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        args.backbone, image_size=args.image_size, pretrained=not args.no_pretrained
    )
    cfg = DataConfig(hf_name=args.dataset, image_size=model.cfg.image_size, seed=args.seed)
    if args.synthetic:
        splits = split_dataset(
            make_synthetic_dataset(args.synthetic, seed=args.seed),
            cfg,
            balance=not args.no_balance,
        )
    else:
        splits = load_splits(cfg, max_samples=args.max_samples, balance=not args.no_balance)
    class_names = tuple(splits[3])
    model.cfg.class_names = class_names
    model.cfg.num_classes = len(class_names)
    if len(class_names) != 2:
        raise ValueError(f"Expected a binary dataset, got classes {class_names}")

    train_loader, val_loader, test_loader, _ = make_loaders_from_splits(
        cfg, splits, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(
        f"device={device} backbone={args.backbone} image_size={model.cfg.image_size} "
        f"classes={class_names} train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}"
    )

    result = fit(
        model,
        train_loader,
        val_loader,
        device,
        out_dir=args.out_dir,
        epochs_frozen=args.epochs_frozen,
        epochs_unfrozen=args.epochs_unfrozen,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        amp=not args.no_amp,
        log_fn=_init_wandb(args),
        extra_ckpt={"train_args": vars(args)},
    )

    best_model, _ = load_checkpoint(result["best_path"], device)
    test_metrics = evaluate_loader(best_model, test_loader, device, args.reports_dir)
    print("test:", json.dumps(test_metrics, indent=2))
    Path(args.reports_dir, "history.json").write_text(json.dumps(result["history"], indent=2))
    if args.wandb:
        import wandb

        wandb.summary.update({f"test_{k}": v for k, v in test_metrics.items()})
        wandb.finish()
    result["test_metrics"] = test_metrics
    return result


if __name__ == "__main__":
    main()
