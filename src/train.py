"""Training loop with transfer learning, mixed precision, and W&B logging.

Run as:
    python -m src.train --config configs/default.yaml

Supports --smoke-test for quick CPU verification (tiny subset, 2 epochs).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from src.dataset import PlantDiseaseDataset, get_weighted_sampler
from src.model import create_model
from src.transforms import build_train_transforms, build_val_transforms
from src.utils import compute_mAP, get_param_count, load_config, set_seed


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def cosine_warmup_schedule(optimizer, warmup_epochs: int, total_epochs: int):
    """Linear warmup for `warmup_epochs`, then cosine decay to 0."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def build_optimizer(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> AdamW:
    """Discriminative LR: backbone params get small lr, head gets larger lr."""
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    return AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )


class Trainer:
    def __init__(self, config: dict, device: torch.device, wandb_run=None):
        self.config = config
        self.device = device
        self.wandb_run = wandb_run

        # Data
        train_tf = build_train_transforms(
            image_size=config["data"]["image_size"],
            aug=config.get("augmentation", {}),
        )
        val_tf = build_val_transforms(image_size=config["data"]["image_size"])

        self.train_ds = PlantDiseaseDataset(split="train", transform=train_tf)
        self.val_ds = PlantDiseaseDataset(split="val", transform=val_tf)

        sampler = (
            get_weighted_sampler(self.train_ds)
            if config["data"]["use_weighted_sampler"]
            else None
        )
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=config["data"]["batch_size"],
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=config["data"]["num_workers"],
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            pin_memory=(device.type == "cuda"),
        )

        # Model
        self.model = create_model(
            backbone=config["model"]["backbone"],
            num_classes=config["model"]["num_classes"],
            pretrained=config["model"]["pretrained"],
            dropout=config["model"]["dropout"],
        ).to(device)

        total, trainable = get_param_count(self.model)
        print(f"Model: {config['model']['backbone']} | params: {total/1e6:.2f}M total")

        # Training state
        self.freeze_epochs = config["training"]["freeze_backbone_epochs"]
        if self.freeze_epochs > 0:
            self.model.freeze_backbone()

        self.optimizer = build_optimizer(
            self.model,
            lr_backbone=config["training"]["lr_backbone"],
            lr_head=config["training"]["lr_head"],
            weight_decay=config["training"]["weight_decay"],
        )
        self.scheduler = cosine_warmup_schedule(
            self.optimizer,
            warmup_epochs=config["training"]["warmup_epochs"],
            total_epochs=config["training"]["epochs"],
        )
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config["training"]["label_smoothing"]
        )
        # GradScaler only meaningful on CUDA; on CPU it's a no-op.
        self.use_amp = config["training"]["mixed_precision"] and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.best_mAP = 0.0
        self.patience = config["training"]["early_stopping_patience"]
        self.epochs_without_improvement = 0

        self.ckpt_dir = PROJECT_ROOT / config["paths"]["checkpoint_dir"]
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = PROJECT_ROOT / config["paths"]["best_model_path"]
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        n_skipped = 0
        start = time.time()
        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            # Skip a batch if loss is non-finite (NaN/Inf). Protects against
            # rare mixed-precision overflows at freeze->unfreeze transitions.
            if not torch.isfinite(loss):
                n_skipped += 1
                continue

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config["training"]["grad_clip_norm"],
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config["training"]["grad_clip_norm"],
                )
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if n_skipped:
            print(f"  (skipped {n_skipped} batches with non-finite loss)")
        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        all_probs = []
        all_targets = []
        total_loss = 0.0
        n_batches = 0
        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            total_loss += loss.item()
            n_batches += 1

        probs_arr = np.concatenate(all_probs, axis=0)
        targets_arr = np.concatenate(all_targets, axis=0)
        metrics = compute_mAP(
            targets_arr, probs_arr, num_classes=self.config["model"]["num_classes"]
        )
        metrics["val_loss"] = total_loss / max(1, n_batches)
        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_mAP": self.best_mAP,
            "config": self.config,
        }
        latest_path = self.ckpt_dir / "latest.pth"
        torch.save(state, latest_path)
        if is_best:
            torch.save(state, self.best_model_path)

    def fit(self) -> dict:
        total_epochs = self.config["training"]["epochs"]
        for epoch in range(total_epochs):
            if epoch == self.freeze_epochs and self.freeze_epochs > 0:
                print(f"Epoch {epoch}: unfreezing backbone")
                self.model.unfreeze_backbone()

            train_loss = self.train_one_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()

            val_mAP = val_metrics["mAP"]
            is_best = val_mAP > self.best_mAP
            if is_best:
                self.best_mAP = val_mAP
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            log = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["val_loss"],
                "val_mAP": val_mAP,
                "val_top1_acc": val_metrics["top1_acc"],
                "lr_backbone": self.optimizer.param_groups[0]["lr"],
                "lr_head": self.optimizer.param_groups[1]["lr"],
            }
            print(
                f"epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                f"val_loss {val_metrics['val_loss']:.4f} | "
                f"val_mAP {val_mAP:.4f} | val_top1 {val_metrics['top1_acc']:.4f}"
            )
            if self.wandb_run is not None:
                self.wandb_run.log(log)

            self.save_checkpoint(epoch, val_metrics, is_best)

            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping at epoch {epoch} (patience={self.patience})")
                break

        return {"best_mAP": self.best_mAP}


def maybe_init_wandb(config: dict):
    """Init wandb if available and project is set; otherwise return None."""
    try:
        import wandb
    except ImportError:
        return None
    wb_cfg = config.get("wandb", {})
    project = wb_cfg.get("project")
    if not project:
        return None
    return wandb.init(
        project=project,
        entity=wb_cfg.get("entity"),
        config=config,
    )


def subset_for_smoke_test(trainer: Trainer, n_train: int = 16, n_val: int = 8) -> None:
    """Slice the datasets to a tiny subset for CPU smoke testing."""
    trainer.train_ds = Subset(trainer.train_ds, list(range(n_train)))
    trainer.val_ds = Subset(trainer.val_ds, list(range(n_val)))
    # Rebuild loaders (no weighted sampler for smoke test).
    trainer.train_loader = DataLoader(
        trainer.train_ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )
    trainer.val_loader = DataLoader(
        trainer.val_ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 2 epochs on a tiny subset for verification.")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])

    if args.smoke_test:
        config["training"]["epochs"] = 2
        config["training"]["freeze_backbone_epochs"] = 1
        config["training"]["warmup_epochs"] = 0
        config["data"]["num_workers"] = 0
        config["data"]["batch_size"] = 4
        config["data"]["use_weighted_sampler"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wandb_run = None if args.no_wandb or args.smoke_test else maybe_init_wandb(config)

    trainer = Trainer(config, device, wandb_run)

    if args.smoke_test:
        subset_for_smoke_test(trainer)

    result = trainer.fit()
    print(f"Best val mAP: {result['best_mAP']:.4f}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
