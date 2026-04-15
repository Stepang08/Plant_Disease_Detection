"""Evaluation: run a trained model on a dataloader and return mAP + details.

Used by the evaluation notebook (and can be run as a script for quick checks).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import PlantDiseaseDataset
from src.model import create_model
from src.transforms import build_val_transforms, build_tta_transforms
from src.utils import compute_mAP, load_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model over dataloader, return (probs, targets) as numpy arrays."""
    model.eval()
    all_probs, all_targets = [], []
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_targets)


@torch.no_grad()
def run_inference_tta(
    model: torch.nn.Module,
    dataset: PlantDiseaseDataset,
    tta_transforms: list,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """TTA: for each TTA transform, run full-set inference; average softmax outputs."""
    model.eval()
    n_samples = len(dataset)
    num_classes = dataset.num_classes
    summed_probs = np.zeros((n_samples, num_classes), dtype=np.float64)
    targets = np.array(dataset.targets)

    for tta_idx, transform in enumerate(tta_transforms):
        dataset.transform = transform
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        probs, _ = run_inference(model, loader, device)
        summed_probs += probs
        print(f"  TTA pass {tta_idx + 1}/{len(tta_transforms)} done")

    averaged = summed_probs / len(tta_transforms)
    return averaged.astype(np.float32), targets


def compute_confusion_matrix(
    targets: np.ndarray, predictions: np.ndarray, num_classes: int
) -> np.ndarray:
    """Plain confusion matrix without sklearn dependency."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, predictions):
        cm[t, p] += 1
    return cm


def top_k_accuracy(targets: np.ndarray, probs: np.ndarray, k: int = 5) -> float:
    top_k = np.argsort(-probs, axis=1)[:, :k]
    correct = np.any(top_k == targets[:, None], axis=1)
    return float(correct.mean())


def evaluate_model(
    model: torch.nn.Module,
    dataset: PlantDiseaseDataset,
    device: torch.device,
    num_classes: int,
    batch_size: int = 32,
    use_tta: bool = False,
    image_size: int = 224,
) -> dict:
    """Full evaluation: returns mAP, per-class AP, top-k accuracy, confusion matrix."""
    if use_tta:
        tta_transforms = build_tta_transforms(image_size)
        probs, targets = run_inference_tta(model, dataset, tta_transforms, device, batch_size)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        probs, targets = run_inference(model, loader, device)

    metrics = compute_mAP(targets, probs, num_classes)
    predictions = probs.argmax(axis=1)
    metrics["top5_acc"] = top_k_accuracy(targets, probs, k=5)
    metrics["confusion_matrix"] = compute_confusion_matrix(targets, predictions, num_classes)
    metrics["predictions"] = predictions
    metrics["probs"] = probs
    metrics["targets"] = targets
    return metrics


def load_checkpoint(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    """Load a checkpoint saved by Trainer.save_checkpoint and return (model, config)."""
    state = torch.load(checkpoint_path, map_location=device)
    config = state["config"]
    model = create_model(
        backbone=config["model"]["backbone"],
        num_classes=config["model"]["num_classes"],
        pretrained=False,  # weights come from checkpoint
        dropout=config["model"]["dropout"],
    )
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, config = load_checkpoint(PROJECT_ROOT / args.checkpoint, device)
    image_size = config["data"]["image_size"]
    num_classes = config["model"]["num_classes"]

    dataset = PlantDiseaseDataset(split=args.split, transform=build_val_transforms(image_size))
    print(f"Evaluating on {args.split}: {len(dataset)} images")

    metrics = evaluate_model(
        model, dataset, device, num_classes,
        batch_size=args.batch_size, use_tta=args.tta, image_size=image_size,
    )

    print(f"\n=== Results ({args.split}, TTA={args.tta}) ===")
    print(f"mAP:       {metrics['mAP']:.4f}")
    print(f"Top-1 acc: {metrics['top1_acc']:.4f}")
    print(f"Top-5 acc: {metrics['top5_acc']:.4f}")

    # Per-class AP table
    class_names = dataset.classes
    per_class = [
        (class_names[i], metrics["per_class_AP"][i])
        for i in range(num_classes)
    ]
    per_class.sort(key=lambda x: x[1] if not np.isnan(x[1]) else -1)
    print("\n--- Weakest 10 classes ---")
    for name, ap in per_class[:10]:
        print(f"  {ap:.4f}  {name}")
    print("\n--- Strongest 10 classes ---")
    for name, ap in per_class[-10:]:
        print(f"  {ap:.4f}  {name}")


if __name__ == "__main__":
    main()
