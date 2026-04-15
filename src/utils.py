"""Shared utilities: seeding, config loading, metrics."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score


def set_seed(seed: int) -> None:
    """Seed all sources of randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> dict:
    """Load YAML config into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def compute_mAP(
    targets: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
) -> dict:
    """Compute mAP (mean Average Precision) and related metrics.

    Args:
        targets: int array, shape (N,), ground-truth class indices.
        probs:   float array, shape (N, num_classes), softmax probabilities.
        num_classes: total number of classes.

    Returns:
        dict with keys: mAP, per_class_AP (dict), top1_acc
    """
    # One-vs-rest binarization for multi-class mAP.
    targets_onehot = np.eye(num_classes)[targets]

    per_class_ap: dict[int, float] = {}
    for cls in range(num_classes):
        # Classes with no positive samples in this batch get AP=NaN and are excluded.
        if targets_onehot[:, cls].sum() == 0:
            per_class_ap[cls] = float("nan")
            continue
        per_class_ap[cls] = float(
            average_precision_score(targets_onehot[:, cls], probs[:, cls])
        )

    valid_aps = [ap for ap in per_class_ap.values() if not np.isnan(ap)]
    mAP = float(np.mean(valid_aps)) if valid_aps else 0.0

    predictions = probs.argmax(axis=1)
    top1_acc = float((predictions == targets).mean())

    return {
        "mAP": mAP,
        "per_class_AP": per_class_ap,
        "top1_acc": top1_acc,
    }


def get_param_count(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params) for logging to W&B."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
