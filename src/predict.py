"""Single-image inference for the FastAPI endpoint and evaluation notebook."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.evaluate import load_checkpoint
from src.transforms import build_tta_transforms, build_val_transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_label_mapping(path: str | Path | None = None) -> list[str]:
    """Load the disease-only label mapping and return it as an index-ordered list."""
    path = Path(path) if path else PROJECT_ROOT / "configs" / "label_mapping.json"
    with open(path) as f:
        mapping: dict[str, int] = json.load(f)
    # Reverse: index -> disease name
    names = [""] * len(mapping)
    for disease, idx in mapping.items():
        names[idx] = disease
    return names


def prepare_image(image: Image.Image | str | Path, image_size: int = 224) -> torch.Tensor:
    """Convert a PIL image (or a path) into the standard model input tensor."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    image = image.convert("RGB")
    transform = build_val_transforms(image_size)
    return transform(image).unsqueeze(0)  # add batch dim


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image: Image.Image | str | Path,
    label_names: list[str],
    device: torch.device,
    image_size: int = 224,
    top_k: int = 5,
) -> list[dict]:
    """Return top-k predictions as [{disease, confidence}] sorted by confidence desc."""
    model.eval()
    tensor = prepare_image(image, image_size).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    top_idx = np.argsort(-probs)[:top_k]
    return [
        {"disease": label_names[i], "confidence": float(probs[i])}
        for i in top_idx
    ]


@torch.no_grad()
def predict_image_tta(
    model: torch.nn.Module,
    image: Image.Image | str | Path,
    label_names: list[str],
    device: torch.device,
    image_size: int = 224,
    top_k: int = 5,
) -> list[dict]:
    """Same as predict_image but averages softmax across TTA views for better accuracy."""
    model.eval()
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    image = image.convert("RGB")

    tta_transforms = build_tta_transforms(image_size)
    summed = np.zeros(len(label_names), dtype=np.float64)
    for transform in tta_transforms:
        tensor = transform(image).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        summed += probs
    avg = (summed / len(tta_transforms)).astype(np.float32)
    top_idx = np.argsort(-avg)[:top_k]
    return [
        {"disease": label_names[i], "confidence": float(avg[i])}
        for i in top_idx
    ]


def load_inference_model(
    checkpoint_path: str | Path = "models/best_model.pth",
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, list[str]]:
    """Convenience loader: returns (model, label_names) ready for predict_image()."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    model, _config = load_checkpoint(checkpoint_path, device)
    label_names = load_label_mapping()
    return model, label_names
