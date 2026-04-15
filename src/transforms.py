"""Augmentation pipelines for training, validation, and test-time augmentation.

Keep hue jitter small: disease color (orange rust, white mildew, dark blight)
is a critical signal we don't want to wash out.
"""

from __future__ import annotations

try:
    from torchvision import transforms as T
    _TORCH_AVAILABLE = True
except ImportError:
    T = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError("torchvision is required. Install with: pip install torchvision")


def build_train_transforms(image_size: int = 224, aug: dict | None = None):
    """Training pipeline: moderate augmentation tuned for disease-feature preservation."""
    _require_torch()
    aug = aug or {}
    resize_size = int(image_size * 256 / 224)
    return T.Compose([
        T.Resize(resize_size),
        T.RandomResizedCrop(
            image_size,
            scale=tuple(aug.get("crop_scale", (0.7, 1.0))),
        ),
        T.RandomHorizontalFlip(p=aug.get("hflip_p", 0.5)),
        T.RandomVerticalFlip(p=aug.get("vflip_p", 0.3)),
        T.RandomRotation(degrees=aug.get("rotation_degrees", 15)),
        T.ColorJitter(
            brightness=aug.get("brightness", 0.2),
            contrast=aug.get("contrast", 0.2),
            saturation=aug.get("saturation", 0.2),
            hue=aug.get("hue", 0.05),  # Small — disease color matters.
        ),
        T.RandomAffine(degrees=0, translate=tuple(aug.get("translate", (0.05, 0.05)))),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_val_transforms(image_size: int = 224):
    """Deterministic pipeline for validation and final test inference."""
    _require_torch()
    resize_size = int(image_size * 256 / 224)
    return T.Compose([
        T.Resize(resize_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_tta_transforms(image_size: int = 224) -> list:
    """Five deterministic views for test-time augmentation. Average softmax outputs."""
    _require_torch()
    resize_size = int(image_size * 256 / 224)
    base = [T.Resize(resize_size), T.CenterCrop(image_size)]
    to_tensor = [T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]

    return [
        T.Compose(base + to_tensor),                                    # 1. Original
        T.Compose(base + [T.RandomHorizontalFlip(p=1.0)] + to_tensor),  # 2. HFlip
        T.Compose(base + [T.RandomRotation(degrees=(5, 5))] + to_tensor),    # 3. +5° rotation
        T.Compose(base + [T.RandomRotation(degrees=(-5, -5))] + to_tensor),  # 4. -5° rotation
        T.Compose([
            T.Resize(resize_size + 16),
            T.CenterCrop(image_size),
        ] + to_tensor),                                                 # 5. Zoomed view
    ]
