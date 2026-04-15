"""Dataset, label mapping, and stratified split for plant disease classification.

Run as a script to (re)generate label_mapping.json and split.json:
    python -m src.dataset
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable

from PIL import Image
from sklearn.model_selection import train_test_split

# Torch is optional: label mapping and split generation must run without it
# (for local dev on machines without PyTorch installed). The Dataset class
# and weighted sampler require torch and will raise at construction if missing.
try:
    from torch.utils.data import Dataset as _TorchDataset
    from torch.utils.data import WeightedRandomSampler
    _TORCH_AVAILABLE = True
except ImportError:
    _TorchDataset = object  # type: ignore[assignment,misc]
    WeightedRandomSampler = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIRS = [PROJECT_ROOT / "data" / "raw" / "train", PROJECT_ROOT / "data" / "raw" / "val"]
LABEL_MAPPING_PATH = PROJECT_ROOT / "configs" / "label_mapping.json"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "split.json"

# Sorted longest-first so multi-word plant names match before their single-word substrings.
PLANT_NAMES: list[str] = sorted(
    [
        "bell pepper",
        "apple", "banana", "basil", "bean", "blueberry",
        "broccoli", "cabbage", "cauliflower", "celery", "cherry",
        "citrus", "coffee", "corn", "cucumber", "garlic", "ginger",
        "grape", "lettuce", "maple", "peach", "plum", "potato",
        "raspberry", "rice", "soybean", "squash", "strawberry",
        "tobacco", "tomato", "wheat", "zucchini",
    ],
    key=len,
    reverse=True,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def extract_disease_name(folder_name: str) -> str:
    """Strip the plant prefix from a folder name to get the disease-only label.

    Examples:
        "apple black rot"   -> "black rot"
        "tomato late blight" -> "late blight"
        "bell pepper powdery mildew" -> "powdery mildew"
    """
    name = folder_name.strip().lower()
    for plant in PLANT_NAMES:
        prefix = plant + " "
        if name.startswith(prefix):
            return name[len(prefix):]
    raise ValueError(f"No known plant prefix in folder name: {folder_name!r}")


def _iter_image_records() -> list[tuple[Path, str, str]]:
    """Walk all raw data directories and return (image_path, folder_name, disease_name)."""
    records = []
    for raw_dir in RAW_DIRS:
        if not raw_dir.exists():
            continue
        for folder in sorted(raw_dir.iterdir()):
            if not folder.is_dir():
                continue
            disease = extract_disease_name(folder.name)
            for img in sorted(folder.iterdir()):
                if img.suffix.lower() in IMAGE_EXTENSIONS:
                    records.append((img, folder.name, disease))
    return records


def build_label_mapping() -> dict[str, int]:
    """Scan raw data and write a sorted disease -> integer mapping to disk."""
    records = _iter_image_records()
    if not records:
        raise RuntimeError(f"No images found under {RAW_DIRS}. Did you move the dataset?")

    diseases = sorted({disease for _, _, disease in records})
    mapping = {disease: idx for idx, disease in enumerate(diseases)}

    LABEL_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LABEL_MAPPING_PATH.open("w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Found {len(mapping)} disease classes (from {len(records)} images).")
    print(f"Wrote {LABEL_MAPPING_PATH.relative_to(PROJECT_ROOT)}")
    print("\nFull mapping:")
    for disease, idx in mapping.items():
        print(f"  {idx:>3}  {disease}")
    return mapping


def create_split(val_fraction: float = 0.15, seed: int = 42) -> dict[str, list[str]]:
    """Stratified split of all images by folder-level class. Saves split.json.

    Paths are stored relative to PROJECT_ROOT so the file is portable across
    machines (local, Colab, Docker container, etc.).
    """
    records = _iter_image_records()
    paths = [str(p.relative_to(PROJECT_ROOT)) for p, _, _ in records]
    folder_labels = [folder for _, folder, _ in records]

    folder_counts = Counter(folder_labels)
    rare_folders = {f for f, c in folder_counts.items() if c < 2}
    if rare_folders:
        # train_test_split needs >= 2 samples per stratum. Move rare singletons to train.
        keep_idx = [i for i, f in enumerate(folder_labels) if f not in rare_folders]
        rare_idx = [i for i, f in enumerate(folder_labels) if f in rare_folders]
        keep_paths = [paths[i] for i in keep_idx]
        keep_labels = [folder_labels[i] for i in keep_idx]
        train_paths, val_paths = train_test_split(
            keep_paths,
            test_size=val_fraction,
            random_state=seed,
            stratify=keep_labels,
        )
        train_paths.extend(paths[i] for i in rare_idx)
    else:
        train_paths, val_paths = train_test_split(
            paths,
            test_size=val_fraction,
            random_state=seed,
            stratify=folder_labels,
        )

    split = {"train": sorted(train_paths), "val": sorted(val_paths)}

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SPLIT_PATH.open("w") as f:
        json.dump(split, f, indent=2)

    print(f"\nSplit: {len(split['train'])} train / {len(split['val'])} val")
    print(f"Wrote {SPLIT_PATH.relative_to(PROJECT_ROOT)}")
    return split


class PlantDiseaseDataset(_TorchDataset):
    """Loads images from a precomputed split and maps folder names to disease labels."""

    def __init__(
        self,
        split: str,
        transform: Callable | None = None,
        split_file: Path = SPLIT_PATH,
        label_mapping_file: Path = LABEL_MAPPING_PATH,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PlantDiseaseDataset requires PyTorch. Install with: pip install torch torchvision"
            )
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        with label_mapping_file.open() as f:
            self.label_mapping: dict[str, int] = json.load(f)
        self.classes: list[str] = sorted(self.label_mapping, key=self.label_mapping.get)
        self.num_classes = len(self.classes)

        with split_file.open() as f:
            split_data = json.load(f)
        # Paths in split.json are relative to PROJECT_ROOT.
        self.image_paths: list[Path] = [PROJECT_ROOT / p for p in split_data[split]]

        # Compute integer labels once at init.
        self.targets: list[int] = [
            self.label_mapping[extract_disease_name(p.parent.name)] for p in self.image_paths
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple:
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[idx]


def get_weighted_sampler(dataset: "PlantDiseaseDataset"):
    """Inverse-frequency weighted sampler so rare classes are seen as often as common ones."""
    if not _TORCH_AVAILABLE:
        raise ImportError("get_weighted_sampler requires PyTorch.")
    class_counts = Counter(dataset.targets)
    sample_weights = [1.0 / class_counts[t] for t in dataset.targets]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )


if __name__ == "__main__":
    build_label_mapping()
    create_split()
