"""Model factory using timm for pretrained backbones.

The classification head is deliberately simple: dropout + linear layer. Keeping
the head minimal lets the backbone do the work and makes fair comparisons
between different backbones in the ablation study.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


SUPPORTED_BACKBONES = {
    "efficientnet_b0",
    "efficientnet_b3",
    "convnext_tiny",
    "resnet50",
}


class PlantDiseaseModel(nn.Module):
    """Backbone + Dropout + Linear head."""

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"backbone {backbone!r} not in supported set {SUPPORTED_BACKBONES}"
            )
        # num_classes=0 returns the feature extractor (no head).
        self.backbone_name = backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True


class DINOv2Model(nn.Module):
    """DINOv2 frozen backbone + linear head. Only the head is trained."""

    def __init__(self, num_classes: int, model_name: str = "dinov2_vitb14", dropout: float = 0.3):
        super().__init__()
        self.backbone_name = model_name
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        num_features = self.backbone.embed_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self) -> None:
        pass  # Already frozen

    def unfreeze_backbone(self) -> None:
        pass  # Keep frozen for foundation model approach


def create_model(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    if backbone.startswith("dinov2"):
        return DINOv2Model(num_classes=num_classes, model_name=backbone, dropout=dropout)
    return PlantDiseaseModel(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )
