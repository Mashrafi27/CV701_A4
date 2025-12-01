"""ResNet-based keypoint regression model."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torchvision.models as models


class KeypointResNet(nn.Module):
    """ResNet-based keypoint regressor (supports ResNet18/34)."""

    def __init__(
        self,
        num_keypoints: int = 68,
        pretrained: bool = True,
        dropout: float = 0.3,
        backbone_name: str = "resnet18",
    ):
        super().__init__()
        backbone = self._build_backbone(backbone_name, pretrained)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_keypoints * 2),
        )
        self.num_keypoints = num_keypoints

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        return self.regressor(feats)

    def freeze_backbone(self, freeze: bool = True) -> None:
        for param in self.feature_extractor.parameters():
            param.requires_grad = not freeze

    @staticmethod
    def _build_backbone(name: str, pretrained: bool):
        name = name.lower()
        valid = {"resnet18": models.resnet18, "resnet34": models.resnet34}
        if name not in valid:
            raise ValueError(f"Unsupported backbone '{name}'. Choose from {sorted(valid)}")

        builder = valid[name]
        weight_enum = None

        if hasattr(models, "ResNet18_Weights"):
            enum_map = {
                "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
                "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
            }
            weight_enum = enum_map.get(name) if pretrained else None
            return builder(weights=weight_enum)

        warnings.warn(
            "Using legacy torchvision API for ResNet weights; consider upgrading torchvision.",
            stacklevel=2,
        )
        return builder(pretrained=pretrained)
