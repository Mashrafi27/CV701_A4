"""ResNet-based keypoint regression model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class KeypointResNet(nn.Module):
    """A lightweight ResNet18 backbone with a regression head."""

    def __init__(self, num_keypoints: int = 68, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
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
