"""Heatmap-based facial keypoint regressor."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torchvision.models as models


class KeypointHeatmapNet(nn.Module):
    """ResNet backbone with deconvolution head that outputs heatmaps."""

    def __init__(
        self,
        num_keypoints: int = 68,
        backbone: str = "resnet18",
        pretrained: bool = True,
        heatmap_size: int = 56,
    ):
        super().__init__()
        self.heatmap_size = heatmap_size
        backbone = backbone.lower()
        if backbone not in {"resnet18", "resnet34"}:
            raise ValueError("Heatmap head currently supports resnet18 or resnet34")

        builder = getattr(models, backbone)
        weights = None
        if pretrained:
            enum_name = f"{backbone.capitalize()}_Weights"
            if hasattr(models, enum_name):
                weights = getattr(models, enum_name).IMAGENET1K_V1
            else:
                warnings.warn(
                    "Using legacy torchvision weights API; consider upgrading torchvision.",
                    stacklevel=2,
                )
                weights = pretrained
        base = builder(weights=weights)
        self.backbone = nn.Sequential(*list(base.children())[:-2])  # keep conv through layer4
        in_channels = 512 if backbone == "resnet18" else 512

        self.head = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_keypoints, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        heatmaps = self.head(feats)
        return heatmaps
