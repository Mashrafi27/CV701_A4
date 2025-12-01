"""Heatmap generation and decoding utilities."""

from __future__ import annotations

import torch

from .keypoints import denormalize_keypoints


def generate_gaussian_heatmaps(
    normalized_keypoints: torch.Tensor,
    normalization_factors: torch.Tensor,
    heatmap_size: int,
    image_size: int,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Create Gaussian heatmaps from normalized keypoints."""

    device = normalized_keypoints.device
    keypoints_px = denormalize_keypoints(normalized_keypoints, normalization_factors)
    scale = heatmap_size / float(image_size)
    coords = keypoints_px * scale

    grid_x = torch.arange(heatmap_size, device=device).view(1, 1, 1, heatmap_size)
    grid_y = torch.arange(heatmap_size, device=device).view(1, 1, heatmap_size, 1)

    cx = coords[..., 0].unsqueeze(-1).unsqueeze(-1)
    cy = coords[..., 1].unsqueeze(-1).unsqueeze(-1)

    dist = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
    heatmaps = torch.exp(-dist / (2 * sigma**2))
    return heatmaps


def heatmaps_to_keypoints(
    heatmaps: torch.Tensor,
    normalization_factors: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    """Decode heatmaps to normalized keypoint coordinates."""

    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)
    indices = flat.argmax(dim=-1)
    y = (indices // W).float()
    x = (indices % W).float()

    scale = float(image_size) / float(W)
    x_px = x * scale
    y_px = y * scale

    width = normalization_factors[:, 1].unsqueeze(-1)
    height = normalization_factors[:, 0].unsqueeze(-1)

    x_norm = (x_px / width - 0.5) * 2.0
    y_norm = (y_px / height - 0.5) * 2.0
    return torch.stack([x_norm, y_norm], dim=-1)
