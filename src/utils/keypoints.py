"""Utility helpers for working with facial keypoints."""

from __future__ import annotations

from typing import Dict, Iterable

import torch

LEFT_EYE_OUTER = 36
RIGHT_EYE_OUTER = 45
MOUTH_LEFT = 48
MOUTH_RIGHT = 54
MOUTH_TOP = 51
MOUTH_BOTTOM = 57
MOUTH_CENTER = 62


def flatten_keypoints(keypoints: torch.Tensor) -> torch.Tensor:
    """Flatten keypoints from (B, K, 2) to (B, K*2)."""

    return keypoints.view(keypoints.size(0), -1)


def denormalize_keypoints(normalized: torch.Tensor, normalization_factors: torch.Tensor) -> torch.Tensor:
    """Convert normalized keypoints back to pixel coordinates."""

    if normalized.ndim != 3:
        raise ValueError("Expected keypoints to have shape (B, K, 2).")

    height = normalization_factors[:, 0].view(-1, 1)
    width = normalization_factors[:, 1].view(-1, 1)

    x = (normalized[:, :, 0] / 2.0 + 0.5) * width
    y = (normalized[:, :, 1] / 2.0 + 0.5) * height
    return torch.stack([x, y], dim=-1)


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    normalization_factors: torch.Tensor,
    *,
    pck_thresholds: Iterable[float] = (0.05, 0.1),
    auc_max: float = 0.5,
    auc_steps: int = 101,
) -> Dict[str, float]:
    """Compute regression metrics plus PCK/AUC statistics."""

    preds_px = denormalize_keypoints(preds, normalization_factors)
    targets_px = denormalize_keypoints(targets, normalization_factors)

    diff = preds_px - targets_px
    mae = torch.mean(torch.abs(diff)).item()
    rmse = torch.sqrt(torch.mean(diff ** 2)).item()

    point_errors = torch.linalg.norm(diff, dim=-1)
    mean_point_error = point_errors.mean().item()

    interocular = torch.linalg.norm(
        targets_px[:, RIGHT_EYE_OUTER, :] - targets_px[:, LEFT_EYE_OUTER, :], dim=-1
    )
    interocular = torch.clamp(interocular, min=1e-6)
    norm_errors = point_errors / interocular.unsqueeze(-1)
    nme = torch.mean(norm_errors.mean(dim=1)).item()

    metrics = {
        "pixel_mae": mae,
        "pixel_rmse": rmse,
        "mean_point_error": mean_point_error,
        "nme": nme,
    }

    norm_errors_flat = norm_errors.reshape(-1)
    total_points = norm_errors_flat.numel()
    if total_points > 0:
        for thr in pck_thresholds:
            key = f"pck_{thr:.2f}"
            metrics[key] = (norm_errors_flat <= thr).float().mean().item()

        thresholds = torch.linspace(0.0, auc_max, steps=auc_steps, device=norm_errors_flat.device)
        cdf = (norm_errors_flat.unsqueeze(-1) <= thresholds).float().mean(dim=0)
        auc = torch.trapz(cdf, thresholds) / auc_max
        metrics["auc_0.5"] = auc.item()

    return metrics
