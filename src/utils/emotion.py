"""Rule-based sentiment classification using predicted keypoints."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from .keypoints import (
    LEFT_EYE_OUTER,
    MOUTH_BOTTOM,
    MOUTH_CENTER,
    MOUTH_LEFT,
    MOUTH_RIGHT,
    MOUTH_TOP,
    RIGHT_EYE_OUTER,
)


class EmotionClassifier:
    """Simple heuristic classifier (positive/negative/neutral)."""

    LABELS = ("negative", "neutral", "positive")

    def __init__(
        self,
        positive_curve: float = 0.015,
        positive_width: float = 0.7,
        negative_curve: float = 0.02,
        negative_height: float = 0.32,
    ):
        self.positive_curve = positive_curve
        self.positive_width = positive_width
        self.negative_curve = negative_curve
        self.negative_height = negative_height

    def predict(self, keypoints: Sequence[Sequence[float]]) -> str:
        """Predict emotion from a single set of keypoints (pixel coords)."""

        points = np.asarray(keypoints, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Expected keypoints with shape (K, 2).")

        interocular = np.linalg.norm(points[RIGHT_EYE_OUTER] - points[LEFT_EYE_OUTER])
        scale = max(interocular, 1e-6)

        mouth_width = (points[MOUTH_RIGHT, 0] - points[MOUTH_LEFT, 0]) / scale
        mouth_height = (points[MOUTH_BOTTOM, 1] - points[MOUTH_TOP, 1]) / scale
        smile_curve = (
            (points[MOUTH_LEFT, 1] + points[MOUTH_RIGHT, 1]) / 2.0 - points[MOUTH_CENTER, 1]
        ) / scale

        if smile_curve < -self.positive_curve and mouth_width > self.positive_width:
            return "positive"
        if smile_curve > self.negative_curve or mouth_height > self.negative_height:
            return "negative"
        return "neutral"

    def predict_batch(self, keypoints_batch: Iterable[Sequence[Sequence[float]]]) -> List[str]:
        return [self.predict(points) for points in keypoints_batch]
