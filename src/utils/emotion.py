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

LEFT_EYE_TOP = 37
LEFT_EYE_BOTTOM = 41
RIGHT_EYE_TOP = 43
RIGHT_EYE_BOTTOM = 47
LEFT_BROW_POINTS = (18, 19, 20)
RIGHT_BROW_POINTS = (23, 24, 25)


class EmotionClassifier:
    """Rule-based classifier (negative/neutral/positive) using geometric cues."""

    LABELS = ("negative", "neutral", "positive")

    def __init__(
        self,
        positive_curve: float = 0.010,
        positive_width: float = 0.56,
        positive_height: float = 0.24,
        positive_eye: float = 0.060,
        negative_curve: float = 0.050,
        negative_height: float = 0.18,
        negative_eye: float = 0.090,
        negative_brow: float = -0.18,
    ):
        self.positive_curve = positive_curve
        self.positive_width = positive_width
        self.positive_height = positive_height
        self.positive_eye = positive_eye
        self.negative_curve = negative_curve
        self.negative_height = negative_height
        self.negative_eye = negative_eye
        self.negative_brow = negative_brow

    def predict(self, keypoints: Sequence[Sequence[float]]) -> str:
        """Predict emotion from a single set of keypoints (pixel coords)."""

        features = self._compute_features(keypoints)
        score = 0

        # Positive cues
        if features["smile_curve"] < self.positive_curve:
            score += 1
        if features["mouth_width"] > self.positive_width:
            score += 1
        if features["mouth_height"] > self.positive_height:
            score += 1
        if features["eye_aspect"] < self.positive_eye:
            score += 1

        # Negative cues
        if features["smile_curve"] > self.negative_curve:
            score -= 2
        if features["mouth_height"] > self.negative_height:
            score -= 1
        if features["eye_aspect"] > self.negative_eye:
            score -= 1
        if features["brow_eye_dist"] > self.negative_brow:
            score -= 1

        if score >= 2:
            return "positive"
        if score <= -1:
            return "negative"
        return "neutral"

    def predict_batch(self, keypoints_batch: Iterable[Sequence[Sequence[float]]]) -> List[str]:
        return [self.predict(points) for points in keypoints_batch]

    @staticmethod
    def _compute_features(keypoints: Sequence[Sequence[float]]):
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

        left_eye_height = abs(points[LEFT_EYE_BOTTOM, 1] - points[LEFT_EYE_TOP, 1]) / scale
        right_eye_height = abs(points[RIGHT_EYE_BOTTOM, 1] - points[RIGHT_EYE_TOP, 1]) / scale
        eye_aspect = (left_eye_height + right_eye_height) / 2.0

        left_brow = points[list(LEFT_BROW_POINTS)].mean(axis=0)
        right_brow = points[list(RIGHT_BROW_POINTS)].mean(axis=0)
        eye_center = (
            (points[LEFT_EYE_TOP] + points[LEFT_EYE_BOTTOM] + points[RIGHT_EYE_TOP] + points[RIGHT_EYE_BOTTOM])
            / 4.0
        )
        brow_eye_dist = ((left_brow[1] + right_brow[1]) / 2 - eye_center[1]) / scale

        return {
            "mouth_width": mouth_width,
            "mouth_height": mouth_height,
            "smile_curve": smile_curve,
            "eye_aspect": eye_aspect,
            "brow_eye_dist": brow_eye_dist,
        }
