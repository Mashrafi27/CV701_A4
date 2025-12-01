"""Custom transforms for facial keypoint detection."""

from __future__ import annotations

import random
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


class Compose:
    """Compose multiple transforms that operate on dict samples."""

    def __init__(self, transforms: Sequence):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class Rescale:
    """Resize the image to a given (height, width) while scaling keypoints."""

    def __init__(self, output_size: Tuple[int, int] | int):
        if isinstance(output_size, Iterable) and not isinstance(output_size, (list, tuple)):
            output_size = tuple(output_size)
        self.output_size = output_size

    def __call__(self, sample):
        image = sample["image"]
        keypoints = sample["keypoints"].copy()
        orig_h, orig_w = image.shape[:2]

        if isinstance(self.output_size, int):
            new_h = new_w = self.output_size
        else:
            new_h, new_w = self.output_size

        pil_image = Image.fromarray(_to_uint8(image))
        resized = pil_image.resize((new_w, new_h), Image.BILINEAR)
        image = np.asarray(resized, dtype=np.float32) / 255.0

        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        keypoints *= np.array([scale_x, scale_y], dtype=np.float32)

        sample.update(
            {
                "image": image,
                "keypoints": keypoints,
                "image_size": np.array([new_h, new_w], dtype=np.float32),
                "keypoint_normalization": np.array([new_h, new_w], dtype=np.float32),
            }
        )
        return sample


class RandomHorizontalFlip:
    """Horizontally flip the sample with a given probability."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() > self.prob:
            return sample

        image = np.fliplr(sample["image"]).copy()
        width = sample["image_size"][1]
        keypoints = sample["keypoints"].copy()
        keypoints[:, 0] = width - keypoints[:, 0]

        sample.update({"image": image, "keypoints": keypoints})
        return sample


class RandomRotation:
    """Rotate image and keypoints around the image center."""

    def __init__(self, degrees: float = 0.0):
        self.degrees = degrees

    def __call__(self, sample):
        if self.degrees <= 0:
            return sample
        angle = random.uniform(-self.degrees, self.degrees)
        if abs(angle) < 1e-3:
            return sample

        image = sample["image"]
        height, width = sample["image_size"]
        pil_image = Image.fromarray(_to_uint8(image))
        rotated = pil_image.rotate(angle, resample=Image.BILINEAR)
        sample["image"] = np.asarray(rotated, dtype=np.float32) / 255.0

        radians = np.deg2rad(angle)
        rot_matrix = np.array(
            [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]],
            dtype=np.float32,
        )
        center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        keypoints = sample["keypoints"].copy()
        keypoints = (rot_matrix @ (keypoints - center).T).T + center
        sample["keypoints"] = keypoints
        return sample


class ColorJitter:
    """Randomly perturb image brightness and contrast."""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2):
        self.brightness = max(0.0, brightness)
        self.contrast = max(0.0, contrast)

    def __call__(self, sample):
        image = sample["image"].astype(np.float32)

        if self.brightness > 0:
            factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            image *= factor

        if self.contrast > 0:
            mean = image.mean(axis=(0, 1), keepdims=True)
            factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            image = (image - mean) * factor + mean

        image = np.clip(image, 0.0, 1.0)
        sample["image"] = image
        return sample


class ToTensor:
    """Convert ndarrays in sample to PyTorch tensors."""

    def __call__(self, sample):
        image = torch.from_numpy(sample["image"].transpose((2, 0, 1))).float()
        keypoints = torch.from_numpy(sample["keypoints"]).float()

        sample["image"] = image
        sample["keypoints"] = keypoints

        for field in ("image_size", "keypoint_normalization"):
            if field in sample:
                sample[field] = torch.from_numpy(sample[field]).float()

        return sample


class NormalizeImage:
    """Normalize tensor image using ImageNet statistics."""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, sample):
        image = sample["image"]
        # Broadcast mean/std to device when invoked inside DataLoader workers.
        mean = self.mean.to(image.device if isinstance(image, torch.Tensor) else "cpu")
        std = self.std.to(image.device if isinstance(image, torch.Tensor) else "cpu")
        sample["image"] = (image - mean) / std
        return sample


class NormalizeKeypoints:
    """Scale keypoints from pixel coordinates to [-1, 1]."""

    def __call__(self, sample):
        keypoints = sample["keypoints"].copy()
        height, width = sample["keypoint_normalization"].astype(np.float32)

        if width == 0 or height == 0:
            raise ValueError("Invalid keypoint normalization factors.")

        keypoints[:, 0] = (keypoints[:, 0] / width - 0.5) * 2.0
        keypoints[:, 1] = (keypoints[:, 1] / height - 0.5) * 2.0

        sample["keypoints"] = keypoints
        return sample


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert float image in [0,1] to uint8 if necessary."""

    if image.dtype == np.uint8:
        return image
    image = np.clip(image * 255.0, 0, 255)
    return image.astype(np.uint8)
