"""Custom transforms for facial keypoint detection."""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter


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


class RandomAffine:
    """Apply random affine transformation (rotation/translation/scale)."""

    def __init__(
        self,
        degrees: float = 0.0,
        translate: Tuple[float, float] = (0.0, 0.0),
        scale: Tuple[float, float] = (1.0, 1.0),
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale

    def __call__(self, sample):
        if self.degrees == 0 and self.translate == (0.0, 0.0) and self.scale == (1.0, 1.0):
            return sample

        height, width = sample["image_size"]
        height, width = int(height), int(width)
        angle = random.uniform(-self.degrees, self.degrees)
        max_dx = self.translate[0] * width
        max_dy = self.translate[1] * height
        translate = (random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy))
        scale = random.uniform(self.scale[0], self.scale[1])

        pil_image = Image.fromarray(_to_uint8(sample["image"]))
        transformed = TF.affine(
            pil_image,
            angle=angle,
            translate=(int(translate[0]), int(translate[1])),
            scale=scale,
            shear=0.0,
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=None,
        )
        sample["image"] = np.asarray(transformed, dtype=np.float32) / 255.0

        radians = math.radians(angle)
        rot_matrix = np.array(
            [[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]],
            dtype=np.float32,
        )
        center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        keypoints = sample["keypoints"].copy()
        keypoints = (keypoints - center) * scale
        keypoints = (rot_matrix @ keypoints.T).T + center + np.array(translate, dtype=np.float32)
        keypoints[:, 0] = np.clip(keypoints[:, 0], 0, width - 1)
        keypoints[:, 1] = np.clip(keypoints[:, 1], 0, height - 1)
        sample["keypoints"] = keypoints
        return sample


class RandomPerspective:
    """Apply a random perspective warp."""

    def __init__(self, distortion_scale: float = 0.25, prob: float = 0.5):
        self.distortion_scale = distortion_scale
        self.prob = prob

    def __call__(self, sample):
        if random.random() > self.prob:
            return sample

        height, width = sample["image_size"]
        height, width = int(height), int(width)
        margin = self.distortion_scale * min(height, width)
        startpoints = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
        )
        endpoints = startpoints + np.random.uniform(-margin, margin, size=startpoints.shape)

        try:
            coeffs = _get_perspective_coeffs(startpoints, endpoints)
        except np.linalg.LinAlgError:
            return sample

        pil_image = Image.fromarray(_to_uint8(sample["image"]))
        transformed = pil_image.transform(
            (width, height), Image.PERSPECTIVE, coeffs, Image.BILINEAR
        )
        sample["image"] = np.asarray(transformed, dtype=np.float32) / 255.0

        matrix = _coeffs_to_matrix(coeffs)
        keypoints = sample["keypoints"].copy()
        keypoints_h = np.concatenate(
            [keypoints, np.ones((keypoints.shape[0], 1), dtype=np.float32)], axis=1
        )
        warped = (matrix @ keypoints_h.T).T
        warped = warped[:, :2] / warped[:, 2:3]
        warped[:, 0] = np.clip(warped[:, 0], 0, width - 1)
        warped[:, 1] = np.clip(warped[:, 1], 0, height - 1)
        sample["keypoints"] = warped
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


class RandomGaussianBlur:
    """Apply Gaussian blur with random radius."""

    def __init__(self, max_radius: float = 1.5, prob: float = 0.2):
        self.max_radius = max_radius
        self.prob = prob

    def __call__(self, sample):
        if random.random() > self.prob or self.max_radius <= 0:
            return sample
        radius = random.uniform(0.1, self.max_radius)
        pil_image = Image.fromarray(_to_uint8(sample["image"]))
        blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))
        sample["image"] = np.asarray(blurred, dtype=np.float32) / 255.0
        return sample


class RandomErasing:
    """Randomly zero out a rectangular region (cutout)."""

    def __init__(
        self,
        prob: float = 0.3,
        scale: Tuple[float, float] = (0.02, 0.12),
        ratio: Tuple[float, float] = (0.3, 3.3),
    ):
        self.prob = prob
        self.scale = scale
        self.ratio = ratio

    def __call__(self, sample):
        if random.random() > self.prob:
            return sample

        image = sample["image"].copy()
        h, w, _ = image.shape
        area = h * w
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            erase_h = int(round(math.sqrt(target_area * aspect_ratio)))
            erase_w = int(round(math.sqrt(target_area / aspect_ratio)))

            if erase_h < h and erase_w < w:
                top = random.randint(0, h - erase_h)
                left = random.randint(0, w - erase_w)
                image[top : top + erase_h, left : left + erase_w, :] = 0.0
                sample["image"] = image
                return sample
        return sample


def _get_perspective_coeffs(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Solve for perspective transform coefficients used by PIL."""

    matrix = []
    for (x_src, y_src), (x_dst, y_dst) in zip(src, dst):
        matrix.append([x_src, y_src, 1, 0, 0, 0, -x_dst * x_src, -x_dst * y_src])
        matrix.append([0, 0, 0, x_src, y_src, 1, -y_dst * x_src, -y_dst * y_src])
    A = np.array(matrix, dtype=np.float32)
    B = dst.flatten()
    coeffs = np.linalg.solve(A, B)
    return coeffs


def _coeffs_to_matrix(coeffs: np.ndarray) -> np.ndarray:
    a, b, c, d, e, f, g, h = coeffs
    return np.array([[a, b, c], [d, e, f], [g, h, 1.0]], dtype=np.float32)
