import glob
import os
from typing import Iterable, Optional

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        transform=None,
        subset_indices: Optional[Iterable[int]] = None,
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            subset_indices (Iterable[int], optional): Optional iterable of row
                indices to subset the CSV file. Useful for train/val splits.
        """

        self.key_pts_frame = pd.read_csv(csv_file)
        if subset_indices is not None:
            self.key_pts_frame = (
                self.key_pts_frame.iloc[list(subset_indices)].reset_index(drop=True)
            )
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        rel_path = self.key_pts_frame.iloc[idx, 0]
        image_path = os.path.join(self.root_dir, rel_path)

        image = mpimg.imread(image_path)
        image = self._ensure_three_channels(image)
        orig_h, orig_w = image.shape[:2]

        image = image.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0

        key_pts = self.key_pts_frame.iloc[idx, 1:].to_numpy()
        key_pts = key_pts.astype("float32").reshape(-1, 2)
        sample = {
            "image": image,
            "keypoints": key_pts,
            "image_name": rel_path,
            "original_size": np.array([orig_h, orig_w], dtype=np.float32),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def _ensure_three_channels(image):
        """Convert grayscale/alpha images to 3-channel RGB."""

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[2] == 4:
            image = image[:, :, 0:3]
        return image
