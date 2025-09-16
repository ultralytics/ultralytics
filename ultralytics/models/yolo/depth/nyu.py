# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
NYU Depth V2 dataset.

This module contains the class for the NYU Depth V2 dataset.
"""

from pathlib import Path

import numpy as np
import cv2

from ultralytics.data.base import BaseDataset
from ultralytics.utils import LOGGER, colorstr


class NYUDepthV2(BaseDataset):
    """
    NYU Depth V2 dataset for depth estimation.

    Args:
        data (dict, optional): A dictionary containing data information. Defaults to None.
        use_keypoints (bool): Whether to use keypoints.
    """

    def __init__(self, *args, data=None, use_keypoints=False, **kwargs):
        """Initializes the NYUDepthV2 dataset."""
        super().__init__(*args, **kwargs)
        self.data = data
        self.use_keypoints = use_keypoints

    def get_img_files(self, img_path):
        """
        Returns a list of image files with specified extensions.
        """
        try:
            f = []  # image files
            for p in Path(img_path).rglob("*.jpg"):
                f.append(str(p))
            return f
        except Exception as e:
            raise FileNotFoundError(f"Error loading data from {img_path}, check 'path' and 'split' in data YAML") from e

    def load_image(self, i, rect_mode=True):
        """
        Loads an image and its corresponding depth map.
        """
        path = self.im_files[i]
        img = cv2.imread(path)  # BGR
        assert img is not None, f"Image Not Found {path}"
        h0, w0 = img.shape[:2]
        r = self.imgsz / max(h0, w0)
        if r != 1:
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
        return img, (h0, w0), img.shape[:2]

    def build_transforms(self, hyp=None):
        """
        Builds transformations for the dataset.
        """
        return None

    def get_labels(self):
        """
        Returns the labels for the dataset.
        """
        labels = []
        for path in self.im_files:
            depth_path = path.replace(".jpg", ".png")
            labels.append(dict(im_file=path, shape=(480, 640), depth_path=depth_path))
        return labels
