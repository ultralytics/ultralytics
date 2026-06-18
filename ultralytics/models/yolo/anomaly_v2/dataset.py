# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Dataset with precomputed memory-bank heatmaps attached as a 4th image channel.

Sidecar layout (written by the offline precompute job): ``<root>/heatmaps_v1/<split>/<stem>.png``
— uint8 grayscale, pixel = anomaly score * 255, any resolution (resized to the image here).
Because the heatmap rides as an image channel, every geometric augmentation (mosaic, affine,
flips, letterbox) transforms it in lockstep with the pixels; color augs skip non-3ch inputs.
``YOLOAnomalyV2Model._predict_once`` splits the channel off before the backbone and serves it
via ``prior_mode="cached"``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER


class CachedPriorDataset(YOLODataset):
    """YOLODataset that appends a precomputed heatmap PNG as a 4th image channel."""

    def __init__(self, *args, prior_dir: str | Path, **kwargs):
        self.prior_dir = Path(prior_dir)
        self._prior_warned = False
        super().__init__(*args, **kwargs)
        if not self.prior_dir.is_dir():
            LOGGER.warning(f"CachedPriorDataset: {self.prior_dir} missing — 4th channel will be all-zero")

    def load_image(self, i: int, rect_mode: bool = True):
        """Load image via the base path, then stack the heatmap sidecar as channel 4."""
        im, hw0, hw = super().load_image(i, rect_mode)
        if im.shape[-1] != 3:  # already stacked (RAM-cached entry)
            return im, hw0, hw
        prior = cv2.imread(str(self.prior_dir / (Path(self.im_files[i]).stem + ".png")), cv2.IMREAD_GRAYSCALE)
        if prior is None:
            if not self._prior_warned:
                LOGGER.warning(f"CachedPriorDataset: no sidecar for {Path(self.im_files[i]).name} "
                               f"(zero prior; warning once)")
                self._prior_warned = True
            prior = np.zeros(im.shape[:2], np.uint8)
        elif prior.shape[:2] != im.shape[:2]:
            prior = cv2.resize(prior, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
        return np.dstack([im, prior]), hw0, hw
