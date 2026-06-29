# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Per-image property extraction for object detection datasets.

``ImagePropertyExtractor(yolo_dataset)`` augments ``dataset.labels`` in place with per-image properties
(brightness, blurriness, crowdedness, object-size counts, ...). No model, no metrics, no I/O. The augmented
labels are platform-consumable as JSON for JS/TS plotting.

References:
    - Hendrycks & Dietterich, ICLR 2019 (brightness/contrast as detection corruptions).
    - Pech-Pacheco et al., ICPR 2000 (variance-of-Laplacian blur).
    - Canny, IEEE TPAMI 1986 (edge detector).
    - Krotkov, IJCV 1988 (Tenengrad sharpness).
    - Shannon, BSTJ 1948 (entropy).
    - Lin et al., ECCV 2014 (COCO small/medium/large area thresholds).
    - Shao et al., CrowdHuman 2018 (per-image crowdedness via pairwise IoU).
    - Everingham et al., Pascal VOC IJCV 2010 (boundary-truncated objects).
"""

from __future__ import annotations

from multiprocessing.pool import ThreadPool
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.utils import NUM_THREADS, TQDM
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xywh2xyxy, xywhn2xyxy
from ultralytics.utils.patches import imread

COCO_AREA_SMALL = 32**2  # COCO small-object area threshold (px^2), Lin et al. 2014
COCO_AREA_MEDIUM = 96**2  # COCO medium/large boundary, Lin et al. 2014
EDGE_PROXIMITY_FRAC = 0.05  # near-edge tolerance as fraction of min(W,H), motivates num_near_edge

_PIXEL_PROPERTIES = (
    "brightness",
    "blurriness",
    "contrast",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    "entropy",
    "edge_density",
    "sharpness",
)


class ImagePropertyExtractor:
    """Augment a ``YOLODataset``'s labels in place with 31 per-image properties.

    Computes pixel-level properties (brightness, blurriness, contrast, dark/bright-pixel ratio, entropy, edge density,
    sharpness), cache-derived properties (W/H, aspect ratio, COCO-bucket object counts, near-edge counts, class entropy,
    center spread, ...), and annotation-interaction properties (max/mean pairwise IoU). Each label dict in
    ``dataset.labels`` gains the property keys; the ``dataset.labels`` list is mutated in place and re-exposed as
    ``self.labels`` for chaining.

    Has no model, metrics, or I/O dependency: the property step is platform-consumable. Serialize ``self.labels`` (after
    dropping numpy arrays) to feed a JS/TS visualizer.

    Attributes:
        labels (list[dict]): The same list as ``dataset.labels``, with property keys added per image.

    Examples:
        >>> from ultralytics.data.build import build_yolo_dataset
        >>> from ultralytics.data.utils import check_det_dataset
        >>> from ultralytics.utils.analysis import ImagePropertyExtractor
        >>> data = check_det_dataset("coco128.yaml")
        >>> ds = build_yolo_dataset(None, data["val"], 1, data, mode="val", rect=False, stride=32)
        >>> labels = ImagePropertyExtractor(ds).labels
        >>> labels[0]["brightness"], labels[0]["num_small"]
    """

    def __init__(self, dataset: Any):
        """Extract per-image properties and mutate ``dataset.labels`` in place.

        Args:
            dataset (Any): A ``YOLODataset`` instance with non-empty ``labels`` and ``im_file`` per entry.
        """
        labels = getattr(dataset, "labels", None)
        if not labels:
            raise ValueError("ImagePropertyExtractor requires a YOLODataset with non-empty labels.")
        with ThreadPool(NUM_THREADS) as pool:
            for _ in TQDM(pool.imap_unordered(self._augment_label, labels), total=len(labels), desc="Image properties"):
                pass
        self.labels = labels

    @staticmethod
    def _augment_label(lbl: dict) -> dict:
        """Compute pixel, cache, and pairwise-IoU properties for one label dict and merge them in place."""
        im_file = lbl["im_file"]
        cls_arr = np.asarray(lbl.get("cls", np.zeros((0, 1)))).reshape(-1).astype(int)
        bboxes_n = np.asarray(lbl.get("bboxes", np.zeros((0, 4)))).reshape(-1, 4)
        img = imread(im_file)
        if img is not None and img.size:
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lbl.update(ImagePropertyExtractor._pixel_properties(img, gray))
        else:
            h, w = 0, 0
            for k in _PIXEL_PROPERTIES:
                lbl[k] = np.nan
        lbl.update(ImagePropertyExtractor._cache_properties(w, h, bboxes_n, cls_arr))
        if bboxes_n.shape[0] >= 2 and h > 0 and w > 0:
            lbl["max_pairwise_iou"], lbl["mean_pairwise_iou"] = ImagePropertyExtractor._pairwise_iou_stats(
                xywhn2xyxy(bboxes_n, w=w, h=h)
            )
        else:
            lbl["max_pairwise_iou"] = np.nan
            lbl["mean_pairwise_iou"] = np.nan
        return lbl

    @staticmethod
    def _cache_properties(w: int, h: int, bboxes_n: np.ndarray, cls_arr: np.ndarray) -> dict[str, Any]:
        """Compute cache-derived properties (counts, sizes, centers, class diversity) from cached W/H and boxes."""
        n = int(bboxes_n.shape[0])
        out: dict[str, Any] = {
            "width": w,
            "height": h,
            "aspect_ratio": (w / h) if h else np.nan,
            "total_pixels": w * h,
            "num_objects": n,
        }
        if n == 0 or h == 0 or w == 0:
            out.update({"num_small": 0, "num_medium": 0, "num_large": 0, "num_near_edge": 0})
            for k in (
                "small_object_ratio",
                "mean_center_x",
                "mean_center_y",
                "center_spread",
                "box_area_std_norm",
                "object_scale_variance",
            ):
                out[k] = np.nan
            out["num_classes_present"] = int(np.unique(cls_arr).size)
            out["class_entropy"] = 0.0
            return out

        cx, cy, bw, bh = bboxes_n[:, 0], bboxes_n[:, 1], bboxes_n[:, 2], bboxes_n[:, 3]
        area_px = (bw * w) * (bh * h)
        out["num_small"] = int(np.sum(area_px < COCO_AREA_SMALL))
        out["num_medium"] = int(np.sum((area_px >= COCO_AREA_SMALL) & (area_px < COCO_AREA_MEDIUM)))
        out["num_large"] = int(np.sum(area_px >= COCO_AREA_MEDIUM))
        out["small_object_ratio"] = out["num_small"] / max(n, 1)

        xyxy_n = xywh2xyxy(bboxes_n)
        min_edge = np.minimum(np.minimum(xyxy_n[:, 0], xyxy_n[:, 1]), np.minimum(1 - xyxy_n[:, 2], 1 - xyxy_n[:, 3]))
        out["num_near_edge"] = int(np.sum(min_edge < EDGE_PROXIMITY_FRAC))

        out["mean_center_x"] = float(np.mean(cx))
        out["mean_center_y"] = float(np.mean(cy))
        out["center_spread"] = float(np.sqrt(np.var(cx) + np.var(cy)))

        area_n = bw * bh
        out["box_area_std_norm"] = float(np.std(area_n))
        out["object_scale_variance"] = float(np.std(area_n) / max(np.mean(area_n), 1e-9))

        uniq, counts = np.unique(cls_arr, return_counts=True)
        out["num_classes_present"] = int(uniq.size)
        p = counts / counts.sum()
        out["class_entropy"] = float(-np.sum(p * np.log2(p + 1e-12)))
        return out

    @staticmethod
    def _pixel_properties(img_bgr: np.ndarray, gray: np.ndarray) -> dict[str, float]:
        """Compute the 8 pixel-reading properties from a BGR image and its grayscale conversion."""
        return {
            "brightness": ImagePropertyExtractor._brightness(img_bgr),
            "blurriness": ImagePropertyExtractor._blurriness(gray),
            "contrast": ImagePropertyExtractor._contrast(gray),
            "dark_pixel_ratio": ImagePropertyExtractor._dark_pixel_ratio(gray),
            "bright_pixel_ratio": ImagePropertyExtractor._bright_pixel_ratio(gray),
            "entropy": ImagePropertyExtractor._entropy(gray),
            "edge_density": ImagePropertyExtractor._edge_density(gray),
            "sharpness": ImagePropertyExtractor._sharpness(gray),
        }

    @staticmethod
    def _brightness(img_bgr: np.ndarray) -> float:
        """HSP perceptual brightness, ``sqrt(0.241 R^2 + 0.691 G^2 + 0.068 B^2) / 255`` (Finley 2006)."""
        f = img_bgr.astype(np.float32)
        b, g, r = f[..., 0], f[..., 1], f[..., 2]
        return float(np.sqrt(0.241 * r * r + 0.691 * g * g + 0.068 * b * b).mean() / 255.0)

    @staticmethod
    def _blurriness(gray: np.ndarray) -> float:
        """Variance-of-Laplacian focus measure, mapped to ``1 / (1 + var)`` (Pech-Pacheco et al. 2000)."""
        return float(1.0 / (1.0 + cv2.Laplacian(gray, cv2.CV_64F).var()))

    @staticmethod
    def _contrast(gray: np.ndarray) -> float:
        """Grayscale standard deviation normalized to [0,1] (Hendrycks & Dietterich 2019 corruption axis)."""
        return float(gray.std() / 255.0)

    @staticmethod
    def _dark_pixel_ratio(gray: np.ndarray) -> float:
        """Fraction of pixels with intensity < 25 (Hendrycks & Dietterich 2019 dark/low-light axis)."""
        return float((gray < 25).mean())

    @staticmethod
    def _bright_pixel_ratio(gray: np.ndarray) -> float:
        """Fraction of pixels with intensity > 230 (Hendrycks & Dietterich 2019 saturation axis)."""
        return float((gray > 230).mean())

    @staticmethod
    def _entropy(gray: np.ndarray) -> float:
        """Shannon entropy over the 256-bin grayscale histogram (Shannon 1948)."""
        hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
        s = hist.sum()
        if s == 0:
            return 0.0
        p = hist[hist > 0] / s
        return float(-np.sum(p * np.log2(p)))

    @staticmethod
    def _edge_density(gray: np.ndarray) -> float:
        """Mean of the Canny edge map normalized to [0,1] (Canny 1986)."""
        return float(cv2.Canny(gray, 100, 200).mean() / 255.0)

    @staticmethod
    def _sharpness(gray: np.ndarray) -> float:
        """Tenengrad sharpness, mean magnitude of Sobel gradient (Krotkov 1988)."""
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.hypot(gx, gy).mean())

    @staticmethod
    def _pairwise_iou_stats(xyxy_pixels: np.ndarray) -> tuple[float, float]:
        """Max and mean upper-triangular pairwise IoU among boxes (CrowdHuman 2018 crowdedness proxy)."""
        t = torch.as_tensor(xyxy_pixels, dtype=torch.float32)
        n = t.shape[0]
        if n < 2:
            return 0.0, 0.0
        iou = box_iou(t, t).triu_(diagonal=1)
        return float(iou.max()), float(iou.sum() / (n * (n - 1) / 2))
