"""Per-image occlusion proxy via yolo11n-seg.

occlusion_score = 1 - (person_mask_pixels / total_pixels), in [0, 1].
NaN when the segmenter finds no person (caller distinguishes from full occlusion).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

_SEG_MODEL = None  # lazy-loaded singleton


def mask_to_occlusion_score(mask: Optional[np.ndarray]) -> float:
    """Compute occlusion score from a boolean person mask.

    Args:
        mask: bool array (H, W). None if segmenter found no person.

    Returns:
        Float in [0, 1], or NaN if mask is None.
    """
    if mask is None:
        return float("nan")
    return 1.0 - float(mask.sum()) / float(mask.size)


def segment_person(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Return the union of all person masks in a BGR image (or None if no person).

    Lazy-loads yolo11n-seg on first call.
    """
    global _SEG_MODEL
    if _SEG_MODEL is None:
        from ultralytics import YOLO

        _SEG_MODEL = YOLO("yolo11n-seg.pt")

    results = _SEG_MODEL(image_bgr, classes=[0], verbose=False)  # class 0 = person
    if len(results) == 0 or results[0].masks is None:
        return None
    masks = results[0].masks.data  # (n_instances, H, W) on device
    union = masks.any(dim=0).cpu().numpy()
    return union.astype(bool)


def occlusion_score(image_bgr: np.ndarray) -> float:
    """End-to-end: BGR image -> occlusion score in [0, 1] (or NaN)."""
    return mask_to_occlusion_score(segment_person(image_bgr))
