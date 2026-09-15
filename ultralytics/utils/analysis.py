# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Extract image properties, correlations, and label-review scores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ultralytics.utils import DataExportMixin, plt_settings
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xywh2xyxy

COCO_AREA_SMALL = 32**2  # COCO small-object area threshold (px^2), Lin et al. 2014
_LABEL_ISSUES = ("possible_fp", "possible_fn", "possible_label_confusion")


@dataclass
class AnalysisReport(DataExportMixin):
    """Store per-image metrics and property correlations.

    Attributes:
        per_image (dict[str, dict]): Per-image metrics and properties keyed by image path.
        correlations (dict[str, dict]): Per-property Spearman correlation and sample count against F1.
        label_issues (list[dict]): Three highest-priority label-review candidates.
    """

    per_image: dict[str, dict]
    correlations: dict[str, dict]

    def summary(self, normalize: bool = False, decimals: int = 5) -> list[dict]:
        """Return one numeric row per image property."""
        return [
            {
                "property": prop,
                "spearman_r": None if row["spearman_r"] is None else round(row["spearman_r"], decimals),
                "n": row["n"],
            }
            for prop, row in self.correlations.items()
        ]

    @property
    def label_issues(self) -> list[dict]:
        """Return the three strongest label-review candidates."""
        return sorted(
            (
                {"image": image, "issue": issue, "score": score}
                for issue in _LABEL_ISSUES
                for image, row in self.per_image.items()
                if (score := row.get(issue, 0.0)) > 0.5
            ),
            key=lambda row: row["score"],
            reverse=True,
        )[:3]

    @plt_settings()
    def plot(self) -> np.ndarray:
        """Return an RGB plot of the three strongest correlations."""
        import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

        properties = sorted(
            self.correlations, key=lambda prop: abs(self.correlations[prop]["spearman_r"] or 0), reverse=True
        )[:3]
        values = list(self.per_image.values())
        f1 = np.array([row.get("f1", np.nan) for row in values], dtype=float)
        fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
        for ax, prop in zip(axes, properties):
            x = np.array([row.get(prop, np.nan) for row in values], dtype=float)
            mask = np.isfinite(x) & np.isfinite(f1)
            ax.scatter(x[mask], f1[mask], s=4, alpha=0.5)
            r = self.correlations[prop]["spearman_r"]
            ax.set_title(f"{prop}\nSpearman r={r:.2f}" if r is not None else prop, fontsize=8)
            ax.set_xlabel(prop, fontsize=7)
            ax.set_ylabel("f1", fontsize=7)
            ax.tick_params(axis="both", labelsize=6)
        fig.tight_layout()
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return image


class ImagePropertyExtractor:
    """Augment a ``YOLODataset``'s labels in place with six per-image properties.

    Compute object count, small-object ratio, object-scale variation, class count, center spread, and maximum pairwise
    IoU from image headers and annotations.

    Attributes:
        labels (list[dict]): The same list as ``dataset.labels``, with an ``im_properties`` dict added per image.
    """

    def __init__(self, dataset):
        """Extract properties into dataset labels."""
        self.labels = dataset.labels
        for label in self.labels:
            self._augment_label(label)

    @staticmethod
    def _augment_label(lbl: dict) -> None:
        """Compute the six properties for one label into its ``im_properties`` dict."""
        cls_arr = lbl["cls"].reshape(-1)
        bboxes_n = lbl["bboxes"].reshape(-1, 4)
        with Image.open(lbl["im_file"]) as image:
            w, h = image.size
        n = len(bboxes_n)
        areas_n = bboxes_n[:, 2] * bboxes_n[:, 3]
        lbl["im_properties"] = {
            "num_objects": n,
            "small_object_ratio": float(np.mean(areas_n * w * h < COCO_AREA_SMALL)) if n else np.nan,
            "object_scale_variance": (float(np.std(areas_n) / max(np.mean(areas_n), 1e-9)) if n else np.nan),
            "num_classes_present": int(np.unique(cls_arr).size),
            "center_spread": (float(np.sqrt(np.var(bboxes_n[:, 0]) + np.var(bboxes_n[:, 1]))) if n else np.nan),
            "max_pairwise_iou": (ImagePropertyExtractor._max_pairwise_iou(xywh2xyxy(bboxes_n)) if n >= 2 else np.nan),
        }

    @staticmethod
    def _max_pairwise_iou(xyxy: np.ndarray) -> float:
        """Calculate the maximum pairwise IoU among boxes in xyxy format."""
        boxes, maximum = torch.as_tensor(xyxy, dtype=torch.float32), 0.0
        for i in range(0, len(boxes), 1024):
            for j in range(i, len(boxes), 1024):
                iou = box_iou(boxes[i : i + 1024], boxes[j : j + 1024])
                if i == j:
                    iou.triu_(diagonal=1)
                maximum = max(maximum, float(iou.max()))
        return maximum


def analyze_correlations(labels: list[dict], metrics) -> AnalysisReport:
    """Correlate image properties with per-image F1."""
    per_image = {}
    for label in labels:
        im_file = str(Path(label["im_file"]).absolute())
        per_image[im_file] = {
            **metrics.box.image_metrics.get(im_file, {}),
            **label["im_properties"],
        }

    f1 = np.array([row.get("f1", np.nan) for row in per_image.values()], dtype=float)
    correlations = {}
    for prop in labels[0]["im_properties"]:
        values = np.array([row[prop] for row in per_image.values()], dtype=float)
        mask = np.isfinite(values) & np.isfinite(f1)
        r = None
        if mask.sum() > 1 and np.ptp(values[mask]) and np.ptp(f1[mask]):
            r = float(np.corrcoef(_rankdata(values[mask]), _rankdata(f1[mask]))[0, 1])
        correlations[prop] = {"spearman_r": r, "n": int(mask.sum())}
    return AnalysisReport(per_image, correlations)


def _label_issue_scores(
    iou: np.ndarray,
    pred_cls: np.ndarray,
    pred_conf: np.ndarray,
    gt_cls: np.ndarray,
) -> dict[str, float]:
    """Return image-level possible FP, FN, and label-confusion scores."""
    pred_cls, gt_cls = pred_cls.astype(int), gt_cls.astype(int)
    same_class = gt_cls[:, None] == pred_cls[None]
    weighted_iou = iou * pred_conf[None]
    scores = (
        np.max(pred_conf * (1 - np.max(iou, axis=0, initial=0)), initial=0),
        np.max(1 - np.max(np.where(same_class, weighted_iou, 0), axis=1, initial=0), initial=0),
        np.max(np.where(~same_class, weighted_iou, 0), initial=0),
    )
    return dict(zip(_LABEL_ISSUES, map(float, scores)))


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks, assigning tied values their mean rank."""
    sorter = np.argsort(values, kind="stable")
    inverse = np.empty(values.size, dtype=int)
    inverse[sorter] = np.arange(values.size)
    sorted_values = values[sorter]
    observed = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    dense = observed.cumsum()[inverse]
    count = np.r_[np.nonzero(observed)[0], values.size]
    return 0.5 * (count[dense] + count[dense - 1] + 1)
