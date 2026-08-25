# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Extract image properties and map F1 correlations to dataset actions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ultralytics.utils import LOGGER, DataExportMixin, SimpleClass
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xywh2xyxy

COCO_AREA_SMALL = 32**2  # COCO small-object area threshold (px^2), Lin et al. 2014
_IMAGE_PROPERTIES = (
    "num_objects",
    "small_object_ratio",
    "object_scale_variance",
    "num_classes_present",
    "center_spread",
    "max_pairwise_iou",
)
_PROPERTY_INSIGHTS = {
    "num_objects": ("dense scenes reduce F1", "add crowded-scene training images or use tiled crops"),
    "small_object_ratio": ("small-object-heavy images reduce F1", "increase imgsz and add small-object examples"),
    "object_scale_variance": (
        "mixed object scales reduce F1",
        "enable multi-scale training and add mixed-scale scenes",
    ),
    "num_classes_present": ("multi-class scenes reduce F1", "add training images containing more classes together"),
    "center_spread": ("widely spread objects reduce F1", "add full-frame scenes with objects near image edges"),
    "max_pairwise_iou": ("overlapping objects reduce F1", "add occluded examples or copy-paste crowded scenes"),
}


@dataclass
class AnalysisReport(SimpleClass, DataExportMixin):
    """Container for raw analysis evidence and concise actionable insights.

    Attributes:
        per_image (dict[str, dict]): Per-image evidence keyed by image basename.
        correlations (dict[str, dict]): Per-property Spearman correlation and sample count against F1.
        insights (list[dict]): Flat actionable records with ``target``, ``issue``, ``score``, ``evidence``, and
            ``action``.
    """

    per_image: dict[str, dict]
    correlations: dict[str, dict]
    insights: list[dict]

    def summary(self, normalize: bool = False, decimals: int = 5) -> list[dict]:
        """Return actionable insight rows."""
        return [{**x, "score": round(x["score"], decimals)} for x in self.insights]

    def plot(self) -> np.ndarray | None:
        """Return an RGB plot of the strongest actionable F1 drivers."""
        import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

        drivers = _performance_drivers(self.correlations)
        if not drivers:
            LOGGER.warning("AnalysisReport.plot: no meaningful F1-lowering property found")
            return
        scored = list(self.per_image.values())
        fig, axes = plt.subplots(1, len(drivers), figsize=(len(drivers) * 3.4, 3.0))
        axes = np.atleast_1d(axes)
        ys = np.array([v.get("f1", np.nan) for v in scored], dtype=float)
        for ax, (prop, corr) in zip(axes, drivers):
            xs = np.array([v.get(prop, np.nan) for v in scored], dtype=float)
            m = np.isfinite(xs) & np.isfinite(ys)
            ax.scatter(xs[m], ys[m], s=4, alpha=0.5, c="tab:blue")
            if m.sum() > 1 and np.std(xs[m]) > 0:
                coef = np.polyfit(xs[m], ys[m], 1)
                xline = np.linspace(xs[m].min(), xs[m].max(), 50)
                ax.plot(xline, np.polyval(coef, xline), color="tab:red", lw=1.0)
            ax.set_title(f"{prop}\nSpearman r={corr['spearman_r']:.2f}", fontsize=8)
            ax.set_xlabel(prop, fontsize=7)
            ax.set_ylabel("f1", fontsize=7)
            ax.tick_params(axis="both", labelsize=6)
        fig.suptitle("Strongest actionable F1 drivers", fontsize=10)
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

    def __init__(self, dataset: Any):
        """Extract per-image properties and mutate ``dataset.labels`` in place.

        Args:
            dataset (Any): A ``YOLODataset`` with non-empty labels.
        """
        labels = getattr(dataset, "labels", None)
        if not labels:
            raise ValueError("ImagePropertyExtractor requires a YOLODataset with non-empty labels.")
        for label in labels:
            self._augment_label(label)
        self.labels = labels

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


class CorrelationAnalysis:
    """Join image properties with per-image F1 and return up to three dataset actions.

    Attributes:
        labels (list[dict]): Property-augmented label dicts from ``ImagePropertyExtractor.labels``.
        metrics: A metrics object exposing ``.box.image_metrics`` (e.g. ``DetMetrics``).
    """

    def __init__(self, labels: list[dict], metrics: Any):
        """Bind labels and metrics; no computation runs until :meth:`run`.

        Args:
            labels (list[dict]): Property-augmented labels (output of ``ImagePropertyExtractor``).
            metrics (Any): Metrics object from ``model.val()`` exposing ``.box.image_metrics``.
        """
        if not labels:
            raise ValueError("CorrelationAnalysis requires non-empty labels from ImagePropertyExtractor.")
        if metrics is None:
            raise ValueError("CorrelationAnalysis requires a metrics object from model.val().")
        self.labels = labels
        self.metrics = metrics

    def run(self) -> AnalysisReport:
        """Return per-image evidence, correlations, and actionable insights."""
        per_image = self._join(self.labels, self.metrics)

        f1s = np.array([rec.get("f1", np.nan) for rec in per_image.values()], dtype=float)
        median_f1 = float(np.nanmedian(f1s)) if np.isfinite(f1s).any() else float("nan")
        if median_f1 < 0.1:
            LOGGER.warning(
                f"CorrelationAnalysis: per-image F1 median is {median_f1:.3f}. "
                f"Re-run model.val(..., conf=0.25) for meaningful per-image F1."
            )

        correlations = self._compute_correlations(per_image)
        return AnalysisReport(
            per_image=per_image,
            correlations=correlations,
            insights=self._build_insights(correlations),
        )

    @staticmethod
    def _join(labels: list[dict], metrics: Any) -> dict[str, dict]:
        """Join per-image validator metrics and properties by image basename."""
        metric_obj = getattr(metrics, "box", metrics)
        image_metrics = getattr(metric_obj, "image_metrics", {})
        per_image: dict[str, dict] = {}
        dup_names: list[str] = []
        for lbl in labels:
            im_file = lbl["im_file"]
            im_name = Path(im_file).name
            if im_name in per_image:
                dup_names.append(im_name)
                continue
            rec = dict(image_metrics.get(im_name, {}))
            rec["im_file"] = im_file
            props = lbl.get("im_properties", {})
            for k in _IMAGE_PROPERTIES:
                if k in props:
                    rec[k] = props[k]
            per_image[im_name] = rec
        if dup_names:
            LOGGER.warning(
                f"CorrelationAnalysis: dropped {len(dup_names)} duplicate basename(s) from join, "
                f"e.g. {', '.join(dup_names[:3])}. Per-image records are keyed by basename."
            )
        return per_image

    @staticmethod
    def _compute_correlations(per_image: dict) -> dict[str, dict]:
        """Compute Spearman correlation between each image property and F1."""
        f1 = np.array([rec.get("f1", np.nan) for rec in per_image.values()], dtype=float)
        out: dict[str, dict] = {}
        for prop in _IMAGE_PROPERTIES:
            xs = np.array([rec.get(prop, np.nan) for rec in per_image.values()], dtype=float)
            m = np.isfinite(xs) & np.isfinite(f1)
            if m.sum() < 30 or np.std(xs[m]) == 0 or np.std(f1[m]) == 0:
                out[prop] = {"spearman_r": None, "n": int(m.sum())}
                continue
            out[prop] = {
                "spearman_r": float(np.corrcoef(_rankdata(xs[m]), _rankdata(f1[m]))[0, 1]),
                "n": int(m.sum()),
            }
        return out

    @staticmethod
    def _build_insights(correlations: dict) -> list[dict]:
        """Translate the strongest F1-lowering correlations into specific dataset actions."""
        return [
            {
                "target": "dataset",
                "issue": _PROPERTY_INSIGHTS[prop][0],
                "score": corr["spearman_r"],
                "evidence": f"{prop} Spearman correlation, n={corr['n']}",
                "action": _PROPERTY_INSIGHTS[prop][1],
            }
            for prop, corr in _performance_drivers(correlations)
        ]


def _performance_drivers(correlations: dict) -> list[tuple[str, dict]]:
    """Return up to three meaningful property correlations where higher values lower F1."""
    return sorted(
        (
            (prop, correlations[prop])
            for prop in _PROPERTY_INSIGHTS
            if isinstance(correlations.get(prop, {}).get("spearman_r"), (int, float))
            and correlations[prop]["spearman_r"] <= -0.1
        ),
        key=lambda x: x[1]["spearman_r"],
    )[:3]


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks of a 1D array (tied values share their mean rank), a NumPy stand-in for scipy.stats.rankdata."""
    sorter = np.argsort(a, kind="stable")
    inv = np.empty(a.size, dtype=int)
    inv[sorter] = np.arange(a.size)
    arr = a[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], a.size]
    return 0.5 * (count[dense] + count[dense - 1] + 1)
