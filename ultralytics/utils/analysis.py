# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Extract image properties and map model evidence to dataset or label actions."""

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
_LABEL_TEMPERATURE = 0.1
_LABEL_ALPHA = 0.9
_LABEL_HIGH_CONF = 0.95
_LABEL_LOW_CONF = 0.5
_LABEL_MIN_IOU = 0.1
_LABEL_BADLOC_IOU = 0.4
_IMAGE_PROPERTIES = (
    "num_objects",
    "small_object_ratio",
    "object_scale_variance",
    "num_classes_present",
    "center_spread",
    "max_pairwise_iou",
)
_LABEL_PROPERTIES = ("overlooked_score", "badloc_score", "swap_score")
_LABEL_INSIGHTS = {
    "overlooked_score": (
        "possible missing label or model false positive",
        "review the overlay; add a box if the prediction is correct, otherwise add the image as a hard negative",
    ),
    "badloc_score": (
        "possible incorrect box or model localization error",
        "review the overlay; correct the label box if wrong, otherwise add the image as a localization example",
    ),
    "swap_score": (
        "possible incorrect class or model classification error",
        "review the overlay; correct the label class if wrong, otherwise add the image as a confusing-class example",
    ),
}
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
        """Return actionable insight rows for ``DataExportMixin``.

        Args:
            normalize (bool, optional): Reserved for ``DataExportMixin`` API symmetry, unused here.
            decimals (int, optional): Decimal precision for float fields.

        Returns:
            (list[dict]): Actionable issues with their evidence and next step.
        """
        return [{**x, "score": round(x["score"], decimals)} for x in self.insights]

    def plot(self) -> np.ndarray | None:
        """Return one compact plot of the strongest actionable F1 drivers.

        Returns:
            (np.ndarray | None): RGB plot image, or None when no property has a meaningful negative correlation.
        """
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
        t = torch.as_tensor(xyxy, dtype=torch.float32)
        iou = box_iou(t, t).triu_(diagonal=1)
        return float(iou.max())


class CorrelationAnalysis:
    """Join image properties and optional label scores with F1, then return actions.

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
        """Compute correlations and return raw evidence plus actionable insights without writing files.

        Returns:
            (AnalysisReport): Raw per-image evidence, correlations, and concise next-step insights.
        """
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
            insights=self._build_insights(per_image, correlations),
        )

    @staticmethod
    def _join(labels: list[dict], metrics: Any) -> dict[str, dict]:
        """Merge per-image validator metrics with property fields, keyed by image basename.

        Warns if two labels share the same basename (the join is keyed on it and would silently collide).
        """
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
            for k in _LABEL_PROPERTIES:
                rec.setdefault(k, np.nan)
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
    def _build_insights(per_image: dict, correlations: dict) -> list[dict]:
        """Translate model evidence into specific dataset and label-review actions."""
        performance = [
            {
                "target": "dataset",
                "issue": _PROPERTY_INSIGHTS[prop][0],
                "score": corr["spearman_r"],
                "evidence": f"{prop} Spearman correlation, n={corr['n']}",
                "action": _PROPERTY_INSIGHTS[prop][1],
            }
            for prop, corr in _performance_drivers(correlations)
        ]
        label_issues = sorted(
            (
                {"target": im_name, "issue": issue, "score": score, "evidence": prop, "action": action}
                for prop, (issue, action) in _LABEL_INSIGHTS.items()
                for im_name, rec in per_image.items()
                if isinstance((score := rec.get(prop)), (int, float)) and np.isfinite(score) and score < 0.5
            ),
            key=lambda x: x["score"],
        )[:3]
        return performance + label_issues


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


def _label_issue_scores(
    iou: np.ndarray | None,
    pred_boxes: np.ndarray,
    pred_cls: np.ndarray,
    pred_conf: np.ndarray,
    gt_boxes: np.ndarray,
    gt_cls: np.ndarray,
) -> dict[str, float]:
    """Score possible missing labels, incorrect boxes, and incorrect classes.

    Args:
        iou (np.ndarray | None): Pairwise IoU matrix, or None when the image has no ground truth.
        pred_boxes (np.ndarray): Prediction boxes in xyxy format with shape (N, 4).
        pred_cls (np.ndarray): Prediction class IDs with shape (N,).
        pred_conf (np.ndarray): Prediction confidences with shape (N,).
        gt_boxes (np.ndarray): Ground-truth boxes in xyxy format with shape (M, 4).
        gt_cls (np.ndarray): Ground-truth class IDs with shape (M,).

    Returns:
        (dict[str, float]): Three scores in [0, 1], where lower values have higher review priority.
    """
    pred_cls, gt_cls = pred_cls.astype(int), gt_cls.astype(int)
    if not len(gt_boxes):
        scores = (_softmin(1.0 - pred_conf[pred_conf >= _LABEL_HIGH_CONF]), 1.0, 1.0)
        return dict(zip(_LABEL_PROPERTIES, np.clip(scores, 0, 1).tolist()))

    gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
    pred_centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
    all_boxes = np.concatenate((gt_boxes, pred_boxes))
    diagonal = max(float(np.linalg.norm(all_boxes[:, 2:].max(0) - all_boxes[:, :2].min(0))), 1e-6)
    distance = np.linalg.norm(gt_centers[:, None] - pred_centers[None], axis=2) / diagonal
    similarity = _LABEL_ALPHA * iou + (1 - _LABEL_ALPHA) * (1 - np.clip(distance, 0, 1))
    same_class = gt_cls[:, None] == pred_cls[None]

    keep = (pred_conf >= _LABEL_HIGH_CONF) & (iou.max(0) < _LABEL_MIN_IOU)
    same_similarity = np.where(same_class, similarity, -np.inf)
    floor = float(similarity[similarity > 0].min()) if np.any(similarity > 0) else 1.0
    overlooked = _softmin(np.where(same_class.any(0), same_similarity.max(0), floor * (1 - pred_conf))[keep])

    candidates = same_class & (pred_conf[None] >= _LABEL_LOW_CONF) & (iou >= _LABEL_BADLOC_IOU)
    best = np.where(candidates, similarity, -np.inf).max(1)
    badloc = _softmin(np.where(candidates.any(1), best, 1.0))

    candidates = ~same_class & (pred_conf[None] >= _LABEL_HIGH_CONF)
    best = np.where(candidates, similarity, -np.inf).max(1)
    swap = _softmin(np.where(candidates.any(1), 1 - best, 1.0))
    return dict(zip(_LABEL_PROPERTIES, np.clip((overlooked, badloc, swap), 0, 1).tolist()))


def _softmin(scores: np.ndarray) -> float:
    """Pool scores toward their minimum."""
    if not len(scores):
        return 1.0
    weights = np.exp((scores.min() - scores) / _LABEL_TEMPERATURE)
    return float(np.average(scores, weights=weights))


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
