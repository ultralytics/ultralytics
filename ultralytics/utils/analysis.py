# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Per-image property extraction and correlation analysis for object detection.

Two single-purpose pieces:
    - ``ImagePropertyExtractor(yolo_dataset)``: augment each ``dataset.labels`` entry in place with six scalar
      properties from image headers and annotations. No model, metrics, pixel decoding, or output.
    - ``CorrelationAnalysis(labels, metrics).run()``: join properties with per-image F1 from ``model.val()`` and
      optional ObjectLab scores from ``model.val(score_labels=True)``, then return actionable next steps.

References:
    - Lin et al., ECCV 2014 (COCO small-object area threshold).
    - Shao et al., CrowdHuman 2018 (per-image crowdedness via pairwise IoU).
    - Tkachenko, Thyagarajan & Mueller, ObjectLab, ICML Workshop 2023 (label-quality scores).
    - Pearson, Proc. Royal Society 1895. Spearman, Am. J. Psychology 1904 (correlation coefficients).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ultralytics.utils import LOGGER, DataExportMixin, SimpleClass
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xywh2xyxy

COCO_AREA_SMALL = 32**2  # COCO small-object area threshold (px^2), Lin et al. 2014

# ObjectLab constants (Tkachenko, Thyagarajan & Mueller, ICML Workshop 2023, arXiv:2309.00832).
_OBJECTLAB_TEMPERATURE = 0.1
_OBJECTLAB_ALPHA = 0.9  # similarity = alpha*IoU + (1-alpha)*(1 - centroid_distance)
_OBJECTLAB_HIGH_PROB = 0.95
_OBJECTLAB_LOW_PROB = 0.5
_OBJECTLAB_MIN_IOU = 0.1  # ignore incidental overlap between different object instances
_OBJECTLAB_BADLOC_IOU = 0.4  # require enough overlap to compare box boundaries for the same instance
_OBJECTLAB_TINY = 1e-100  # log-clip floor in label_quality_score aggregation
_IMAGE_PROPERTIES = (
    "num_objects",
    "small_object_ratio",
    "object_scale_variance",
    "num_classes_present",
    "center_spread",
    "max_pairwise_iou",
)
_OBJECTLAB_PROPERTIES = ("overlooked_score", "badloc_score", "swap_score", "label_quality_score")
_OBJECTLAB_INSIGHTS = {
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
_ALL_PROPERTIES = _IMAGE_PROPERTIES + _OBJECTLAB_PROPERTIES
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
        correlations (dict[str, dict]): Per-property correlation summary against F1. Each entry has ``pearson_r``,
            ``pearson_p``, ``spearman_r``, ``spearman_p``, ``n``, ``effect_band``, ``direction``.
        insights (list[dict]): Flat actionable records with ``target``, ``issue``, ``score``, ``evidence``, and
            ``action``.
    """

    per_image: dict[str, dict] = field(default_factory=dict)
    correlations: dict[str, dict] = field(default_factory=dict)
    insights: list[dict] = field(default_factory=list)

    def summary(self, normalize: bool = False, decimals: int = 5) -> list[dict]:
        """Return actionable insight rows for ``DataExportMixin``.

        Args:
            normalize (bool, optional): Reserved for ``DataExportMixin`` API symmetry, unused here.
            decimals (int, optional): Decimal precision for float fields.

        Returns:
            (list[dict]): Actionable issues with their evidence and next step.
        """
        return [{**x, "score": round(x["score"], decimals)} for x in self.insights]

    def plot(self, save: bool = False, filename: str = "analysis.png") -> np.ndarray | None:
        """Return one compact plot of the strongest actionable F1 drivers.

        Args:
            save (bool, optional): Save the plot to ``filename``.
            filename (str, optional): Output filename when ``save`` is enabled.

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
        if save:
            fig.savefig(filename, dpi=120)
        plt.close(fig)
        return image


class ImagePropertyExtractor:
    """Augment a ``YOLODataset``'s labels in place with six per-image properties.

    Computes object count, small-object ratio, object-scale variation, class count, center spread, and maximum pairwise
    IoU from image headers and annotations. Each label in ``dataset.labels`` gains an ``im_properties`` dict, and the
    same label list is exposed as ``self.labels`` for chaining.

    Attributes:
        labels (list[dict]): The same list as ``dataset.labels``, with an ``im_properties`` dict added per image.

    Examples:
        >>> from ultralytics.cfg import get_cfg
        >>> from ultralytics.data.build import build_yolo_dataset
        >>> from ultralytics.data.utils import check_det_dataset
        >>> from ultralytics.utils.analysis import ImagePropertyExtractor
        >>> data = check_det_dataset("coco128.yaml")
        >>> cfg = get_cfg(overrides={"task": "detect", "imgsz": 320})
        >>> ds = build_yolo_dataset(cfg, data["val"], 1, data, mode="val", rect=False, stride=32)
        >>> labels = ImagePropertyExtractor(ds).labels
        >>> num_objects = labels[0]["im_properties"]["num_objects"]
    """

    def __init__(self, dataset: Any):
        """Extract per-image properties and mutate ``dataset.labels`` in place.

        Args:
            dataset (Any): A ``YOLODataset`` instance with non-empty ``labels`` and ``im_file`` per entry.
        """
        labels = getattr(dataset, "labels", None)
        if not labels:
            raise ValueError("ImagePropertyExtractor requires a YOLODataset with non-empty labels.")
        for label in labels:
            self._augment_label(label)
        self.labels = labels

    @staticmethod
    def _augment_label(lbl: dict) -> dict:
        """Compute the six properties for one label into its ``im_properties`` dict."""
        cls_arr = np.asarray(lbl.get("cls", np.zeros((0, 1)))).reshape(-1).astype(int)
        bboxes_n = np.asarray(lbl.get("bboxes", np.zeros((0, 4)))).reshape(-1, 4)
        with Image.open(lbl["im_file"]) as image:
            w, h = image.size
        n = int(bboxes_n.shape[0])
        areas_n = bboxes_n[:, 2] * bboxes_n[:, 3]
        lbl["im_properties"] = {
            "num_objects": n,
            "small_object_ratio": float(np.mean(areas_n * w * h < COCO_AREA_SMALL)) if n else np.nan,
            "object_scale_variance": (float(np.std(areas_n) / max(np.mean(areas_n), 1e-9)) if n else np.nan),
            "num_classes_present": int(np.unique(cls_arr).size),
            "center_spread": (float(np.sqrt(np.var(bboxes_n[:, 0]) + np.var(bboxes_n[:, 1]))) if n else np.nan),
            "max_pairwise_iou": (ImagePropertyExtractor._max_pairwise_iou(xywh2xyxy(bboxes_n)) if n >= 2 else np.nan),
        }
        return lbl

    @staticmethod
    def _max_pairwise_iou(xyxy: np.ndarray) -> float:
        """Calculate the maximum pairwise IoU among boxes in xyxy format."""
        t = torch.as_tensor(xyxy, dtype=torch.float32)
        iou = box_iou(t, t).triu_(diagonal=1)
        return float(iou.max())


class CorrelationAnalysis:
    """Join image properties and optional ObjectLab scores with F1, then return actionable insights.

    Consumes the labels returned by :class:`ImagePropertyExtractor` together with a metrics object from ``model.val()``.
    Computes correlation-based dataset actions and, with ``model.val(score_labels=True)``, per-image label-review
    actions for possible missing labels, incorrect boxes, and incorrect classes.

    Attributes:
        labels (list[dict]): Property-augmented label dicts from ``ImagePropertyExtractor.labels``.
        metrics: A metrics object exposing ``.box.image_metrics`` (e.g. ``DetMetrics``).

    Examples:
        >>> from ultralytics import YOLO
        >>> from ultralytics.utils.analysis import ImagePropertyExtractor, CorrelationAnalysis
        >>> m = YOLO("yolo11n.pt")
        >>> metrics = m.val(data="coco128.yaml")
        >>> labels = ImagePropertyExtractor(m.validator.dataloader.dataset).labels
        >>> report = CorrelationAnalysis(labels, metrics).run()
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
            im_file = lbl.get("im_file")
            if im_file is None:
                continue
            im_name = Path(im_file).name
            if im_name in per_image:
                dup_names.append(im_name)
                continue
            rec = dict(image_metrics.get(im_name, {}))
            rec["im_file"] = im_file
            props = lbl.get("im_properties", {})
            for k in _ALL_PROPERTIES:
                if k in props:
                    rec[k] = props[k]
            for k in _OBJECTLAB_PROPERTIES:
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
        """Compute Pearson + Spearman per property vs F1 with effect-size band and direction string.

        Uses SciPy for exact two-sided p-values when installed, otherwise falls back to NumPy correlation
        coefficients with p-values left None, matching the optional-SciPy convention in ops.py.
        """
        try:
            from scipy.stats import (  # exact p-values when SciPy is installed
                pearsonr,
                spearmanr,
            )
        except ImportError:
            pearsonr = spearmanr = None

        f1 = np.array([rec.get("f1", np.nan) for rec in per_image.values()], dtype=float)
        out: dict[str, dict] = {}
        for prop in _ALL_PROPERTIES:
            xs = np.array([rec.get(prop, np.nan) for rec in per_image.values()], dtype=float)
            m = np.isfinite(xs) & np.isfinite(f1)
            if m.sum() < 30 or np.std(xs[m]) == 0 or np.std(f1[m]) == 0:
                out[prop] = {
                    "pearson_r": None,
                    "pearson_p": None,
                    "spearman_r": None,
                    "spearman_p": None,
                    "n": int(m.sum()),
                    "effect_band": "n/a",
                    "direction": "n/a",
                }
                continue
            if pearsonr is not None:
                pr, sr = pearsonr(xs[m], f1[m]), spearmanr(xs[m], f1[m])
                pearson_r, pearson_p = float(pr.statistic), float(pr.pvalue)
                spearman_r, spearman_p = float(sr.correlation), float(sr.pvalue)
            else:
                pearson_r = float(np.corrcoef(xs[m], f1[m])[0, 1])
                spearman_r = float(np.corrcoef(_rankdata(xs[m]), _rankdata(f1[m]))[0, 1])
                pearson_p = spearman_p = None  # t-distribution p-values need SciPy; r drives band/direction
            out[prop] = {
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "n": int(m.sum()),
                "effect_band": _strength_band(spearman_r),
                "direction": _direction_phrase(prop, spearman_r),
            }
        return out

    @staticmethod
    def _build_insights(per_image: dict, correlations: dict) -> list[dict]:
        """Translate performance correlations and low ObjectLab scores into specific actions."""
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
        label_issues = []
        for prop, (issue, action) in _OBJECTLAB_INSIGHTS.items():
            candidates = [
                {"target": im_name, "issue": issue, "score": score, "evidence": prop, "action": action}
                for im_name, rec in per_image.items()
                if isinstance((score := rec.get(prop)), (int, float)) and np.isfinite(score) and score < 0.5
            ]
            label_issues.extend(sorted(candidates, key=lambda x: x["score"])[:3])
        return performance + sorted(label_issues, key=lambda x: x["score"])


def _strength_band(r: float | None) -> str:
    """Map a Spearman r magnitude to a strength descriptor (negligible/weak/moderate/strong)."""
    if r is None:
        return "n/a"
    a = abs(r)
    if a >= 0.5:
        return "strong"
    if a >= 0.3:
        return "moderate"
    if a >= 0.1:
        return "weak"
    return "negligible"


def _direction_phrase(prop: str, r: float | None) -> str:
    """Render a correlation direction using the raw property name (`higher num_objects -> lower F1`)."""
    if r is None or abs(r) < 0.1:
        return "no clear effect"
    return f"higher {prop} -> lower F1" if r < 0 else f"higher {prop} -> higher F1"


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


def _objectlab_score_dict(overlooked: float, badloc: float, swap: float) -> dict[str, float]:
    """Clip the 3 ObjectLab subtype scores to [0, 1] and append their weighted geometric mean as label_quality_score."""
    w = 1.0 / 3.0
    lq = float(
        np.exp(
            w * np.log(_OBJECTLAB_TINY + overlooked)
            + w * np.log(_OBJECTLAB_TINY + badloc)
            + w * np.log(_OBJECTLAB_TINY + swap)
        )
    )
    return {
        "overlooked_score": float(np.clip(overlooked, 0.0, 1.0)),
        "badloc_score": float(np.clip(badloc, 0.0, 1.0)),
        "swap_score": float(np.clip(swap, 0.0, 1.0)),
        "label_quality_score": float(np.clip(lq, 0.0, 1.0)),
    }


def compute_objectlab_scores(
    iou: np.ndarray | None,
    pred_bb: np.ndarray,
    pred_cls: np.ndarray,
    pred_conf: np.ndarray,
    gt_bb: np.ndarray,
    gt_cls: np.ndarray,
) -> dict[str, float]:
    """Compute the 4 ObjectLab subtype scores (Tkachenko et al., ICML Workshop 2023) from per-image predictions and GTs.

    Args:
        iou (np.ndarray | None): Pairwise (n_gt, n_pred) IoU matrix, or None when either side is empty (unused then).
        pred_bb (np.ndarray): Prediction xyxy boxes, shape (n_pred, 4).
        pred_cls (np.ndarray): Prediction class IDs, shape (n_pred,).
        pred_conf (np.ndarray): Prediction confidences, shape (n_pred,).
        gt_bb (np.ndarray): Ground-truth xyxy boxes, shape (n_gt, 4).
        gt_cls (np.ndarray): Ground-truth class IDs, shape (n_gt,).

    Returns:
        (dict): ``overlooked_score``, ``badloc_score``, ``swap_score``, ``label_quality_score`` in ``[0, 1]``, quality
            convention (low = likely label issue, high = clean label).
    """
    pred_cls = pred_cls.astype(int)
    gt_cls = gt_cls.astype(int)
    n_gt, n_pred = gt_bb.shape[0], pred_bb.shape[0]
    if n_pred == 0:
        return {k: float("nan") for k in _OBJECTLAB_PROPERTIES}
    if n_gt == 0:
        keep = pred_conf >= _OBJECTLAB_HIGH_PROB
        overlooked = _softmin1d(1.0 - pred_conf[keep], _OBJECTLAB_TEMPERATURE)
        return _objectlab_score_dict(overlooked, 1.0, 1.0)

    gt_cx, gt_cy = (gt_bb[:, 0] + gt_bb[:, 2]) / 2, (gt_bb[:, 1] + gt_bb[:, 3]) / 2
    pr_cx, pr_cy = (pred_bb[:, 0] + pred_bb[:, 2]) / 2, (pred_bb[:, 1] + pred_bb[:, 3]) / 2
    all_xy = np.concatenate([gt_bb, pred_bb], axis=0)
    diag = max(
        float(np.hypot(all_xy[:, 2].max() - all_xy[:, 0].min(), all_xy[:, 3].max() - all_xy[:, 1].min())),
        1e-6,
    )
    cd = np.hypot(gt_cx[:, None] - pr_cx[None, :], gt_cy[:, None] - pr_cy[None, :]) / diag
    sim = _OBJECTLAB_ALPHA * iou + (1 - _OBJECTLAB_ALPHA) * (1.0 - np.clip(cd, 0, 1))
    same_class = gt_cls[:, None] == pred_cls[None, :]

    keep_pred = (pred_conf >= _OBJECTLAB_HIGH_PROB) & (iou.max(axis=0) < _OBJECTLAB_MIN_IOU)
    sim_same = np.where(same_class, sim, -np.inf)
    best_same_per_pred = sim_same.max(axis=0)
    min_similarity = float(sim[sim > 0].min()) if np.any(sim > 0) else 1.0
    per_pred = np.where(same_class.any(axis=0), best_same_per_pred, min_similarity * (1.0 - pred_conf))
    overlooked = _softmin1d(per_pred[keep_pred], _OBJECTLAB_TEMPERATURE)

    cand_low = same_class & (pred_conf[None, :] >= _OBJECTLAB_LOW_PROB) & (iou >= _OBJECTLAB_BADLOC_IOU)
    rowmax_low = np.where(cand_low, sim, -np.inf).max(axis=1)
    badloc_per_box = np.where(cand_low.any(axis=1), rowmax_low, 1.0)
    badloc = _softmin1d(badloc_per_box, _OBJECTLAB_TEMPERATURE)

    cand_high = ~same_class & (pred_conf[None, :] >= _OBJECTLAB_HIGH_PROB)
    rowmax_high = np.where(cand_high, sim, -np.inf).max(axis=1)
    swap_per_box = np.where(cand_high.any(axis=1), np.maximum(_OBJECTLAB_TINY, 1.0 - rowmax_high), 1.0)
    swap = _softmin1d(swap_per_box, _OBJECTLAB_TEMPERATURE)
    return _objectlab_score_dict(overlooked, badloc, swap)


def _softmin1d(scores: np.ndarray, T: float) -> float:
    """Softmin-pool a 1D score array as the softmax-weighted mean.

    Args:
        scores (np.ndarray): 1D array of per-box scores in ``[0, 1]``.
        T (float): Temperature parameter, must be > 0.

    Returns:
        (float): Pooled per-image score in ``[0, 1]``.
    """
    if scores.size == 0:
        return 1.0
    a = -scores / T
    a -= a.max()
    w = np.exp(a)
    w /= w.sum()
    return float(np.dot(w, scores))


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
