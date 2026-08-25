# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Per-image property extraction and correlation analysis for object detection.

Two single-purpose pieces:
    - ``ImagePropertyExtractor(yolo_dataset)``: augment each ``dataset.labels`` entry in place with six scalar
      properties from image headers and annotations. No model, metrics, pixel decoding, or output.
    - ``CorrelationAnalysis(labels, metrics).run()``: join properties with per-image F1 from ``model.val()`` and
      return a compact report whose default exports are actionable issues and next steps.

References:
    - Lin et al., ECCV 2014 (COCO small-object area threshold).
    - Shao et al., CrowdHuman 2018 (per-image crowdedness via pairwise IoU).
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
_IMAGE_PROPERTIES = (
    "num_objects",
    "small_object_ratio",
    "object_scale_variance",
    "num_classes_present",
    "center_spread",
    "max_pairwise_iou",
)
_OBJECTLAB_PROPERTIES = ("overlooked_score", "badloc_score", "swap_score", "label_quality_score")
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
        insights (list[dict]): Flat actionable records with ``target``, ``issue``, ``score``, ``evidence``, and ``action``.
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
    """Join image properties with validation metrics and return actionable performance insights.

    Consumes the labels returned by :class:`ImagePropertyExtractor` together with a metrics object from ``model.val()``.
    Computes Pearson + Spearman correlations against per-image F1 and translates the three strongest meaningful
    F1-lowering relationships into plain-language issues and next actions. Raw evidence remains on the returned report.

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
    def _build_insights(_per_image: dict, correlations: dict) -> list[dict]:
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
