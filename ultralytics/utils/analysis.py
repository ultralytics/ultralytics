# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Per-image property extraction and correlation analysis for object detection.

Two single-purpose pieces:
    - ``ImagePropertyExtractor(yolo_dataset)``: augment each ``dataset.labels`` entry in place with six scalar
      properties from image headers and annotations. No model, metrics, pixel decoding, or output.
    - ``CorrelationAnalysis(labels, metrics).run()``: join property-augmented labels with the per-image F1
      from ``model.val()``, compute Pearson + Spearman correlations, rank worst-performing images, and
      write CSV/JSON/plots/``summary.md``.

References:
    - Lin et al., ECCV 2014 (COCO small-object area threshold).
    - Shao et al., CrowdHuman 2018 (per-image crowdedness via pairwise IoU).
    - Pearson, Proc. Royal Society 1895. Spearman, Am. J. Psychology 1904 (correlation coefficients).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics.utils import LOGGER, RUNS_DIR, DataExportMixin, SimpleClass
from ultralytics.utils.files import increment_path
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.patches import imread

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
_METRIC_FIELDS = ("precision", "recall", "f1", "tp", "fp", "fn")


@dataclass
class AnalysisReport(SimpleClass, DataExportMixin):
    """Container for per-image properties, correlations against F1, and worst-image ranking.

    Attributes:
        per_image (dict[str, dict]): Per-image record keyed by image basename. Each record holds metric fields
            (precision/recall/f1/tp/fp/fn), all computed image properties, and an ``anomaly_score`` for ranking.
        correlations (dict[str, dict]): Per-property correlation summary against F1. Each entry has ``pearson_r``,
            ``pearson_p``, ``spearman_r``, ``spearman_p``, ``n``, ``effect_band``, ``direction``.
        save_dir (Path): Output directory for CSV / JSON / plots / summary.md.
        names (dict[int, str] | None): Optional class id to name mapping used to label boxes on the worst-image strip.
    """

    per_image: dict[str, dict] = field(default_factory=dict)
    correlations: dict[str, dict] = field(default_factory=dict)
    save_dir: Path = field(default_factory=Path)
    names: dict[int, str] | None = None

    def summary(self, normalize: bool = False, decimals: int = 5) -> list[dict]:
        """Return per-image summary rows for ``DataExportMixin`` (powers ``to_csv``/``to_json``/``to_df``).

        Args:
            normalize (bool, optional): Reserved for ``DataExportMixin`` API symmetry, unused here.
            decimals (int, optional): Decimal precision for float fields.

        Returns:
            (list[dict]): One dict per image, sorted ascending by F1.
        """
        rows = []
        for im_name, rec in sorted(self.per_image.items(), key=lambda kv: _worst_record_score(kv[1])):
            out = {"im_name": im_name, "im_file": rec.get("im_file", "")}
            for k in _METRIC_FIELDS:
                v = rec.get(k)
                out[k] = round(float(v), decimals) if isinstance(v, (int, float)) else v
            for k in (*_ALL_PROPERTIES, "anomaly_score"):
                v = rec.get(k)
                out[k] = round(float(v), decimals) if isinstance(v, (int, float)) else v
            rows.append(out)
        return rows

    def plot(self, save_dir: Path | str | None = None, n_strip: int = 20) -> None:
        """Render scatter grid (F1 vs property), property correlation heatmap, and worst-image strip.

        Args:
            save_dir (Path | str, optional): Directory to write PNGs into.
            n_strip (int, optional): Number of thumbnails on the worst-image strip plot.
        """
        import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

        out_dir = Path(save_dir or self.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        scored = [v for v in self.per_image.values() if any(p in v for p in _ALL_PROPERTIES)]
        if not scored:
            LOGGER.warning("AnalysisReport.plot: no per-image properties available, nothing to render")
            return
        plotted_props = [p for p in _ALL_PROPERTIES if any(np.isfinite(v.get(p, np.nan)) for v in scored)]

        ncols = min(4, int(np.ceil(np.sqrt(len(plotted_props)))))
        nrows = (len(plotted_props) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.8))
        axes = np.atleast_2d(axes).ravel()
        ys = np.array([v.get("f1", np.nan) for v in scored], dtype=float)
        for ax, prop in zip(axes, plotted_props):
            xs = np.array([v.get(prop, np.nan) for v in scored], dtype=float)
            m = np.isfinite(xs) & np.isfinite(ys)
            ax.scatter(xs[m], ys[m], s=4, alpha=0.5, c="tab:blue")
            if m.sum() > 1 and np.std(xs[m]) > 0:
                coef = np.polyfit(xs[m], ys[m], 1)
                xline = np.linspace(xs[m].min(), xs[m].max(), 50)
                ax.plot(xline, np.polyval(coef, xline), color="tab:red", lw=1.0)
            r = self.correlations.get(prop, {}).get("pearson_r")
            title = f"{prop}\nr={r:.2f}" if isinstance(r, (int, float)) else f"{prop}\nr=n/a"
            ax.set_title(title, fontsize=8)
            ax.set_xlabel(prop, fontsize=7)
            ax.set_ylabel("f1", fontsize=7)
            ax.tick_params(axis="both", labelsize=6)
        for ax in axes[len(plotted_props) :]:
            ax.set_visible(False)
        fig.suptitle(
            "Per-image F1 vs each property (one dot = one image, red line = linear fit, r = Pearson)",
            fontsize=10,
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        fig.savefig(out_dir / "correlation_scatter.png", dpi=120)
        plt.close(fig)

        prop_columns = [*plotted_props, "f1"]
        mat = np.full((len(prop_columns), len(prop_columns)), np.nan)
        cols = {p: np.array([v.get(p, np.nan) for v in scored], dtype=float) for p in prop_columns}
        for i, p1 in enumerate(prop_columns):
            for j in range(i + 1, len(prop_columns)):  # upper triangle only, mirror to lower (Pearson r is symmetric)
                a, b = cols[p1], cols[prop_columns[j]]
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() >= 30 and np.std(a[m]) > 0 and np.std(b[m]) > 0:
                    r = float(np.corrcoef(a[m], b[m])[0, 1])
                    mat[i, j] = r
                    mat[j, i] = r
        fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(prop_columns)), max(5, 0.4 * len(prop_columns))))
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad(color="white")
        im = ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(len(prop_columns)))
        ax.set_xticklabels(prop_columns, rotation=70, fontsize=7, ha="right")
        ax.set_yticks(range(len(prop_columns)))
        ax.set_yticklabels(prop_columns, fontsize=7)
        ax.set_title(
            "Property correlation matrix (Pearson r)\n"
            "red = positively correlated, blue = negatively correlated, white = self/undefined",
            fontsize=10,
        )
        fig.colorbar(im, ax=ax, fraction=0.04, label="Pearson r")
        fig.tight_layout()
        fig.savefig(out_dir / "correlation_heatmap.png", dpi=120)
        plt.close(fig)

        candidates = (v for v in scored if v.get("im_file") and isinstance(v.get("f1"), (int, float)))
        worst = sorted(candidates, key=_worst_record_score)[:n_strip]
        if worst:
            from matplotlib.patches import (
                Rectangle,  # scope for faster 'import ultralytics'
            )

            ncols = 5
            nrows = (len(worst) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 3.0))
            axes = np.atleast_1d(axes).ravel()
            for ax, rec in zip(axes, worst):
                img = imread(rec["im_file"])
                if img is None:
                    continue
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax.set_xticks([])
                ax.set_yticks([])
                for box in rec.get("gt_bboxes", []):
                    x1, y1, x2, y2 = box
                    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, lw=1.2, ec="lime", fc="none"))
                for box, conf, cid in zip(
                    rec.get("pred_bboxes", []), rec.get("pred_conf", []), rec.get("pred_cls", [])
                ):
                    x1, y1, x2, y2 = box
                    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, lw=1.2, ec="red", fc="none", ls="--"))
                    label = self.names.get(int(cid), str(int(cid))) if self.names else str(int(cid))
                    ax.text(
                        x1,
                        max(y1 - 4, 10),
                        f"{label} {conf:.2f}",
                        fontsize=7,
                        color="red",
                        bbox={"facecolor": "white", "alpha": 0.7, "pad": 0.6, "edgecolor": "none"},
                    )
                ax.set_title(f"{Path(rec['im_file']).stem}\nF1={rec['f1']:.2f}", fontsize=8)
            for ax in axes[len(worst) :]:
                ax.set_visible(False)
            fig.suptitle(
                f"{len(worst)} worst-performing images (lowest F1). Green = ground truth, red dashed = model "
                f"predictions.",
                fontsize=11,
                y=0.995,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.985))
            fig.savefig(out_dir / "worst_images_strip.png", dpi=140)
            plt.close(fig)

    def write_summary_md(self, save_dir: Path | str | None = None, n_strip: int = 20) -> None:
        """Write a plain-English ``summary.md`` with a headline finding, top correlations, worst images, and plots.

        Args:
            save_dir (Path | str, optional): Directory to write into.
            n_strip (int, optional): Number of worst-image rows shown in the summary table.
        """
        out_dir = Path(save_dir or self.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        top_corr = sorted(
            self.correlations.items(),
            key=lambda kv: abs(kv[1].get("spearman_r") or 0),
            reverse=True,
        )[:3]
        worst = sorted(self.per_image.items(), key=lambda kv: _worst_record_score(kv[1]))[:n_strip]

        lines = [
            "# Image Property Analysis Report",
            "",
            f"**Dataset:** {len(self.per_image)} images.",
            "",
            "## Top 3 things that hurt F1",
            "",
        ]
        strong_enough = [(p, c) for p, c in top_corr if abs(c.get("spearman_r") or 0) >= 0.1]
        if strong_enough:
            for prop, c in strong_enough:
                r = c.get("spearman_r")
                lines.append(f"- **`{prop}`** ({_strength_band(r)}, {_direction_phrase(prop, r)})")
        else:
            lines.append("- No image property strongly predicts F1 in this dataset (all correlations negligible).")

        lines += [
            "",
            f"## Worst {len(worst)} images (lowest F1)",
            "",
            "**Why it stands out** lists the properties where this image is most extreme in the F1-lowering direction.",
            "",
            "| Image | F1 | Why it stands out |",
            "|---|---|---|",
        ]
        for im_name, rec in worst:
            top3 = ", ".join(f"`{p}`" for p in rec.get("top_3_problematic", []))
            f1 = rec.get("f1")
            f1_s = f"{f1:.2f}" if isinstance(f1, (int, float)) else "-"
            lines.append(f"| `{im_name}` | {f1_s} | {top3} |")

        lines += [
            "",
            "## Plots",
            "",
            "- **F1 vs each property** (`correlation_scatter.png`): one dot per image, red line is a linear fit.",
            "- **Property correlation heatmap** (`correlation_heatmap.png`): how each pair of properties moves together.",
            f"- **Worst-image strip** (`worst_images_strip.png`): the {len(worst)} worst images with **green** ground-truth boxes and **red dashed** model predictions.",
        ]

        notes: list[str] = []
        f1_vals = np.array([r.get("f1", np.nan) for r in self.per_image.values()], dtype=float)
        median_f1 = float(np.nanmedian(f1_vals)) if np.isfinite(f1_vals).any() else float("nan")
        if median_f1 < 0.1:
            notes.append(
                f"Per-image F1 median is **{median_f1:.3f}**, which is low. The validator default `conf=0.001` "
                f"lets ~300 false positives through per image (max_det), which dominates the F1 denominator. "
                f"For meaningful per-image F1 re-run with `model.val(..., conf=0.25)`."
            )
        if any(np.isfinite(r.get("label_quality_score", np.nan)) for r in self.per_image.values()):
            notes.append(
                "`label_quality_score` is the geometric mean of `overlooked_score`, `badloc_score`, and "
                "`swap_score`. When two of those subtypes saturate at 1.0 on a clean dataset (common on COCO), "
                "`label_quality_score` collapses into a monotonic transform of the third, so a near-perfect "
                "correlation in the heatmap between it and that subtype is expected."
            )
        notes.extend(
            [
                (
                    "**Strength** is based on the Spearman rank correlation magnitude: "
                    "strong (>=0.5), moderate (>=0.3), weak (>=0.1), otherwise negligible. Full numeric correlations are "
                    "available in `correlations.json` and `per_image_analysis.csv`."
                ),
                (
                    "Property definitions, how to interpret each score, and suggestions for improving your model or "
                    "dataset from these results are in the [analysis guide](https://docs.ultralytics.com/guides/analysis/)."
                ),
            ]
        )
        lines += ["", "## How to read this report", ""] + [f"- {n}" for n in notes]
        (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    """Join property-augmented labels with per-image metrics, correlate, rank, and write reports.

    Consumes the labels returned by :class:`ImagePropertyExtractor` together with a metrics object from ``model.val()``.
    Computes Pearson + Spearman correlations of every property against per-image F1, ranks the worst-performing images
    by F1 with anomaly score as tiebreak, and writes ``per_image_analysis.csv``, ``correlations.json``,
    ``worst_images.json``, ``summary.md``, and three plots into ``save_dir``.

    Attributes:
        labels (list[dict]): Property-augmented label dicts from ``ImagePropertyExtractor.labels``.
        metrics: A metrics object exposing ``.box.image_metrics`` (e.g. ``DetMetrics``).
        names (dict[int, str] | None): Optional class id to name mapping. Auto-resolved from ``metrics`` when ``None``.

    Examples:
        >>> from ultralytics import YOLO
        >>> from ultralytics.utils.analysis import ImagePropertyExtractor, CorrelationAnalysis
        >>> m = YOLO("yolo11n.pt")
        >>> metrics = m.val(data="coco128.yaml")
        >>> labels = ImagePropertyExtractor(m.validator.dataloader.dataset).labels
        >>> report = CorrelationAnalysis(labels, metrics).run()
    """

    def __init__(self, labels: list[dict], metrics: Any, names: dict[int, str] | None = None):
        """Bind labels and metrics; no computation runs until :meth:`run`.

        Args:
            labels (list[dict]): Property-augmented labels (output of ``ImagePropertyExtractor``).
            metrics (Any): Metrics object from ``model.val()`` exposing ``.box.image_metrics``.
            names (dict[int, str], optional): Class id to name mapping. Auto-resolved from metrics if omitted.
        """
        if not labels:
            raise ValueError("CorrelationAnalysis requires non-empty labels from ImagePropertyExtractor.")
        if metrics is None:
            raise ValueError("CorrelationAnalysis requires a metrics object from model.val().")
        self.labels = labels
        self.metrics = metrics
        self.names = names

    def run(self, save_dir: Path | str | None = None, n_worst: int = 100, n_strip: int = 20) -> AnalysisReport:
        """Compute correlations, rank worst images, write outputs, and return the populated report.

        Args:
            save_dir (Path | str, optional): Output directory. Defaults to an auto-incremented ``runs/analyze``.
            n_worst (int, optional): Number of worst-performing images saved in ``worst_images.json``.
            n_strip (int, optional): Number of thumbnails on the worst-image strip plot.

        Returns:
            (AnalysisReport): A fully populated report. CSV / JSON / plots / summary.md are written to ``save_dir``.
        """
        save_dir = Path(save_dir) if save_dir else increment_path(RUNS_DIR / "analyze", exist_ok=False)
        save_dir.mkdir(parents=True, exist_ok=True)
        per_image = self._join(self.labels, self.metrics)

        f1s = np.array([rec.get("f1", np.nan) for rec in per_image.values()], dtype=float)
        median_f1 = float(np.nanmedian(f1s)) if np.isfinite(f1s).any() else float("nan")
        if median_f1 < 0.1:
            LOGGER.warning(
                f"CorrelationAnalysis: per-image F1 median is {median_f1:.3f}. "
                f"Re-run model.val(..., conf=0.25) for meaningful per-image F1 (see summary.md)."
            )

        correlations = self._compute_correlations(per_image)
        self._rank_and_score(per_image, correlations)

        names = self.names
        if names is None:
            names = getattr(self.metrics, "names", None)
            if names and not isinstance(names, dict):
                names = dict(enumerate(names))

        report = AnalysisReport(
            per_image=per_image,
            correlations=correlations,
            save_dir=save_dir,
            names=names,
        )
        (save_dir / "per_image_analysis.csv").write_text(report.to_csv(), encoding="utf-8")
        (save_dir / "correlations.json").write_text(
            json.dumps(correlations, indent=2, default=_json_default), encoding="utf-8"
        )
        worst = self._top_worst_records(per_image, top_n=n_worst)
        (save_dir / "worst_images.json").write_text(
            json.dumps(worst, indent=2, default=_json_default), encoding="utf-8"
        )
        report.plot(n_strip=n_strip)
        report.write_summary_md(n_strip=n_strip)
        return report

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
    def _rank_and_score(per_image: dict, correlations: dict) -> None:
        """Compute per-image ``anomaly_score`` (sign-aligned z-mean) + top-3 problematic properties."""
        prop_arrays = {}
        for prop in _ALL_PROPERTIES:
            xs = np.array([rec.get(prop, np.nan) for rec in per_image.values()], dtype=float)
            m = np.isfinite(xs)
            if m.sum() < 2 or np.std(xs[m]) == 0:
                continue
            pr = correlations.get(prop, {}).get("pearson_r")
            if pr is None:  # insufficient evidence to pick a direction, skip rather than default to +1
                continue
            mu, sd = float(xs[m].mean()), float(xs[m].std())
            sign = -1.0 if pr > 0 else 1.0  # bad direction = side that lowers F1
            prop_arrays[prop] = (mu, sd, sign)

        for rec in per_image.values():
            zs, names = [], []
            for prop, (mu, sd, sign) in prop_arrays.items():
                v = rec.get(prop)
                if not isinstance(v, (int, float)) or np.isnan(v):
                    continue
                z = sign * (v - mu) / sd
                zs.append(z)
                names.append(prop)
            rec["anomaly_score"] = float(np.mean(zs)) if zs else 0.0
            if zs:
                # Rank by signed z (sign-aligned so positive = F1-lowering) and keep only bad-direction props, so a
                # value that is extreme in the F1-raising direction is never listed as problematic.
                ranked = sorted(zip(zs, names), key=lambda zn: zn[0], reverse=True)
                rec["top_3_problematic"] = [name for z, name in ranked if z > 0][:3]

    @staticmethod
    def _top_worst_records(per_image: dict, top_n: int = 100) -> list[dict]:
        """Return ranked worst-image dicts ready for ``worst_images.json``."""
        ranked = sorted(per_image.items(), key=lambda kv: _worst_record_score(kv[1]))[:top_n]
        return [
            {
                "im_name": name,
                "im_file": rec.get("im_file"),
                "f1": rec.get("f1"),
                "anomaly_score": rec.get("anomaly_score"),
                "top_3_problematic": rec.get("top_3_problematic", []),
            }
            for name, rec in ranked
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


def _worst_record_score(rec: dict) -> tuple[float, float]:
    """Sortable tuple where lower is worse: F1 ascending with anomaly score descending as the tiebreak.

    Empty-GT images (``num_objects == 0``) have undefined per-image F1 and are pushed past every real image so they
    never pollute the worst-image table.
    """
    f1 = (
        float("inf")
        if not rec.get("num_objects", 0) or (v := rec.get("f1")) is None or not np.isfinite(v)
        else float(v)
    )
    return (f1, -float(rec.get("anomaly_score", 0.0)))


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


def _json_default(obj: Any) -> Any:
    """Fallback JSON encoder for numpy scalars, arrays, and Path objects."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")
