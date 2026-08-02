# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, RANK, nms, ops
from ultralytics.utils.geometry3d import backproject_points_torch
from ultralytics.utils.kitti_eval import (
    KITTI_CLASSES,
    KittiAnnotation,
    build_kitti_predictions,
    d3_box_overlap,
    evaluate_kitti_metric,
    evaluate_kitti_r40,
    format_kitti_r40,
    paired_d3_box_overlap,
    parse_kitti_label,
    plot_kitti_r40,
    write_kitti_predictions,
)
from ultralytics.utils.metrics import DetMetrics, Metric, ap_per_class, box_iou
from ultralytics.utils.plotting import plot_images

from .utils import set_detect3d_quality_power


_D3_ERROR_NAMES = ("dist", "xc", "yc", "w3d", "h3d", "l3d", "ry_deg")
_D3_DISTANCE_RANGES = (
    ("0_20m", 0.0, 20.0),
    ("20_40m", 20.0, 40.0),
    ("40m_plus", 40.0, math.inf),
)


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Return zero-based average ranks, assigning equal values the same rank."""
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values, dtype=torch.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Return a finite Pearson correlation, or zero when either input is constant."""
    x, y = x.double(), y.double()
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    return float((x * y).sum() / denominator) if denominator > torch.finfo(torch.float64).eps else 0.0


def _camera_boxes_to_kitti(boxes: torch.Tensor) -> KittiAnnotation:
    """Convert ``(z, x, y_bottom, w, h, l, ry)`` camera boxes to a compact KITTI annotation."""
    array = boxes.detach().double().cpu().numpy()
    n = len(array)
    return build_kitti_predictions(
        ["Diagnostic"] * n,
        np.zeros((n, 4), dtype=np.float64),
        np.zeros(n, dtype=np.float64),
        np.zeros(n, dtype=np.float64),
        array[:, [4, 3, 5]],  # h, w, l
        array[:, [1, 2, 0]],  # x, y_bottom, z
        array[:, 6],
    )


def _paired_camera_iou3d(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Compute exact aligned rotated 3D IoU in KITTI camera coordinates."""
    overlaps = paired_d3_box_overlap(_camera_boxes_to_kitti(gt), _camera_boxes_to_kitti(pred))
    return torch.from_numpy(overlaps).to(device=gt.device, dtype=torch.float32)


def _camera_iou3d(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Return exact pairwise rotated 3D IoU, assigning zero overlap to invalid camera boxes."""
    overlaps = torch.zeros((len(gt), len(pred)), dtype=torch.float32)
    valid_gt = torch.isfinite(gt).all(1) & (gt[:, 0] > 0) & (gt[:, 3:6] > 0).all(1)
    valid_pred = torch.isfinite(pred).all(1) & (pred[:, 0] > 0) & (pred[:, 3:6] > 0).all(1)
    gt_indices = valid_gt.nonzero(as_tuple=False).squeeze(1)
    pred_indices = valid_pred.nonzero(as_tuple=False).squeeze(1)
    if not len(gt_indices) or not len(pred_indices):
        return overlaps

    valid_overlaps = d3_box_overlap(
        _camera_boxes_to_kitti(gt[gt_indices]),
        _camera_boxes_to_kitti(pred[pred_indices]),
    )
    gt_indices = gt_indices.detach().cpu()
    pred_indices = pred_indices.detach().cpu()
    overlaps[gt_indices[:, None], pred_indices[None, :]] = torch.from_numpy(valid_overlaps).to(dtype=torch.float32)
    return overlaps


def _summarize_d3_diagnostics(matched: torch.Tensor, gt_depths: torch.Tensor) -> dict[str, float]:
    """Summarize matched-pair errors using all valid 3D GT as recall denominators.

    ``matched`` columns are GT depth, seven absolute errors, exact 3D IoU and raw sigmoid(q3d). Matching itself is
    performed earlier using same-class 2D IoU >= 0.5, one prediction per GT and one GT per prediction.
    """
    gt_depths = gt_depths.double()
    gt_depths = gt_depths[torch.isfinite(gt_depths) & (gt_depths > 0)]
    matched = matched.double().reshape(-1, 10)
    valid_matched = torch.isfinite(matched[:, :9]).all(1) & (matched[:, 0] > 0)
    matched = matched[valid_matched]
    results: dict[str, float] = {}

    def add_scope(prefix: str, match_mask: torch.Tensor, gt_mask: torch.Tensor, percentiles: bool) -> None:
        rows = matched[match_mask]
        gt_count = int(gt_mask.sum())
        results[f"{prefix}/gt_count"] = float(gt_count)
        results[f"{prefix}/matched_count"] = float(len(rows))
        results[f"{prefix}/match_recall"] = len(rows) / gt_count if gt_count else 0.0
        for threshold in (0.5, 0.7):
            passed = int((rows[:, 8] >= threshold).sum()) if len(rows) else 0
            results[f"{prefix}/iou3d_recall_{threshold:.1f}"] = passed / gt_count if gt_count else 0.0
        if not len(rows):
            return
        errors = rows[:, 1:8]
        for index, name in enumerate(_D3_ERROR_NAMES):
            values = errors[:, index]
            results[f"{prefix}/{name}_MAE"] = float(values.mean())
            if percentiles:
                results[f"{prefix}/{name}_P50"] = float(torch.quantile(values, 0.5))
                results[f"{prefix}/{name}_P90"] = float(torch.quantile(values, 0.9))
        results[f"{prefix}/iou3d_mean"] = float(rows[:, 8].mean())
        quality_valid = torch.isfinite(rows[:, 9])
        results[f"{prefix}/q3d_count"] = float(quality_valid.sum())
        if quality_valid.sum() >= 2:
            quality, iou3d = rows[quality_valid, 9], rows[quality_valid, 8]
            results[f"{prefix}/q3d_iou3d_pearson"] = _correlation(quality, iou3d)
            results[f"{prefix}/q3d_iou3d_spearman"] = _correlation(_average_ranks(quality), _average_ranks(iou3d))

    all_matches = torch.ones(len(matched), dtype=torch.bool)
    all_gt = torch.ones(len(gt_depths), dtype=torch.bool)
    add_scope("3d/diagnostic", all_matches, all_gt, percentiles=True)
    for label, lower, upper in _D3_DISTANCE_RANGES:
        match_mask = (matched[:, 0] >= lower) & (matched[:, 0] < upper)
        gt_mask = (gt_depths >= lower) & (gt_depths < upper)
        add_scope(f"3d/range_{label}", match_mask, gt_mask, percentiles=False)
    return results


class Detection3DMetrics(DetMetrics):
    """Ultralytics-style box and exact rotated camera-box 3D detection metrics."""

    def __init__(self, names: dict[int, str] | None = None) -> None:
        """Initialize independent box/3D metrics and auxiliary diagnostic results."""
        super().__init__(names)
        self.d3 = Metric()
        self.stats.update({"tp_d3": [], "target_cls_d3": [], "target_img_d3": []})
        self.nt_per_class_d3 = None
        self.nt_per_image_d3 = None
        self.d3_results: dict[str, float] = {}

    def update_stats(self, stat: dict[str, Any]) -> None:
        """Accumulate aligned box and 3D true-positive statistics for one image."""
        super().update_stats(stat)
        self.d3.update_image_metrics(stat["tp_d3"], stat["target_cls_d3"], stat["pred_cls"], stat["im_name"])

    def clear_image_metrics(self) -> None:
        """Clear box and 3D per-image metric caches."""
        super().clear_image_metrics()
        self.d3.clear_image_metrics()

    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
        """Process box metrics followed by generic 3D P/R and AP over IoU 0.50:0.95."""
        stats = super().process(save_dir, plot, on_plot=on_plot)
        if not stats:
            return stats
        if len(stats["target_cls_d3"]):
            results_d3 = ap_per_class(
                stats["tp_d3"],
                stats["conf"],
                stats["pred_cls"],
                stats["target_cls_d3"],
                plot=plot,
                save_dir=save_dir,
                names=self.names,
                on_plot=on_plot,
                prefix="3D",
            )[2:]
        else:
            empty = np.zeros(0, dtype=np.float64)
            empty_curves = np.zeros((0, 1000), dtype=np.float64)
            results_d3 = (
                empty,
                empty,
                empty,
                np.zeros((0, 10), dtype=np.float64),
                np.zeros(0, dtype=int),
                empty_curves,
                empty_curves.copy(),
                empty_curves.copy(),
                np.linspace(0.0, 1.0, 1000),
                np.zeros((1, 1000), dtype=np.float64),
            )
        self.d3.nc = len(self.names)
        self.d3.update(results_d3)
        self.nt_per_class_d3 = np.bincount(stats["target_cls_d3"].astype(int), minlength=len(self.names))
        self.nt_per_image_d3 = np.bincount(stats["target_img_d3"].astype(int), minlength=len(self.names))
        return stats

    @property
    def keys(self) -> list[str]:
        """Return box metrics followed by generic exact-3D metrics."""
        return [
            *DetMetrics.keys.fget(self),
            "metrics/precision(3D)",
            "metrics/recall(3D)",
            "metrics/mAP50(3D)",
            "metrics/mAP50-95(3D)",
        ]

    def mean_results(self) -> list[float]:
        """Return mean box and 3D P/R/mAP results."""
        return DetMetrics.mean_results(self) + self.d3.mean_results()

    def class_result(self, i: int) -> list[float]:
        """Return box and 3D results for the box metric's class at result index ``i``."""
        box_result = list(DetMetrics.class_result(self, i))
        class_id = int(self.box.ap_class_index[i])
        d3_indices = np.flatnonzero(np.asarray(self.d3.ap_class_index, dtype=int) == class_id)
        d3_result = list(self.d3.class_result(int(d3_indices[0]))) if len(d3_indices) else [0.0] * 4
        return box_result + d3_result

    @property
    def maps(self) -> np.ndarray:
        """Return task-primary generic 3D mAP values per class."""
        return self.d3.maps

    @property
    def fitness(self) -> float:
        """Use generic 3D mAP50 exclusively for checkpoint selection."""
        return float(np.nan_to_num(self.d3.map50))

    @property
    def curves(self) -> list[str]:
        """Return box and 3D metric curve names."""
        return [
            *DetMetrics.curves.fget(self),
            "Precision-Recall(3D)",
            "F1-Confidence(3D)",
            "Precision-Confidence(3D)",
            "Recall-Confidence(3D)",
        ]

    @property
    def curves_results(self) -> list[list]:
        """Return box and 3D curve data."""
        return DetMetrics.curves_results.fget(self) + self.d3.curves_results

    @property
    def results_dict(self) -> dict[str, float]:
        """Return standard box metrics together with 3D MAE values."""
        return {**super().results_dict, **self.d3_results}

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]:
        """Return per-class 2D and 3D metrics in the standard exportable summary format."""
        summary = DetMetrics.summary(self, normalize, decimals)
        d3_class_ids = np.asarray(self.d3.ap_class_index, dtype=int)
        for index, row in enumerate(summary):
            class_id = int(self.box.ap_class_index[index])
            matches = np.flatnonzero(d3_class_ids == class_id)
            if len(matches):
                d3_index = int(matches[0])
                d3_p = self.d3.p[d3_index]
                d3_r = self.d3.r[d3_index]
                d3_f1 = self.d3.f1[d3_index]
                d3_map50, d3_map = self.d3.class_result(d3_index)[2:4]
            else:
                d3_p = d3_r = d3_f1 = d3_map50 = d3_map = 0.0
            row.update(
                {
                    "3D-Instances": (int(self.nt_per_class_d3[class_id]) if self.nt_per_class_d3 is not None else 0),
                    "3D-P": round(float(d3_p), decimals),
                    "3D-R": round(float(d3_r), decimals),
                    "3D-F1": round(float(d3_f1), decimals),
                    "3D-mAP50": round(float(d3_map50), decimals),
                    "3D-mAP50-95": round(float(d3_map), decimals),
                }
            )
        return summary


class Detection3DValidator(DetectionValidator):
    """A class extending the DetectionValidator class for validation based on a 3D detection model.

    This validator specializes in evaluating models that predict 3D attributes (depth, position, dimensions, rotation)
    in addition to standard 2D bounding boxes. Also overrides plot methods to draw 3D boxes using P2 calibration.

    Attributes:
        args (dict): Configuration arguments for the validator.
        metrics (DetMetrics): Metrics object for evaluating model performance.

    Methods:
        init_metrics: Initialize evaluation metrics for YOLO.
        postprocess: Post-process 3D predictions.
        _prepare_batch: Prepare batch data for 3D validation.
        _prepare_pred: Prepare predictions for evaluation against ground truth.

    Examples:
        >>> from ultralytics.models.yolo.detect3d import Detection3DValidator
        >>> args = dict(model="yolo11n-3d.pt", data="kitti3d.yaml")
        >>> validator = Detection3DValidator(args=args)
        >>> validator(model=args["model"])
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """Initialize Detection3DValidator and set task to 'detect3d'."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "detect3d"
        self.metrics = Detection3DMetrics()
        self.d3_err: list[torch.Tensor] = []  # matched-pair 3D errors
        self.d3_diagnostics: list[torch.Tensor] = []  # depth, errors, exact IoU3D, raw sigmoid(q3d)
        self.d3_gt_depths: list[torch.Tensor] = []  # all valid 3D GT, including missed detections
        self.extended_d3_diagnostics = False
        self.kitti_records: list[tuple[str, KittiAnnotation, KittiAnnotation]] = []
        self.kitti_gt_cache: dict[Path, KittiAnnotation] = {}
        self.kitti_label_dir: str | None = None
        self.kitti_classes: tuple[str, ...] = ()
        self.kitti_eval_mode = "off"
        self.kitti_plot_paths: list[Path] = []

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize evaluation metrics and reset 3D error accumulators."""
        # Native .pt validation can rescan score calibration without retraining. Exported graphs have this exponent
        # baked into the graph, so the helper intentionally becomes a no-op for immutable backends.
        quality3d_power = getattr(getattr(self, "args", None), "quality3d_power", 0.5)
        set_detect3d_quality_power(model, quality3d_power)
        super().init_metrics(model)
        self.d3_err = []
        self.d3_diagnostics = []
        self.d3_gt_depths = []
        # Exact rotated IoU and q3d correlation are standalone diagnostics. Keeping them out of trainer validation avoids
        # adding CPU polygon intersections to every training epoch.
        self.extended_d3_diagnostics = not self.training
        self.kitti_records = []
        self.kitti_plot_paths = []
        self.metrics.d3_results = {}
        requested_mode = str(getattr(getattr(self, "args", None), "kitti_eval", "off")).lower()
        if requested_mode not in {"off", "fast", "full"}:
            raise ValueError(f"kitti_eval must be one of ('off', 'fast', 'full'), got {requested_mode!r}")
        self.kitti_eval_mode = requested_mode
        configured_label_dir = self.data.get("kitti_label_dir") if self.kitti_eval_mode != "off" else None
        if self.kitti_eval_mode in {"fast", "full"} and not configured_label_dir:
            raise ValueError(f"kitti_eval={self.kitti_eval_mode!r} requires kitti_label_dir in the dataset YAML")
        self.kitti_label_dir = str(configured_label_dir) if configured_label_dir else None
        available_names = {str(name).lower() for name in self.names.values()}
        self.kitti_classes = tuple(name for name in KITTI_CLASSES if name.lower() in available_names)
        if self.kitti_label_dir and not self.kitti_classes:
            raise ValueError(
                f"KITTI R40 requires at least one of {KITTI_CLASSES} in dataset names, got {tuple(self.names.values())}"
            )

    def _raw_q3d_candidates(self, preds: Any) -> torch.Tensor | None:
        """Align raw sigmoid(q3d) with the rows entering NMS without changing the public eight-value 3D output."""
        if (
            getattr(self, "training", False)
            or not isinstance(preds, (tuple, list))
            or len(preds) < 2
            or not isinstance(preds[1], dict)
        ):
            return None
        inference, raw = preds[0], preds[1]
        if self.end2end:
            raw = raw.get("one2one")
        if not isinstance(raw, dict):
            return None
        raw_d3, raw_scores = raw.get("d3_params"), raw.get("scores")
        if not isinstance(raw_d3, torch.Tensor) or raw_d3.ndim != 3 or raw_d3.shape[1] <= 6:
            return None
        quality = raw_d3[:, 6].sigmoid()
        if not self.end2end:
            return quality if quality.shape[-1] == inference.shape[-1] else None
        if (
            not isinstance(raw_scores, torch.Tensor)
            or raw_scores.ndim != 3
            or raw_scores.shape[0] != quality.shape[0]
            or raw_scores.shape[-1] != quality.shape[-1]
        ):
            return None

        # YOLO26 performs an anchor-level top-k in Detect3D.postprocess before the validator sees predictions. Recreate
        # only that index selection from native raw logits so q3d stays aligned with its selected box. This is the same
        # score expression as Detect3D._inference and does not alter the actual decoded predictions or ranking.
        power = float(getattr(self.args, "quality3d_power", 0.5))
        best_score = (raw_scores.sigmoid() * quality.unsqueeze(1).pow(power)).max(1).values
        topk = int(inference.shape[1])
        if topk > best_score.shape[1]:
            return None
        anchor_indices = best_score.topk(topk, dim=1).indices
        return quality.gather(1, anchor_indices)

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Postprocess predictions with one class per anchor and expose the 8 decoded 3D parameters as ``extra``."""
        # Detect3D decodes dimensions using the highest-scoring class prior before NMS. Duplicating one anchor for
        # secondary classes (the base validator's multi_label=True behavior) would attach that highest-class geometry
        # to a different class, producing invalid 3D boxes and R40 records.
        quality_candidates = self._raw_q3d_candidates(preds)
        outputs, keep_indices = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            nc=self.nc,
            multi_label=False,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            return_idxs=True,
        )
        processed = []
        for batch_index, output in enumerate(outputs):
            item = {
                "bboxes": output[:, :4],
                "conf": output[:, 4],
                "cls": output[:, 5],
                "extra": output[:, 6:],
            }
            if quality_candidates is not None:
                item["q3d"] = quality_candidates[batch_index, keep_indices[batch_index].long()]
            processed.append(item)
        return processed

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare batch data for 3D validation with proper scaling and formatting.

        Label format: 12 columns -> cls + cxcywh + dist,xc,yc,w3d,h3d,l3d,rz (7 params)
        Also passes P2 calibration for proper camera-coordinate decoding.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]  # 11 columns: cx,cy,w,h,dist,xc,yc,w3d,h3d,l3d,rz
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        # Get P2 for this image
        p2s = batch.get("p2s", None)
        p2 = p2s[si] if p2s is not None and si < len(p2s) else None
        p2s_aug = batch.get("p2s_aug", None)
        p2_aug = p2s_aug[si] if p2s_aug is not None and si < len(p2s_aug) else None
        d3_valid = batch.get("d3_valid")
        d3_valid = d3_valid[idx].reshape(-1).bool() if d3_valid is not None else None
        if cls.shape[0]:
            # Convert xywh to xyxy and scale to pixel coords; only use first 4 columns for IoU evaluation
            bbox_2d = ops.xywh2xyxy(bbox[..., :4]) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]
            d3_gt = bbox[..., 4:]  # (N, 7): dist, xc, yc, w3d, h3d, l3d, rz
        else:
            bbox_2d = bbox[..., :4]
            d3_gt = bbox[..., :4]  # dummy, won't be used
        return {
            "cls": cls,
            "bboxes": bbox_2d,  # (N, 4) in xyxy pixel coords for IoU computation
            "d3": d3_gt,
            "d3_valid": d3_valid,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
            "p2": p2,
            "p2_aug": p2_aug,
        }

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Compute independent 2D and exact rotated 3D true-positive matrices for one image."""
        stats = super()._process_batch(preds, batch)
        d3_gt = batch.get("d3")
        if d3_gt is None or d3_gt.shape[0] == 0:
            valid = torch.zeros(0, dtype=torch.bool, device=batch["cls"].device)
            target_cls_d3 = batch["cls"][:0]
        else:
            # d3_valid contains training-policy gates such as depth range, pixel height and projected-center offset.
            # Evaluation must instead include every geometrically valid ground-truth 3D box.
            valid = torch.isfinite(d3_gt).all(1) & (d3_gt[:, 0] > 0) & (d3_gt[:, 3:6] > 0).all(1)
            target_cls_d3 = batch["cls"][valid]
        num_predictions = int(preds["cls"].shape[0])
        extra = preds.get("extra")
        if num_predictions and (
            not isinstance(extra, torch.Tensor)
            or extra.ndim != 2
            or extra.shape[0] != num_predictions
            or extra.shape[1] < 8
        ):
            shape = None if extra is None else getattr(extra, "shape", type(extra).__name__)
            raise ValueError(
                f"Detect3D validation requires one row with at least 8 geometry values per prediction, got {shape} "
                f"for {num_predictions} predictions"
            )
        if not len(target_cls_d3) or not num_predictions:
            tp_d3 = np.zeros((num_predictions, self.niou), dtype=bool)
        else:
            decoded = self._decode_d3(extra[:, :8], batch)
            iou3d = _camera_iou3d(d3_gt[valid], decoded)
            # Geometry intersection is CPU/OpenCV based, so keep matching on CPU and avoid a CPU->GPU->CPU round trip.
            tp_d3 = self.match_predictions(preds["cls"].detach().cpu(), target_cls_d3.detach().cpu(), iou3d).numpy()
        target_cls_d3 = target_cls_d3.detach().cpu().numpy()
        stats.update(
            {
                "tp_d3": tp_d3,
                "target_cls_d3": target_cls_d3,
                "target_img_d3": np.unique(target_cls_d3),
            }
        )
        return stats

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Update 2D metrics and accumulate matched-pair 3D regression errors."""
        super().update_metrics(preds, batch)
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            self._update_d3_stats(pred, pbatch)
            if self.kitti_label_dir:
                self._collect_kitti_record(pred, pbatch)

    def _kitti_ground_truth_path(self, im_file: str) -> Path:
        """Resolve an original KITTI `label_2` file for an image in a split-local dataset layout."""
        label_dir = Path(self.kitti_label_dir or "")
        if not label_dir.is_absolute():
            label_dir = Path(im_file).parent.parent / label_dir
        path = label_dir / f"{Path(im_file).stem}.txt"
        if not path.is_file():
            raise FileNotFoundError(
                f"KITTI R40 ground truth not found for {im_file}: {path}. "
                "Set kitti_label_dir to the original KITTI label_2 directory."
            )
        return path

    def _collect_kitti_record(self, pred: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Decode one image prediction and retain an official-format record for rank-zero R40 evaluation."""
        image_id = Path(pbatch["im_file"]).stem
        ground_truth_path = self._kitti_ground_truth_path(pbatch["im_file"])
        cache = getattr(self, "kitti_gt_cache", None)
        if cache is None:
            cache = self.kitti_gt_cache = {}
        ground_truth = cache.get(ground_truth_path)
        if ground_truth is None:
            ground_truth = cache[ground_truth_path] = parse_kitti_label(ground_truth_path)
        if pred["cls"].shape[0] == 0:
            predictions = build_kitti_predictions([], [], [], [], [], [], [])
            self.kitti_records.append((image_id, ground_truth, predictions))
            return

        native_boxes = pred["bboxes"].clone()
        ratio_pad = pbatch.get("ratio_pad")
        if ratio_pad is None:
            native_boxes = ops.scale_boxes(pbatch["imgsz"], native_boxes, pbatch["ori_shape"])
        else:
            # Dataset image loading rounds width and height independently, so their effective gains can differ slightly.
            # ``ops.scale_boxes`` assumes one shared gain, which can shift native KITTI boxes by a few pixels.
            gain_y, gain_x = float(ratio_pad[0][0]), float(ratio_pad[0][1])
            pad_x, pad_y = float(ratio_pad[1][0]), float(ratio_pad[1][1])
            if gain_x <= 0.0 or gain_y <= 0.0:
                raise ValueError(f"Invalid validation resize gains: gain_x={gain_x}, gain_y={gain_y}")
            native_boxes[:, [0, 2]] = (native_boxes[:, [0, 2]] - pad_x) / gain_x
            native_boxes[:, [1, 3]] = (native_boxes[:, [1, 3]] - pad_y) / gain_y
            native_boxes = ops.clip_boxes(native_boxes, pbatch["ori_shape"])
        decoded = self._decode_d3(pred["extra"][:, :8], pbatch)
        names = [str(self.names[int(index)]) for index in pred["cls"].long().tolist()]
        arrays = [native_boxes, pred["conf"], pred["extra"], decoded]
        finite = torch.stack([torch.isfinite(value).reshape(value.shape[0], -1).all(1) for value in arrays]).all(0)
        finite &= decoded[:, 0] > 0
        names = [name for name, keep in zip(names, finite.tolist()) if keep]
        native_boxes = native_boxes[finite].cpu().numpy()
        confidence = pred["conf"][finite].cpu().numpy()
        extra = pred["extra"][finite]
        decoded = decoded[finite]
        alpha = torch.atan2(extra[:, 3], extra[:, 4]).cpu().numpy()
        dimensions = decoded[:, [4, 3, 5]].cpu().numpy()  # h, w, l
        location = decoded[:, [1, 2, 0]].cpu().numpy()  # x, y_bottom, z
        predictions = build_kitti_predictions(
            names,
            native_boxes,
            confidence,
            alpha,
            dimensions,
            location,
            decoded[:, 6].cpu().numpy(),
        )
        self.kitti_records.append((image_id, ground_truth, predictions))

    def _update_d3_stats(self, pred: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Accumulate 3D errors for prediction/GT pairs greedily matched by 2D IoU >= 0.5 (same class).

        The 8 decoded params from head (center_x, center_y, depth, sin_alpha, cos_alpha, h, w, l)
        are decoded to camera coordinates for comparison with GT (dist, xc, yc, w3d, h3d, l3d, rz).
        """
        d3_gt = pbatch.get("d3")
        extra = pred.get("extra")
        if d3_gt is None or d3_gt.shape[0] == 0:
            return
        valid = torch.isfinite(d3_gt).all(1) & (d3_gt[:, 0] > 0) & (d3_gt[:, 3:6] > 0).all(1)
        # Keep regression diagnostics aligned with the 3D training policy. Generic 3D AP and KITTI R40 use the
        # complete geometrically valid target set in _process_batch() and remain intentionally unaffected.
        if pbatch.get("d3_valid") is not None:
            valid &= pbatch["d3_valid"].to(device=valid.device, dtype=torch.bool)
        if not valid.any():
            return
        d3_gt = d3_gt[valid]
        if getattr(self, "extended_d3_diagnostics", False):
            self.d3_gt_depths.append(d3_gt[:, 0].detach().cpu())
        if extra is None or extra.shape[0] == 0:
            return
        gt_boxes = pbatch["bboxes"][valid]
        gt_cls = pbatch["cls"][valid]
        # IoU-sorted one-to-one matching, consistent with the detection metric matching policy.
        iou = box_iou(gt_boxes, pred["bboxes"])  # (n_gt, n_pred)
        same_cls = gt_cls.view(-1, 1) == pred["cls"].view(1, -1)
        gt_idx, pred_idx = torch.where((iou >= 0.5) & same_cls)
        if gt_idx.numel() == 0:
            return
        order = iou[gt_idx, pred_idx].argsort(descending=True)
        unique_gt, unique_pred = [], []
        used_gt, used_pred = set(), set()
        for candidate in order.tolist():
            gi, pi = int(gt_idx[candidate]), int(pred_idx[candidate])
            if gi not in used_gt and pi not in used_pred:
                unique_gt.append(gi)
                unique_pred.append(pi)
                used_gt.add(gi)
                used_pred.add(pi)

        gt = d3_gt[unique_gt]  # (n, 7): dist, xc, yc, w3d, h3d, l3d, rz
        pr = self._decode_d3(extra[unique_pred, :8], pbatch)

        err = torch.zeros((gt.shape[0], 7), device=gt.device)
        err[:, :6] = (pr[:, :6] - gt[:, :6]).abs()  # meters
        # rotation_y absolute angular error in degrees, wrap-aware
        d_ang = (pr[:, 6] - gt[:, 6] + torch.pi) % (2 * torch.pi) - torch.pi
        err[:, 6] = d_ang.abs() * 180.0 / torch.pi
        self.d3_err.append(err.cpu())
        if getattr(self, "extended_d3_diagnostics", False):
            # Invalid decoded boxes cannot form a 3D true positive. Their GT remains in the recall denominator, while
            # non-finite values are excluded from percentiles and q3d correlation.
            pair_valid = torch.isfinite(pr).all(1) & (pr[:, 0] > 0) & (pr[:, 3:6] > 0).all(1)
            if not pair_valid.any():
                return
            diag_gt, diag_pr, diag_err = gt[pair_valid], pr[pair_valid], err[pair_valid]
            unique_pred_tensor = torch.as_tensor(unique_pred, device=extra.device, dtype=torch.long)[pair_valid]
            iou3d = _paired_camera_iou3d(diag_gt, diag_pr)
            q3d = pred.get("q3d")
            if q3d is None:
                quality = torch.full_like(iou3d, float("nan"))
            else:
                quality = q3d[unique_pred_tensor].to(device=iou3d.device, dtype=iou3d.dtype)
            diagnostic = torch.cat((diag_gt[:, :1], diag_err, iou3d[:, None], quality[:, None]), dim=1)
            self.d3_diagnostics.append(diagnostic.detach().cpu())

    def _decode_d3(self, extra: torch.Tensor, image_info: dict[str, Any]) -> torch.Tensor:
        """Decode 8 head outputs to label-format camera coordinates.

        Input centers are absolute pixels in the model's letterboxed image. They are first mapped back to the original
        camera image, then back-projected through P2. Output columns are ``z, x, y_bottom, w, h, l, rotation_y``.
        """
        if extra.shape[0] == 0:
            return extra.new_zeros((0, 7))

        center = extra[:, :2].clone()
        depth = extra[:, 2]
        h3d, w3d, l3d = extra[:, 5], extra[:, 6], extra[:, 7]
        p2 = image_info.get("p2_aug")
        if p2 is None:
            # Compatibility path for callers that only provide original P2 + letterbox metadata.
            p2 = image_info.get("p2")
            if p2 is None:
                raise ValueError("Decoding detect3d predictions requires a valid P2 or p2_aug matrix")
            ratio_pad = image_info.get("ratio_pad")
            if ratio_pad is not None:
                gain_y, gain_x = float(ratio_pad[0][0]), float(ratio_pad[0][1])
                pad_x, pad_y = ratio_pad[1]
                center[:, 0] = (center[:, 0] - float(pad_x)) / gain_x
                center[:, 1] = (center[:, 1] - float(pad_y)) / gain_y
        p2 = torch.as_tensor(p2, device=extra.device, dtype=extra.dtype)
        location_center = backproject_points_torch(center, depth, p2)
        x_cam, z_cam = location_center[:, 0], location_center[:, 2]
        y_bottom = location_center[:, 1] + h3d * 0.5

        alpha = torch.atan2(extra[:, 3], extra[:, 4])
        rotation_y = alpha + torch.atan2(x_cam, z_cam.clamp_min(1e-12))
        rotation_y = (rotation_y + torch.pi) % (2 * torch.pi) - torch.pi
        return torch.stack((z_cam, x_cam, y_bottom, w3d, h3d, l3d, rotation_y), dim=1)

    def get_stats(self) -> dict[str, Any]:
        """Calculate and return 2D metrics, diagnostic 3D MAE and optional official KITTI R40 metrics."""
        super().get_stats()
        if self.d3_err:
            mae = torch.cat(self.d3_err).mean(0)
            self.metrics.d3_results = {f"3d/{n}_MAE": mae[i].item() for i, n in enumerate(_D3_ERROR_NAMES)}
        if self.d3_gt_depths:
            matched = torch.cat(self.d3_diagnostics) if self.d3_diagnostics else torch.empty((0, 10))
            self.metrics.d3_results.update(_summarize_d3_diagnostics(matched, torch.cat(self.d3_gt_depths)))
        if self.kitti_records:
            records = sorted(self.kitti_records, key=lambda record: record[0])
            image_ids = [record[0] for record in records]
            if len(image_ids) != len(set(image_ids)):
                raise ValueError("KITTI R40 image stems must be unique within the validation split")
            ground_truth = [record[1] for record in records]
            detections = [record[2] for record in records]
            if self.kitti_eval_mode == "fast":
                moderate_ap3d = []
                for class_name in self.kitti_classes:
                    ap3d = evaluate_kitti_metric(
                        ground_truth,
                        detections,
                        class_name=class_name,
                        difficulty="moderate",
                        metric="3d",
                    )
                    self.metrics.d3_results[f"kitti/{class_name}_AP3D_R40_moderate"] = ap3d
                    moderate_ap3d.append(ap3d)
                self.metrics.d3_results["kitti/AP3D_R40_moderate"] = (
                    float(np.mean(moderate_ap3d)) if moderate_ap3d else 0.0
                )
            else:
                kitti_results, kitti_curves = evaluate_kitti_r40(
                    ground_truth, detections, self.kitti_classes, return_curves=True
                )
                self.metrics.d3_results.update(kitti_results)
                if self.args.plots:
                    self.kitti_plot_paths = plot_kitti_r40(
                        kitti_results, kitti_curves, self.kitti_classes, self.save_dir
                    )
                prediction_dir = self.save_dir / "kitti_predictions"
                for image_id, _, prediction in records:
                    write_kitti_predictions(prediction, prediction_dir / f"{image_id}.txt")
        return self.metrics.results_dict

    def gather_stats(self) -> None:
        """Gather standard statistics, 3D errors and per-image KITTI records during distributed validation."""
        if RANK == 0:
            gathered = [None] * dist.get_world_size()
            dist.gather_object(self.d3_err, gathered, dst=0)
            self.d3_err = [error for rank_errors in gathered for error in (rank_errors or [])]
            gathered_diagnostics = [None] * dist.get_world_size()
            dist.gather_object((self.d3_diagnostics, self.d3_gt_depths), gathered_diagnostics, dst=0)
            self.d3_diagnostics = [
                row for rank_diagnostics, _ in gathered_diagnostics for row in (rank_diagnostics or [])
            ]
            self.d3_gt_depths = [depth for _, rank_depths in gathered_diagnostics for depth in (rank_depths or [])]
            gathered_kitti = [None] * dist.get_world_size()
            dist.gather_object(self.kitti_records, gathered_kitti, dst=0)
            self.kitti_records = [record for rank_records in gathered_kitti for record in (rank_records or [])]
        elif RANK > 0:
            dist.gather_object(self.d3_err, None, dst=0)
            self.d3_err = []
            dist.gather_object((self.d3_diagnostics, self.d3_gt_depths), None, dst=0)
            self.d3_diagnostics = []
            self.d3_gt_depths = []
            dist.gather_object(self.kitti_records, None, dst=0)
            self.kitti_records = []
        super().gather_stats()
        self._gather_image_metrics(self.metrics.d3)

    def print_results(self) -> None:
        """Print training/validation set metrics per class, plus a 3D error summary."""
        super().print_results()
        if self.d3_err:
            mae = torch.cat(self.d3_err).mean(0)
            n = sum(e.shape[0] for e in self.d3_err)
            LOGGER.info(
                f"3D errors ({n} matched pairs, IoU>=0.5):\n"
                f"  dist_MAE={mae[0]:.3f}m,  xc_MAE={mae[1]:.3f}m,  yc_MAE={mae[2]:.3f}m\n"
                f"  w_MAE={mae[3]:.3f}m,  h_MAE={mae[4]:.3f}m,  l_MAE={mae[5]:.3f}m\n"
                f"  ry_MAE={mae[6]:.2f}°"
            )
        if self.d3_gt_depths:
            result = self.metrics.d3_results
            prefix = "3d/diagnostic"
            gt_count = int(result[f"{prefix}/gt_count"])
            matched_count = int(result[f"{prefix}/matched_count"])
            LOGGER.info(
                "3D diagnostic (same-class 2D IoU>=0.5 one-to-one matches; recall denominator is all valid 3D GT):\n"
                f"  matched={matched_count}/{gt_count} ({result[f'{prefix}/match_recall'] * 100:.2f}%),  "
                f"IoU3D recall@0.5={result[f'{prefix}/iou3d_recall_0.5'] * 100:.2f}%,  "
                f"recall@0.7={result[f'{prefix}/iou3d_recall_0.7'] * 100:.2f}%\n"
                f"  P50/P90: dist={result.get(f'{prefix}/dist_P50', 0.0):.3f}/"
                f"{result.get(f'{prefix}/dist_P90', 0.0):.3f}m,  "
                f"xc={result.get(f'{prefix}/xc_P50', 0.0):.3f}/"
                f"{result.get(f'{prefix}/xc_P90', 0.0):.3f}m,  "
                f"yc={result.get(f'{prefix}/yc_P50', 0.0):.3f}/"
                f"{result.get(f'{prefix}/yc_P90', 0.0):.3f}m,  "
                f"ry={result.get(f'{prefix}/ry_deg_P50', 0.0):.2f}/"
                f"{result.get(f'{prefix}/ry_deg_P90', 0.0):.2f}°"
            )
            quality_count = int(result.get(f"{prefix}/q3d_count", 0.0))
            if quality_count >= 2:
                LOGGER.info(
                    f"  raw sigmoid(q3d) vs IoU3D ({quality_count} matched pairs): "
                    f"Pearson={result[f'{prefix}/q3d_iou3d_pearson']:.3f}, "
                    f"Spearman={result[f'{prefix}/q3d_iou3d_spearman']:.3f}"
                )
            else:
                LOGGER.info("  raw sigmoid(q3d) vs IoU3D: unavailable (native .pt raw head outputs are required)")
            lines = ["  Range       GT  Matched   R@0.5   R@0.7  dist_MAE  IoU3D_mean"]
            for label, _, _ in _D3_DISTANCE_RANGES:
                range_prefix = f"3d/range_{label}"
                lines.append(
                    f"  {label:<9} {int(result[f'{range_prefix}/gt_count']):5d} "
                    f"{int(result[f'{range_prefix}/matched_count']):8d} "
                    f"{result[f'{range_prefix}/iou3d_recall_0.5'] * 100:7.2f}% "
                    f"{result[f'{range_prefix}/iou3d_recall_0.7'] * 100:7.2f}% "
                    f"{result.get(f'{range_prefix}/dist_MAE', 0.0):9.3f} "
                    f"{result.get(f'{range_prefix}/iou3d_mean', 0.0):11.3f}"
                )
            LOGGER.info("3D diagnostic by GT camera depth:\n" + "\n".join(lines))
        if self.kitti_records:
            if self.kitti_eval_mode == "fast":
                values = "  ".join(
                    f"{class_name}={self.metrics.d3_results[f'kitti/{class_name}_AP3D_R40_moderate']:.2f}"
                    for class_name in self.kitti_classes
                )
                LOGGER.info(f"Fast KITTI AP3D R40 Moderate (auxiliary report): {values}")
            else:
                LOGGER.info(format_kitti_r40(self.metrics.d3_results, self.kitti_classes))

    def get_desc(self) -> str:
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "3D(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot validation image samples with 3D boxes."""
        p2s = batch.get("p2s", None)
        ori_shapes = batch.get("ori_shape", None)
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
            p2s=p2s,
            ori_shapes=ori_shapes,
        )

    def plot_predictions(
        self,
        batch: dict[str, Any],
        preds: list[dict[str, torch.Tensor]],
        ni: int,
        max_det: int | None = None,
    ) -> None:
        """Plot predicted 2D and projected 3D bounding boxes on input images."""
        if not preds:
            return
        decoded_preds = []
        for i, pred in enumerate(preds):
            pred = pred.copy()
            pred["batch_idx"] = torch.ones_like(pred["conf"]) * i
            p2s = batch.get("p2s")
            image_info = {
                "p2": p2s[i] if p2s is not None and i < len(p2s) else None,
                "p2_aug": batch["p2s_aug"][i],
                "ratio_pad": batch["ratio_pad"][i],
                "imgsz": batch["img"].shape[2:],
                "ori_shape": batch["ori_shape"][i],
            }
            pred["extra"] = self._decode_d3(pred["extra"][:, :8], image_info)
            decoded_preds.append(pred)
        keys = decoded_preds[0].keys()
        max_det = max_det or self.args.max_det
        batched_preds = {k: torch.cat([x[k][:max_det] for x in decoded_preds], dim=0) for k in keys}
        batched_preds["bboxes"] = ops.xyxy2xywh(batched_preds["bboxes"])
        batched_preds["bboxes"] = torch.cat((batched_preds["bboxes"], batched_preds.pop("extra")), dim=1)
        plot_images(
            images=batch["img"],
            labels=batched_preds,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
            p2s=batch.get("p2s"),
            ori_shapes=batch.get("ori_shape"),
        )
