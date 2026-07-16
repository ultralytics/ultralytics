# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.data.stereo.box3d import Box3D
from ultralytics.data.stereo.calib import CalibrationParameters
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.s3d.metrics import (
    DIFFICULTY_EASY,
    DIFFICULTY_HARD,
    DIFFICULTY_MODERATE,
    Stereo3DDetMetrics,
    classify_difficulty,
)
from ultralytics.models.yolo.s3d.preprocess import (
    compute_letterbox_params,
    decode_and_refine_predictions,
    preprocess_stereo_batch,
)
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.metrics import DetMetrics, box_iou, compute_3d_iou, compute_bev_iou
from ultralytics.utils.plotting import plot_stereo3d_boxes


def _reverse_letterbox_calib(
    calib: dict | CalibrationParameters,
    letterbox_scale: float,
    pad_left: int,
    pad_top: int,
    actual_w: int,
    actual_h: int,
) -> dict:
    """Reverse letterbox transformation on calibration to get original image coordinates.

    Args:
        calib: Calibration in letterboxed space (dict or CalibrationParameters).
        letterbox_scale: Scale factor applied during letterbox.
        pad_left: Left padding added by letterbox.
        pad_top: Top padding added by letterbox.
        actual_w: Original image width.
        actual_h: Original image height.

    Returns:
        Calibration dict in original image coordinates.
    """
    if isinstance(calib, dict):
        return {
            "fx": calib["fx"] / letterbox_scale,
            "fy": calib["fy"] / letterbox_scale,
            "cx": (calib["cx"] - pad_left) / letterbox_scale,
            "cy": (calib["cy"] - pad_top) / letterbox_scale,
            "baseline": calib.get("baseline", 0.54),
            "image_width": actual_w,
            "image_height": actual_h,
        }
    elif isinstance(calib, CalibrationParameters):
        return {
            "fx": calib.fx / letterbox_scale,
            "fy": calib.fy / letterbox_scale,
            "cx": (calib.cx - pad_left) / letterbox_scale,
            "cy": (calib.cy - pad_top) / letterbox_scale,
            "baseline": calib.baseline,
            "image_width": actual_w,
            "image_height": actual_h,
        }
    return calib


def compute_3d_iou_batch(pred_boxes: list[Box3D], gt_boxes: list[Box3D], eps: float = 1e-7) -> np.ndarray:
    """Compute 3D IoU matrix between prediction and ground truth boxes.

    Only computes IoU for boxes with matching class_id; others are set to 0.0.

    Args:
        pred_boxes: List of predicted Box3D objects.
        gt_boxes: List of ground truth Box3D objects.
        eps: Small value to avoid division by zero.

    Returns:
        IoU matrix of shape (len(pred_boxes), len(gt_boxes)).
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)))

    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))

    # Extract class IDs for filtering
    pred_class_ids = np.array([box.class_id for box in pred_boxes])
    gt_class_ids = np.array([box.class_id for box in gt_boxes])
    class_match = pred_class_ids[:, None] == gt_class_ids[None, :]

    # Compute IoU for matching pairs
    for i in range(len(pred_boxes)):
        for j in range(len(gt_boxes)):
            if not class_match[i, j]:
                continue
            iou_matrix[i, j] = compute_3d_iou(pred_boxes[i], gt_boxes[j], eps=eps)

    return iou_matrix


def compute_bev_iou_batch(pred_boxes: list[Box3D], gt_boxes: list[Box3D], eps: float = 1e-7) -> np.ndarray:
    """Compute the bird's-eye-view (BEV) IoU matrix between prediction and GT boxes.

    Mirrors :func:`compute_3d_iou_batch` but uses ground-plane footprint overlap (ignoring height) for KITTI AP_BEV.
    Only matching-class pairs are filled.

    Args:
        pred_boxes: List of predicted Box3D objects.
        gt_boxes: List of ground truth Box3D objects.
        eps: Small value to avoid division by zero.

    Returns:
        BEV IoU matrix of shape (len(pred_boxes), len(gt_boxes)).
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)))

    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    pred_class_ids = np.array([box.class_id for box in pred_boxes])
    gt_class_ids = np.array([box.class_id for box in gt_boxes])
    class_match = pred_class_ids[:, None] == gt_class_ids[None, :]

    for i in range(len(pred_boxes)):
        for j in range(len(gt_boxes)):
            if not class_match[i, j]:
                continue
            iou_matrix[i, j] = compute_bev_iou(pred_boxes[i], gt_boxes[j], eps=eps)

    return iou_matrix


class Stereo3DDetValidator(BaseValidator):
    """Stereo 3D Detection Validator.

    Extends BaseValidator to implement 3D detection validation with AP3D metrics. Computes 3D IoU, matches predictions
    to ground truth, and calculates AP3D at IoU 0.5 and 0.7.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """Initialize Stereo3DDetValidator.

        Args:
            dataloader: DataLoader for validation data.
            save_dir: Directory to save results.
            args: Configuration arguments.
            _callbacks: Callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "s3d"
        self.iouv = torch.tensor([0.5, 0.7])  # IoU thresholds for AP3D
        self.niou = len(self.iouv)
        self.metrics = Stereo3DDetMetrics()

        # 2D bbox metrics (YOLO-style mAP50/mAP50-95) for debugging bbox head quality.
        self.det_iouv = torch.linspace(0.5, 0.95, 10)
        self.det_metrics = DetMetrics()

        # Mean and std dimensions from dataset config (for decoding)
        self.mean_dims = None
        self.std_dims = None

    def get_dataset(self) -> dict[str, Any]:
        """Parse stereo dataset YAML and return metadata.

        Uses check_det_dataset() for proper path resolution and automatic downloads,
        then transforms the result into stereo descriptor format.

        Returns:
            dict: Dataset dictionary with stereo descriptor dicts for train/val splits.
        """
        from ultralytics.data.utils import check_det_dataset

        data_cfg = check_det_dataset(self.args.data, autodownload=True)

        root = Path(data_cfg["path"])
        train_split = data_cfg.get("train_split", "train")
        val_split = data_cfg.get("val_split", "val")

        names = data_cfg.get("names")
        if names is None:
            raise ValueError("Dataset configuration must include 'names' mapping")
        nc = data_cfg.get("nc", len(names))

        return {
            "yaml_file": str(self.args.data) if isinstance(self.args.data, (str, Path)) else None,
            "path": str(root),
            "channels": 6,
            "train": {"type": "kitti_stereo", "root": str(root), "split": train_split},
            "val": {"type": "kitti_stereo", "root": str(root), "split": val_split},
            "names": names,
            "nc": nc,
            "stereo": data_cfg.get("stereo", True),
            "baseline": data_cfg.get("baseline"),
            "mean_dims": data_cfg.get("mean_dims"),
            "std_dims": data_cfg.get("std_dims"),
        }

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Normalize 6-channel images to float [0,1] and move targets to device.

        Uses shared preprocessing from preprocess.py for consistency with trainer.
        Also stores the current batch for postprocess() access.
        """
        batch = preprocess_stereo_batch(batch, self.device, half=self.args.quantize == 16)
        # Make the *current* batch available to postprocess(); BaseValidator calls postprocess() before update_metrics().
        self._current_batch = batch
        return batch

    def postprocess(self, preds) -> list[list[Box3D]]:
        """Postprocess model outputs to Box3D objects.

        Uses shared decode_and_refine_predictions from preprocess.py which handles
        decoding, geometric construction, and dense alignment refinement.

        Args:
            preds: Tuple of (inference_output, preds_dict) from model forward.

        Returns:
            List of Box3D lists (one per batch item).
        """
        # Unpack (y, preds_dict) from Detect.forward — AutoBackend may convert tuple to list
        if isinstance(preds, (tuple, list)) and len(preds) == 2 and isinstance(preds[1], dict):
            y, preds_dict = preds
            preds_dict = {**preds_dict, "det": y}  # add inference output for decode
        else:
            preds_dict = preds

        # Get batch for calibration and images
        batch = self._current_batch if hasattr(self, "_current_batch") else None

        # Get use_geometric and use_dense_alignment from args.
        # Disable dense alignment during training validation to prevent DDP hangs:
        # dense alignment is CPU-intensive and takes variable time per batch, causing
        # rank divergence when different ranks finish at different times and deadlock
        # at dist.reduce().
        use_geometric = getattr(self.args, "use_geometric", None)
        use_dense_alignment = getattr(self.args, "use_dense_alignment", None)
        if self.training:
            use_dense_alignment = False
            use_geometric = False

        return decode_and_refine_predictions(
            preds=preds_dict,
            batch=batch,
            args=self.args,
            use_geometric=use_geometric,
            use_dense_alignment=use_dense_alignment,
            conf_threshold=self.args.conf,
            top_k=100,
            iou_thres=getattr(self.args, "iou", 0.45),
            imgsz=getattr(self.args, "imgsz", 384),
            mean_dims=self.mean_dims if hasattr(self, "mean_dims") else None,
            std_dims=self.std_dims if hasattr(self, "std_dims") else None,
            class_names=self.names if hasattr(self, "names") else None,
            score_k=getattr(self.args, "score_k", 0.5),
        )

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics with model information.

        Args:
            model: Model being validated.
        """
        # Use model's class names — they define the class IDs the model predicts.
        # The dataset YAML may have more classes (e.g., 8 KITTI classes) than the model (e.g., 3).
        # Using model names ensures class IDs match between predictions and GT filtering.
        model_names = getattr(model, "names", None)
        data_names = self.data.get("names") if hasattr(self, "data") and self.data else None
        if model_names:
            self.names = model_names
        elif data_names:
            self.names = data_names
        else:
            raise ValueError("Model or dataset configuration must include 'names' mapping")

        self.nc = len(self.names) if isinstance(self.names, (dict, list, tuple)) else 0

        self.seen = 0
        self.metrics.names = self.names
        self.metrics.nc = self.nc

        # Map mean_dims/std_dims from dataset YAML (keyed by class name) to model class IDs.
        # E.g., model {0: "Car", 1: "Pedestrian"} + YAML {"Car": [L,W,H]} → {0: (H,W,L)}
        def _parse_dims(raw_dims):
            """Convert {class_name: [L,W,H]} to {model_class_id: (H,W,L)}."""
            if raw_dims is None:
                return None
            result = {}
            for class_key, dims in raw_dims.items():
                if not (isinstance(dims, (list, tuple)) and len(dims) == 3):
                    continue
                l, w, h = dims  # YAML has [L, W, H]
                # Find model class ID by name match
                matched = False
                for cid, cname in self.names.items() if isinstance(self.names, dict) else enumerate(self.names):
                    if str(class_key) == str(cname) or (isinstance(class_key, int) and class_key == cid):
                        result[cid] = (h, w, l)  # Store as (H, W, L)
                        matched = True
                        break
                if not matched and isinstance(class_key, int):
                    result[class_key] = (h, w, l)
            # Fallback: if name matching failed, assign by iteration order to class IDs 0..N
            if not result and raw_dims:
                for i, (_, dims) in enumerate(raw_dims.items()):
                    if isinstance(dims, (list, tuple)) and len(dims) == 3:
                        l, w, h = dims
                        result[i] = (h, w, l)
            return result if result else None

        mean_dims_raw = self.data.get("mean_dims") if hasattr(self, "data") and self.data else None
        std_dims_raw = self.data.get("std_dims") if hasattr(self, "data") and self.data else None
        self.mean_dims = _parse_dims(mean_dims_raw)
        self.std_dims = _parse_dims(std_dims_raw)
        if self.mean_dims is None:
            LOGGER.info("No mean_dims in dataset config, will use defaults")
        if self.std_dims is None:
            LOGGER.info("No std_dims in dataset config, will use defaults")

        # Build dataset→model class ID mapping when dataset has more classes than model.
        # E.g., dataset {0:Car, 1:Van, 3:Pedestrian} + model {0:Car, 1:Pedestrian}
        # → remap {0:0, 3:1} so GT Pedestrian (dataset id=3) maps to model id=1.
        self._dataset_to_model_cls = None
        if data_names and model_names and len(data_names) != len(model_names):
            model_name_to_id = {
                name: idx
                for idx, name in (model_names.items() if isinstance(model_names, dict) else enumerate(model_names))
            }
            remap = {}
            for did, dname in data_names.items() if isinstance(data_names, dict) else enumerate(data_names):
                if dname in model_name_to_id:
                    remap[int(did)] = model_name_to_id[dname]
            if remap:
                self._dataset_to_model_cls = remap

        # Clear accumulated stats from previous validation epoch
        self.metrics.clear_stats()

        # Init 2D detection metrics (bbox mAP)
        self.det_metrics.names = self.names
        self.det_metrics.clear_stats()

    def update_metrics(self, preds: list[list[Box3D]], batch: dict[str, Any]) -> None:
        """Update metrics with predictions and ground truth.

        Args:
            preds: List of predicted Box3D lists (one per image).
            batch: Batch containing ground truth labels.
        """
        self._current_batch = batch  # Store for calibration access

        labels_list = batch.get("labels", [])
        calibs = batch.get("calib", [])
        ori_shapes = batch.get("ori_shape", [])
        im_files = batch.get("im_file", [])

        for si, (pred_boxes, labels) in enumerate(zip(preds, labels_list)):
            self.seen += 1

            # Get calibration for this sample
            calib = calibs[si] if si < len(calibs) and isinstance(calibs[si], dict) else None

            # Convert labels to Box3D - need to reverse letterbox transformation on calibration
            # Labels use original image normalized coordinates, but calib is letterboxed
            if calib is not None and si < len(ori_shapes):
                ori_shape = ori_shapes[si]
                if isinstance(ori_shape, (list, tuple)) and len(ori_shape) >= 2:
                    actual_h, actual_w = ori_shape[0], ori_shape[1]
                    imgsz = getattr(self.args, "imgsz", 384)

                    letterbox_scale, pad_left, pad_top = compute_letterbox_params(actual_h, actual_w, imgsz)
                    calib = _reverse_letterbox_calib(calib, letterbox_scale, pad_left, pad_top, actual_w, actual_h)
                    in_h, in_w = (imgsz, imgsz) if isinstance(imgsz, int) else (int(imgsz[0]), int(imgsz[1]))

            # Convert labels to Box3D
            data_names = self.data.get("names") if hasattr(self, "data") and self.data else self.names
            calib_dict = calib.to_dict() if hasattr(calib, "to_dict") else calib
            image_hw = (
                (int(calib.get("image_height", 375)), int(calib.get("image_width", 1242))) if calib else (375, 1242)
            )
            gt_boxes = [
                b
                for lab in labels
                if (b := Box3D.from_label(lab, calib_dict, class_names=data_names, image_hw=image_hw)) is not None
            ]

            # Remap GT class IDs from dataset space to model space and drop unmapped classes
            if self._dataset_to_model_cls:
                remapped = []
                for box in gt_boxes:
                    new_id = self._dataset_to_model_cls.get(box.class_id)
                    if new_id is not None:
                        box.class_id = new_id
                        box.class_label = (
                            self.names.get(new_id, str(new_id)) if isinstance(self.names, dict) else str(new_id)
                        )
                        remapped.append(box)
                gt_boxes = remapped

            # ------------------------------------------------------------
            # 2D bbox metrics (original-image xyxy)
            # ------------------------------------------------------------
            # Pred boxes2d from 3D projection (original coords)
            pred_bboxes2d = []
            pred_conf2d = []
            pred_cls2d = []
            for pb in pred_boxes:
                bbox_2d = pb.project_to_2d(calib) if calib else None
                if bbox_2d is None or bbox_2d[2] <= bbox_2d[0] or bbox_2d[3] <= bbox_2d[1]:
                    continue
                pred_bboxes2d.append([float(bbox_2d[0]), float(bbox_2d[1]), float(bbox_2d[2]), float(bbox_2d[3])])
                pred_conf2d.append(float(pb.confidence))
                pred_cls2d.append(int(pb.class_id))

            # GT boxes2d from labels (labels are normalized to *letterboxed* input space).
            gt_bboxes2d = []
            gt_cls2d = []

            # Use ori_shapes for inverse-letterbox.
            imgsz = getattr(self.args, "imgsz", 384)
            if si < len(ori_shapes) and isinstance(ori_shapes[si], (list, tuple)) and len(ori_shapes[si]) >= 2:
                ori_h, ori_w = int(ori_shapes[si][0]), int(ori_shapes[si][1])
            else:
                ori_h, ori_w = 375, 1242
            letterbox_scale, pad_left, pad_top = compute_letterbox_params(ori_h, ori_w, imgsz)
            if isinstance(imgsz, int):
                in_h, in_w = imgsz, imgsz
            else:
                in_h, in_w = int(imgsz[0]), int(imgsz[1])

            for lab in labels:
                lb = lab.get("left_box", None)
                if lb is None:
                    continue
                cls_i = int(lab.get("class_id", 0))
                # Remap dataset class ID to model class ID
                if self._dataset_to_model_cls:
                    mapped = self._dataset_to_model_cls.get(cls_i)
                    if mapped is None:
                        continue  # Skip classes not in model
                    cls_i = mapped
                cx = float(lb.get("center_x", 0.0)) * in_w
                cy = float(lb.get("center_y", 0.0)) * in_h
                bw = float(lb.get("width", 0.0)) * in_w
                bh = float(lb.get("height", 0.0)) * in_h
                x1_l = cx - bw / 2
                y1_l = cy - bh / 2
                x2_l = cx + bw / 2
                y2_l = cy + bh / 2
                # letterbox -> original
                x1 = (x1_l - pad_left) / letterbox_scale
                y1 = (y1_l - pad_top) / letterbox_scale
                x2 = (x2_l - pad_left) / letterbox_scale
                y2 = (y2_l - pad_top) / letterbox_scale
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                gt_bboxes2d.append([x1, y1, x2, y2])
                gt_cls2d.append(cls_i)

            # Compute tp matrix (N,10) for bbox metrics
            n_pred = len(pred_bboxes2d)
            if n_pred == 0:
                tp2d = np.zeros((0, self.det_iouv.numel()), dtype=bool)
                conf2d = np.zeros((0,), dtype=np.float32)
                pred_cls_np = np.zeros((0,), dtype=np.int64)
            else:
                pred_boxes_t = torch.tensor(pred_bboxes2d, dtype=torch.float32)
                pred_cls_t = torch.tensor(pred_cls2d, dtype=torch.int64)
                gt_boxes_t = (
                    torch.tensor(gt_bboxes2d, dtype=torch.float32)
                    if gt_bboxes2d
                    else torch.zeros((0, 4), dtype=torch.float32)
                )
                gt_cls_t = (
                    torch.tensor(gt_cls2d, dtype=torch.int64) if gt_cls2d else torch.zeros((0,), dtype=torch.int64)
                )

                if gt_boxes_t.shape[0] == 0:
                    tp2d = np.zeros((n_pred, self.det_iouv.numel()), dtype=bool)
                else:
                    iou2d = box_iou(gt_boxes_t, pred_boxes_t)  # MxN (gt x pred)
                    # Use BaseValidator matching but with det_iouv.
                    # IMPORTANT: guard with try/finally so self.iouv is always restored (prevents leaking 10 IoUs
                    # into the 3D metrics path where tp/fp are shape (N, 2)).
                    old_iouv = self.iouv
                    try:
                        self.iouv = self.det_iouv
                        correct = self.match_predictions(pred_cls_t, gt_cls_t, iou2d).cpu().numpy()
                    finally:
                        self.iouv = old_iouv
                    tp2d = correct

                conf2d = np.asarray(pred_conf2d, dtype=np.float32)
                pred_cls_np = np.asarray(pred_cls2d, dtype=np.int64)

            target_cls_np = np.asarray(gt_cls2d, dtype=np.int64)
            self.det_metrics.update_stats(
                {
                    "tp": tp2d,
                    "conf": conf2d,
                    "pred_cls": pred_cls_np,
                    "target_cls": target_cls_np,
                    "target_img": np.unique(target_cls_np),
                    "im_name": Path(im_files[si]).name if si < len(im_files) else str(si),
                }
            )

            # Handle empty predictions or ground truth
            if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                continue

            # Classify GT difficulty (KITTI standard)
            gt_difficulties = np.full(len(gt_boxes), -1, dtype=int)
            for gi, gt_box in enumerate(gt_boxes):
                trunc = gt_box.truncated if gt_box.truncated is not None else 0.0
                occ = gt_box.occluded if gt_box.occluded is not None else 0
                # Get 2D bbox height in original image pixels
                bbox_2d = gt_box.project_to_2d(calib) if calib else None
                h_2d = float(bbox_2d[3] - bbox_2d[1]) if bbox_2d is not None else 0.0
                gt_difficulties[gi] = classify_difficulty(trunc, occ, h_2d)

            # Compute pred 2D bbox heights for min-height filtering
            pred_heights_2d = np.zeros(len(pred_boxes), dtype=np.float32)
            for pi, pb in enumerate(pred_boxes):
                bbox_2d = pb.project_to_2d(calib) if calib else None
                pred_heights_2d[pi] = float(bbox_2d[3] - bbox_2d[1]) if bbox_2d is not None else 0.0

            # Compute 3D and BEV IoU matrices
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                iou_matrix = compute_3d_iou_batch(pred_boxes, gt_boxes)
                bev_iou_matrix = compute_bev_iou_batch(pred_boxes, gt_boxes)
            else:
                iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
                bev_iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))

            # Store raw data for per-difficulty matching in metrics.process()
            self.metrics.update_stats(
                {
                    "pred_boxes": pred_boxes,
                    "gt_boxes": gt_boxes,
                    "iou_matrix": iou_matrix,
                    "bev_iou_matrix": bev_iou_matrix,
                    "gt_difficulties": gt_difficulties,
                    "pred_heights_2d": pred_heights_2d,
                }
            )

            # Update progress bar with intermediate metrics (every batch for real-time feedback)
        if hasattr(self, "_progress_bar") and self._progress_bar is not None and RANK in {-1, 0}:
            # Update progress bar periodically to avoid performance impact
            if hasattr(self, "_batch_count"):
                self._batch_count += 1
            else:
                self._batch_count = 1

            # Update every 5 batches or if we're near the end (more frequent than before)
            if self._batch_count % 5 == 0 or (
                hasattr(self, "_total_batches") and self._batch_count >= self._total_batches - 1
            ):
                metrics_str = self._format_progress_metrics()
                if metrics_str:
                    self._progress_bar.set_description(metrics_str)

        # Generate visualization images if plots enabled.
        # NOTE: This stereo validator saves 1 file per sample, so keep defaults conservative to avoid generating
        # thousands of images when using large validation batch sizes.
        #
        # Default to 3 batches (matching Detect task style), but can be overridden via `max_plot_batches`.
        # Additionally cap samples per batch (default=1) via `max_plot_samples`.
        max_plot_batches = getattr(self.args, "max_plot_batches", 3)
        if self.args.plots and hasattr(self, "batch_i") and self.batch_i < max_plot_batches and RANK in {-1, 0}:
            self.plot_validation_samples(batch, preds, self.batch_i)

    def get_desc(self) -> str:
        """Return a formatted string summarizing validation metrics header for progress bar."""
        return ("%22s" + "%11s" * 8) % (
            "",
            "Images",
            "Instances",
            "AP3D@.5(E)",
            "AP3D@.5(M)",
            "AP3D@.5(H)",
            "AP3D@.7(E)",
            "AP3D@.7(M)",
            "AP3D@.7(H)",
        )

    def finalize_metrics(self) -> None:
        """Finalize metrics computation."""
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir
        self.det_metrics.speed = self.speed
        self.det_metrics.save_dir = self.save_dir

    def get_stats(self) -> dict[str, Any]:
        """Calculate and return metrics statistics.

        Returns:
            Dictionary containing metrics results.
        """
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        try:
            self.det_metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        except (KeyError, IndexError):
            # 2D det metrics are auxiliary — plotting can fail when model nc != dataset nc
            self.det_metrics.process(save_dir=self.save_dir, plot=False)
        # Merge so training logs/CSV show both 3D and 2D bbox metrics.
        # 3D metrics go LAST so fitness=AP3D@0.5 (not 2D mAP) drives best-model selection.
        # 3D metrics FIRST to preserve CSV column order, then 2D metrics appended.
        # Pop 2D fitness so 3D fitness (AP3D@0.5) drives best-model selection.
        det_results = self.det_metrics.results_dict
        det_results.pop("fitness", None)
        return {**self.metrics.results_dict, **det_results}

    def plot_validation_samples(
        self,
        batch: dict[str, Any],
        pred_boxes3d: list[list[Box3D]],
        batch_idx: int,
    ) -> None:
        """Generate and save validation visualization images with 3D bounding boxes in up-down layout.

        Creates simple up-down layout: predictions on top, ground truth on bottom.
        Uses original image sizes (not resized/letterboxed).

        Args:
            batch: Batch dictionary containing images, labels, and calibration data.
            pred_boxes3d: List of predicted Box3D lists (one per image).
            batch_idx: Batch index for file naming.
        """
        if not self.args.plots:
            return

        import cv2

        labels_list = batch.get("labels", [])
        calibs = batch.get("calib", [])
        im_files = batch.get("im_file", [])

        if not im_files:
            LOGGER.warning("No image files in batch for visualization")
            return

        batch_size = len(im_files)
        max_samples = getattr(self.args, "max_plot_samples", 4)
        num_samples = min(batch_size, max_samples)

        for si in range(num_samples):
            im_file = im_files[si] if si < len(im_files) else None
            if not im_file:
                continue

            # Load original images from file paths
            left_path = Path(im_file)
            if not left_path.exists():
                LOGGER.debug("Left image not found: %s, skipping visualization", left_path)
                continue

            # Get right image path (same filename, different directory)
            # im_file format: images/{split}/left/{image_id}.png
            # right path: images/{split}/right/{image_id}.png
            right_path = left_path.parent.parent / "right" / left_path.name
            if not right_path.exists():
                LOGGER.debug("Right image not found: %s, skipping visualization", right_path)
                continue

            # Load original images (BGR format from OpenCV)
            left_img = cv2.imread(str(left_path))
            right_img = cv2.imread(str(right_path))

            if left_img is None or right_img is None:
                LOGGER.debug("Failed to load images for %s, skipping", left_path)
                continue

            # Get predictions and ground truth for this sample
            pred_boxes = pred_boxes3d[si] if si < len(pred_boxes3d) else []
            labels = labels_list[si] if si < len(labels_list) else []
            calib = calibs[si] if si < len(calibs) and isinstance(calibs[si], dict) else None

            # Skip visualization if no calibration available
            if calib is None:
                continue

            # Get actual image dimensions and compute letterbox parameters
            actual_h, actual_w = left_img.shape[:2]
            imgsz = getattr(self.args, "imgsz", 384)

            letterbox_scale, pad_left, pad_top = compute_letterbox_params(actual_h, actual_w, imgsz)
            calib_orig = _reverse_letterbox_calib(calib, letterbox_scale, pad_left, pad_top, actual_w, actual_h)

            # Convert labels to Box3D for ground truth
            gt_boxes = (
                [b for lab in labels if (b := Box3D.from_label(lab, calib_orig, class_names=self.names)) is not None]
                if labels
                else []
            )

            # Filter out predictions with confidence == 0 or below threshold before visualization
            if pred_boxes:
                conf_threshold = self.args.conf
                pred_boxes = [
                    box for box in pred_boxes if hasattr(box, "confidence") and box.confidence > conf_threshold
                ]

            # Generate visualization with predictions only (top image)
            _, _, combined_pred = plot_stereo3d_boxes(
                left_img=left_img.copy(),
                right_img=right_img.copy(),
                pred_boxes3d=pred_boxes,
                gt_boxes3d=[],
                left_calib=calib_orig,
            )

            # Generate visualization with ground truth only (bottom image)
            _, _, combined_gt = plot_stereo3d_boxes(
                left_img=left_img.copy(),
                right_img=right_img.copy(),
                pred_boxes3d=[],
                gt_boxes3d=gt_boxes,
                left_calib=calib_orig,
            )
            # Stack vertically: predictions on top, ground truth on bottom
            h_pred, w_pred = combined_pred.shape[:2]
            h_gt, w_gt = combined_gt.shape[:2]

            # Ensure both images have the same width
            if w_pred != w_gt:
                target_w = max(w_pred, w_gt)
                if w_pred < target_w:
                    combined_pred = cv2.resize(
                        combined_pred,
                        (target_w, h_pred),
                        interpolation=cv2.INTER_LINEAR,
                    )
                if w_gt < target_w:
                    combined_gt = cv2.resize(
                        combined_gt,
                        (target_w, h_gt),
                        interpolation=cv2.INTER_LINEAR,
                    )

            # Stack vertically
            stacked = np.vstack([combined_pred, combined_gt])

            # Add labels
            label_height = 30
            stacked_with_labels = np.zeros(
                (stacked.shape[0] + label_height * 2, stacked.shape[1], 3),
                dtype=np.uint8,
            )
            stacked_with_labels[label_height : label_height + stacked.shape[0], :, :] = stacked

            # Add text labels
            cv2.putText(
                stacked_with_labels,
                "Predictions",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                stacked_with_labels,
                "Ground Truth",
                (10, label_height + stacked.shape[0] + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Save individual image (one file per sample)
            image_id = left_path.stem
            save_path = self.save_dir / f"val_batch{batch_idx}_sample{si}_{image_id}.jpg"
            cv2.imwrite(str(save_path), stacked_with_labels)
            if self.on_plot:
                self.on_plot(save_path)

    def print_results(self) -> None:
        """Print training/validation set metrics per class with KITTI difficulty splits."""
        if not self.metrics.stats:
            LOGGER.warning(f"no labels found in {self.args.task} set, can not compute metrics without labels")
            return

        # Count ground truth objects per class from raw stats
        nt_per_class = np.zeros(self.metrics.nc, dtype=int)
        nt_per_image = np.zeros(self.metrics.nc, dtype=int)
        for stat in self.metrics.stats:
            gt_classes = set()
            for box in stat.get("gt_boxes", []):
                if 0 <= box.class_id < self.metrics.nc:
                    nt_per_class[box.class_id] += 1
                    gt_classes.add(box.class_id)
            for cls_id in gt_classes:
                nt_per_image[cls_id] += 1

        total_gt = int(nt_per_class.sum())
        if total_gt == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, can not compute metrics without labels")
            return

        # Get per-difficulty AP values
        ap3d = self.metrics.ap3d

        def mean_ap_diff(iou_t, diff):
            """Mean AP across classes for given IoU and difficulty."""
            if not ap3d or iou_t not in ap3d:
                return 0.0
            d = ap3d[iou_t].get(diff, {})
            return float(np.mean(list(d.values()))) if d else 0.0

        # Print: class, images, instances, AP3D@0.5(E/M/H), AP3D@0.7(E/M/H)
        pf = "%22s" + "%11i" * 2 + "%11.3g" * 6
        LOGGER.info(
            pf
            % (
                "all",
                self.seen,
                total_gt,
                mean_ap_diff(0.5, DIFFICULTY_EASY),
                mean_ap_diff(0.5, DIFFICULTY_MODERATE),
                mean_ap_diff(0.5, DIFFICULTY_HARD),
                mean_ap_diff(0.7, DIFFICULTY_EASY),
                mean_ap_diff(0.7, DIFFICULTY_MODERATE),
                mean_ap_diff(0.7, DIFFICULTY_HARD),
            )
        )

        # Print results per class
        if self.args.verbose and not self.training and self.metrics.nc > 1 and ap3d:
            for class_id, class_name in self.metrics.names.items():
                if class_name.startswith("Aux_"):
                    continue
                nt_class = int(nt_per_class[class_id]) if class_id < len(nt_per_class) else 0
                nt_images = int(nt_per_image[class_id]) if class_id < len(nt_per_image) else 0

                def cls_ap(iou_t, diff):
                    if not ap3d or iou_t not in ap3d:
                        return 0.0
                    return ap3d[iou_t].get(diff, {}).get(class_id, 0.0)

                LOGGER.info(
                    pf
                    % (
                        class_name,
                        nt_images,
                        nt_class,
                        cls_ap(0.5, DIFFICULTY_EASY),
                        cls_ap(0.5, DIFFICULTY_MODERATE),
                        cls_ap(0.5, DIFFICULTY_HARD),
                        cls_ap(0.7, DIFFICULTY_EASY),
                        cls_ap(0.7, DIFFICULTY_MODERATE),
                        cls_ap(0.7, DIFFICULTY_HARD),
                    )
                )

    def _format_progress_metrics(self) -> str:
        """Format current metrics for progress bar display."""
        if not hasattr(self.metrics, "stats") or len(self.metrics.stats) == 0:
            return ("%11i" + "%11s" * 6) % (int(self.seen), "-", "-", "-", "-", "-", "-")

        saved_stats = self.metrics.stats.copy()
        self.metrics.process(save_dir=self.save_dir, plot=False)
        ap3d = self.metrics.ap3d
        self.metrics.stats = saved_stats

        if not ap3d:
            return ("%11i" + "%11s" * 6) % (int(self.seen), "-", "-", "-", "-", "-", "-")

        def mean_ap(iou_t, diff):
            d = ap3d.get(iou_t, {}).get(diff, {})
            return float(np.mean(list(d.values()))) if d else 0.0

        return ("%11i" + "%11.4g" * 6) % (
            int(self.seen),
            mean_ap(0.5, DIFFICULTY_EASY),
            mean_ap(0.5, DIFFICULTY_MODERATE),
            mean_ap(0.5, DIFFICULTY_HARD),
            mean_ap(0.7, DIFFICULTY_EASY),
            mean_ap(0.7, DIFFICULTY_MODERATE),
            mean_ap(0.7, DIFFICULTY_HARD),
        )

    def build_dataset(
        self, img_path: str | dict[str, Any], mode: str = "val", batch: int | None = None
    ) -> torch.utils.data.Dataset:
        """Build Stereo3DDetDataset for validation.

        Args:
            img_path: Stereo descriptor dict from self.data[split], or a dataset root path string.
            mode: 'train' or 'val' mode.
            batch: Batch size (unused, kept for compatibility).

        Returns:
            Stereo3DDetDataset: Dataset instance for validation.
        """
        from ultralytics.models.yolo.s3d.dataset import Stereo3DDetDataset

        # Resolve descriptor: prefer the dict passed in, fall back to self.data[mode]
        desc = img_path if isinstance(img_path, dict) else self.data.get(mode)
        if not isinstance(desc, dict) or desc.get("type") != "kitti_stereo":
            raise ValueError(
                f"Cannot build stereo dataset from img_path={img_path} (type: {type(img_path)}). "
                f"Expected a descriptor dict with type='kitti_stereo'."
            )

        imgsz = getattr(self.args, "imgsz", 384)
        if isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
            imgsz = (int(imgsz[0]), int(imgsz[1]))
        elif isinstance(imgsz, (list, tuple)):
            imgsz = (int(imgsz[0]), int(imgsz[0]))
        else:
            imgsz = (int(imgsz), int(imgsz))

        return Stereo3DDetDataset(
            root=str(desc["root"]),
            split=str(desc.get("split", mode)),
            imgsz=imgsz,
            names=self.data.get("names"),
            mean_dims=self.data.get("mean_dims"),
            std_dims=self.data.get("std_dims"),
            augment=False,
            filter_occluded=False,
        )

    def get_dataloader(self, dataset_path: str | dict[str, Any], batch_size: int) -> torch.utils.data.DataLoader:
        """Construct and return dataloader for validation.

        Args:
            dataset_path: Path to the dataset, or a descriptor dict from self.data.get(split).
            batch_size: Size of each batch.

        Returns:
            torch.utils.data.DataLoader: Dataloader for validation.
        """
        from ultralytics.data import build_dataloader

        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")

        # build_dataloader automatically uses dataset.collate_fn if available
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers,
            shuffle=False,  # No shuffling for validation
            rank=-1,  # Single GPU validation
            drop_last=False,  # Don't drop last batch in validation
            pin_memory=self.device.type == "cuda" if hasattr(self, "device") else True,
        )
