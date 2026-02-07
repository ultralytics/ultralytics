# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.data.stereo.box3d import Box3D
from ultralytics.data.stereo.calib import CalibrationParameters
from ultralytics.models.yolo.stereo3ddet.preprocess import (
    preprocess_stereo_batch,
    compute_letterbox_params,
    decode_and_refine_predictions,
    clear_config_cache,
)
from ultralytics.models.yolo.stereo3ddet.metrics import Stereo3DDetMetrics
from ultralytics.utils import LOGGER, RANK, YAML
from ultralytics.utils.metrics import DetMetrics, box_iou, compute_3d_iou
from ultralytics.utils.plotting import plot_stereo3d_boxes
from ultralytics.utils.profiling import profile_function, profile_section
from ultralytics.engine.validator import BaseValidator




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


@profile_function(name="compute_3d_iou_batch")
def compute_3d_iou_batch(
    pred_boxes: list[Box3D],
    gt_boxes: list[Box3D],
    eps: float = 1e-7,
) -> np.ndarray:
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
            try:
                iou_matrix[i, j] = compute_3d_iou(pred_boxes[i], gt_boxes[j], eps=eps)
            except Exception:
                iou_matrix[i, j] = 0.0

    return iou_matrix


def _labels_to_box3d_list(
    labels: list[dict[str, Any]],
    calib: dict[str, float] | None = None,
    names: dict[int, str] | None = None,
    **kwargs,
) -> list[Box3D]:
    """Convert label dictionaries to Box3D objects.

    Delegates to Box3D.from_label() for 3D reconstruction.

    Args:
        labels: List of label dictionaries from dataset.
        calib: Calibration parameters dict.
        names: Class names mapping {class_id: class_name}.
        **kwargs: Ignored (kept for backward compatibility with callers passing letterbox params).

    Returns:
        List of Box3D objects.
    """
    if names is None:
        raise ValueError("class_names mapping must be provided")

    # Convert calib to dict if CalibrationParameters
    calib_dict = calib.to_dict() if hasattr(calib, "to_dict") else calib

    # Determine image_hw for disparity fallback
    if calib is not None:
        image_hw = (int(calib.get("image_height", 375)), int(calib.get("image_width", 1242)))
    else:
        image_hw = (375, 1242)

    return [b for lab in labels if (b := Box3D.from_label(lab, calib_dict, class_names=names, image_hw=image_hw)) is not None]


class Stereo3DDetValidator(BaseValidator):
    """Stereo 3D Detection Validator.

    Extends BaseValidator to implement 3D detection validation with AP3D metrics.
    Computes 3D IoU, matches predictions to ground truth, and calculates AP3D at IoU 0.5 and 0.7.
    """

    def __init__(
        self, dataloader=None, save_dir=None, args=None, _callbacks=None
    ) -> None:
        """Initialize Stereo3DDetValidator.

        Args:
            dataloader: DataLoader for validation data.
            save_dir: Directory to save results.
            args: Configuration arguments.
            _callbacks: Callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "stereo3ddet"
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
        """Parse stereo dataset YAML and return metadata for KITTIStereoDataset.

        This overrides the base implementation to avoid the default YOLO detection dataset checks
        and instead wire up paths/splits intended for the custom `KITTIStereoDataset` loader.

        Returns:
            dict: Dataset dictionary with fields used by the validator and model.
        """
        # Load YAML if a path is provided; accept dicts directly
        data_cfg = self.args.data
        if isinstance(data_cfg, (str, Path)):
            data_cfg = YAML.load(str(data_cfg))

        if not isinstance(data_cfg, dict):
            raise RuntimeError("stereo3ddet: data must be a YAML path or dict")

        # Validate channels for stereo (must be 6 = left RGB + right RGB)
        channels = data_cfg.get("channels", 6)
        if channels != 6:
            raise ValueError(
                f"Stereo3DDet requires 6 input channels (left + right RGB), "
                f"but dataset config has channels={channels}. "
                f"Please set 'channels: 6' in your dataset YAML."
            )

        # Root path and splits
        root_path = data_cfg.get("path") or "."
        root = Path(str(root_path)).resolve()
        # Accept either directory-style train/val or txt; KITTIStereoDataset uses split names
        train_split = data_cfg.get("train_split", "train")
        val_split = data_cfg.get("val_split", "val")

        # Names/nc - must be provided by dataset configuration
        names = data_cfg.get("names")
        if names is None:
            raise ValueError("Dataset configuration must include 'names' mapping")
        nc = data_cfg.get("nc", len(names))

        # Mean dimensions per class (for dimension decoding)
        mean_dims = data_cfg.get("mean_dims")

        # Standard deviation of dimensions per class (for normalized offset decoding)
        std_dims = data_cfg.get("std_dims")

        # Return a dict compatible with BaseValidator expectations, plus stereo descriptors
        return {
            "yaml_file": (
                str(self.args.data) if isinstance(self.args.data, (str, Path)) else None
            ),
            "path": str(root),
            # Channels for model input (6 = left+right stacked)
            "channels": 6,
            # Signal to our get_dataloader/build_dataset that this is a stereo dataset
            "train": {"type": "kitti_stereo", "root": str(root), "split": train_split},
            "val": {"type": "kitti_stereo", "root": str(root), "split": val_split},
            "names": names,
            "nc": nc,
            # carry over optional stereo metadata if present
            "stereo": data_cfg.get("stereo", True),
            "image_size": data_cfg.get("image_size", [375, 1242]),
            "baseline": data_cfg.get("baseline"),
            "focal_length": data_cfg.get("focal_length"),
            "mean_dims": mean_dims,
            "std_dims": std_dims,
        }

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Normalize 6-channel images to float [0,1] and move targets to device.

        Uses shared preprocessing from preprocess.py for consistency with trainer.
        Also stores the current batch for postprocess() access.
        """
        batch = preprocess_stereo_batch(batch, self.device, half=self.args.half)
        # Make the *current* batch available to postprocess(); BaseValidator calls postprocess() before update_metrics().
        self._current_batch = batch
        return batch

    def postprocess(self, preds: dict[str, torch.Tensor]) -> list[list[Box3D]]:
        """Postprocess model outputs to Box3D objects.

        Uses shared decode_and_refine_predictions from preprocess.py which handles
        decoding, geometric construction, and dense alignment refinement.

        Args:
            preds: Dictionary of 10-branch model outputs.

        Returns:
            List of Box3D lists (one per batch item).
        """
        # Get batch for calibration and images
        batch = self._current_batch if hasattr(self, "_current_batch") else None

        # Get use_geometric and use_dense_alignment from args
        use_geometric = getattr(self.args, "use_geometric", None)
        use_dense_alignment = getattr(self.args, "use_dense_alignment", None)

        return decode_and_refine_predictions(
            preds=preds,
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
        )

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics with model information.

        Args:
            model: Model being validated.
        """
        # Clear config cache at start of validation to reload fresh configs
        clear_config_cache()

        # Get class names from dataset, not from metrics results_dict (which contains metric keys, not class names)
        # Get class names from dataset configuration - names must be provided
        if hasattr(self, "data") and self.data and "names" in self.data:
            self.names = self.data["names"]
        elif hasattr(model, "names") and model.names:
            self.names = model.names
        else:
            raise ValueError("Dataset configuration must include 'names' mapping")

        self.nc = (
            len(self.names)
            if isinstance(self.names, dict)
            else len(self.names) if isinstance(self.names, (list, tuple)) else 0
        )
        self.seen = 0
        self.metrics.names = self.names
        self.metrics.nc = (
            self.nc
        )  # Also update metrics.nc to match the correct number of classes

        # Create reverse mapping from class name to class ID
        # self.names is {0: "Car", 1: "Van", ...} so we need {"Car": 0, "Van": 1, ...}
        name_to_id = {}
        if isinstance(self.names, dict):
            name_to_id = {name: idx for idx, name in self.names.items()}
        elif isinstance(self.names, (list, tuple)):
            name_to_id = {name: idx for idx, name in enumerate(self.names)}

        # Parse and convert mean_dims from YAML format (class name -> [L, W, H]) to (class ID -> (H, W, L))
        mean_dims_raw = (
            self.data.get("mean_dims") if hasattr(self, "data") and self.data else None
        )
        if mean_dims_raw is not None:
            # Convert from {class_name or class_id: [L, W, H]} to {class_id: (H, W, L)}
            self.mean_dims = {}
            for class_key, dims in mean_dims_raw.items():
                if isinstance(dims, (list, tuple)) and len(dims) == 3:
                    l, w, h = dims  # YAML has [L, W, H]
                    # Convert class name to class ID if needed
                    if isinstance(class_key, str) and class_key in name_to_id:
                        class_id = name_to_id[class_key]
                    elif isinstance(class_key, int):
                        class_id = class_key
                    else:
                        LOGGER.warning("Unknown class key in mean_dims: %s", class_key)
                        continue
                    self.mean_dims[class_id] = (h, w, l)  # Store as (H, W, L)
        else:
            self.mean_dims = None
            LOGGER.info("No mean_dims in dataset config, will use defaults")

        # Parse and convert std_dims from YAML format (class name -> [L, W, H]) to (class ID -> (H, W, L))
        std_dims_raw = (
            self.data.get("std_dims") if hasattr(self, "data") and self.data else None
        )
        if std_dims_raw is not None:
            # Convert from {class_name: [L, W, H]} to {class_id: (H, W, L)}
            self.std_dims = {}
            for class_key, dims in std_dims_raw.items():
                if isinstance(dims, (list, tuple)) and len(dims) == 3:
                    l, w, h = dims  # YAML has [L, W, H]
                    # Convert class name to class ID if needed
                    if isinstance(class_key, str) and class_key in name_to_id:
                        class_id = name_to_id[class_key]
                    elif isinstance(class_key, int):
                        class_id = class_key
                    else:
                        LOGGER.warning("Unknown class key in std_dims: %s", class_key)
                        continue
                    self.std_dims[class_id] = (h, w, l)  # Store as (H, W, L)
            LOGGER.info("Loaded std_dims with %d classes: %s", len(self.std_dims), list(self.std_dims.keys()))
        else:
            self.std_dims = None
            LOGGER.info("No std_dims in dataset config, will use defaults")

        # Init 2D detection metrics (bbox mAP)
        self.det_metrics.names = self.names
        self.det_metrics.clear_stats()

    def update_metrics(self, preds: list[list[Box3D]], batch: dict[str, Any]) -> None:
        """Update metrics with predictions and ground truth.

        Args:
            preds: List of predicted Box3D lists (one per image).
            batch: Batch containing ground truth labels.
        """
        with profile_section("update_metrics"):
            self._current_batch = batch  # Store for calibration access

            labels_list = batch.get("labels", [])
            calibs = batch.get("calib", [])
            ori_shapes = batch.get("ori_shape", [])

            for si, (pred_boxes, labels) in enumerate(zip(preds, labels_list)):
                self.seen += 1

                # Get calibration for this sample
                calib = (
                    calibs[si]
                    if si < len(calibs) and isinstance(calibs[si], dict)
                    else None
                )

                # Convert labels to Box3D - need to reverse letterbox transformation on calibration
                # Labels use original image normalized coordinates, but calib is letterboxed
                if calib is not None and si < len(ori_shapes):
                    ori_shape = ori_shapes[si]
                    if isinstance(ori_shape, (list, tuple)) and len(ori_shape) >= 2:
                        actual_h, actual_w = ori_shape[0], ori_shape[1]
                        imgsz = getattr(self.args, "imgsz", 384)

                        letterbox_scale, pad_left, pad_top = compute_letterbox_params(
                            actual_h, actual_w, imgsz
                        )
                        calib = _reverse_letterbox_calib(
                            calib, letterbox_scale, pad_left, pad_top, actual_w, actual_h
                        )
                        in_h, in_w = (imgsz, imgsz) if isinstance(imgsz, int) else (int(imgsz[0]), int(imgsz[1]))

                # Convert labels to Box3D - passing letterbox parameters for bbox_2d conversion
                try:
                    gt_boxes = _labels_to_box3d_list(
                        labels,
                        calib,
                        names=self.names,
                        letterbox_scale=letterbox_scale,
                        pad_left=pad_left,
                        pad_top=pad_top,
                        in_h=in_h,
                        in_w=in_w,
                    )
                except Exception as e:
                    LOGGER.warning(
                        "Error converting labels to Box3D (sample %d): %s", si, e
                    )
                    gt_boxes = []

                # ------------------------------------------------------------
                # 2D bbox metrics (original-image xyxy)
                # ------------------------------------------------------------
                try:
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
                    if (
                        si < len(ori_shapes)
                        and isinstance(ori_shapes[si], (list, tuple))
                        and len(ori_shapes[si]) >= 2
                    ):
                        ori_h, ori_w = int(ori_shapes[si][0]), int(ori_shapes[si][1])
                    else:
                        ori_h, ori_w = 375, 1242
                    letterbox_scale, pad_left, pad_top = compute_letterbox_params(
                        ori_h, ori_w, imgsz
                    )
                    if isinstance(imgsz, int):
                        in_h, in_w = imgsz, imgsz
                    else:
                        in_h, in_w = int(imgsz[0]), int(imgsz[1])

                    for lab in labels:
                        lb = lab.get("left_box", None)
                        if lb is None:
                            continue
                        cls_i = int(lab.get("class_id", 0))
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
                            torch.tensor(gt_cls2d, dtype=torch.int64)
                            if gt_cls2d
                            else torch.zeros((0,), dtype=torch.int64)
                        )

                        if gt_boxes_t.shape[0] == 0:
                            tp2d = np.zeros((n_pred, self.det_iouv.numel()), dtype=bool)
                        else:
                            iou2d = box_iou(gt_boxes_t, pred_boxes_t).T  # NxM
                            # Use BaseValidator matching but with det_iouv.
                            # IMPORTANT: guard with try/finally so self.iouv is always restored (prevents leaking 10 IoUs
                            # into the 3D metrics path where tp/fp are shape (N, 2)).
                            old_iouv = self.iouv
                            try:
                                self.iouv = self.det_iouv
                                correct = (
                                    self.match_predictions(pred_cls_t, gt_cls_t, iou2d)
                                    .cpu()
                                    .numpy()
                                )
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
                        }
                    )
                except Exception as e:
                    LOGGER.debug("bbox metrics update failed (sample %d): %s", si, e)

                # Handle empty predictions or ground truth
                if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                    continue

                # Compute 3D IoU matrix using vectorized batch computation
                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    # Match predictions to ground truth using 3D IoU (vectorized)
                    try:
                        iou_matrix = compute_3d_iou_batch(pred_boxes, gt_boxes)
                    except Exception as e:
                        LOGGER.warning(
                            "Error computing 3D IoU batch: %s, falling back to individual computation",
                            e,
                        )
                        # Fallback to individual computation if batch fails
                        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
                        for i, pred_box in enumerate(pred_boxes):
                            for j, gt_box in enumerate(gt_boxes):
                                if pred_box.class_id == gt_box.class_id:
                                    try:
                                        iou = compute_3d_iou(pred_box, gt_box)
                                        iou_matrix[i, j] = iou
                                    except Exception as e2:
                                        LOGGER.warning("Error computing 3D IoU: %s", e2)
                                        iou_matrix[i, j] = 0.0

                    # Match predictions to ground truth (greedy matching)
                    matched_gt = set()
                    tp = np.zeros((len(pred_boxes), self.niou), dtype=bool)
                    fp = np.zeros((len(pred_boxes), self.niou), dtype=bool)

                    # Sort predictions by confidence
                    pred_indices = sorted(
                        range(len(pred_boxes)),
                        key=lambda i: pred_boxes[i].confidence,
                        reverse=True,
                    )

                    for pred_idx in pred_indices:
                        pred_box = pred_boxes[pred_idx]
                        best_iou = 0.0
                        best_gt_idx = -1

                        # Find best matching ground truth
                        for gt_idx, gt_box in enumerate(gt_boxes):
                            if gt_idx in matched_gt:
                                continue
                            if pred_box.class_id != gt_box.class_id:
                                continue
                            if iou_matrix[pred_idx, gt_idx] > best_iou:
                                best_iou = iou_matrix[pred_idx, gt_idx]
                                best_gt_idx = gt_idx

                        # Check if match exceeds IoU thresholds
                        for iou_idx, iou_thresh in enumerate(self.iouv):
                            if best_iou >= iou_thresh.item():
                                tp[pred_idx, iou_idx] = True
                                if best_gt_idx >= 0:
                                    matched_gt.add(best_gt_idx)
                            else:
                                fp[pred_idx, iou_idx] = True

                else:
                    # No matches possible
                    tp = np.zeros((len(pred_boxes), self.niou), dtype=bool)
                    fp = (
                        np.ones((len(pred_boxes), self.niou), dtype=bool)
                        if len(pred_boxes) > 0
                        else np.zeros((0, self.niou), dtype=bool)
                    )

                # Extract statistics
                conf = (
                    np.array([box.confidence for box in pred_boxes])
                    if pred_boxes
                    else np.array([])
                )
                pred_cls = (
                    np.array([box.class_id for box in pred_boxes])
                    if pred_boxes
                    else np.array([], dtype=int)
                )
                target_cls = (
                    np.array([box.class_id for box in gt_boxes])
                    if gt_boxes
                    else np.array([], dtype=int)
                )

                # DIAGNOSTIC START
                # self._diagnostic_log_statistics_extraction(conf, pred_cls, target_cls, si)
                # DIAGNOSTIC END

                # Update metrics
                self.metrics.update_stats(
                    {
                        "tp": tp,
                        "fp": fp,
                        "conf": conf,
                        "pred_cls": pred_cls,
                        "target_cls": target_cls,
                        "boxes3d_pred": pred_boxes,
                        "boxes3d_target": gt_boxes,
                    }
                )

                # DIAGNOSTIC START
                # self.metrics._diagnostic_log_statistics_accumulation(self.metrics.stats, self.batch_i if hasattr(self, 'batch_i') else 0)
                # DIAGNOSTIC END

                # Update progress bar with intermediate metrics (every batch for real-time feedback)
            if (
                hasattr(self, "_progress_bar")
                and self._progress_bar is not None
                and RANK in {-1, 0}
            ):
                # Update progress bar periodically to avoid performance impact
                if hasattr(self, "_batch_count"):
                    self._batch_count += 1
                else:
                    self._batch_count = 1

                # Update every 5 batches or if we're near the end (more frequent than before)
                if self._batch_count % 5 == 0 or (
                    hasattr(self, "_total_batches")
                    and self._batch_count >= self._total_batches - 1
                ):
                    try:
                        metrics_str = self._format_progress_metrics()
                        if metrics_str:
                            self._progress_bar.set_description(metrics_str)
                    except Exception as e:
                        LOGGER.debug("Error updating progress bar: %s", e)

            # Generate visualization images if plots enabled.
            # NOTE: This stereo validator saves 1 file per sample, so keep defaults conservative to avoid generating
            # thousands of images when using large validation batch sizes.
            #
            # Default to 3 batches (matching Detect task style), but can be overridden via `max_plot_batches`.
            # Additionally cap samples per batch (default=1) via `max_plot_samples`.
            max_plot_batches = getattr(self.args, "max_plot_batches", 3)
            if (
                self.args.plots
                and hasattr(self, "batch_i")
                and self.batch_i < max_plot_batches
                and RANK in {-1, 0}
            ):
                try:
                    self.plot_validation_samples(batch, preds, self.batch_i)
                except Exception as e:
                    LOGGER.warning("Error generating validation visualizations: %s", e)

    def get_desc(self) -> str:
        """Return a formatted string summarizing validation metrics header for progress bar.

        Returns:
            Formatted header string matching the progress bar format.
        """
        # Format: class name (22 chars), then Images (11 chars), Instances (11 chars), then 6 metric columns (11 chars each)
        # This matches the data row format: "%22s" + "%11i" * 2 + "%11.3g" * 6
        return ("%22s" + "%11s" * 8) % (
            "",  # Empty class name column (22 chars)
            "Images".rjust(11),
            "Instances".rjust(11),
            "AP3D@0.5".rjust(11),
            "AP3D@0.7".rjust(11),
            "Precision".rjust(11),
            "Recall".rjust(11),
            "mAP50".rjust(11),
            "mAP50-95".rjust(11),
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
        self.metrics.process(
            save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot
        )
        self.det_metrics.process(
            save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot
        )
        # Merge so training logs/CSV show both 3D and 2D bbox metrics.
        return {**self.metrics.results_dict, **self.det_metrics.results_dict}

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

        try:
            import cv2

            labels_list = batch.get("labels", [])
            calibs = batch.get("calib", [])
            im_files = batch.get("im_file", [])

            if not im_files:
                LOGGER.warning("No image files in batch for visualization")
                return

            batch_size = len(im_files)
            # TEMPORARY: Modified to generate ALL validation samples for error analysis
            max_samples = getattr(self.args, "max_plot_samples", 4)
            num_samples = min(batch_size, max_samples)

            for si in range(num_samples):
                im_file = im_files[si] if si < len(im_files) else None
                if not im_file:
                    continue

                # Load original images from file paths
                left_path = Path(im_file)
                if not left_path.exists():
                    LOGGER.debug(
                        "Left image not found: %s, skipping visualization", left_path
                    )
                    continue

                # Get right image path (same filename, different directory)
                # im_file format: images/{split}/left/{image_id}.png
                # right path: images/{split}/right/{image_id}.png
                right_path = left_path.parent.parent / "right" / left_path.name
                if not right_path.exists():
                    LOGGER.debug(
                        "Right image not found: %s, skipping visualization", right_path
                    )
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
                calib = (
                    calibs[si]
                    if si < len(calibs) and isinstance(calibs[si], dict)
                    else None
                )

                # Skip visualization if no calibration available
                if calib is None:
                    continue

                # Get actual image dimensions and compute letterbox parameters
                actual_h, actual_w = left_img.shape[:2]
                imgsz = getattr(self.args, "imgsz", 384)

                letterbox_scale, pad_left, pad_top = compute_letterbox_params(
                    actual_h, actual_w, imgsz
                )
                calib_orig = _reverse_letterbox_calib(
                    calib, letterbox_scale, pad_left, pad_top, actual_w, actual_h
                )
                in_h, in_w = (imgsz, imgsz) if isinstance(imgsz, int) else (int(imgsz[0]), int(imgsz[1]))

                # Convert labels to Box3D for ground truth
                # Use original calibration, pass letterbox parameters for bbox_2d conversion
                gt_boxes = []
                if labels:
                    gt_boxes = _labels_to_box3d_list(
                        labels,
                        calib_orig,
                        names=self.names,
                        letterbox_scale=letterbox_scale,
                        pad_left=pad_left,
                        pad_top=pad_top,
                        in_h=in_h,
                        in_w=in_w,
                    )

                # Filter out predictions with confidence == 0 or below threshold before visualization
                if pred_boxes:
                    conf_threshold = self.args.conf
                    if conf_threshold < 0.1:
                        LOGGER.warning(
                            f"The prediction conf threshold is less than 0.1, you can set the conf through CLI."
                        )
                    pred_boxes = [
                        box
                        for box in pred_boxes
                        if hasattr(box, "confidence")
                        and box.confidence > conf_threshold
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
                # Use left image only for simplicity (or combine left+right horizontally first)
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
                stacked_with_labels[
                    label_height : label_height + stacked.shape[0], :, :
                ] = stacked

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
                save_path = (
                    self.save_dir / f"val_batch{batch_idx}_sample{si}_{image_id}.jpg"
                )
                cv2.imwrite(str(save_path), stacked_with_labels)
                if self.on_plot:
                    self.on_plot(save_path)

        except Exception as e:
            LOGGER.warning("Error in plot_validation_samples: %s", e)

    def print_results(self) -> None:
        """Print training/validation set metrics per class."""
        if not self.metrics.stats:
            LOGGER.warning(
                f"no labels found in {self.args.task} set, can not compute metrics without labels"
            )
            return

        # Count ground truth objects per class
        all_target_cls = (
            np.concatenate(
                [
                    s["target_cls"]
                    for s in self.metrics.stats
                    if len(s["target_cls"]) > 0
                ],
                axis=0,
            )
            if self.metrics.stats
            else np.array([], dtype=int)
        )
        if len(all_target_cls) == 0:
            LOGGER.warning(
                f"no labels found in {self.args.task} set, can not compute metrics without labels"
            )
            return

        nt_per_class = (
            np.bincount(all_target_cls.astype(int), minlength=self.metrics.nc)
            if len(all_target_cls) > 0
            else np.zeros(self.metrics.nc, dtype=int)
        )
        total_gt = int(nt_per_class.sum())

        # Compute images per class (how many images contain each class)
        # Use det_metrics if available (already computed), otherwise compute from stats
        if hasattr(self, "det_metrics") and hasattr(self.det_metrics, "nt_per_image"):
            nt_per_image = self.det_metrics.nt_per_image
        else:
            # Compute from stats: count unique images (by stat index) that contain each class
            nt_per_image = np.zeros(self.metrics.nc, dtype=int)
            for stat in self.metrics.stats:
                if len(stat.get("target_cls", [])) > 0:
                    unique_classes = np.unique(stat["target_cls"].astype(int))
                    for cls_id in unique_classes:
                        if 0 <= cls_id < self.metrics.nc:
                            nt_per_image[cls_id] += 1

        # Get mean metrics
        maps3d_50 = self.metrics.maps3d_50
        maps3d_70 = self.metrics.maps3d_70

        # Get precision and recall (flatten nested dicts to get mean values)
        precision_mean = 0.0
        recall_mean = 0.0
        if isinstance(self.metrics.precision, dict) and self.metrics.precision:
            all_precisions = []
            for iou_dict in self.metrics.precision.values():
                if isinstance(iou_dict, dict):
                    all_precisions.extend(
                        [v for v in iou_dict.values() if isinstance(v, (int, float))]
                    )
            precision_mean = float(np.mean(all_precisions)) if all_precisions else 0.0

        if isinstance(self.metrics.recall, dict) and self.metrics.recall:
            all_recalls = []
            for iou_dict in self.metrics.recall.values():
                if isinstance(iou_dict, dict):
                    all_recalls.extend(
                        [v for v in iou_dict.values() if isinstance(v, (int, float))]
                    )
            recall_mean = float(np.mean(all_recalls)) if all_recalls else 0.0

        # Get 2D bbox mAP50 and mAP50-95 metrics (for main summary line)
        box_map50 = 0.0
        box_map5095 = 0.0
        try:
            det_res = (
                self.det_metrics.results_dict if hasattr(self, "det_metrics") else {}
            )
            box_map50 = det_res.get("metrics/mAP50(B)", 0.0)
            box_map5095 = det_res.get("metrics/mAP50-95(B)", 0.0)
        except Exception as e:
            LOGGER.debug("Failed to get bbox2d metrics for main summary: %s", e)

        # Print format: class name, images, instances, AP3D@0.5, AP3D@0.7, precision, recall, mAP50, mAP50-95
        # Matches detect task format: "Class", "Images", "Instances", ...
        pf = "%22s" + "%11i" * 2 + "%11.3g" * 6
        LOGGER.info(
            pf
            % (
                "all",
                self.seen,
                total_gt,
                maps3d_50,
                maps3d_70,
                precision_mean,
                recall_mean,
                box_map50,
                box_map5095,
            )
        )

        # Print results per class if verbose and multiple classes
        if self.args.verbose and self.metrics.nc > 1 and self.metrics.ap3d_50:
            # Get per-class 2D bbox metrics for printing
            class_map50_dict = {}
            class_map5095_dict = {}
            try:
                if (
                    hasattr(self, "det_metrics")
                    and hasattr(self.det_metrics, "class_result")
                    and hasattr(self.det_metrics, "ap_class_index")
                ):
                    # Get per-class mAP50 and mAP50-95 from DetMetrics
                    # class_result(i) returns (p, r, map50, map) for index i in ap_class_index
                    # ap_class_index[i] gives the class_id at index i
                    for i, class_id in enumerate(self.det_metrics.ap_class_index):
                        try:
                            class_result = self.det_metrics.class_result(i)
                            if len(class_result) >= 4:
                                class_map50_dict[class_id] = float(
                                    class_result[2]
                                )  # mAP50
                                class_map5095_dict[class_id] = float(
                                    class_result[3]
                                )  # mAP50-95
                        except (IndexError, AttributeError) as e:
                            LOGGER.debug(
                                "Failed to get metrics for class %d at index %d: %s",
                                class_id,
                                i,
                                e,
                            )
                            continue
            except Exception as e:
                LOGGER.debug("Failed to get per-class bbox2d metrics: %s", e)

            for class_id, class_name in self.metrics.names.items():
                ap3d_50_class = self.metrics.ap3d_50.get(class_name, 0.0)
                ap3d_70_class = self.metrics.ap3d_70.get(class_name, 0.0)

                # Get class-specific precision and recall (average across IoU thresholds)
                prec_class = 0.0
                recall_class = 0.0
                if isinstance(self.metrics.precision, dict):
                    prec_values = []
                    for iou_dict in self.metrics.precision.values():
                        if isinstance(iou_dict, dict) and class_id in iou_dict:
                            prec_values.append(iou_dict[class_id])
                    prec_class = float(np.mean(prec_values)) if prec_values else 0.0

                if isinstance(self.metrics.recall, dict):
                    recall_values = []
                    for iou_dict in self.metrics.recall.values():
                        if isinstance(iou_dict, dict) and class_id in iou_dict:
                            recall_values.append(iou_dict[class_id])
                    recall_class = (
                        float(np.mean(recall_values)) if recall_values else 0.0
                    )

                # Get class-specific mAP50 and mAP50-95
                class_map50 = class_map50_dict.get(class_id, 0.0)
                class_map5095 = class_map5095_dict.get(class_id, 0.0)

                nt_class = (
                    int(nt_per_class[class_id]) if class_id < len(nt_per_class) else 0
                )
                nt_images = (
                    int(nt_per_image[class_id]) if class_id < len(nt_per_image) else 0
                )
                # Use same format as main summary: class name, images, instances, then metrics
                LOGGER.info(
                    pf
                    % (
                        class_name,
                        nt_images,  # number of images containing this class
                        nt_class,  # number of ground truth instances for this class
                        ap3d_50_class,
                        ap3d_70_class,
                        prec_class,
                        recall_class,
                        class_map50,
                        class_map5095,
                    )
                )

    def _format_progress_metrics(self) -> str:
        """Format current metrics for progress bar display.

        Returns:
            Formatted string with key metrics in training-style format.
        """
        if not hasattr(self.metrics, "stats") or len(self.metrics.stats) == 0:
            return ("%11i" + "%11s" * 6) % (
                int(self.seen),
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
            )

        # Compute intermediate metrics on accumulated stats
        try:
            # Save current stats
            saved_stats = self.metrics.stats.copy()
            # Process to get metrics
            temp_results = self.metrics.process(save_dir=self.save_dir, plot=False)
            # Also compute bbox metrics if available
            det_temp = (
                self.det_metrics.process(save_dir=self.save_dir, plot=False)
                if hasattr(self, "det_metrics")
                else {}
            )
            # Restore stats for final processing
            self.metrics.stats = saved_stats

            if not temp_results:
                return ("%11i" + "%11s" * 6) % (
                    int(self.seen),
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                )

            # Get AP3D metrics (use mean values)
            ap50 = temp_results.get("maps3d_50", 0.0)
            if isinstance(ap50, dict):
                ap50 = (
                    float(
                        np.mean(
                            [v for v in ap50.values() if isinstance(v, (int, float))]
                        )
                    )
                    if ap50
                    else 0.0
                )
            ap70 = temp_results.get("maps3d_70", 0.0)
            if isinstance(ap70, dict):
                ap70 = (
                    float(
                        np.mean(
                            [v for v in ap70.values() if isinstance(v, (int, float))]
                        )
                    )
                    if ap70
                    else 0.0
                )

            # Get precision and recall (flatten nested dicts)
            precision = temp_results.get("precision", 0.0)
            if isinstance(precision, dict):
                all_precisions = []
                for iou_dict in precision.values():
                    if isinstance(iou_dict, dict):
                        all_precisions.extend(
                            [
                                v
                                for v in iou_dict.values()
                                if isinstance(v, (int, float))
                            ]
                        )
                precision = float(np.mean(all_precisions)) if all_precisions else 0.0

            recall = temp_results.get("recall", 0.0)
            if isinstance(recall, dict):
                all_recalls = []
                for iou_dict in recall.values():
                    if isinstance(iou_dict, dict):
                        all_recalls.extend(
                            [
                                v
                                for v in iou_dict.values()
                                if isinstance(v, (int, float))
                            ]
                        )
                recall = float(np.mean(all_recalls)) if all_recalls else 0.0

            # Get bbox mAPs (DetMetrics uses 'metrics/mAP50(B)' keys)
            map50 = (
                det_temp.get("metrics/mAP50(B)", 0.0)
                if isinstance(det_temp, dict)
                else 0.0
            )
            map5095 = (
                det_temp.get("metrics/mAP50-95(B)", 0.0)
                if isinstance(det_temp, dict)
                else 0.0
            )

            # Format: Images, AP3D@0.5, AP3D@0.7, Precision, Recall, mAP50(B), mAP50-95(B)
            # Use same width format as training for consistency (matches get_desc header)
            # Use %11i for Images (integer count) and %11.4g for float metrics
            return ("%11i" + "%11.4g" * 6) % (
                int(self.seen),  # Images (integer)
                ap50,  # AP3D@0.5
                ap70,  # AP3D@0.7
                precision,  # Precision
                recall,  # Recall
                float(map50),
                float(map5095),
            )
        except Exception as e:
            LOGGER.debug("Error formatting progress metrics: %s", e)
            return ("%11i" + "%11s" * 6) % (
                int(self.seen),
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
            )

    def build_dataset(
        self,
        img_path: str | dict[str, Any],
        mode: str = "val",
        batch: int | None = None,
    ) -> torch.utils.data.Dataset:
        """Build Stereo3DDetDataset for validation.

        Args:
            img_path: Path to dataset root directory, or a descriptor dict from self.data.get(split).
            mode: 'train' or 'val' mode.
            batch: Batch size (unused, kept for compatibility).

        Returns:
            Stereo3DDetDataset: Dataset instance for validation.
        """
        from ultralytics.models.yolo.stereo3ddet.dataset import Stereo3DDetDataset

        # img_path should be a dir

        # means it's a file instead of the path, return it's parent directory
        if isinstance(img_path, str) and not os.path.isdir(img_path):
            img_path = Path(img_path).parent

        # Handle descriptor dict from self.data.get(self.args.split)
        desc = (
            img_path
            if isinstance(img_path, dict)
            else self.data.get(mode) if hasattr(self, "data") else None
        )

        if isinstance(desc, dict) and desc.get("type") == "kitti_stereo":
            # Get image size from args, default to 384
            imgsz = getattr(self.args, "imgsz", 384)
            if isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
                imgsz = (int(imgsz[0]), int(imgsz[1]))  # (H, W)
            elif isinstance(imgsz, (list, tuple)):
                imgsz = (int(imgsz[0]), int(imgsz[0]))  # Fallback to square
            else:
                imgsz = (int(imgsz), int(imgsz))  # Int to square

            # Get max_samples from args if available (for profiling/testing)
            max_samples = getattr(self.args, "max_samples", None)

            # Compute output_size from imgsz with default stride (8x for P3)
            # This can be overridden if model is available later, but default works for most cases
            output_size = None  # Will use dataset default (imgsz // 8)

            # Get mean_dims from dataset config if available
            mean_dims = self.data.get("mean_dims") if hasattr(self, "data") else None
            std_dims = self.data.get("std_dims") if hasattr(self, "data") else None

            return Stereo3DDetDataset(
                root=str(desc.get("root", ".")),
                split=str(desc.get("split", mode)),
                imgsz=imgsz,
                names=self.data.get("names") if hasattr(self, "data") else None,
                max_samples=max_samples,
                output_size=output_size,
                mean_dims=mean_dims,
                std_dims=std_dims,
                augment=False,
            )

        # Fallback: if img_path is a string, try to use it directly
        if isinstance(img_path, str) or isinstance(img_path, Path):
            imgsz = getattr(self.args, "imgsz", 384)
            if isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
                imgsz = (int(imgsz[0]), int(imgsz[1]))  # (H, W)
            elif isinstance(imgsz, (list, tuple)):
                imgsz = (int(imgsz[0]), int(imgsz[0]))  # Fallback to square
            else:
                imgsz = (int(imgsz), int(imgsz))  # Int to square

            return Stereo3DDetDataset(
                root=img_path,
                split=mode,
                imgsz=imgsz,
                names=self.data.get("names") if hasattr(self, "data") else None,
                output_size=None,  # Will use dataset default
                mean_dims=self.data.get("mean_dims") if hasattr(self, "data") else None,
                std_dims=self.data.get("std_dims") if hasattr(self, "data") else None,
                augment=False,
            )

        # If we can't determine the dataset, raise an error
        raise ValueError(
            f"Cannot build dataset from img_path={img_path} (type: {type(img_path)}). "
            f"Expected a string path or a descriptor dict with type='kitti_stereo'."
        )

    def get_dataloader(
        self, dataset_path: str | dict[str, Any], batch_size: int
    ) -> torch.utils.data.DataLoader:
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
