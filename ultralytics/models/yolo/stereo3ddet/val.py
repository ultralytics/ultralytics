# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.data.stereo.box3d import Box3D
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.stereo3ddet.metrics import Stereo3DDetMetrics
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.metrics import compute_3d_iou


def decode_stereo3d_outputs(
    outputs: dict[str, torch.Tensor],
    conf_threshold: float = 0.25,
    top_k: int = 100,
    calib: dict[str, float] | None = None,
) -> list[Box3D]:
    """Decode 10-branch model outputs to 3D bounding boxes.

    Decodes Stereo CenterNet 10-branch outputs following the paper methodology:
    1. Extract top-k detections from heatmap
    2. Apply offset for sub-pixel center refinement
    3. Compute depth from lr_distance using calibration parameters
    4. Decode 3D dimensions from offsets + class means
    5. Decode orientation from Multi-Bin representation
    6. Construct Box3D objects with all attributes

    Args:
        outputs: Dictionary with 10 branch outputs:
            - heatmap: [B, C, H/4, W/4] - center point heatmap
            - offset: [B, 2, H/4, W/4] - sub-pixel offset (δx, δy)
            - bbox_size: [B, 2, H/4, W/4] - 2D box size (w, h)
            - lr_distance: [B, 1, H/4, W/4] - left-right center distance d
            - right_width: [B, 1, H/4, W/4] - right box width wr
            - dimensions: [B, 3, H/4, W/4] - 3D dimension offsets (ΔH, ΔW, ΔL)
            - orientation: [B, 8, H/4, W/4] - Multi-Bin orientation encoding
            - vertices: [B, 8, H/4, W/4] - bottom 4 vertex coordinates
            - vertex_offset: [B, 8, H/4, W/4] - vertex sub-pixel offsets
            - vertex_dist: [B, 4, H/4, W/4] - center to vertex distances
        conf_threshold: Confidence threshold for filtering detections.
        top_k: Maximum number of detections to extract.
        calib: Calibration parameters dict with keys: fx, fy, cx, cy, baseline.

    Returns:
        List of Box3D objects with decoded 3D bounding boxes.

    References:
        Stereo CenterNet paper: Section 3.2 (Decoding)
    """
    # Class names and mean dimensions (KITTI dataset statistics)
    class_names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
    # Mean dimensions: (height, width, length) in meters
    mean_dims = {
        0: (1.52, 1.73, 3.89),  # Car
        1: (1.73, 0.50, 0.80),  # Pedestrian
        2: (1.77, 0.60, 1.76),  # Cyclist
    }

    boxes3d = []
    batch_size = outputs["heatmap"].shape[0]

    for b in range(batch_size):
        heatmap = outputs["heatmap"][b]  # [C, H, W]
        offset = outputs["offset"][b]  # [2, H, W]
        bbox_size = outputs["bbox_size"][b]  # [2, H, W]
        lr_distance = outputs["lr_distance"][b]  # [1, H, W]
        dimensions = outputs["dimensions"][b]  # [3, H, W]
        orientation = outputs["orientation"][b]  # [8, H, W]

        num_classes, h, w = heatmap.shape

        # Flatten heatmap and get top-k detections
        heatmap_flat = heatmap.reshape(num_classes, -1)  # [C, H*W]
        scores, indices = torch.topk(heatmap_flat, k=min(top_k, heatmap_flat.numel()), dim=1)

        for c in range(num_classes):
            class_scores = scores[c]
            class_indices = indices[c]

            for score, idx in zip(class_scores, class_indices):
                # Clamp confidence to [0, 1] range (heatmap values may be unnormalized)
                # Apply sigmoid if score is likely unnormalized (outside [0, 1]), otherwise clamp
                if score < 0 or score > 1:
                    # Apply sigmoid to normalize unnormalized scores
                    score_normalized = torch.sigmoid(score)
                else:
                    score_normalized = score
                confidence = float(torch.clamp(score_normalized, 0.0, 1.0).item())
                
                if confidence < conf_threshold:
                    continue

                # Convert flat index to (y, x) coordinates
                y_idx = idx // w
                x_idx = idx % w

                # Get sub-pixel offset
                dx = offset[0, y_idx, x_idx].item()
                dy = offset[1, y_idx, x_idx].item()

                # Refined 2D center (in feature map coordinates)
                center_x = x_idx + dx
                center_y = y_idx + dy

                # Get 2D box size
                box_w = bbox_size[0, y_idx, x_idx].item()
                box_h = bbox_size[1, y_idx, x_idx].item()

                # Get left-right distance
                d = lr_distance[0, y_idx, x_idx].item()

                # Compute depth from stereo geometry
                # Z = (f × B) / disparity, where disparity ≈ d
                if calib is not None and d > 0:
                    fx = calib.get("fx", 721.5377)  # Default KITTI value
                    baseline = calib.get("baseline", 0.54)  # Default KITTI value
                    depth = (fx * baseline) / (d + 1e-6)
                else:
                    # Fallback: use approximate depth estimation
                    depth = 50.0  # Default depth in meters

                # Convert 2D center to 3D position
                # Scale to original image coordinates (assuming 4x downsampling)
                scale = 4.0
                u = center_x * scale
                v = center_y * scale

                if calib is not None:
                    cx = calib.get("cx", 609.5593)
                    cy = calib.get("cy", 172.8540)
                    fx = calib.get("fx", 721.5377)
                    fy = calib.get("fy", 721.5377)
                else:
                    # Default KITTI calibration
                    cx, cy = 609.5593, 172.8540
                    fx, fy = 721.5377, 721.5377

                # 3D position: X = (u - cx) × Z / fx, Y = (v - cy) × Z / fy, Z = depth
                x_3d = (u - cx) * depth / fx
                y_3d = (v - cy) * depth / fy
                z_3d = depth

                # Decode dimensions (offsets + class mean)
                dim_offsets = dimensions[:, y_idx, x_idx].cpu().numpy()
                mean_h, mean_w, mean_l = mean_dims[c]
                height = mean_h + dim_offsets[0]
                width = mean_w + dim_offsets[1]
                length = mean_l + dim_offsets[2]

                # Ensure positive dimensions
                height = max(0.1, height)
                width = max(0.1, width)
                length = max(0.1, length)

                # Decode orientation from Multi-Bin representation
                # Multi-Bin: 8 values representing 4 bins (2 per bin: sin, cos)
                orient_bins = orientation[:, y_idx, x_idx].cpu().numpy()
                # Find bin with maximum confidence
                bin_confidences = orient_bins[::2] ** 2 + orient_bins[1::2] ** 2
                bin_idx = np.argmax(bin_confidences)
                sin_val = orient_bins[bin_idx * 2]
                cos_val = orient_bins[bin_idx * 2 + 1]
                # Convert to angle
                angle = np.arctan2(sin_val, cos_val)

                # Create Box3D object
                box3d = Box3D(
                    center_3d=(float(x_3d), float(y_3d), float(z_3d)),
                    dimensions=(float(length), float(width), float(height)),
                    orientation=float(angle),
                    class_label=class_names[c],
                    class_id=c,
                    confidence=confidence,
                    bbox_2d=(
                        float((center_x - box_w / 2) * scale),
                        float((center_y - box_h / 2) * scale),
                        float((center_x + box_w / 2) * scale),
                        float((center_y + box_h / 2) * scale),
                    ),
                )
                boxes3d.append(box3d)

    return boxes3d


def _labels_to_box3d_list(labels: list[dict[str, Any]], calib: dict[str, float] | None = None) -> list[Box3D]:
    """Convert label dictionaries to Box3D objects.

    Args:
        labels: List of label dictionaries from dataset.
        calib: Calibration parameters.

    Returns:
        List of Box3D objects.
    """
    boxes3d = []
    class_names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}

    for label in labels:
        try:
            class_id = label.get("class_id", 0)
            if class_id not in class_names:
                continue

            # Get dimensions
            dims = label.get("dimensions", {})
            height = dims.get("height", 1.5)
            width = dims.get("width", 1.5)
            length = dims.get("length", 3.0)

            # Get orientation (alpha is observation angle, convert to rotation_y)
            alpha = label.get("alpha", 0.0)

            # Reconstruct 3D center from 2D box and depth estimation
            # For validation, we'll use a simplified approach: estimate depth from 2D box height
            left_box = label.get("left_box", {})
            box_h = left_box.get("height", 0.1)
            # Rough depth estimation: Z ≈ (f × H_3d) / h_2d
            if calib and box_h > 0:
                fy = calib.get("fy", 721.5377)
                # Estimate depth from box height
                depth = (fy * height) / (box_h * 375.0)  # Assuming original image height ~375
            else:
                depth = 30.0  # Default depth

            # Convert 2D center to 3D
            center_x_2d = left_box.get("center_x", 0.5) * 1242.0  # Assuming original width
            center_y_2d = left_box.get("center_y", 0.5) * 375.0  # Assuming original height

            if calib:
                cx = calib.get("cx", 609.5593)
                cy = calib.get("cy", 172.8540)
                fx = calib.get("fx", 721.5377)
                fy = calib.get("fy", 721.5377)
            else:
                cx, cy = 609.5593, 172.8540
                fx, fy = 721.5377, 721.5377

            x_3d = (center_x_2d - cx) * depth / fx
            y_3d = (center_y_2d - cy) * depth / fy
            z_3d = depth

            # Convert alpha to rotation_y (simplified)
            rotation_y = alpha

            box3d = Box3D(
                center_3d=(float(x_3d), float(y_3d), float(z_3d)),
                dimensions=(float(length), float(width), float(height)),
                orientation=float(rotation_y),
                class_label=class_names[class_id],
                class_id=class_id,
                confidence=1.0,  # Ground truth has confidence 1.0
                bbox_2d=None,
                truncated=label.get("truncated"),
                occluded=label.get("occluded"),
            )
            boxes3d.append(box3d)
        except Exception as e:
            LOGGER.warning(f"Error converting label to Box3D: {e}")
            continue

    return boxes3d


class Stereo3DDetValidator(BaseValidator):
    """Stereo 3D Detection Validator.

    Extends BaseValidator to implement 3D detection validation with AP3D metrics.
    Computes 3D IoU, matches predictions to ground truth, and calculates AP3D at IoU 0.5 and 0.7.
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
        self.args.task = "stereo3ddet"
        self.iouv = torch.tensor([0.5, 0.7])  # IoU thresholds for AP3D
        self.niou = len(self.iouv)
        self.metrics = Stereo3DDetMetrics()

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

        # Root path and splits
        root_path = data_cfg.get("path") or "."
        root = Path(str(root_path)).resolve()
        # Accept either directory-style train/val or txt; KITTIStereoDataset uses split names
        train_split = data_cfg.get("train_split", "train")
        val_split = data_cfg.get("val_split", "val")

        # Names/nc fallback
        names = data_cfg.get("names") or {
            0: "Car",
            1: "Van",
            2: "Truck",
            3: "Pedestrian",
            4: "Person_sitting",
            5: "Cyclist",
            6: "Tram",
            7: "Misc",
        }
        nc = data_cfg.get("nc", len(names))

        # Return a dict compatible with BaseValidator expectations, plus stereo descriptors
        return {
            "yaml_file": str(self.args.data) if isinstance(self.args.data, (str, Path)) else None,
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
        }

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess stereo batch for validation.

        Args:
            batch: Batch containing stereo images [B, 6, H, W] and labels.

        Returns:
            Preprocessed batch.
        """
        # Move to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")

        # Normalize images
        if "img" in batch:
            batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255.0

        return batch

    def postprocess(self, preds: dict[str, torch.Tensor]) -> list[list[Box3D]]:
        """Postprocess model outputs to Box3D objects.

        Args:
            preds: Dictionary of 10-branch model outputs.

        Returns:
            List of Box3D lists (one per batch item).
        """
        batch_size = preds["heatmap"].shape[0]
        results = []

        # Get calibration from batch if available
        calib = None
        if hasattr(self, "_current_batch") and self._current_batch:
            # Try to get calibration from batch
            calibs = self._current_batch.get("calib", [])
            if calibs:
                calib = calibs[0] if isinstance(calibs[0], dict) else None

        for b in range(batch_size):
            # Extract single batch item outputs
            single_outputs = {k: v[b : b + 1] for k, v in preds.items()}
            boxes3d = decode_stereo3d_outputs(
                single_outputs,
                conf_threshold=self.args.conf,
                top_k=100,
                calib=calib,
            )
            results.append(boxes3d)

        return results

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics with model information.

        Args:
            model: Model being validated.
        """
        self.names = model.names
        self.nc = len(model.names)
        self.seen = 0
        self.metrics.names = model.names

    def update_metrics(self, preds: list[list[Box3D]], batch: dict[str, Any]) -> None:
        """Update metrics with predictions and ground truth.

        Args:
            preds: List of predicted Box3D lists (one per image).
            batch: Batch containing ground truth labels.
        """
        self._current_batch = batch  # Store for calibration access

        labels_list = batch.get("labels", [])
        calibs = batch.get("calib", [])

        for si, (pred_boxes, labels) in enumerate(zip(preds, labels_list)):
            self.seen += 1

            # Get calibration for this sample
            calib = calibs[si] if si < len(calibs) and isinstance(calibs[si], dict) else None

            # Convert labels to Box3D
            try:
                gt_boxes = _labels_to_box3d_list(labels, calib)
            except Exception as e:
                LOGGER.warning(f"Error converting labels to Box3D (sample {si}): {e}")
                gt_boxes = []

            # Handle empty predictions or ground truth
            if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                continue

            # Compute 3D IoU matrix
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                # Match predictions to ground truth using 3D IoU
                iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
                for i, pred_box in enumerate(pred_boxes):
                    for j, gt_box in enumerate(gt_boxes):
                        if pred_box.class_id == gt_box.class_id:
                            try:
                                iou = compute_3d_iou(pred_box, gt_box)
                                iou_matrix[i, j] = iou
                            except Exception as e:
                                LOGGER.warning(f"Error computing 3D IoU: {e}")
                                iou_matrix[i, j] = 0.0

                # Match predictions to ground truth (greedy matching)
                matched_gt = set()
                tp = np.zeros((len(pred_boxes), self.niou), dtype=bool)
                fp = np.zeros((len(pred_boxes), self.niou), dtype=bool)

                # Sort predictions by confidence
                pred_indices = sorted(range(len(pred_boxes)), key=lambda i: pred_boxes[i].confidence, reverse=True)

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
                fp = np.ones((len(pred_boxes), self.niou), dtype=bool) if len(pred_boxes) > 0 else np.zeros((0, self.niou), dtype=bool)

            # Extract statistics
            conf = np.array([box.confidence for box in pred_boxes]) if pred_boxes else np.array([])
            pred_cls = np.array([box.class_id for box in pred_boxes]) if pred_boxes else np.array([], dtype=int)
            target_cls = np.array([box.class_id for box in gt_boxes]) if gt_boxes else np.array([], dtype=int)

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

    def finalize_metrics(self) -> None:
        """Finalize metrics computation."""
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir

    def get_stats(self) -> dict[str, Any]:
        """Calculate and return metrics statistics.

        Returns:
            Dictionary containing metrics results.
        """
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        return self.metrics.results_dict

    def __call__(self, trainer=None, model=None):
        """Run validation loop.

        Args:
            trainer: Trainer object (optional).
            model: Model to validate (optional).

        Returns:
            Dictionary of validation metrics.
        """
        # Initialize metrics if model is provided
        if model is not None:
            self.init_metrics(model)

        # Call parent implementation which handles the validation loop
        # BaseValidator.__call__ will see self.data is already set and skip the dataset loading logic
        result = super().__call__(trainer=trainer, model=model)

        # Return metrics
        if hasattr(self.metrics, "results_dict"):
            return self.metrics.results_dict
        return result if result else {}

    def build_dataset(self, img_path: str | dict[str, Any], mode: str = "val", batch: int | None = None) -> torch.utils.data.Dataset:
        """Build Stereo3DDetAdapterDataset for validation.

        Args:
            img_path: Path to dataset root directory, or a descriptor dict from self.data.get(split).
            mode: 'train' or 'val' mode.
            batch: Batch size (unused, kept for compatibility).

        Returns:
            Stereo3DDetAdapterDataset: Dataset instance for validation.
        """
        from ultralytics.models.yolo.stereo3ddet.dataset import Stereo3DDetAdapterDataset
        # img_path should be a dir 
        if isinstance(img_path, str) and not os.path.isdir(img_path):
            # means it's a file instead of the path, return it's parent directory
            img_path = Path(img_path).parent


        # Handle descriptor dict from self.data.get(self.args.split)
        desc = img_path if isinstance(img_path, dict) else self.data.get(mode) if hasattr(self, "data") else None
        
        if isinstance(desc, dict) and desc.get("type") == "kitti_stereo":
            # Get image size from args, default to 384
            imgsz = getattr(self.args, "imgsz", 384)
            if isinstance(imgsz, (list, tuple)):
                imgsz = imgsz[0] if len(imgsz) > 0 else 384
            
            return Stereo3DDetAdapterDataset(
                root=str(desc.get("root", ".")),
                split=str(desc.get("split", mode)),
                imgsz=imgsz,
                names=self.data.get("names") if hasattr(self, "data") else None,
            )
        
        # Fallback: if img_path is a string, try to use it directly
        if isinstance(img_path, str) or isinstance(img_path, Path):
            imgsz = getattr(self.args, "imgsz", 384)
            if isinstance(imgsz, (list, tuple)):
                imgsz = imgsz[0] if len(imgsz) > 0 else 384
            
            return Stereo3DDetAdapterDataset(
                root=img_path,
                split=mode,
                imgsz=imgsz,
                names=self.data.get("names") if hasattr(self, "data") else None,
            )
        
        # If we can't determine the dataset, raise an error
        raise ValueError(
            f"Cannot build dataset from img_path={img_path} (type: {type(img_path)}). "
            f"Expected a string path or a descriptor dict with type='kitti_stereo'."
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
