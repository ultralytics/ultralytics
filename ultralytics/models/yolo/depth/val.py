# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
MDE (Monocular Depth Estimation) validation module.

This module provides validation functionality specifically for MDE models,
handling the unique output format with depth estimation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import ConfusionMatrix, MDEMetrics, box_iou


class MDEValidator(BaseValidator):
    """
    MDE Validator for YOLO models with depth estimation.

    This validator handles the unique output format of MDE models that includes
    bounding boxes, class probabilities, and depth values.

    Attributes:
        model: The MDE model to validate.
        dataloader: Validation data loader.
        device: Device to run validation on.
        metrics: Detection metrics.

    Methods:
        __call__: Run validation on the model.
        postprocess: Post-process MDE model predictions.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """
        Initialize MDE validator with necessary variables and settings.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (SimpleNamespace, optional): Arguments for validation.
            _callbacks (dict, optional): Callbacks for validation.
        """
        # Initialize base validator
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.metrics = MDEMetrics()
        self.seen = 0

    def preprocess(self, batch):
        """
        Preprocess batch of images for MDE validation.

        Args:
            batch (dict): Batch containing images and annotations.

        Returns:
            (dict): Preprocessed batch.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        return batch

    def postprocess(self, preds):
        """
        Post-process MDE model predictions.

        Args:
            preds: Raw model predictions.

        Returns:
            List of dictionaries containing processed predictions.
        """
        from ultralytics.utils import nms

        # Apply standard NMS like DetectionValidator
        outputs = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            nc=0 if self.args.task == "detect" else self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
        )

        # Return in the same format as DetectionValidator
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs]

    def update_metrics(self, preds, batch):
        """
        Update metrics with new predictions and ground truth.

        Args:
            preds: List of predictions from the model.
            batch: Batch data containing ground truth.
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0

            # Match predictions to ground truth and extract matched depth pairs with class labels
            matched_pred_depths, matched_target_depths, matched_pred_cls = self._match_depth_pairs(
                predn, pbatch, batch, si
            )

            # Update all metrics (detection + depth)
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                    "pred_depths": matched_pred_depths,
                    "target_depths": matched_target_depths,
                    "depth_pred_cls": matched_pred_cls,  # Add matched class labels for depth pairs
                }
            )

    def _prepare_batch(self, si, batch):
        """Prepare a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if cls.shape[0]:
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }

    def _prepare_pred(self, pred):
        """Prepare predictions for evaluation against ground truth."""
        if self.args.single_cls:
            pred["cls"] *= 0
        return pred

    def _process_batch(self, preds, batch):
        """Return correct prediction matrix."""
        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            return {"tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)}
        iou = box_iou(batch["bboxes"], preds["bboxes"])
        result = {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}
        return result

    def _match_depth_pairs(self, predn, pbatch, batch, si):
        """
        Match predictions to ground truth and extract depth pairs for matched detections.

        Args:
            predn: Prepared predictions dictionary
            pbatch: Prepared batch dictionary
            batch: Original batch data
            si: Sample index in batch

        Returns:
            tuple: (matched_pred_depths, matched_target_depths, matched_pred_cls) as numpy arrays
        """
        # Extract depth predictions
        if "extra" in predn and predn["extra"] is not None and predn["extra"].shape[1] > 0:
            pred_depths = predn["extra"][:, 0]  # First extra channel is depth
        else:
            return np.array([]), np.array([]), np.array([])

        # Extract depth targets
        if "depths" not in batch:
            return np.array([]), np.array([]), np.array([])

        idx = batch["batch_idx"] == si
        if not idx.any():
            return np.array([]), np.array([]), np.array([])

        target_depths = batch["depths"][idx]

        # If no predictions or no targets, return empty
        if pbatch["cls"].shape[0] == 0 or predn["cls"].shape[0] == 0:
            return np.array([]), np.array([]), np.array([])

        # Calculate IoU between predictions and ground truth
        iou = box_iou(pbatch["bboxes"], predn["bboxes"])

        # Match predictions to ground truth
        # Use IoU threshold of 0.5 (could be made configurable)
        iou_threshold = 0.5
        matched_pred_depths_list = []
        matched_target_depths_list = []
        matched_pred_cls_list = []

        # For each ground truth, find the best matching prediction
        for gt_idx in range(pbatch["cls"].shape[0]):
            # Find predictions that match this ground truth class
            class_match = predn["cls"] == pbatch["cls"][gt_idx]

            # Get IoU values for this ground truth with all predictions
            iou_values = iou[gt_idx, :]

            # Apply class matching constraint
            iou_values = iou_values * class_match.float()

            # Find best matching prediction
            if iou_values.max() >= iou_threshold:
                best_pred_idx = iou_values.argmax()
                matched_pred_depths_list.append(pred_depths[best_pred_idx].item())
                matched_target_depths_list.append(target_depths[gt_idx].item())
                matched_pred_cls_list.append(predn["cls"][best_pred_idx].item())

        # Convert to numpy arrays
        matched_pred_depths = np.array(matched_pred_depths_list)
        matched_target_depths = np.array(matched_target_depths_list)
        matched_pred_cls = np.array(matched_pred_cls_list)
        return matched_pred_depths, matched_target_depths, matched_pred_cls

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None) -> torch.utils.data.Dataset:
        """
        Build YOLO Dataset for MDE validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (Dataset): YOLO dataset.
        """
        from ultralytics.data import build_yolo_dataset

        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path: str, batch_size: int) -> torch.utils.data.DataLoader:
        """
        Construct and return dataloader for MDE validation.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Size of each batch.

        Returns:
            (torch.utils.data.DataLoader): Dataloader for validation.
        """
        from ultralytics.data import build_dataloader

        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(
            dataset, batch_size, self.args.workers, shuffle=False, rank=-1, drop_last=self.args.compile
        )

    def init_metrics(self, model: torch.nn.Module) -> None:
        """
        Initialize evaluation metrics for MDE validation.

        Args:
            model (torch.nn.Module): The model to validate.
        """
        from ultralytics.utils.metrics import MDEMetrics

        # Set names from model (same as DetectionValidator)
        self.names = model.names if hasattr(model, "names") and model.names is not None else {}
        self.nc = len(self.names)

        self.metrics = MDEMetrics(names=self.names)
        self.confusion_matrix = ConfusionMatrix(names=self.names, task="detect")
        self.seen = 0
        self.jdict = []
        self.stats = []

    def get_desc(self) -> str:
        """Return a formatted string summarizing class metrics of MDE model."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95",
            "Depth_Err",
            "Depth_MAE",
            "Depth_RMSE",
            "Depth_Acc",
        )

    def finalize_metrics(self) -> None:
        """Set final values for metrics speed and confusion matrix."""
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir, normalize=normalize, on_plot=self.on_plot)
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self) -> dict[str, Any]:
        """Return validation statistics."""
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        self.metrics.clear_stats()
        return self.metrics.results_dict

    def print_results(self) -> None:
        """Print training/validation set metrics per class for MDE."""
        import numpy as np

        from ultralytics.utils import LOGGER

        pf = "%22s" + "%11i" * 2 + "%11.3g" * 8  # print format for MDE metrics

        # Get mean results including depth metrics
        mean_results = self.metrics.mean_results()
        depth_error = np.mean(self.metrics.depth_errors) if self.metrics.depth_errors else 0.0
        depth_mae = np.mean(self.metrics.depth_abs_errors) if self.metrics.depth_abs_errors else 0.0
        depth_rmse = np.sqrt(np.mean(self.metrics.depth_sq_errors)) if self.metrics.depth_sq_errors else 0.0
        depth_acc = (
            np.mean(self.metrics.depth_accuracies["δ < 1.250"]) if self.metrics.depth_accuracies["δ < 1.250"] else 0.0
        )

        # Print overall results
        LOGGER.info(
            pf
            % (
                "all",
                self.seen,
                self.metrics.nt_per_class.sum() if self.metrics.nt_per_class is not None else 0,
                *mean_results[:4],  # Box metrics: P, R, mAP50, mAP50-95
                depth_error,
                depth_mae,
                depth_rmse,
                depth_acc,
            )
        )

        if self.metrics.nt_per_class is None or self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.metrics.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                class_results = self.metrics.class_result(i)
                LOGGER.info(
                    pf
                    % (
                        self.names[c],
                        self.metrics.nt_per_image[c],
                        self.metrics.nt_per_class[c],
                        *class_results[:4],  # Box metrics: P, R, mAP50, mAP50-95
                        *class_results[4:8],  # Per-class depth metrics: Depth_Err, Depth_MAE, Depth_RMSE, Depth_Acc
                    )
                )

        # Print depth statistics if available
        if self.metrics.depth_errors:
            LOGGER.info("\n📊 Depth Estimation Metrics:")
            LOGGER.info(f"   Depth Error Rate: {depth_error:.2f}%")
            LOGGER.info(f"   Depth MAE: {depth_mae:.2f}m")
            LOGGER.info(f"   Depth RMSE: {depth_rmse:.2f}m")
            LOGGER.info(f"   Depth Accuracy (δ<1.25): {depth_acc:.2f}%")

            for key, values in self.metrics.depth_accuracies.items():
                if values:
                    acc_value = np.mean(values)
                    LOGGER.info(f"   Depth Accuracy ({key}): {acc_value:.2f}%")

    def plot_val_samples(self, batch, ni):
        """
        Plot validation image samples with ground truth labels and depth values.

        Args:
            batch (dict): Batch containing images and annotations.
            ni (int): Batch index.
        """
        from ultralytics.utils.plotting import plot_images

        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """
        Plot predicted bounding boxes with depth values on input images and save the result.

        Args:
            batch (dict): Batch containing images and annotations.
            preds (list[dict]): List of predictions from the model.
            ni (int): Batch index.
        """
        from ultralytics.utils.plotting import plot_images

        # Add batch index to predictions
        for i, pred in enumerate(preds):
            pred["batch_idx"] = torch.ones_like(pred["conf"]) * i

        # Get keys from predictions
        keys = preds[0].keys()
        max_det = self.args.max_det

        # Concatenate all predictions into single batch
        batched_preds = {k: torch.cat([x[k][:max_det] for x in preds], dim=0) for k in keys}

        # Convert bboxes to xywh format for plotting
        batched_preds["bboxes"][:, :4] = ops.xyxy2xywh(batched_preds["bboxes"][:, :4])

        # Extract depth values from extra channel if available
        if "extra" in batched_preds and batched_preds["extra"].shape[1] > 0:
            batched_preds["depths"] = batched_preds["extra"][:, 0]  # First extra channel is depth

        plot_images(
            images=batch["img"],
            labels=batched_preds,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )
