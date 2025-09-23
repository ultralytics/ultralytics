# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
MDE (Monocular Depth Estimation) validation module.

This module provides validation functionality specifically for MDE models,
handling the unique output format with depth estimation.
"""

from __future__ import annotations

import numpy as np
import torch

from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import DetMetrics, box_iou


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
        self.args.task = "detect"  # Ensure task is set for detection metrics
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.metrics = DetMetrics()
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
        # For now, use the same update logic as DetectionValidator
        # This can be extended later for MDE-specific metrics
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
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
        return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}

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
