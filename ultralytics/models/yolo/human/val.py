# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.data.dataset import HumanDataset
from ultralytics.engine.results import Human, Results
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr
from ultralytics.utils.metrics import HumanMetrics, box_iou


class HumanValidator(DetectionValidator):
    """A class extending the DetectionValidator class for validation based on a human model.

    This validator evaluates detection quality with the usual box metrics and, for every true positive box, the accuracy
    of the predicted weight, height, gender, age and ethnicity.

    Attributes:
        args (dict): Arguments for the validator including task set to "human".
        metrics (HumanMetrics): Metrics object for detection and human attribute evaluation.

    Methods:
        build_dataset: Build the HumanDataset for validation.
        preprocess: Preprocess batch by converting human attributes to float.
        postprocess: Postprocess YOLO predictions to extract human attributes.
        get_desc: Return description of evaluation metrics in string format.
        save_one_txt: Save YOLO-Human detections to a text file in normalized coordinates.

    Examples:
        >>> from ultralytics.models.yolo.human import HumanValidator
        >>> args = dict(model="yolov8n-human.pt", data="coco8-human.yaml")
        >>> validator = HumanValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """Initialize a HumanValidator object for human attribute validation.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to be used for validation.
            save_dir (Path | str, optional): Directory to save results.
            args (dict, optional): Arguments for the validator including task set to "human".
            _callbacks (dict, optional): Dictionary of callback functions to be executed during validation.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "human"
        self.metrics = HumanMetrics()

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None) -> HumanDataset:
        """Build the HumanDataset for validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (HumanDataset): Dataset object configured for the specified mode.
        """
        cfg = self.args
        return HumanDataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",  # augmentation
            hyp=cfg,
            rect=cfg.rect or mode == "val",  # rectangular batches
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=int(self.stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            classes=cfg.classes,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch by converting human attributes to float."""
        batch = super().preprocess(batch)
        batch["attributes"] = batch["attributes"].float()
        return batch

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Postprocess YOLO predictions, moving the human attributes out of the 'extra' field."""
        preds = super().postprocess(preds)
        for pred in preds:
            pred["attributes"] = pred.pop("extra")
        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare a batch for processing by attaching the ground truth human attributes.

        Args:
            si (int): Sample index within the batch.
            batch (dict[str, Any]): Dictionary containing batch data.

        Returns:
            (dict[str, Any]): Prepared batch including the ground truth `attributes` of the sample.
        """
        pbatch = super()._prepare_batch(si, batch)
        pbatch["attributes"] = batch["attributes"][batch["batch_idx"] == si]
        return pbatch

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Return the correct prediction matrix and accumulate the human attribute accuracies.

        Args:
            preds (dict[str, torch.Tensor]): Dictionary containing prediction data with 'bboxes', 'cls' and 'attributes'
                keys.
            batch (dict[str, Any]): Batch dictionary containing ground truth 'bboxes', 'cls' and 'attributes'.

        Returns:
            (dict[str, np.ndarray]): Dictionary containing 'tp' key with correct prediction matrix of shape (N, 10) for
                10 IoU levels.
        """
        tp = super()._process_batch(preds, batch)
        if batch["cls"].shape[0] and preds["cls"].shape[0]:
            iou = box_iou(batch["bboxes"], preds["bboxes"])
            self._process_attributes(preds["attributes"], batch["attributes"], iou)
        return tp

    def _process_attributes(
        self, pred_attrs: torch.Tensor, gt_attrs: torch.Tensor, iou: torch.Tensor, iou_thres: float = 0.5
    ) -> None:
        """Accumulate the per-attribute accuracy of the true positive predictions.

        Args:
            pred_attrs (torch.Tensor): The predicted attributes with shape (M, 11).
            gt_attrs (torch.Tensor): The ground truth attributes with shape (N, 5).
            iou (torch.Tensor): The IoU values between ground truth and predicted boxes with shape (N, M), used to
                choose the true positives whose attributes are evaluated.
            iou_thres (float): The IoU threshold that determines true positive samples.
        """
        values, indices = iou.max(1)
        tp = values >= iou_thres
        gt_attrs = gt_attrs[tp]
        if not gt_attrs.shape[0]:
            return
        pred_attrs = Human(pred_attrs[indices[tp]])
        weight, height, gender, age, ethnicity = (gt_attrs[:, i] for i in range(5))
        self.metrics.attrs_stats["weight"].append((1 - (pred_attrs.weight - weight).abs() / weight).clip(0, 1))
        self.metrics.attrs_stats["height"].append((1 - (pred_attrs.height - height).abs() / height).clip(0, 1))
        self.metrics.attrs_stats["gender"].append((pred_attrs.cls_gender == gender).float())
        self.metrics.attrs_stats["age"].append((1 - (pred_attrs.age - age).abs() / age).clip(0, 1))
        self.metrics.attrs_stats["ethnicity"].append((pred_attrs.cls_ethnicity == ethnicity).float())

    def get_desc(self) -> str:
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 11) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "acc(W)",  # weight
            "acc(H)",  # height
            "acc(G)",  # gender
            "acc(A)",  # age
            "acc(E)",  # ethnicity
        )

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """Save YOLO-Human detections to a text file in normalized coordinates.

        Args:
            predn (dict[str, torch.Tensor]): Prediction dict with keys 'bboxes', 'conf', 'cls' and 'attributes'.
            save_conf (bool): Whether to save confidence scores.
            shape (tuple[int, int]): Shape of the original image (height, width).
            file (Path): Output file path to save detections.
        """
        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            human=predn["attributes"],
        ).save_txt(file, save_conf=save_conf)
