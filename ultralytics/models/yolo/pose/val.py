# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import OKS_SIGMA, PoseMetrics, kpt_iou


class PoseValidator(DetectionValidator):
    """A class extending the DetectionValidator class for validation based on a pose model.

    This validator is specifically designed for pose estimation tasks, handling keypoints and implementing specialized
    metrics for pose evaluation.

    Attributes:
        sigma (np.ndarray): Sigma values for OKS calculation: dataset `kpt_oks_sigmas` if provided, else OKS_SIGMA for
            COCO keypoints or ones divided by number of keypoints.
        kpt_shape (list[int]): Shape of the keypoints, typically [17, 3] for COCO format.
        args (dict): Arguments for the validator including task set to "pose".
        metrics (PoseMetrics): Metrics object for pose evaluation.

    Methods:
        preprocess: Preprocess batch by converting keypoints data to float and moving it to the device.
        get_desc: Return description of evaluation metrics in string format.
        init_metrics: Initialize pose estimation metrics for YOLO model.
        postprocess: Postprocess YOLO predictions to extract and reshape keypoints for pose estimation.
        _prepare_batch: Prepare a batch for processing by scaling normalized keypoints to model input image dimensions.
        _process_batch: Return correct prediction matrices from box IoU and keypoint OKS between detections and ground
            truth.
        gather_stats: Gather stats from all GPUs.
        scale_preds: Scale predictions to the original image size.
        save_one_txt: Save YOLO pose detections to a text file in normalized coordinates.
        pred_to_json: Convert YOLO predictions to COCO JSON format.
        eval_json: Evaluate pose estimation model using COCO JSON format.

    Examples:
        >>> from ultralytics.models.yolo.pose import PoseValidator
        >>> args = dict(model="yolo26n-pose.pt", data="coco8-pose.yaml")
        >>> validator = PoseValidator(args=args)
        >>> validator()

    Notes:
        This class extends DetectionValidator with pose-specific functionality. It initializes with sigma values
        for OKS calculation and sets up PoseMetrics for evaluation.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """Initialize a PoseValidator object for pose estimation validation.

        This validator is specifically designed for pose estimation tasks, handling keypoints and implementing
        specialized metrics for pose evaluation.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to be used for validation.
            save_dir (Path | str, optional): Directory to save results.
            args (dict, optional): Arguments for the validator including task set to "pose".
            _callbacks (dict, optional): Dictionary of callback functions to be executed during validation.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.args.task = "pose"
        self.metrics = PoseMetrics()

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch by converting keypoints data to float and moving it to the device."""
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].float()
        return batch

    def get_desc(self) -> str:
        """Return description of evaluation metrics in string format."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize evaluation metrics for YOLO pose validation.

        Args:
            model (torch.nn.Module): Model to validate.

        Raises:
            ValueError: If dataset `kpt_oks_sigmas` does not contain one positive value per keypoint.
        """
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        if sigmas := self.data.get("kpt_oks_sigmas"):  # optional custom OKS sigmas from the dataset YAML
            self.sigma = np.array(sigmas, dtype=np.float32).flatten()
            if len(self.sigma) != nkpt or not np.all(self.sigma > 0):
                raise ValueError(f"'kpt_oks_sigmas' must be {nkpt} positive values, got {sigmas}")
        else:
            self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Postprocess YOLO predictions to extract and reshape keypoints for pose estimation.

        This method extends the parent class postprocessing by extracting keypoints from the 'extra' field of
        predictions and reshaping them according to the keypoint shape configuration. The keypoints are reshaped from a
        flattened format to the proper dimensional structure (typically [N, 17, 3] for COCO pose format).

        Args:
            preds (torch.Tensor): Raw prediction tensor from the YOLO pose model containing bounding boxes, confidence
                scores, class predictions, and keypoint data.

        Returns:
            (list[dict[str, torch.Tensor]]): List of processed prediction dictionaries, each containing:
                - 'bboxes': Bounding box coordinates
                - 'conf': Confidence scores
                - 'cls': Class predictions
                - 'keypoints': Reshaped keypoint coordinates with shape (-1, *self.kpt_shape)

        Notes:
            The keypoints are extracted from the 'extra' field which contains additional task-specific data beyond
            basic detection.
        """
        preds = super().postprocess(preds)
        for pred in preds:
            pred["keypoints"] = pred.pop("extra").view(-1, *self.kpt_shape)  # remove extra if exists
        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare a batch for processing by scaling normalized keypoints to model input image dimensions.

        Args:
            si (int): Sample index within the batch.
            batch (dict[str, Any]): Dictionary containing batch data with keys like 'keypoints', 'batch_idx', etc.

        Returns:
            (dict[str, Any]): Prepared batch with keypoints scaled to model input (letterboxed) image dimensions.

        Notes:
            This method extends the parent class's _prepare_batch method by adding keypoint processing.
            Keypoints are scaled from normalized coordinates to the model input (letterboxed) image dimensions.
        """
        pbatch = super()._prepare_batch(si, batch)
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = pbatch["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        pbatch["keypoints"] = kpts
        return pbatch

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Return correct prediction matrices from box IoU and keypoint OKS between detections and ground truth.

        Args:
            preds (dict[str, torch.Tensor]): Dictionary containing prediction data with keys 'cls' for class predictions
                and 'keypoints' for keypoint predictions.
            batch (dict[str, Any]): Dictionary containing ground truth data with keys 'cls' for class labels, 'bboxes'
                for bounding boxes, and 'keypoints' for keypoint annotations.

        Returns:
            (dict[str, np.ndarray]): Dictionary containing the box true positives 'tp', the pose true positives 'tp_p'
                and the pose ignore flags 'ignore_p', each with shape (N, 10) for 10 IoU/OKS thresholds, plus the
                classes 'target_cls_p' of the GTs with at least one labeled keypoint.

        Notes:
            `0.53` scale factor used in area computation is referenced from
            https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384.
            Like COCO, GTs without labeled keypoints are excluded from the pose targets, and predictions matched only
            to them are ignored instead of being counted as false positives.
        """
        tp = super()._process_batch(preds, batch)
        gt_cls, kpts = batch["cls"], batch["keypoints"]
        # COCO ignores GTs without labeled keypoints (num_keypoints == 0): they are neither targets nor sources of FPs
        valid = (kpts[..., 2] > 0).any(-1) if kpts.shape[-1] == 3 else torch.ones_like(gt_cls, dtype=torch.bool)
        tp_p = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
        ignore_p = np.zeros_like(tp_p)
        if gt_cls.shape[0] and preds["cls"].shape[0]:
            # `0.53` is from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            area = ops.xyxy2xywh(batch["bboxes"])[:, 2:].prod(1) * 0.53
            if valid.any():
                iou = kpt_iou(kpts[valid], preds["keypoints"], sigma=self.sigma, area=area[valid])
                tp_p = self.match_predictions(preds["cls"], gt_cls[valid], iou).cpu().numpy()
            if not valid.all():
                ignore_p = self._match_ignored(preds, batch["bboxes"][~valid], gt_cls[~valid], area[~valid], tp_p)
        tp.update({"tp_p": tp_p, "ignore_p": ignore_p, "target_cls_p": gt_cls[valid].cpu().numpy()})
        return tp

    def _match_ignored(
        self,
        preds: dict[str, torch.Tensor],
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        area: torch.Tensor,
        tp_p: np.ndarray,
    ) -> np.ndarray:
        """Find unmatched predictions that COCO would match to GTs without labeled keypoints, and thus ignore.

        OKS against such a GT follows pycocotools `computeOks` for `k1 == 0`: each predicted keypoint is scored by its
        distance to the GT box expanded by its width and height on each side. Predictions are visited in descending
        confidence and each ignored GT absorbs at most one prediction per OKS threshold, as in COCO for non-crowd GTs.

        Args:
            preds (dict[str, torch.Tensor]): Predictions with keys 'cls', 'conf' and 'keypoints'.
            gt_bboxes (torch.Tensor): Boxes of the ignored GTs in xyxy format, shape (M, 4).
            gt_cls (torch.Tensor): Classes of the ignored GTs, shape (M,).
            area (torch.Tensor): Areas of the ignored GTs used for OKS, shape (M,).
            tp_p (np.ndarray): Pose true positives against the valid GTs, shape (N, 10).

        Returns:
            (np.ndarray): Boolean array of shape (N, 10), True where a prediction is ignored at that OKS threshold.
        """
        wh = gt_bboxes[:, 2:] - gt_bboxes[:, :2]
        lo = (gt_bboxes[:, :2] - wh)[:, None, None]  # (M, 1, 1, 2), box expanded by its size on each side
        hi = (gt_bboxes[:, 2:] + wh)[:, None, None]
        xy = preds["keypoints"][None, ..., :2]  # (1, N, K, 2)
        d = ((lo - xy).clamp(min=0) + (xy - hi).clamp(min=0)).pow(2).sum(-1)  # (M, N, K)
        sigma = torch.as_tensor(self.sigma, device=d.device, dtype=d.dtype)
        e = d / ((2 * sigma).pow(2) * (area[:, None, None] + 1e-7) * 2)  # same scaling as kpt_iou
        oks = ((-e).exp().mean(-1) * (gt_cls[:, None] == preds["cls"][None])).cpu().numpy()  # (M, N)

        ignore_p = np.zeros_like(tp_p)
        thr = self.iouv.cpu().numpy()[:, None]  # (T, 1)
        used = np.zeros((len(thr), oks.shape[0]), dtype=bool)  # (T, M) ignored GTs already taken per threshold
        order = preds["conf"].argsort(descending=True).cpu().numpy()
        for j in order[oks.max(0)[order] >= thr.min()]:  # greedy by confidence, only predictions that can match
            cand = np.where(used, -1.0, oks[None, :, j])  # (T, M)
            best = cand.argmax(1)
            hit = (cand.max(1) >= thr[:, 0]) & ~tp_p[j]
            used[hit, best[hit]] = True
            ignore_p[j] = hit
        return ignore_p

    def gather_stats(self) -> None:
        """Gather stats from all GPUs."""
        super().gather_stats()  # gather stats from DetectionValidator
        self._gather_image_metrics(self.metrics.pose)

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """Save YOLO pose detections to a text file in normalized coordinates.

        Args:
            predn (dict[str, torch.Tensor]): Prediction dict with keys 'bboxes', 'conf', 'cls', and 'keypoints'.
            save_conf (bool): Whether to save confidence scores.
            shape (tuple[int, int]): Shape of the original image (height, width).
            file (Path): Output file path to save detections.

        Notes:
            The output format is: class_id x_center y_center width height confidence keypoints where keypoints are
            normalized (x, y, visibility) values for each point.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            keypoints=predn["keypoints"],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Convert YOLO predictions to COCO JSON format.

        This method takes prediction tensors and batch data, converts the bounding boxes from YOLO format to COCO
        format, and appends the results with keypoints to the internal JSON dictionary (self.jdict).

        Args:
            predn (dict[str, torch.Tensor]): Prediction dictionary containing 'bboxes', 'conf', 'cls', and 'keypoints'
                tensors.
            pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.

        Notes:
            The method extracts the image ID from the filename stem (either as an integer if numeric, or as a string),
            converts bounding boxes from xyxy to xywh format, and adjusts coordinates from center to top-left corner
            before saving to the JSON dictionary.
        """
        super().pred_to_json(predn, pbatch)
        kpts = predn["keypoints"]
        for i, k in enumerate(kpts.flatten(1, 2).tolist()):
            self.jdict[-len(kpts) + i]["keypoints"] = k  # keypoints

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scale boxes and keypoints to the original image size."""
        return {
            **super().scale_preds(predn, pbatch),
            "keypoints": ops.scale_coords(
                pbatch["imgsz"],
                predn["keypoints"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            ),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Evaluate pose estimation model using COCO JSON format."""
        anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"  # annotations
        pred_json = self.save_dir / "predictions.json"  # predictions
        return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "keypoints"], suffix=["Box", "Pose"])
