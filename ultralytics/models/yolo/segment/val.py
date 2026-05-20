# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import SegmentMetrics, mask_biou, mask_dice, mask_iou


class SegmentationValidator(DetectionValidator):
    """A class extending the DetectionValidator class for validation based on a segmentation model.

    This validator handles the evaluation of segmentation models, processing both bounding box and mask predictions to
    compute metrics such as mAP for both detection and segmentation tasks.

    Attributes:
        plot_masks (list): List to store masks for plotting.
        process (callable): Function to process masks based on save_json and save_txt flags.
        args (namespace): Arguments for the validator.
        metrics (SegmentMetrics): Metrics calculator for segmentation tasks.
        stats (dict): Dictionary to store statistics during validation.

    Examples:
        >>> from ultralytics.models.yolo.segment import SegmentationValidator
        >>> args = dict(model="yolo26n-seg.pt", data="coco8-seg.yaml")
        >>> validator = SegmentationValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (namespace, optional): Arguments for the validator.
            _callbacks (list, optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.process = None
        self.args.task = "segment"

        # Validate fitness_weight length for segment task
        fitness_weight = getattr(self.args, "fitness_weight", None)
        if fitness_weight is not None and len(fitness_weight) not in [4, 8, 10]:
            LOGGER.warning(
                f"fitness_weight must have 4, 8 or 10 values for segment task, got {len(fitness_weight)}. "
                f"Using default weights. Expected: [P, R, mAP@0.5, mAP@0.5:0.95] (4 values), "
                f"[box_P, box_R, box_mAP@0.5, box_mAP@0.5:0.95, mask_P, mask_R, mask_mAP@0.5, mask_mAP@0.5:0.95] (8 values), "
                f"or the 8-value layout plus [mask_Dice, mask_BIoU] for boundary-aware fitness (10 values)."
            )
            fitness_weight = None

        # Reuse the boundary kernel from the loss config so val/train agree on rim thickness.
        boundary_kernel = int(getattr(self.args, "seg_boundary_kernel", 3) or 3)

        self.metrics = SegmentMetrics(
            fitness_weight=fitness_weight,
            class_weights=getattr(self.args, "class_weights_resolved", None),
            boundary_kernel=boundary_kernel,
        )

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch of images for YOLO segmentation validation.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.

        Returns:
            (dict[str, Any]): Preprocessed batch.
        """
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].float()
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics and select mask processing function based on save_json flag.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        super().init_metrics(model)
        if self.args.save_json:
            check_requirements("faster-coco-eval>=1.6.7")
        # More accurate vs faster
        self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask

    def get_desc(self) -> str:
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 12) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95",
            "Dice",
            "BIoU)",
        )

    def postprocess(self, preds: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Post-process YOLO predictions and return output detections with proto.

        Args:
            preds (list[torch.Tensor]): Raw predictions from the model.

        Returns:
            list[dict[str, torch.Tensor]]: Processed detection predictions with masks.
        """
        proto = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        preds = super().postprocess(preds[0])
        imgsz = [4 * x for x in proto.shape[2:]]  # get image size from proto
        for i, pred in enumerate(preds):
            coefficient = pred.pop("extra")
            pred["masks"] = (
                self.process(proto[i], coefficient, pred["bboxes"], shape=imgsz)
                if coefficient.shape[0]
                else torch.zeros(
                    (0, *(imgsz if self.process is ops.process_mask_native else proto.shape[2:])),
                    dtype=torch.uint8,
                    device=pred["bboxes"].device,
                )
            )
        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare a batch for training or inference by processing images and targets.

        Args:
            si (int): Batch index.
            batch (dict[str, Any]): Batch data containing images and annotations.

        Returns:
            (dict[str, Any]): Prepared batch with processed annotations.
        """
        prepared_batch = super()._prepare_batch(si, batch)
        nl = prepared_batch["cls"].shape[0]
        if self.args.overlap_mask:
            masks = batch["masks"][si]
            index = torch.arange(1, nl + 1, device=masks.device).view(nl, 1, 1)
            masks = (masks == index).float()
        else:
            masks = batch["masks"][batch["batch_idx"] == si]
        if nl:
            mask_size = [s if self.process is ops.process_mask_native else s // 4 for s in prepared_batch["imgsz"]]
            if masks.shape[1:] != mask_size:
                masks = F.interpolate(masks[None], mask_size, mode="bilinear", align_corners=False)[0]
                masks = masks.gt_(0.5)
        prepared_batch["masks"] = masks
        return prepared_batch

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Compute correct prediction matrix for a batch based on bounding boxes and optional masks.

        Args:
            preds (dict[str, torch.Tensor]): Dictionary containing predictions with keys like 'cls' and 'masks'.
            batch (dict[str, Any]): Dictionary containing batch data with keys like 'cls' and 'masks'.

        Returns:
            (dict[str, np.ndarray]): A dictionary containing correct prediction matrices including 'tp_m' for mask IoU,
                and per matched-pair scalar arrays `mask_dice`, `mask_biou`, `matched_cls` consumed by
                :class:`SegmentMetrics` for Dice / Boundary-IoU reporting.

        Examples:
            >>> preds = {"cls": torch.tensor([1, 0]), "masks": torch.rand(2, 640, 640), "bboxes": torch.rand(2, 4)}
            >>> batch = {"cls": torch.tensor([1, 0]), "masks": torch.rand(2, 640, 640), "bboxes": torch.rand(2, 4)}
            >>> correct_preds = validator._process_batch(preds, batch)

        Notes:
            - If `masks` is True, the function computes IoU between predicted and ground truth masks.
            - If `overlap` is True and `masks` is True, overlapping masks are taken into account when computing IoU.
            - Dice / Boundary-IoU are recorded for pairs matched at IoU >= 0.5 with class agreement, mirroring the
              greedy matching used by :py:meth:`BaseValidator.match_predictions`.
        """
        tp = super()._process_batch(preds, batch)
        gt_cls = batch["cls"]
        empty_scores = np.zeros(0, dtype=np.float32)
        empty_cls = np.zeros(0, dtype=np.int64)
        if gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0:
            tp_m = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
            dice_scores, biou_scores, matched_cls = empty_scores, empty_scores, empty_cls
        else:
            gt_masks = batch["masks"]
            pred_masks = preds["masks"].float()
            iou = mask_iou(gt_masks.flatten(1), pred_masks.flatten(1))  # (M, N)
            tp_m = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
            dice_scores, biou_scores, matched_cls = self._matched_mask_quality(
                gt_masks, pred_masks, gt_cls, preds["cls"], iou
            )
        tp.update(
            {
                "tp_m": tp_m,
                "mask_dice": dice_scores,
                "mask_biou": biou_scores,
                "matched_cls": matched_cls,
            }
        )
        return tp

    def _matched_mask_quality(
        self,
        gt_masks: torch.Tensor,
        pred_masks: torch.Tensor,
        gt_cls: torch.Tensor,
        pred_cls: torch.Tensor,
        iou: torch.Tensor,
        iou_threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Greedy-match predictions to GT at `iou_threshold` (with class agreement) and score each pair.

        Mirrors the non-scipy branch of :py:meth:`BaseValidator.match_predictions` but returns the actual
        (gt_idx, pred_idx) pairs so we can compute scalar Dice / Boundary-IoU per matched instance.

        Args:
            gt_masks (torch.Tensor): GT binary masks (M, H, W).
            pred_masks (torch.Tensor): Predicted binary masks (N, H, W).
            gt_cls (torch.Tensor): GT class indices (M,).
            pred_cls (torch.Tensor): Predicted class indices (N,).
            iou (torch.Tensor): Pairwise mask IoU (M, N).
            iou_threshold (float, optional): Matching threshold.

        Returns:
            (tuple[np.ndarray, np.ndarray, np.ndarray]): (dice_scores, biou_scores, matched_cls), each of shape (K,)
                where K is the number of matched pairs. Empty arrays when no matches.
        """
        # Mask out cross-class candidates (class agreement, like match_predictions).
        correct_class = gt_cls[:, None] == pred_cls  # (M, N)
        iou_masked = (iou * correct_class).cpu().numpy()

        matches = np.nonzero(iou_masked >= iou_threshold)
        matches = np.array(matches).T  # (K, 2): columns are (gt_idx, pred_idx)
        if not matches.shape[0]:
            empty_scores = np.zeros(0, dtype=np.float32)
            return empty_scores, empty_scores, np.zeros(0, dtype=np.int64)

        if matches.shape[0] > 1:
            # Standard greedy reduction: keep highest-IoU per GT and per pred (same order as match_predictions).
            order = iou_masked[matches[:, 0], matches[:, 1]].argsort()[::-1]
            matches = matches[order]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

        gi = matches[:, 0].astype(np.int64)
        pi = matches[:, 1].astype(np.int64)
        gt_sel = gt_masks[gi]  # (K, H, W)
        pred_sel = pred_masks[pi]  # (K, H, W), float in {0,1}

        # Pairwise (K, K) wastes work for large K — compute only the K matched scalars via the diagonal.
        dice_mat = mask_dice(gt_sel.flatten(1), pred_sel.flatten(1))
        biou_mat = mask_biou(gt_sel, pred_sel, self.metrics.boundary_kernel)
        dice_scores = dice_mat.diagonal().detach().cpu().numpy().astype(np.float32)
        biou_scores = biou_mat.diagonal().detach().cpu().numpy().astype(np.float32)
        matched_cls = gt_cls[gi].detach().cpu().numpy().astype(np.int64)
        return dice_scores, biou_scores, matched_cls

    def plot_predictions(self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int) -> None:
        """Plot batch predictions with masks and bounding boxes.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.
            preds (list[dict[str, torch.Tensor]]): List of predictions from the model.
            ni (int): Batch index.
        """
        for p in preds:
            masks = p["masks"]
            if masks.shape[0] > self.args.max_det:
                LOGGER.warning(f"Limiting validation plots to 'max_det={self.args.max_det}' items.")
            p["masks"] = torch.as_tensor(masks[: self.args.max_det], dtype=torch.uint8).cpu()
        super().plot_predictions(batch, preds, ni, max_det=self.args.max_det)  # plot bboxes

    def save_one_txt(self, predn: torch.Tensor, save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """Save YOLO detections to a txt file in normalized coordinates in a specific format.

        Args:
            predn (torch.Tensor): Predictions in the format (x1, y1, x2, y2, conf, class).
            save_conf (bool): Whether to save confidence scores.
            shape (tuple[int, int]): Shape of the original image.
            file (Path): File path to save the detections.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            masks=torch.as_tensor(predn["masks"], dtype=torch.uint8),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Save one JSON result for COCO evaluation.

        Args:
            predn (dict[str, torch.Tensor]): Predictions containing bboxes, masks, confidence scores, and classes.
            pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.
        """

        def to_string(counts: list[int]) -> str:
            """Converts the RLE object into a compact string representation. Each count is delta-encoded and
            variable-length encoded as a string.

            Args:
                counts (list[int]): List of RLE counts.
            """
            result = []

            for i in range(len(counts)):
                x = int(counts[i])

                # Apply delta encoding for all counts after the second entry
                if i > 2:
                    x -= int(counts[i - 2])

                # Variable-length encode the value
                while True:
                    c = x & 0x1F  # Take 5 bits
                    x >>= 5

                    # If the sign bit (0x10) is set, continue if x != -1;
                    # otherwise, continue if x != 0
                    more = (x != -1) if (c & 0x10) else (x != 0)
                    if more:
                        c |= 0x20  # Set continuation bit
                    c += 48  # Shift to ASCII
                    result.append(chr(c))
                    if not more:
                        break

            return "".join(result)

        def multi_encode(pixels: torch.Tensor) -> list[int]:
            """Convert multiple binary masks using Run-Length Encoding (RLE).

            Args:
                pixels (torch.Tensor): A 2D tensor where each row represents a flattened binary mask with shape [N,
                    H*W].

            Returns:
                (list[int]): A list of RLE counts for each mask.
            """
            transitions = pixels[:, 1:] != pixels[:, :-1]
            row_idx, col_idx = torch.where(transitions)
            col_idx = col_idx + 1

            # Compute run lengths
            counts = []
            for i in range(pixels.shape[0]):
                positions = col_idx[row_idx == i]
                if len(positions):
                    count = torch.diff(positions).tolist()
                    count.insert(0, positions[0].item())
                    count.append(len(pixels[i]) - positions[-1].item())
                else:
                    count = [len(pixels[i])]

                # Ensure starting with background (0) count
                if pixels[i][0].item() == 1:
                    count = [0, *count]
                counts.append(count)

            return counts

        pred_masks = predn["masks"].transpose(2, 1).contiguous().view(len(predn["masks"]), -1)  # N, H*W
        h, w = predn["masks"].shape[1:3]
        counts = multi_encode(pred_masks)
        rles = []
        for c in counts:
            rles.append({"size": [h, w], "counts": to_string(c)})
        super().pred_to_json(predn, pbatch)
        for i, r in enumerate(rles):
            self.jdict[-len(rles) + i]["segmentation"] = r  # segmentation

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scales predictions to the original image size."""
        return {
            **super().scale_preds(predn, pbatch),
            "masks": ops.scale_masks(predn["masks"][None], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])[
                0
            ].byte(),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Return COCO-style instance segmentation evaluation metrics."""
        pred_json = self.save_dir / "predictions.json"  # predictions
        anno_json = (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )  # annotations
        return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "segm"], suffix=["Box", "Mask"])
