# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model validation metrics."""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.utils import LOGGER, DataExportMixin, SimpleClass, TryExcept, checks, plt_settings

OKS_SIGMA = (
    np.array(
        [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89],
        dtype=np.float32,
    )
    / 10.0
)


def bbox_ioa(box1: np.ndarray, box2: np.ndarray, iou: bool = False, eps: float = 1e-7) -> np.ndarray:
    """Calculate the intersection over box2 area given box1 and box2.

    Args:
        box1 (np.ndarray): A numpy array of shape (N, 4) representing N bounding boxes in x1y1x2y2 format.
        box2 (np.ndarray): A numpy array of shape (M, 4) representing M bounding boxes in x1y1x2y2 format.
        iou (bool, optional): Calculate the standard IoU if True else return inter_area/box2_area.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (np.ndarray): A numpy array of shape (N, M) representing the intersection over box2 area.
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)

    # Box2 area
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # Intersection over box2 area
    return inter_area / (area + eps)


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Calculate intersection-over-union (IoU) of boxes.

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes in (x1, y1, x2, y2) format.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes in (x1, y1, x2, y2) format.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.

    References:
        https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py
    """
    # NOTE: Need .float() to get accurate iou values
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Calculate the Intersection over Union (IoU) between bounding boxes.

    This function supports various shapes for `box1` and `box2` as long as the last dimension is 4. For instance, you
    may pass tensors shaped like (4,), (N, 4), (B, N, 4), or (B, N, 1, 4). Internally, the code will split the last
    dimension into (x, y, w, h) if `xywh=True`, or (x1, y1, x2, y2) if `xywh=False`.

    Args:
        box1 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        box2 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in (x1, y1,
            x2, y2) format.
        GIoU (bool, optional): If True, calculate Generalized IoU.
        DIoU (bool, optional): If True, calculate Distance IoU.
        CIoU (bool, optional): If True, calculate Complete IoU.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def mask_iou(mask1: torch.Tensor, mask2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Calculate masks IoU.

    Args:
        mask1 (torch.Tensor): A tensor of shape (N, n) where N is the number of ground truth objects and n is the
            product of image width and height.
        mask2 (torch.Tensor): A tensor of shape (M, n) where M is the number of predicted objects and n is the product
            of image width and height.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing masks IoU.
    """
    intersection = torch.matmul(mask1, mask2.T).clamp_(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


def kpt_iou(
    kpt1: torch.Tensor, kpt2: torch.Tensor, area: torch.Tensor, sigma: list[float], eps: float = 1e-7
) -> torch.Tensor:
    """Calculate Object Keypoint Similarity (OKS).

    Args:
        kpt1 (torch.Tensor): A tensor of shape (N, 17, 3) representing ground truth keypoints.
        kpt2 (torch.Tensor): A tensor of shape (M, 17, 3) representing predicted keypoints.
        area (torch.Tensor): A tensor of shape (N,) representing areas from ground truth.
        sigma (list): A list containing 17 values representing keypoint scales.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing keypoint similarities.
    """
    d = (kpt1[:, None, :, 0] - kpt2[..., 0]).pow(2) + (kpt1[:, None, :, 1] - kpt2[..., 1]).pow(2)  # (N, M, 17)
    sigma = torch.tensor(sigma, device=kpt1.device, dtype=kpt1.dtype)  # (17, )
    kpt_mask = kpt1[..., 2] != 0  # (N, 17)
    e = d / ((2 * sigma).pow(2) * (area[:, None, None] + eps) * 2)  # from cocoeval
    # e = d / ((area[None, :, None] + eps) * sigma) ** 2 / 2  # from formula
    return ((-e).exp() * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)


def _get_covariance_matrix(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate covariance matrix from oriented bounding boxes.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance matrices corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def probiou(obb1: torch.Tensor, obb2: torch.Tensor, CIoU: bool = False, eps: float = 1e-7) -> torch.Tensor:
    """Calculate probabilistic IoU between oriented bounding boxes.

    Args:
        obb1 (torch.Tensor): Ground truth OBBs, shape (N, 5), format xywhr.
        obb2 (torch.Tensor): Predicted OBBs, shape (N, 5), format xywhr.
        CIoU (bool, optional): If True, calculate CIoU.
        eps (float, optional): Small value to avoid division by zero.

    Returns:
        (torch.Tensor): OBB similarities, shape (N,).

    Notes:
        OBB format: [center_x, center_y, width, height, rotation_angle].

    References:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    if CIoU:  # only include the wh aspect ratio part
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU
    return iou


def batch_probiou(obb1: torch.Tensor | np.ndarray, obb2: torch.Tensor | np.ndarray, eps: float = 1e-7) -> torch.Tensor:
    """Calculate the probabilistic IoU between oriented bounding boxes.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.

    References:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def smooth_bce(eps: float = 0.1) -> tuple[float, float]:
    """Compute smoothed positive and negative Binary Cross-Entropy targets.

    Args:
        eps (float, optional): The epsilon value for label smoothing.

    Returns:
        pos (float): Positive label smoothing BCE target.
        neg (float): Negative label smoothing BCE target.

    References:
        https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class ConfusionMatrix(DataExportMixin):
    """A class for calculating and updating a confusion matrix for object detection and classification tasks.

    Attributes:
        task (str): The type of task, either 'detect' or 'classify'.
        matrix (np.ndarray): The confusion matrix, with dimensions depending on the task.
        nc (int): The number of classes.
        names (list[str]): The names of the classes, used as labels on the plot.
        matches (dict): Contains the indices of ground truths and predictions categorized into TP, FP and FN.
    """

    def __init__(self, names: dict[int, str] = [], task: str = "detect", save_matches: bool = False):
        """Initialize a ConfusionMatrix instance.

        Args:
            names (dict[int, str], optional): Names of classes, used as labels on the plot.
            task (str, optional): Type of task, either 'detect' or 'classify'.
            save_matches (bool, optional): Save the indices of GTs, TPs, FPs, FNs for visualization.
        """
        self.task = task
        self.nc = len(names)  # number of classes
        self.matrix = np.zeros((self.nc, self.nc)) if self.task == "classify" else np.zeros((self.nc + 1, self.nc + 1))
        self.names = names  # name of classes
        self.matches = {} if save_matches else None

    def _append_matches(self, mtype: str, batch: dict[str, Any], idx: int) -> None:
        """Append the matches to TP, FP, FN or GT list for the last batch.

        This method updates the matches dictionary by appending specific batch data to the appropriate match type (True
        Positive, False Positive, or False Negative).

        Args:
            mtype (str): Match type identifier ('TP', 'FP', 'FN' or 'GT').
            batch (dict[str, Any]): Batch data containing detection results with keys like 'bboxes', 'cls', 'conf',
                'keypoints', 'masks'.
            idx (int): Index of the specific detection to append from the batch.

        Notes:
            For masks, handles both overlap and non-overlap cases. When masks.max() > 1.0, it indicates
            overlap_mask=True with shape (1, H, W), otherwise uses direct indexing.
        """
        if self.matches is None:
            return
        for k, v in batch.items():
            if k in {"bboxes", "cls", "conf", "keypoints"}:
                self.matches[mtype][k] += v[[idx]]
            elif k == "masks":
                # NOTE: masks.max() > 1.0 means overlap_mask=True with (1, H, W) shape
                self.matches[mtype][k] += [v[0] == idx + 1] if v.max() > 1.0 else [v[idx]]

    def process_cls_preds(self, preds: list[torch.Tensor], targets: list[torch.Tensor]) -> None:
        """Update confusion matrix for classification task.

        Args:
            preds (list[N, min(nc,5)]): Predicted class labels.
            targets (list[N, 1]): Ground truth class labels.
        """
        preds, targets = torch.cat(preds)[:, 0], torch.cat(targets)
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[p][t] += 1

    def process_batch(
        self,
        detections: dict[str, torch.Tensor],
        batch: dict[str, Any],
        conf: float = 0.25,
        iou_thres: float = 0.45,
    ) -> None:
        """Update confusion matrix for object detection task.

        Args:
            detections (dict[str, torch.Tensor]): Dictionary containing detected bounding boxes and their associated
                information. Should contain 'cls', 'conf', and 'bboxes' keys, where 'bboxes' can be Array[N, 4] for
                regular boxes or Array[N, 5] for OBB with angle.
            batch (dict[str, Any]): Batch dictionary containing ground truth data with 'bboxes' (Array[M, 4]| Array[M,
                5]) and 'cls' (Array[M]) keys, where M is the number of ground truth objects.
            conf (float, optional): Confidence threshold for detections.
            iou_thres (float, optional): IoU threshold for matching detections to ground truth.
        """
        gt_cls, gt_bboxes = batch["cls"], batch["bboxes"]
        if self.matches is not None:  # only if visualization is enabled
            self.matches = {k: defaultdict(list) for k in {"TP", "FP", "FN", "GT"}}
            for i in range(gt_cls.shape[0]):
                self._append_matches("GT", batch, i)  # store GT
        is_obb = gt_bboxes.shape[1] == 5  # check if boxes contains angle for OBB
        conf = 0.25 if conf in {None, 0.01 if is_obb else 0.001} else conf  # apply 0.25 if default val conf is passed
        no_pred = detections["cls"].shape[0] == 0
        if gt_cls.shape[0] == 0:  # Check if labels is empty
            if not no_pred:
                detections = {k: detections[k][detections["conf"] > conf] for k in detections}
                detection_classes = detections["cls"].int().tolist()
                for i, dc in enumerate(detection_classes):
                    self.matrix[dc, self.nc] += 1  # FP
                    self._append_matches("FP", detections, i)
            return
        if no_pred:
            gt_classes = gt_cls.int().tolist()
            for i, gc in enumerate(gt_classes):
                self.matrix[self.nc, gc] += 1  # FN
                self._append_matches("FN", batch, i)
            return

        detections = {k: detections[k][detections["conf"] > conf] for k in detections}
        gt_classes = gt_cls.int().tolist()
        detection_classes = detections["cls"].int().tolist()
        bboxes = detections["bboxes"]
        iou = batch_probiou(gt_bboxes, bboxes) if is_obb else box_iou(gt_bboxes, bboxes)

        x = torch.where(iou > iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                dc = detection_classes[m1[j].item()]
                self.matrix[dc, gc] += 1  # TP if class is correct else both an FP and an FN
                if dc == gc:
                    self._append_matches("TP", detections, m1[j].item())
                else:
                    self._append_matches("FP", detections, m1[j].item())
                    self._append_matches("FN", batch, i)
            else:
                self.matrix[self.nc, gc] += 1  # FN
                self._append_matches("FN", batch, i)

        for i, dc in enumerate(detection_classes):
            if not any(m1 == i):
                self.matrix[dc, self.nc] += 1  # FP
                self._append_matches("FP", detections, i)

    def matrix(self):
        """Return the confusion matrix."""
        return self.matrix

    def tp_fp(self) -> tuple[np.ndarray, np.ndarray]:
        """Return true positives and false positives.

        Returns:
            tp (np.ndarray): True positives.
            fp (np.ndarray): False positives.
        """
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return (tp, fp) if self.task == "classify" else (tp[:-1], fp[:-1])  # remove background class if task=detect

    def plot_matches(self, img: torch.Tensor, im_file: str, save_dir: Path) -> None:
        """Plot grid of GT, TP, FP, FN for each image.

        Args:
            img (torch.Tensor): Image to plot onto.
            im_file (str): Image filename to save visualizations.
            save_dir (Path): Location to save the visualizations to.
        """
        if not self.matches:
            return
        from .ops import xyxy2xywh
        from .plotting import plot_images

        # Create batch of 4 (GT, TP, FP, FN)
        labels = defaultdict(list)
        for i, mtype in enumerate(["GT", "FP", "TP", "FN"]):
            mbatch = self.matches[mtype]
            if "conf" not in mbatch:
                mbatch["conf"] = torch.tensor([1.0] * len(mbatch["bboxes"]), device=img.device)
            mbatch["batch_idx"] = torch.ones(len(mbatch["bboxes"]), device=img.device) * i
            for k in mbatch.keys():
                labels[k] += mbatch[k]

        labels = {k: torch.stack(v, 0) if len(v) else torch.empty(0) for k, v in labels.items()}
        if self.task != "obb" and labels["bboxes"].shape[0]:
            labels["bboxes"] = xyxy2xywh(labels["bboxes"])
        (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)
        plot_images(
            labels,
            img.repeat(4, 1, 1, 1),
            paths=["Ground Truth", "False Positives", "True Positives", "False Negatives"],
            fname=save_dir / "visualizations" / Path(im_file).name,
            names=self.names,
            max_subplots=4,
            conf_thres=0.001,
        )

    @TryExcept(msg="ConfusionMatrix plot failure")
    @plt_settings()
    def plot(self, normalize: bool = True, save_dir: str = "", on_plot=None):
        """Plot the confusion matrix using matplotlib and save it to a file.

        Args:
            normalize (bool, optional): Whether to normalize the confusion matrix.
            save_dir (str, optional): Directory where the plot will be saved.
            on_plot (callable, optional): An optional callback to pass plots path and data when they are rendered.
        """
        import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        names, n = list(self.names.values()), self.nc
        if self.nc >= 100:  # downsample for large class count
            k = max(2, self.nc // 60)  # step size for downsampling, always > 1
            keep_idx = slice(None, None, k)  # create slice instead of array
            names = names[keep_idx]  # slice class names
            array = array[keep_idx, :][:, keep_idx]  # slice matrix rows and cols
            n = (self.nc + k - 1) // k  # number of retained classes
        nc = nn = n if self.task == "classify" else n + 1  # adjust for background if needed
        ticklabels = ([*names, "background"]) if (0 < nn < 99) and (nn == nc) else "auto"
        xy_ticks = np.arange(len(ticklabels))
        tick_fontsize = max(6, 15 - 0.1 * nc)  # Minimum size is 6
        label_fontsize = max(6, 12 - 0.1 * nc)
        title_fontsize = max(6, 12 - 0.1 * nc)
        btm = max(0.1, 0.25 - 0.001 * nc)  # Minimum value is 0.1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            im = ax.imshow(array, cmap="Blues", vmin=0.0, interpolation="none")
            ax.xaxis.set_label_position("bottom")
            if nc < 30:  # Add score for each cell of confusion matrix
                color_threshold = 0.45 * (1 if normalize else np.nanmax(array))  # text color threshold
                for i, row in enumerate(array[:nc]):
                    for j, val in enumerate(row[:nc]):
                        val = array[i, j]
                        if np.isnan(val):
                            continue
                        ax.text(
                            j,
                            i,
                            f"{val:.2f}" if normalize else f"{int(val)}",
                            ha="center",
                            va="center",
                            fontsize=10,
                            color="white" if val > color_threshold else "black",
                        )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
        title = "Confusion Matrix" + " Normalized" * normalize
        ax.set_xlabel("True", fontsize=label_fontsize, labelpad=10)
        ax.set_ylabel("Predicted", fontsize=label_fontsize, labelpad=10)
        ax.set_title(title, fontsize=title_fontsize, pad=20)
        ax.set_xticks(xy_ticks)
        ax.set_yticks(xy_ticks)
        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
        ax.tick_params(axis="y", left=True, right=False, labelleft=True, labelright=False)
        if ticklabels != "auto":
            ax.set_xticklabels(ticklabels, fontsize=tick_fontsize, rotation=90, ha="center")
            ax.set_yticklabels(ticklabels, fontsize=tick_fontsize)
        for s in {"left", "right", "bottom", "top", "outline"}:
            if s != "outline":
                ax.spines[s].set_visible(False)  # Confusion matrix plot don't have outline
            cbar.ax.spines[s].set_visible(False)
        fig.subplots_adjust(left=0, right=0.84, top=0.94, bottom=btm)  # Adjust layout to ensure equal margins
        plot_fname = Path(save_dir) / f"{title.lower().replace(' ', '_')}.png"
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)
        if on_plot:
            on_plot(plot_fname)

    def print(self):
        """Print the confusion matrix to the console."""
        for i in range(self.matrix.shape[0]):
            LOGGER.info(" ".join(map(str, self.matrix[i])))

    def summary(self, normalize: bool = False, decimals: int = 5) -> list[dict[str, float]]:
        """Generate a summarized representation of the confusion matrix as a list of dictionaries, with optional
        normalization. This is useful for exporting the matrix to various formats such as CSV, XML, HTML, JSON,
        or SQL.

        Args:
            normalize (bool): Whether to normalize the confusion matrix values.
            decimals (int): Number of decimal places to round the output values to.

        Returns:
            (list[dict[str, float]]): A list of dictionaries, each representing one predicted class with corresponding
                values for all actual classes.

        Examples:
            >>> results = model.val(data="coco8.yaml", plots=True)
            >>> cm_dict = results.confusion_matrix.summary(normalize=True, decimals=5)
            >>> print(cm_dict)
        """
        import re

        names = list(self.names.values()) if self.task == "classify" else [*list(self.names.values()), "background"]
        clean_names, seen = [], set()
        for name in names:
            clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
            original_clean = clean_name
            counter = 1
            while clean_name.lower() in seen:
                clean_name = f"{original_clean}_{counter}"
                counter += 1
            seen.add(clean_name.lower())
            clean_names.append(clean_name)
        array = (self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)).round(decimals)
        return [
            dict({"Predicted": clean_names[i]}, **{clean_names[j]: array[i, j] for j in range(len(clean_names))})
            for i in range(len(clean_names))
        ]


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


@plt_settings()
def plot_pr_curve(
    px: np.ndarray,
    py: np.ndarray,
    ap: np.ndarray,
    save_dir: Path = Path("pr_curve.png"),
    names: dict[int, str] = {},
    on_plot=None,
):
    """Plot precision-recall curve.

    Args:
        px (np.ndarray): X values for the PR curve.
        py (np.ndarray): Y values for the PR curve.
        ap (np.ndarray): Average precision values.
        save_dir (Path, optional): Path to save the plot.
        names (dict[int, str], optional): Dictionary mapping class indices to class names.
        on_plot (callable, optional): Function to call after plot is saved.
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="gray")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


@plt_settings()
def plot_mc_curve(
    px: np.ndarray,
    py: np.ndarray,
    save_dir: Path = Path("mc_curve.png"),
    names: dict[int, str] = {},
    xlabel: str = "Confidence",
    ylabel: str = "Metric",
    on_plot=None,
):
    """Plot metric-confidence curve.

    Args:
        px (np.ndarray): X values for the metric-confidence curve.
        py (np.ndarray): Y values for the metric-confidence curve.
        save_dir (Path, optional): Path to save the plot.
        names (dict[int, str], optional): Dictionary mapping class indices to class names.
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.
        on_plot (callable, optional): Function to call after plot is saved.
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="gray")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.1)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


def compute_ap(recall: list[float], precision: list[float]) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        ap (float): Average precision.
        mpre (np.ndarray): Precision envelope curve.
        mrec (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        func = np.trapezoid if checks.check_version(np.__version__, ">=2.0") else np.trapz  # np.trapz deprecated
        ap = func(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    plot: bool = False,
    on_plot=None,
    save_dir: Path = Path(),
    names: dict[int, str] = {},
    eps: float = 1e-16,
    prefix: str = "",
) -> tuple:
    """Compute the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        plot (bool, optional): Whether to plot PR curves or not.
        on_plot (callable, optional): A callback to pass plots path and data when they are rendered.
        save_dir (Path, optional): Directory to save the PR curves.
        names (dict[int, str], optional): Dictionary of class names to plot PR curves.
        eps (float, optional): A small value to avoid division by zero.
        prefix (str, optional): A prefix string for saving the plot files.

    Returns:
        tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.
        fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class.
        p (np.ndarray): Precision values at threshold given by max F1 metric for each class.
        r (np.ndarray): Recall values at threshold given by max F1 metric for each class.
        f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class.
        ap (np.ndarray): Average precision for each class at different IoU thresholds.
        unique_classes (np.ndarray): An array of unique classes that have data.
        p_curve (np.ndarray): Precision curves for each class.
        r_curve (np.ndarray): Recall curves for each class.
        f1_curve (np.ndarray): F1-score curves for each class.
        x (np.ndarray): X-axis values for the curves.
        prec_values (np.ndarray): Precision values at mAP@0.5 for each class.
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    x, prec_values = np.linspace(0, 1, 1000), []

    # Average precision, precision and recall curves
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

    prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))  # (nc, 1000)

    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = {i: names[k] for i, k in enumerate(unique_classes) if k in names}  # dict: only classes that have data
    if plot:
        plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
        plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
        plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
        plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)

    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values


class Metric(SimpleClass):
    """Class for computing evaluation metrics for Ultralytics YOLO models.

    Attributes:
        p (list): Precision for each class. Shape: (nc,).
        r (list): Recall for each class. Shape: (nc,).
        f1 (list): F1 score for each class. Shape: (nc,).
        all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
        ap_class_index (list): Index of class for each AP score. Shape: (nc,).
        nc (int): Number of classes.

    Methods:
        ap50: AP at IoU threshold of 0.5 for all classes.
        ap: AP at IoU thresholds from 0.5 to 0.95 for all classes.
        mp: Mean precision of all classes.
        mr: Mean recall of all classes.
        map50: Mean AP at IoU threshold of 0.5 for all classes.
        map75: Mean AP at IoU threshold of 0.75 for all classes.
        map: Mean AP at IoU thresholds from 0.5 to 0.95 for all classes.
        mean_results: Mean of results, returns mp, mr, map50, map.
        class_result: Class-aware result, returns p[i], r[i], ap50[i], ap[i].
        maps: mAP of each class.
        fitness: Model fitness as a weighted combination of metrics.
        update: Update metric attributes with new evaluation results.
        curves: Provides a list of curves for accessing specific metrics like precision, recall, F1, etc.
        curves_results: Provide a list of results for accessing specific metrics like precision, recall, F1, etc.
    """

    def __init__(self) -> None:
        """Initialize a Metric instance for computing evaluation metrics for the YOLOv8 model."""
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )
        self.nc = 0

    @property
    def ap50(self) -> np.ndarray | list:
        """Return the Average Precision (AP) at an IoU threshold of 0.5 for all classes.

        Returns:
            (np.ndarray | list): Array of shape (nc,) with AP50 values per class, or an empty list if not available.
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self) -> np.ndarray | list:
        """Return the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes.

        Returns:
            (np.ndarray | list): Array of shape (nc,) with AP50-95 values per class, or an empty list if not available.
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self) -> float:
        """Return the Mean Precision of all classes.

        Returns:
            (float): The mean precision of all classes.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self) -> float:
        """Return the Mean Recall of all classes.

        Returns:
            (float): The mean recall of all classes.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self) -> float:
        """Return the mean Average Precision (mAP) at an IoU threshold of 0.5.

        Returns:
            (float): The mAP at an IoU threshold of 0.5.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self) -> float:
        """Return the mean Average Precision (mAP) at an IoU threshold of 0.75.

        Returns:
            (float): The mAP at an IoU threshold of 0.75.
        """
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

    @property
    def map(self) -> float:
        """Return the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95 in steps of 0.05.

        Returns:
            (float): The mAP over IoU thresholds of 0.5 - 0.95 in steps of 0.05.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self) -> list[float]:
        """Return mean of results, mp, mr, map50, map."""
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i: int) -> tuple[float, float, float, float]:
        """Return class-aware result, p[i], r[i], ap50[i], ap[i]."""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self) -> np.ndarray:
        """Return mAP of each class."""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def fitness(self) -> float:
        """Return model fitness as a weighted combination of metrics."""
        w = [0.0, 0.0, 0.0, 1.0]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        return (np.nan_to_num(np.array(self.mean_results())) * w).sum()

    def update(self, results: tuple):
        """Update the evaluation metrics with a new set of results.

        Args:
            results (tuple): A tuple containing evaluation metrics:
                - p (list): Precision for each class.
                - r (list): Recall for each class.
                - f1 (list): F1 score for each class.
                - all_ap (list): AP scores for all classes and all IoU thresholds.
                - ap_class_index (list): Index of class for each AP score.
                - p_curve (list): Precision curve for each class.
                - r_curve (list): Recall curve for each class.
                - f1_curve (list): F1 curve for each class.
                - px (list): X values for the curves.
                - prec_values (list): Precision values for each class.
        """
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
            self.p_curve,
            self.r_curve,
            self.f1_curve,
            self.px,
            self.prec_values,
        ) = results

    @property
    def curves(self) -> list:
        """Return a list of curves for accessing specific metrics curves."""
        return []

    @property
    def curves_results(self) -> list[list]:
        """Return a list of curves for accessing specific metrics curves."""
        return [
            [self.px, self.prec_values, "Recall", "Precision"],
            [self.px, self.f1_curve, "Confidence", "F1"],
            [self.px, self.p_curve, "Confidence", "Precision"],
            [self.px, self.r_curve, "Confidence", "Recall"],
        ]


class DetMetrics(SimpleClass, DataExportMixin):
    """Utility class for computing detection metrics such as precision, recall, and mean average precision (mAP).

    Attributes:
        names (dict[int, str]): A dictionary of class names.
        box (Metric): An instance of the Metric class for storing detection results.
        speed (dict[str, float]): A dictionary for storing execution times of different parts of the detection process.
        task (str): The task type, set to 'detect'.
        stats (dict[str, list]): A dictionary containing lists for true positives, confidence scores, predicted classes,
            target classes, and target images.
        nt_per_class: Number of targets per class.
        nt_per_image: Number of targets per image.

    Methods:
        update_stats: Update statistics by appending new values to existing stat collections.
        process: Process predicted results for object detection and update metrics.
        clear_stats: Clear the stored statistics.
        keys: Return a list of keys for accessing specific metrics.
        mean_results: Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95.
        class_result: Return the result of evaluating the performance of an object detection model on a specific class.
        maps: Return mean Average Precision (mAP) scores per class.
        fitness: Return the fitness of box object.
        ap_class_index: Return the average precision index per class.
        results_dict: Return dictionary of computed performance metrics and statistics.
        curves: Return a list of curves for accessing specific metrics curves.
        curves_results: Return a list of computed performance metrics and statistics.
        summary: Generate a summarized representation of per-class detection metrics as a list of dictionaries.
    """

    def __init__(self, names: dict[int, str] = {}) -> None:
        """Initialize a DetMetrics instance with a save directory, plot flag, and class names.

        Args:
            names (dict[int, str], optional): Dictionary of class names.
        """
        self.names = names
        self.box = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "detect"
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.nt_per_class = None
        self.nt_per_image = None

    def update_stats(self, stat: dict[str, Any]) -> None:
        """Update statistics by appending new values to existing stat collections.

        Args:
            stat (dict[str, any]): Dictionary containing new statistical values to append. Keys should match existing
                keys in self.stats.
        """
        for k in self.stats.keys():
            self.stats[k].append(stat[k])

    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
        """Process predicted results for object detection and update metrics.

        Args:
            save_dir (Path): Directory to save plots. Defaults to Path(".").
            plot (bool): Whether to plot precision-recall curves. Defaults to False.
            on_plot (callable, optional): Function to call after plots are generated. Defaults to None.

        Returns:
            (dict[str, np.ndarray]): Dictionary containing concatenated statistics arrays.
        """
        stats = {k: np.concatenate(v, 0) for k, v in self.stats.items()}  # to numpy
        if not stats:
            return stats
        results = ap_per_class(
            stats["tp"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            plot=plot,
            save_dir=save_dir,
            names=self.names,
            on_plot=on_plot,
            prefix="Box",
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results)
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=len(self.names))
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=len(self.names))
        return stats

    def clear_stats(self):
        """Clear the stored statistics."""
        for v in self.stats.values():
            v.clear()

    @property
    def keys(self) -> list[str]:
        """Return a list of keys for accessing specific metrics."""
        return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]

    def mean_results(self) -> list[float]:
        """Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95."""
        return self.box.mean_results()

    def class_result(self, i: int) -> tuple[float, float, float, float]:
        """Return the result of evaluating the performance of an object detection model on a specific class."""
        return self.box.class_result(i)

    @property
    def maps(self) -> np.ndarray:
        """Return mean Average Precision (mAP) scores per class."""
        return self.box.maps

    @property
    def fitness(self) -> float:
        """Return the fitness of box object."""
        return self.box.fitness()

    @property
    def ap_class_index(self) -> list:
        """Return the average precision index per class."""
        return self.box.ap_class_index

    @property
    def results_dict(self) -> dict[str, float]:
        """Return dictionary of computed performance metrics and statistics."""
        keys = [*self.keys, "fitness"]
        values = ((float(x) if hasattr(x, "item") else x) for x in ([*self.mean_results(), self.fitness]))
        return dict(zip(keys, values))

    @property
    def curves(self) -> list[str]:
        """Return a list of curves for accessing specific metrics curves."""
        return ["Precision-Recall(B)", "F1-Confidence(B)", "Precision-Confidence(B)", "Recall-Confidence(B)"]

    @property
    def curves_results(self) -> list[list]:
        """Return a list of computed performance metrics and statistics."""
        return self.box.curves_results

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]:
        """Generate a summarized representation of per-class detection metrics as a list of dictionaries. Includes
        shared scalar metrics (mAP, mAP50, mAP75) alongside precision, recall, and F1-score for each class.

        Args:
            normalize (bool): For Detect metrics, everything is normalized by default [0-1].
            decimals (int): Number of decimal places to round the metrics values to.

        Returns:
            (list[dict[str, Any]]): A list of dictionaries, each representing one class with corresponding metric
                values.

        Examples:
           >>> results = model.val(data="coco8.yaml")
           >>> detection_summary = results.summary()
           >>> print(detection_summary)
        """
        per_class = {
            "Box-P": self.box.p,
            "Box-R": self.box.r,
            "Box-F1": self.box.f1,
        }
        return [
            {
                "Class": self.names[self.ap_class_index[i]],
                "Images": self.nt_per_image[self.ap_class_index[i]],
                "Instances": self.nt_per_class[self.ap_class_index[i]],
                **{k: round(v[i], decimals) for k, v in per_class.items()},
                "mAP50": round(self.class_result(i)[2], decimals),
                "mAP50-95": round(self.class_result(i)[3], decimals),
            }
            for i in range(len(per_class["Box-P"]))
        ]


class SegmentMetrics(DetMetrics):
    """Calculate and aggregate detection and segmentation metrics over a given set of classes.

    Attributes:
        names (dict[int, str]): Dictionary of class names.
        box (Metric): An instance of the Metric class for storing detection results.
        seg (Metric): An instance of the Metric class to calculate mask segmentation metrics.
        speed (dict[str, float]): A dictionary for storing execution times of different parts of the detection process.
        task (str): The task type, set to 'segment'.
        stats (dict[str, list]): A dictionary containing lists for true positives, confidence scores, predicted classes,
            target classes, and target images.
        nt_per_class: Number of targets per class.
        nt_per_image: Number of targets per image.

    Methods:
        process: Process the detection and segmentation metrics over the given set of predictions.
        keys: Return a list of keys for accessing metrics.
        mean_results: Return the mean metrics for bounding box and segmentation results.
        class_result: Return classification results for a specified class index.
        maps: Return mAP scores for object detection and semantic segmentation models.
        fitness: Return the fitness score for both segmentation and bounding box models.
        curves: Return a list of curves for accessing specific metrics curves.
        curves_results: Provide a list of computed performance metrics and statistics.
        summary: Generate a summarized representation of per-class segmentation metrics as a list of dictionaries.
    """

    def __init__(self, names: dict[int, str] = {}) -> None:
        """Initialize a SegmentMetrics instance with a save directory, plot flag, and class names.

        Args:
            names (dict[int, str], optional): Dictionary of class names.
        """
        DetMetrics.__init__(self, names)
        self.seg = Metric()
        self.task = "segment"
        self.stats["tp_m"] = []  # add additional stats for masks

    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
        """Process the detection and segmentation metrics over the given set of predictions.

        Args:
            save_dir (Path): Directory to save plots. Defaults to Path(".").
            plot (bool): Whether to plot precision-recall curves. Defaults to False.
            on_plot (callable, optional): Function to call after plots are generated. Defaults to None.

        Returns:
            (dict[str, np.ndarray]): Dictionary containing concatenated statistics arrays.
        """
        stats = DetMetrics.process(self, save_dir, plot, on_plot=on_plot)  # process box stats
        results_mask = ap_per_class(
            stats["tp_m"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            plot=plot,
            on_plot=on_plot,
            save_dir=save_dir,
            names=self.names,
            prefix="Mask",
        )[2:]
        self.seg.nc = len(self.names)
        self.seg.update(results_mask)
        return stats

    @property
    def keys(self) -> list[str]:
        """Return a list of keys for accessing metrics."""
        return [
            *DetMetrics.keys.fget(self),
            "metrics/precision(M)",
            "metrics/recall(M)",
            "metrics/mAP50(M)",
            "metrics/mAP50-95(M)",
        ]

    def mean_results(self) -> list[float]:
        """Return the mean metrics for bounding box and segmentation results."""
        return DetMetrics.mean_results(self) + self.seg.mean_results()

    def class_result(self, i: int) -> list[float]:
        """Return classification results for a specified class index."""
        return DetMetrics.class_result(self, i) + self.seg.class_result(i)

    @property
    def maps(self) -> np.ndarray:
        """Return mAP scores for object detection and semantic segmentation models."""
        return DetMetrics.maps.fget(self) + self.seg.maps

    @property
    def fitness(self) -> float:
        """Return the fitness score for both segmentation and bounding box models."""
        return self.seg.fitness() + DetMetrics.fitness.fget(self)

    @property
    def curves(self) -> list[str]:
        """Return a list of curves for accessing specific metrics curves."""
        return [
            *DetMetrics.curves.fget(self),
            "Precision-Recall(M)",
            "F1-Confidence(M)",
            "Precision-Confidence(M)",
            "Recall-Confidence(M)",
        ]

    @property
    def curves_results(self) -> list[list]:
        """Return a list of computed performance metrics and statistics."""
        return DetMetrics.curves_results.fget(self) + self.seg.curves_results

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]:
        """Generate a summarized representation of per-class segmentation metrics as a list of dictionaries. Includes
        both box and mask scalar metrics (mAP, mAP50, mAP75) alongside precision, recall, and F1-score for
        each class.

        Args:
            normalize (bool): For Segment metrics, everything is normalized by default [0-1].
            decimals (int): Number of decimal places to round the metrics values to.

        Returns:
            (list[dict[str, Any]]): A list of dictionaries, each representing one class with corresponding metric
                values.

        Examples:
            >>> results = model.val(data="coco8-seg.yaml")
            >>> seg_summary = results.summary(decimals=4)
            >>> print(seg_summary)
        """
        per_class = {
            "Mask-P": self.seg.p,
            "Mask-R": self.seg.r,
            "Mask-F1": self.seg.f1,
        }
        summary = DetMetrics.summary(self, normalize, decimals)  # get box summary
        for i, s in enumerate(summary):
            s.update({**{k: round(v[i], decimals) for k, v in per_class.items()}})
        return summary


class PoseMetrics(DetMetrics):
    """Calculate and aggregate detection and pose metrics over a given set of classes.

    Attributes:
        names (dict[int, str]): Dictionary of class names.
        pose (Metric): An instance of the Metric class to calculate pose metrics.
        box (Metric): An instance of the Metric class for storing detection results.
        speed (dict[str, float]): A dictionary for storing execution times of different parts of the detection process.
        task (str): The task type, set to 'pose'.
        stats (dict[str, list]): A dictionary containing lists for true positives, confidence scores, predicted classes,
            target classes, and target images.
        nt_per_class: Number of targets per class.
        nt_per_image: Number of targets per image.

    Methods:
        process: Process the detection and pose metrics over the given set of predictions. R
        keys: Return a list of keys for accessing metrics.
        mean_results: Return the mean results of box and pose.
        class_result: Return the class-wise detection results for a specific class i.
        maps: Return the mean average precision (mAP) per class for both box and pose detections.
        fitness: Return combined fitness score for pose and box detection.
        curves: Return a list of curves for accessing specific metrics curves.
        curves_results: Provide a list of computed performance metrics and statistics.
        summary: Generate a summarized representation of per-class pose metrics as a list of dictionaries.
    """

    def __init__(self, names: dict[int, str] = {}) -> None:
        """Initialize the PoseMetrics class with directory path, class names, and plotting options.

        Args:
            names (dict[int, str], optional): Dictionary of class names.
        """
        super().__init__(names)
        self.pose = Metric()
        self.task = "pose"
        self.stats["tp_p"] = []  # add additional stats for pose

    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
        """Process the detection and pose metrics over the given set of predictions.

        Args:
            save_dir (Path): Directory to save plots. Defaults to Path(".").
            plot (bool): Whether to plot precision-recall curves. Defaults to False.
            on_plot (callable, optional): Function to call after plots are generated.

        Returns:
            (dict[str, np.ndarray]): Dictionary containing concatenated statistics arrays.
        """
        stats = DetMetrics.process(self, save_dir, plot, on_plot=on_plot)  # process box stats
        results_pose = ap_per_class(
            stats["tp_p"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            plot=plot,
            on_plot=on_plot,
            save_dir=save_dir,
            names=self.names,
            prefix="Pose",
        )[2:]
        self.pose.nc = len(self.names)
        self.pose.update(results_pose)
        return stats

    @property
    def keys(self) -> list[str]:
        """Return a list of evaluation metric keys."""
        return [
            *DetMetrics.keys.fget(self),
            "metrics/precision(P)",
            "metrics/recall(P)",
            "metrics/mAP50(P)",
            "metrics/mAP50-95(P)",
        ]

    def mean_results(self) -> list[float]:
        """Return the mean results of box and pose."""
        return DetMetrics.mean_results(self) + self.pose.mean_results()

    def class_result(self, i: int) -> list[float]:
        """Return the class-wise detection results for a specific class i."""
        return DetMetrics.class_result(self, i) + self.pose.class_result(i)

    @property
    def maps(self) -> np.ndarray:
        """Return the mean average precision (mAP) per class for both box and pose detections."""
        return DetMetrics.maps.fget(self) + self.pose.maps

    @property
    def fitness(self) -> float:
        """Return combined fitness score for pose and box detection."""
        return self.pose.fitness() + DetMetrics.fitness.fget(self)

    @property
    def curves(self) -> list[str]:
        """Return a list of curves for accessing specific metrics curves."""
        return [
            *DetMetrics.curves.fget(self),
            "Precision-Recall(B)",
            "F1-Confidence(B)",
            "Precision-Confidence(B)",
            "Recall-Confidence(B)",
            "Precision-Recall(P)",
            "F1-Confidence(P)",
            "Precision-Confidence(P)",
            "Recall-Confidence(P)",
        ]

    @property
    def curves_results(self) -> list[list]:
        """Return a list of computed performance metrics and statistics."""
        return DetMetrics.curves_results.fget(self) + self.pose.curves_results

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]:
        """Generate a summarized representation of per-class pose metrics as a list of dictionaries. Includes both box
        and pose scalar metrics (mAP, mAP50, mAP75) alongside precision, recall, and F1-score for each class.

        Args:
            normalize (bool): For Pose metrics, everything is normalized by default [0-1].
            decimals (int): Number of decimal places to round the metrics values to.

        Returns:
            (list[dict[str, Any]]): A list of dictionaries, each representing one class with corresponding metric
                values.

        Examples:
            >>> results = model.val(data="coco8-pose.yaml")
            >>> pose_summary = results.summary(decimals=4)
            >>> print(pose_summary)
        """
        per_class = {
            "Pose-P": self.pose.p,
            "Pose-R": self.pose.r,
            "Pose-F1": self.pose.f1,
        }
        summary = DetMetrics.summary(self, normalize, decimals)  # get box summary
        for i, s in enumerate(summary):
            s.update({**{k: round(v[i], decimals) for k, v in per_class.items()}})
        return summary


class ClassifyMetrics(SimpleClass, DataExportMixin):
    """Class for computing classification metrics including top-1 and top-5 accuracy.

    Attributes:
        top1 (float): The top-1 accuracy.
        top5 (float): The top-5 accuracy.
        speed (dict): A dictionary containing the time taken for each step in the pipeline.
        task (str): The task type, set to 'classify'.

    Methods:
        process: Process target classes and predicted classes to compute metrics.
        fitness: Return mean of top-1 and top-5 accuracies as fitness score.
        results_dict: Return a dictionary with model's performance metrics and fitness score.
        keys: Return a list of keys for the results_dict property.
        curves: Return a list of curves for accessing specific metrics curves.
        curves_results: Provide a list of computed performance metrics and statistics.
        summary: Generate a single-row summary of classification metrics (Top-1 and Top-5 accuracy).
    """

    def __init__(self) -> None:
        """Initialize a ClassifyMetrics instance."""
        self.top1 = 0
        self.top5 = 0
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "classify"

    def process(self, targets: torch.Tensor, pred: torch.Tensor):
        """Process target classes and predicted classes to compute metrics.

        Args:
            targets (torch.Tensor): Target classes.
            pred (torch.Tensor): Predicted classes.
        """
        pred, targets = torch.cat(pred), torch.cat(targets)
        correct = (targets[:, None] == pred).float()
        acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
        self.top1, self.top5 = acc.mean(0).tolist()

    @property
    def fitness(self) -> float:
        """Return mean of top-1 and top-5 accuracies as fitness score."""
        return (self.top1 + self.top5) / 2

    @property
    def results_dict(self) -> dict[str, float]:
        """Return a dictionary with model's performance metrics and fitness score."""
        return dict(zip([*self.keys, "fitness"], [self.top1, self.top5, self.fitness]))

    @property
    def keys(self) -> list[str]:
        """Return a list of keys for the results_dict property."""
        return ["metrics/accuracy_top1", "metrics/accuracy_top5"]

    @property
    def curves(self) -> list:
        """Return a list of curves for accessing specific metrics curves."""
        return []

    @property
    def curves_results(self) -> list:
        """Return a list of curves for accessing specific metrics curves."""
        return []

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, float]]:
        """Generate a single-row summary of classification metrics (Top-1 and Top-5 accuracy).

        Args:
            normalize (bool): For Classify metrics, everything is normalized by default [0-1].
            decimals (int): Number of decimal places to round the metrics values to.

        Returns:
            (list[dict[str, float]]): A list with one dictionary containing Top-1 and Top-5 classification accuracy.

        Examples:
            >>> results = model.val(data="imagenet10")
            >>> classify_summary = results.summary(decimals=4)
            >>> print(classify_summary)
        """
        return [{"top1_acc": round(self.top1, decimals), "top5_acc": round(self.top5, decimals)}]


class OBBMetrics(DetMetrics):
    """Metrics for evaluating oriented bounding box (OBB) detection.

    Attributes:
        names (dict[int, str]): Dictionary of class names.
        box (Metric): An instance of the Metric class for storing detection results.
        speed (dict[str, float]): A dictionary for storing execution times of different parts of the detection process.
        task (str): The task type, set to 'obb'.
        stats (dict[str, list]): A dictionary containing lists for true positives, confidence scores, predicted classes,
            target classes, and target images.
        nt_per_class: Number of targets per class.
        nt_per_image: Number of targets per image.

    References:
        https://arxiv.org/pdf/2106.06072.pdf
    """

    def __init__(self, names: dict[int, str] = {}) -> None:
        """Initialize an OBBMetrics instance with directory, plotting, and class names.

        Args:
            names (dict[int, str], optional): Dictionary of class names.
        """
        DetMetrics.__init__(self, names)
        # TODO: probably remove task as well
        self.task = "obb"
