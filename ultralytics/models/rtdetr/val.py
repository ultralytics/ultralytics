# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import torch

from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, LetterBox, v8_transforms
from ultralytics.data.utils import get_split_fraction
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops

__all__ = ("RTDETRValidator",)  # tuple or list


class RTDETRDataset(YOLODataset):
    """Real-Time DEtection TRansformer (RT-DETR) dataset class extending the base YOLODataset class.

    This specialized dataset class is designed for use with the RT-DETR object detection model and is optimized for
    real-time detection and tracking tasks.

    Attributes:
        augment (bool): Whether to apply data augmentation.
        rect (bool): Whether to use rectangular training.
        use_segments (bool): Whether to use segmentation masks.
        use_keypoints (bool): Whether to use keypoint annotations.
        imgsz (int): Target image size for training.

    Methods:
        load_image: Load one image from dataset index.
        build_transforms: Build transformation pipeline for the dataset.

    Examples:
        Initialize an RT-DETR dataset
        >>> dataset = RTDETRDataset(img_path="path/to/images", imgsz=640)
        >>> image, hw0, hw = dataset.load_image(0)
    """

    def __init__(self, *args, data=None, **kwargs):
        """Initialize the RTDETRDataset class by inheriting from the YOLODataset class.

        This constructor sets up a dataset specifically optimized for the RT-DETR (Real-Time DEtection TRansformer)
        model, building upon the base YOLODataset functionality.

        Args:
            *args (Any): Variable length argument list passed to the parent YOLODataset class.
            data (dict | None): Dictionary containing dataset information. If None, default values will be used.
            **kwargs (Any): Additional keyword arguments passed to the parent YOLODataset class.
        """
        super().__init__(*args, data=data, **kwargs)

    def load_image(self, i, rect_mode=False):
        """Load one image from dataset index 'i'.

        Args:
            i (int): Index of the image to load.
            rect_mode (bool, optional): Whether to use rectangular mode for batch inference.

        Returns:
            im (np.ndarray): Loaded image as a NumPy array.
            hw_original (tuple[int, int]): Original image dimensions in (height, width) format.
            hw_resized (tuple[int, int]): Resized image dimensions in (height, width) format.

        Examples:
            Load an image from the dataset
            >>> dataset = RTDETRDataset(img_path="path/to/images")
            >>> image, hw0, hw = dataset.load_image(0)
        """
        return super().load_image(i=i, rect_mode=rect_mode)


class RTDETRValidator(DetectionValidator):
    """RTDETRValidator extends the DetectionValidator class to provide validation capabilities specifically tailored for
    the RT-DETR (Real-Time DETR) object detection model.

    The class allows building of an RTDETR-specific dataset for validation, applies confidence thresholding for
    post-processing, and updates evaluation metrics accordingly.

    Attributes:
        args (Namespace): Configuration arguments for validation.
        data (dict): Dataset configuration dictionary.

    Methods:
        build_dataset: Build an RTDETR Dataset for validation.
        postprocess: Apply confidence thresholding to prediction outputs.

    Examples:
        Initialize and run RT-DETR validation
        >>> from ultralytics.models.rtdetr import RTDETRValidator
        >>> args = dict(model="rtdetr-l.pt", data="coco8.yaml")
        >>> validator = RTDETRValidator(args=args)
        >>> validator()

    Notes:
        For further details on the attributes and methods, refer to the parent DetectionValidator class.
    """

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build an RTDETR Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str, optional): `train` mode or `val` mode, users are able to customize different augmentations for
                each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (RTDETRDataset): Dataset configured for RT-DETR validation.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # no augmentation
            hyp=self.args,
            rect=False,  # no rect
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=1.0
            if self.data.get("complete")
            else get_split_fraction(self.args.fraction, self.args.split or "val"),
        )

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Return predictions unchanged as RT-DETR handles scaling in postprocessing."""
        return predn

    def postprocess(
        self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        """Apply post-processing to prediction outputs.

        Top-k selection is already performed inside the decoder head. This method converts normalized xywh
        coordinates to pixel xyxy format.

        Args:
            preds (torch.Tensor | list | tuple): Predictions from the model with shape (batch_size, num_queries, 6),
                where the last dimension is [cx, cy, w, h, score, class].

        Returns:
            (list[dict[str, torch.Tensor]]): List of dictionaries for each image, each containing:
                - 'bboxes': Tensor of shape (N, 4) with bounding box coordinates in xyxy pixel format
                - 'conf': Tensor of shape (N,) with confidence scores
                - 'cls': Tensor of shape (N,) with class indices
        """
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        bboxes, scores, labels = preds.split((4, 1, 1), dim=-1)
        bboxes = ops.xywh2xyxy(bboxes) * self.args.imgsz
        scores, labels = scores.squeeze(-1), labels.squeeze(-1)
        masks = [(score > self.args.conf).nonzero().squeeze(1)[: self.args.max_det] for score in scores]

        return [
            {"bboxes": bbox[m], "conf": score[m], "cls": label[m]}
            for bbox, score, label, m in zip(bboxes, scores, labels, masks)
        ]

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Serialize YOLO predictions to COCO json format.

        Args:
            predn (dict[str, torch.Tensor]): Predictions dictionary containing 'bboxes', 'conf', and 'cls' keys with
                bounding box coordinates, confidence scores, and class predictions.
            pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.
        """
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = predn["bboxes"].clone()
        box[..., [0, 2]] *= pbatch["ori_shape"][1] / self.args.imgsz  # native-space pred
        box[..., [1, 3]] *= pbatch["ori_shape"][0] / self.args.imgsz  # native-space pred
        box = ops.xyxy2xywh(box)  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )


_NO_AUG_EPOCH = 4  # DEIM trains the final epochs without augmentation


def compute_deim_scheduled_prob(base_prob: float, epoch: int, stop_epoch: int) -> float:
    """Linearly decay an augmentation probability to 0 by the no-aug stage boundary.

    Args:
        base_prob (float): Probability configured in the hyperparameters.
        epoch (int): Current epoch.
        stop_epoch (int): Epoch at which the probability reaches 0.

    Returns:
        (float): Decayed probability for this epoch.
    """
    base_prob = float(base_prob)
    if base_prob <= 0.0 or stop_epoch <= 0 or epoch >= stop_epoch:
        return 0.0
    return base_prob * max(0.0, 1.0 - (float(epoch) / float(stop_epoch)))


def compute_policy_epochs(hyp) -> tuple[int, int, int]:
    """Compute DEIM stage boundaries from ``epochs`` and the fixed four-epoch no-augmentation tail.

    Args:
        hyp (SimpleNamespace | IterableSimpleNamespace): Hyperparameters carrying the total epoch count.

    Returns:
        start (int): End of stage 1, where the flat learning rate begins.
        mid (int): End of stage 2 and start of stage 3, where the cosine decay begins.
        stop (int): End of stage 3 and start of the no-augmentation tail.
    """
    epochs = max(1, int(hyp.epochs))
    stop = epochs - min(_NO_AUG_EPOCH, epochs)
    start = min(4, max(0, stop - 1))
    mid = start + (stop - start) // 2
    if not (0 <= start <= mid <= stop <= epochs):
        raise ValueError(
            f"compute_policy_epochs produced invalid boundaries: "
            f"start={start}, mid={mid}, stop={stop}, epochs={epochs}."
        )
    return start, mid, stop


class DEIMDataset(RTDETRDataset):
    """RT-DETR dataset variant that linearly decays YOLO augmentation probabilities over epochs.

    All augmentation probabilities (mosaic, mixup, copy_paste) decay from their base hyp value to 0 linearly across
    ``[0, stop_epoch]``, where ``stop_epoch`` leaves the final four epochs for the DEIM no-aug tail. Past stop_epoch
    every augmentation is hard-zeroed.
    """

    def __init__(self, *args, data=None, **kwargs):
        """Stash base hyp values then defer to the parent for normal dataset construction.

        Args:
            *args (Any): Positional arguments forwarded to RTDETRDataset.
            data (dict, optional): Dataset dictionary.
            **kwargs (Any): Keyword arguments forwarded to RTDETRDataset; hyp is required.
        """
        hyp = kwargs["hyp"]
        self.base_hyp = copy(hyp)
        self.policy_epochs = compute_policy_epochs(hyp)
        super().__init__(*args, data=data, **kwargs)
        if self.augment:
            self.set_epoch(0)

    def _build_v8_epoch_hyp(self, epoch: int):
        """Clone the base hyp and apply linear decay; zero everything past the no-aug boundary.

        Args:
            epoch (int): Current epoch.

        Returns:
            (SimpleNamespace | IterableSimpleNamespace): Copy of the base hyperparameters with the augmentation
                probabilities decayed for this epoch.
        """
        hyp = copy(self.base_hyp)
        _, _, stop = self.policy_epochs
        if epoch >= stop:
            for key in (
                "mosaic",
                "mixup",
                "copy_paste",
                "cutmix",
                "degrees",
                "translate",
                "scale",
                "shear",
                "perspective",
                "hsv_h",
                "hsv_s",
                "hsv_v",
            ):
                setattr(hyp, key, 0.0)
            hyp.augmentations = []
        else:
            hyp.mosaic = compute_deim_scheduled_prob(self.base_hyp.mosaic, epoch, stop)
            hyp.mixup = compute_deim_scheduled_prob(self.base_hyp.mixup, epoch, stop)
            hyp.copy_paste = compute_deim_scheduled_prob(self.base_hyp.copy_paste, epoch, stop)
        return hyp

    def build_transforms(self, hyp=None):
        """Build v8 transforms with current (possibly decayed) hyp values.

        Args:
            hyp (SimpleNamespace | IterableSimpleNamespace, optional): Hyperparameters for this epoch.

        Returns:
            (Compose): Transform pipeline ending in the Format transform.
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if not self.rect else 0.0
            hyp.mixup = hyp.mixup if not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if not self.rect else 0.0
            # Keep v8 MixUp inputs same-sized; current v8 Mosaic no longer carries the old mosaic_border crop hint.
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            # Matches YOLODataset/RTDETRDataset: a no-op resize on the already-square val image whose only
            # effect is rewriting ratio_pad into the ((gain_h, gain_w), (pad_w, pad_h)) form scale_boxes needs.
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms

    def set_epoch(self, epoch: int) -> None:
        """Rebuild transforms with decayed hyp probabilities for the current epoch.

        Args:
            epoch (int): Current epoch.
        """
        self.epoch = epoch
        if self.augment:
            self.transforms = self.build_transforms(hyp=self._build_v8_epoch_hyp(epoch))


class DEIMValidator(RTDETRValidator):
    """RT-DETR validator for DEIM-decoder models."""
