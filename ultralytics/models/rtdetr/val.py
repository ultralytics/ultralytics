# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, colorstr, ops

from .detr_augment import rtdetr_transforms

__all__ = ("RTDETRValidator",)  # tuple or list


class RTDETRDataset(YOLODataset):
    """Real-Time DEtection and TRacking (RT-DETR) dataset class extending the base YOLODataset class.

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

        This constructor sets up a dataset specifically optimized for the RT-DETR (Real-Time DEtection and TRacking)
        model, building upon the base YOLODataset functionality.

        Args:
            *args (Any): Variable length argument list passed to the parent YOLODataset class.
            data (dict | None): Dictionary containing dataset information. If None, default values will be used.
            **kwargs (Any): Additional keyword arguments passed to the parent YOLODataset class.
        """
        self.rtdetr_augmentations = kwargs["hyp"].rtdetr_augmentations
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

    def build_transforms(self, hyp=None):
        """Build transformation pipeline for the dataset.

        Args:
            hyp (dict, optional): Hyperparameters for transformations.

        Returns:
            (Compose): Composition of transformation functions.
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            if self.rtdetr_augmentations:
                transforms = rtdetr_transforms(self, self.imgsz, hyp, stretch=True)
            else:
                transforms = v8_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            # transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, scale_fill=True)])
            transforms = Compose([])
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

    @staticmethod
    def _pop_batch_flag(batch: dict[str, Any], key: str) -> bool:
        """Pop a boolean batch marker emitted by RT-DETR transforms."""
        marker = batch.pop(key, None)
        if isinstance(marker, torch.Tensor):
            return bool(marker.view(-1)[0].item()) if marker.numel() else False
        if isinstance(marker, (list, tuple)):
            return bool(marker[0]) if marker else False
        return bool(marker) if marker is not None else False

    @staticmethod
    def _normalize_input(img: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet mean/std normalization to BCHW tensors."""
        if img.shape[1] != 3:
            raise ValueError(f"rtdetr_input_normalize expects 3-channel input, got shape={tuple(img.shape)}.")
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        mean_t = img.new_tensor(mean).view(1, 3, 1, 1)
        std_t = img.new_tensor(std).view(1, 3, 1, 1)
        return (img - mean_t) / std_t

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics and apply optional decoder eval_idx override.

        Walks the model for transformer-decoder submodules that expose both `eval_idx` and `num_layers`
        (covers RT-DETR's DeformableTransformerDecoder and D-FINE/DEIM's DFineTransformerDecoder/
        DeimTransformerDecoder) and rewrites their `eval_idx` to the value requested via `self.args.eval_idx`.
        Negative indices follow the same `num_layers + eval_idx` convention used at construction
        (-1 selects the last layer). No-op when `self.args.eval_idx` is unset or no matching decoder is found.

        Args:
            model (torch.nn.Module): Model being validated.
        """
        super().init_metrics(model)
        requested = getattr(self.args, "eval_idx", None)
        if requested is None:
            return
        requested = int(requested)
        decoders = [m for m in model.modules() if hasattr(m, "eval_idx") and hasattr(m, "num_layers")]
        if not decoders:
            return
        for decoder in decoders:
            idx = requested if requested >= 0 else decoder.num_layers + requested
            n_available = len(decoder.layers) if hasattr(decoder, "layers") else decoder.num_layers
            if not 0 <= idx < n_available:
                raise ValueError(
                    f"eval_idx={requested} resolves to layer {idx}, outside [0, {n_available - 1}]; "
                    f"checkpoint may have been truncated via convert_to_deploy()."
                )
            decoder.eval_idx = idx
        LOGGER.info(f"Decoder eval_idx override applied: layer {decoders[0].eval_idx} (requested={requested}).")

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess validation batch."""
        already_scaled = self._pop_batch_flag(batch, "img_scaled")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        if not already_scaled:
            batch["img"] = batch["img"] / 255
        if bool(self.args.rtdetr_input_normalize):
            batch["img"] = self._normalize_input(batch["img"])
        return batch

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
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Return predictions unchanged as RT-DETR handles scaling in postprocessing."""
        return predn

    def postprocess(
        self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        """Apply post-processing to prediction outputs.

        The top-k selection is already performed inside the decoder head. This method only converts
        coordinates from normalized xywh to pixel xyxy format.

        Args:
            preds (torch.Tensor | list | tuple): Predictions from the model with shape (batch_size, num_queries, 6)
                where last dimension is [cx, cy, w, h, score, class].

        Returns:
            (list[dict[str, torch.Tensor]]): List of dictionaries for each image, each containing:
                - 'bboxes': Tensor of shape (N, 4) with bounding box coordinates in xyxy pixel format
                - 'conf': Tensor of shape (N,) with confidence scores
                - 'cls': Tensor of shape (N,) with class indices
        """
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
            preds = [preds, None]

        bboxes, scores, labels = preds[0].split((4, 1, 1), dim=-1)
        bboxes = ops.xywh2xyxy(bboxes) * self.args.imgsz

        return [
            {"bboxes": b, "conf": s.squeeze(-1), "cls": l.squeeze(-1)}
            for b, s, l in zip(bboxes, scores, labels)
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
