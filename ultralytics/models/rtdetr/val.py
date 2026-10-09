# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch

from ultralytics.data import YOLODataset
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

    Examples:
        Initialize an RT-DETR dataset
        >>> dataset = RTDETRDataset(img_path="path/to/images", data={"names": {0: "person"}}, imgsz=640)
        >>> image, hw0, hw = dataset.load_image(0)
    """

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
            >>> dataset = RTDETRDataset(img_path="path/to/images", data={"names": {0: "person"}})
            >>> image, hw0, hw = dataset.load_image(0)
        """
        return super().load_image(i=i, rect_mode=rect_mode)


class RTDETRValidator(DetectionValidator):
    """Validator extending DetectionValidator for the RT-DETR (Real-Time DETR) object detection model.

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
