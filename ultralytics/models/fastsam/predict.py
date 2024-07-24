# Ultralytics YOLO 🚀, AGPL-3.0 license
import torch

from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils.metrics import box_iou

from .utils import adjust_bboxes_to_image_border


class FastSAMPredictor(SegmentationPredictor):
    """
    FastSAMPredictor is specialized for fast SAM (Segment Anything Model) segmentation prediction tasks in Ultralytics
    YOLO framework.

    This class extends the SegmentationPredictor, customizing the prediction pipeline specifically for fast SAM. It
    adjusts post-processing steps to incorporate mask prediction and non-max suppression while optimizing for single-
    class segmentation.
    """

    def postprocess(self, preds, img, orig_imgs):
        """Applies box postprocess for FastSAM predictions."""
        results = super().postprocess(preds, img, orig_imgs)
        for result in results:
            full_box = torch.tensor(
                [0, 0, result.orig_shape[1], result.orig_shape[0]], device=preds[0].device, dtype=torch.float32
            )
            boxes = adjust_bboxes_to_image_border(result.boxes.xyxy, result.orig_shape)
            idx = torch.nonzero(box_iou(full_box[None], boxes) > 0.9).flatten()
            if idx.numel() != 0:
                result.boxes.xyxy[idx] = full_box
        return results

    def inference(self, im, bboxes=None, points=None, texts=None):
        """
        Perform image segmentation inference based on the given input cues, using the currently loaded image. This
        method leverages FastSAM's (Based on Ultralytics YOLOv8) architecture and CLIP model for real-time and 
        promptable segmentation tasks.

        Args:
            im (torch.Tensor): The preprocessed input image in tensor format, with shape (N, C, H, W).
            bboxes (np.ndarray | List, optional): Bounding boxes with shape (N, 4), in XYXY format.
            points (np.ndarray | List, optional): Points indicating object locations with shape (N, 2), in pixels.
            texts (List, optional): Textual prompts.
        """
        results = super().inference(im)
        if bboxes is not None:
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32, device=self.device)
            bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
            for result in results:
                pass
