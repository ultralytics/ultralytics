# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from multiprocessing.pool import ThreadPool

import numpy as np
import torch
from PIL import Image

from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils import DEFAULT_CFG, NUM_THREADS
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import clip_boxes, clip_coords
from ultralytics.utils.torch_utils import TORCH_1_10

from .utils import adjust_bboxes_to_image_border


class FastSAMPredictor(SegmentationPredictor):
    """FastSAMPredictor is specialized for fast SAM (Segment Anything Model) segmentation prediction tasks.

    This class extends the SegmentationPredictor, customizing the prediction pipeline specifically for fast SAM. It
    adjusts post-processing steps to incorporate mask prediction and non-maximum suppression while optimizing for
    single-class segmentation.

    Attributes:
        prompts (dict): Dictionary containing prompt information for segmentation (bboxes, points, labels, texts).
        device (torch.device): Device on which model and tensors are processed.
        clip (Any, optional): CLIP model used for text-based prompting, loaded on demand.

    Methods:
        finalize_results: Select FastSAM prompts in original-image coordinates.
        prompt: Perform image segmentation inference based on various prompt types.
        set_prompts: Set prompts to be used during inference.
    """

    default_iou = 0.7  # nested single-class masks, not the YOLO predict default
    _defer_prompts = True  # finalize_results reads them for every batch under the inference lock

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """Initialize the FastSAMPredictor with configuration and callbacks.

        This initializes a predictor specialized for Fast SAM (Segment Anything Model) segmentation tasks. The predictor
        extends SegmentationPredictor with custom post-processing for mask prediction and non-maximum suppression
        optimized for single-class segmentation.

        Args:
            cfg (dict): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides.
            _callbacks (dict, optional): Dictionary of callback functions.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.prompts = {}

    def finalize_results(self, results: list) -> list:
        """Adjust source-space masks and select prompts after any tiles have been merged."""
        bboxes = self.prompts.get("bboxes")
        points = self.prompts.get("points")
        labels = self.prompts.get("labels")
        texts = self.prompts.get("texts")
        for result in results:
            full_box = torch.tensor(
                [0, 0, result.orig_shape[1], result.orig_shape[0]], device=result.boxes.data.device, dtype=torch.float32
            )
            boxes = adjust_bboxes_to_image_border(result.boxes.xyxy, result.orig_shape)
            idx = torch.nonzero(box_iou(full_box[None], boxes)[0] > 0.9).flatten()
            if idx.numel() != 0:
                result.boxes.xyxy[idx] = full_box

        return self.prompt(results, bboxes=bboxes, points=points, labels=labels, texts=texts)

    def prompt(self, results, bboxes=None, points=None, labels=None, texts=None):
        """Perform image segmentation inference based on cues like bounding boxes, points, and text prompts.

        Args:
            results (Results | list[Results]): Original inference results from FastSAM models without any prompts.
            bboxes (np.ndarray | list, optional): Bounding boxes with shape (N, 4), in XYXY format.
            points (np.ndarray | list, optional): Points indicating object locations with shape (N, 2), in pixels.
            labels (np.ndarray | list, optional): Labels for point prompts, shape (N, ). 1 = foreground, 0 = background.
            texts (str | list[str], optional): Textual prompts, a list containing string objects.

        Returns:
            (list[Results]): Output results filtered and determined by the provided prompts.
        """
        if bboxes is None and points is None and texts is None:
            return results
        prompt_results = []
        if not isinstance(results, list):
            results = [results]
        for result in results:
            if len(result) == 0:
                prompt_results.append(result)
                continue
            crops = list(result.masks.crops(result.orig_shape))  # each mask inside its box, at the image size

            def window(crop, x1, y1, x2, y2):
                """Return the part of a mask inside an image window."""
                (bx1, by1, _, _), m = crop
                return m[max(y1 - by1, 0) : max(y2 - by1, 0), max(x1 - bx1, 0) : max(x2 - bx1, 0)]

            # bboxes prompt
            idx = torch.zeros(len(result), dtype=torch.bool, device=self.device)
            if bboxes is not None:
                boxes = torch.as_tensor(bboxes, dtype=torch.int32, device=self.device).clone()
                boxes = boxes[None] if boxes.ndim == 1 else boxes
                boxes = clip_boxes(boxes, result.orig_shape)
                bbox_areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                mask_areas = torch.stack([torch.stack([window(c, *b).sum() for c in crops]) for b in boxes.tolist()])
                full_mask_areas = torch.stack([m.sum() for _, m in crops])

                union = bbox_areas[:, None] + full_mask_areas - mask_areas
                idx[torch.argmax(mask_areas / union, dim=1)[mask_areas.any(dim=1)]] = True
            if points is not None:
                coords = torch.as_tensor(points, dtype=torch.int64, device=self.device).clone()
                coords = coords[None] if coords.ndim == 1 else coords
                coords = clip_coords(coords, tuple(x - 1 for x in result.orig_shape))
                if labels is None:
                    labels = torch.ones(coords.shape[0])
                labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
                assert len(labels) == len(coords), "Labels and points must contain the same number of items."
                point_masks = torch.stack(
                    [torch.stack([window(c, x, y, x + 1, y + 1).any() for x, y in coords.tolist()]) for c in crops]
                ).bool()
                point_idx = (
                    torch.ones(len(result), dtype=torch.bool, device=self.device)
                    if labels.sum() == 0  # all negative points
                    else point_masks[:, labels == 1].any(1)
                )
                point_idx &= ~point_masks[:, labels == 0].any(1)
                idx |= point_idx
            if texts is not None:
                if isinstance(texts, str):
                    texts = [texts]
                crop_ims, keep = [], []
                for i, (b, c) in enumerate(zip(result.boxes.xyxy.tolist(), crops)):
                    x1, y1, x2, y2 = (int(x) for x in b)
                    if (c[1].sum() if TORCH_1_10 else c[1].sum(0).sum()) <= 100:  # torch 1.9 bug workaround
                        continue
                    keep.append(i)
                    crop = result.orig_img[y1:y2, x1:x2]
                    part, m = window(c, x1, y1, x2, y2).cpu().numpy(), np.zeros((*crop.shape[:2], 1), crop.dtype)
                    oy, ox = max(c[0][1] - y1, 0), max(c[0][0] - x1, 0)  # where the mask's box starts in the crop
                    m[oy : oy + part.shape[0], ox : ox + part.shape[1], 0] = part
                    crop = crop * m
                    crop_ims.append(Image.fromarray(crop[:, :, ::-1]))
                if crop_ims:  # the area filter can leave no masks to score
                    similarity = self._clip_inference(crop_ims, texts)  # (M, N)
                    idx[torch.tensor(keep, device=self.device)[similarity.argmax(-1)]] = True  # best mask per text

            prompt_results.append(result[idx])

        return prompt_results

    def _clip_inference(self, images, texts):
        """Perform CLIP inference to calculate similarity between images and text prompts.

        Args:
            images (list[PIL.Image]): List of source images, each should be PIL.Image with RGB channel order.
            texts (list[str]): List of prompt texts, each should be a string object.

        Returns:
            (torch.Tensor): Similarity matrix between texts and images with shape (M, N), where M is the number of texts
                and N is the number of images.
        """
        from ultralytics.nn.text_model import MobileCLIPTS

        if not hasattr(self, "clip"):
            self.clip = MobileCLIPTS(self.device, "clip_b32_datacomp.ts")
        if self.device.type == "cuda" and NUM_THREADS > 1 and len(images) >= 2 * NUM_THREADS:
            with ThreadPool(NUM_THREADS) as pool:
                images = pool.map(self.clip.image_preprocess, images)
            images = torch.stack([image.to(self.device) for image in images])
        else:
            images = torch.stack([self.clip.image_preprocess(image).to(self.device) for image in images])
        image_features = self.clip.encode_image(images)
        text_features = self.clip.encode_text(self.clip.tokenize(texts))
        return text_features @ image_features.T  # (M, N)

    def set_prompts(self, prompts):
        """Set prompts to be used during inference."""
        self.prompts = prompts
