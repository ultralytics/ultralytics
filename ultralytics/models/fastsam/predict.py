# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
from PIL import Image

from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils import DEFAULT_CFG, checks
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import scale_masks
from ultralytics.utils.torch_utils import TORCH_1_10

from .utils import adjust_bboxes_to_image_border


class FastSAMPredictor(SegmentationPredictor):
    """
    FastSAMPredictor is specialized for fast SAM (Segment Anything Model) segmentation prediction tasks.

    This class extends the SegmentationPredictor, customizing the prediction pipeline specifically for fast SAM. It
    adjusts post-processing steps to incorporate mask prediction and non-maximum suppression while optimizing for
    single-class segmentation.

    Attributes:
        prompts (dict): Dictionary containing prompt information for segmentation (bboxes, points, labels, texts).
        device (torch.device): Device on which model and tensors are processed.
        clip_model (Any, optional): CLIP model for text-based prompting, loaded on demand.
        clip_preprocess (Any, optional): CLIP preprocessing function for images, loaded on demand.

    Methods:
        postprocess: Apply postprocessing to FastSAM predictions and handle prompts.
        prompt: Perform image segmentation inference based on various prompt types.
        set_prompts: Set prompts to be used during inference.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the FastSAMPredictor with configuration and callbacks.

        This initializes a predictor specialized for Fast SAM (Segment Anything Model) segmentation tasks. The predictor
        extends SegmentationPredictor with custom post-processing for mask prediction and non-maximum suppression
        optimized for single-class segmentation.

        Args:
            cfg (dict): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides.
            _callbacks (list, optional): List of callback functions.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.prompts = {}

    def postprocess(self, preds, img, orig_imgs):
        """
        Apply postprocessing to FastSAM predictions and handle prompts.

        Args:
            preds (list[torch.Tensor]): Raw predictions from the model.
            img (torch.Tensor): Input image tensor that was fed to the model.
            orig_imgs (list[np.ndarray]): Original images before preprocessing.

        Returns:
            (list[Results]): Processed results with prompts applied.
        """
        bboxes = self.prompts.pop("bboxes", None)
        points = self.prompts.pop("points", None)
        labels = self.prompts.pop("labels", None)
        texts = self.prompts.pop("texts", None)
        results = super().postprocess(preds, img, orig_imgs)
        for result in results:
            full_box = torch.tensor(
                [0, 0, result.orig_shape[1], result.orig_shape[0]], device=preds[0].device, dtype=torch.float32
            )
            boxes = adjust_bboxes_to_image_border(result.boxes.xyxy, result.orig_shape)
            idx = torch.nonzero(box_iou(full_box[None], boxes) > 0.9).flatten()
            if idx.numel() != 0:
                result.boxes.xyxy[idx] = full_box

        return self.prompt(results, bboxes=bboxes, points=points, labels=labels, texts=texts)

    def prompt(self, results, bboxes=None, points=None, labels=None, texts=None):
        """
        Perform image segmentation inference based on cues like bounding boxes, points, and text prompts.

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
            masks = result.masks.data
            if masks.shape[1:] != result.orig_shape:
                masks = scale_masks(masks[None], result.orig_shape)[0]
            # bboxes prompt
            idx = torch.zeros(len(result), dtype=torch.bool, device=self.device)
            if bboxes is not None:
                bboxes = torch.as_tensor(bboxes, dtype=torch.int32, device=self.device)
                bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
                bbox_areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
                mask_areas = torch.stack([masks[:, b[1] : b[3], b[0] : b[2]].sum(dim=(1, 2)) for b in bboxes])
                full_mask_areas = torch.sum(masks, dim=(1, 2))

                union = bbox_areas[:, None] + full_mask_areas - mask_areas
                idx[torch.argmax(mask_areas / union, dim=1)] = True
            if points is not None:
                points = torch.as_tensor(points, dtype=torch.int32, device=self.device)
                points = points[None] if points.ndim == 1 else points
                if labels is None:
                    labels = torch.ones(points.shape[0])
                labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
                assert len(labels) == len(points), (
                    f"Expected `labels` with same size as `point`, but got {len(labels)} and {len(points)}"
                )
                point_idx = (
                    torch.ones(len(result), dtype=torch.bool, device=self.device)
                    if labels.sum() == 0  # all negative points
                    else torch.zeros(len(result), dtype=torch.bool, device=self.device)
                )
                for point, label in zip(points, labels):
                    point_idx[torch.nonzero(masks[:, point[1], point[0]], as_tuple=True)[0]] = bool(label)
                idx |= point_idx
            if texts is not None:
                if isinstance(texts, str):
                    texts = [texts]
                crop_ims, filter_idx = [], []
                for i, b in enumerate(result.boxes.xyxy.tolist()):
                    x1, y1, x2, y2 = (int(x) for x in b)
                    if (masks[i].sum() if TORCH_1_10 else masks[i].sum(0).sum()) <= 100:  # torch 1.9 bug workaround
                        filter_idx.append(i)
                        continue
                    crop_ims.append(Image.fromarray(result.orig_img[y1:y2, x1:x2, ::-1]))
                similarity = self._clip_inference(crop_ims, texts)
                text_idx = torch.argmax(similarity, dim=-1)  # (M, )
                if len(filter_idx):
                    text_idx += (torch.tensor(filter_idx, device=self.device)[None] <= int(text_idx)).sum(0)
                idx[text_idx] = True

            prompt_results.append(result[idx])

        return prompt_results

    def _clip_inference(self, images, texts):
        """
        Perform CLIP inference to calculate similarity between images and text prompts.

        Args:
            images (list[PIL.Image]): List of source images, each should be PIL.Image with RGB channel order.
            texts (list[str]): List of prompt texts, each should be a string object.

        Returns:
            (torch.Tensor): Similarity matrix between given images and texts with shape (M, N).
        """
        try:
            import clip
        except ImportError:
            checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip
        if (not hasattr(self, "clip_model")) or (not hasattr(self, "clip_preprocess")):
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        images = torch.stack([self.clip_preprocess(image).to(self.device) for image in images])
        tokenized_text = clip.tokenize(texts).to(self.device)
        image_features = self.clip_model.encode_image(images)
        text_features = self.clip_model.encode_text(tokenized_text)
        image_features /= image_features.norm(dim=-1, keepdim=True)  # (N, 512)
        text_features /= text_features.norm(dim=-1, keepdim=True)  # (M, 512)
        return (image_features * text_features[:, None]).sum(-1)  # (M, N)

    def set_prompts(self, prompts):
        """Set prompts to be used during inference."""
        self.prompts = prompts
