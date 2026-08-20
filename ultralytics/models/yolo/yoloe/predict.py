# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np
import torch

from ultralytics.data.augment import LoadVisualPrompt
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.models.yolo.segment import SegmentationPredictor


class YOLOEVPDetectPredictor(DetectionPredictor):
    """A class extending DetectionPredictor for YOLO-EVP (Enhanced Visual Prompting) predictions.

    This class provides common functionality for YOLO models that use visual prompting, including model setup, prompt
    handling, and preprocessing transformations.

    Attributes:
        model (torch.nn.Module): The YOLO model for inference.
        device (torch.device): Device to run the model on (CPU or CUDA).
        prompts (dict): Visual prompts containing class indices and bounding boxes or masks.
        visuals (torch.Tensor): The prompts rasterized against the shapes of the batch being preprocessed.

    Methods:
        setup_model: Initialize the YOLO model and set it to evaluation mode.
        set_prompts: Set the visual prompts for the model.
        is_per_image: Report whether the prompts hold one array per image.
        preprocess: Preprocess a batch of images and rasterize its visual prompts.
        inference: Run inference with visual prompts.
        get_vpe: Process source to get visual prompt embeddings.
    """

    def setup_model(self, model, verbose: bool = True):
        """Set up the model for prediction.

        Args:
            model (torch.nn.Module): Model to load or use.
            verbose (bool, optional): If True, provides detailed logging.
        """
        super().setup_model(model, verbose=verbose)
        self.done_warmup = True

    def set_prompts(self, prompts):
        """Set the visual prompts for the model.

        Args:
            prompts (dict): Dictionary containing class indices and bounding boxes or masks. Must include a 'cls' key
                with class indices.
        """
        self.prompts = prompts

    @staticmethod
    def is_per_image(prompts: dict) -> bool:
        """Return True if 'bboxes' and 'cls' hold one array per image rather than one set applied to every image."""
        return all(
            isinstance(prompts.get(k), list) and all(isinstance(x, np.ndarray) for x in prompts[k])
            for k in ("bboxes", "cls")
        )

    def preprocess(self, im):
        """Preprocess a batch and rasterize its visual prompts."""
        imgs = super().preprocess(im)
        dst_shape = tuple(imgs.shape[2:])  # one letterboxed shape per batch, since preprocess stacks the images
        # tensor sources skip letterboxing, so their src and dst shapes are identical
        src_shapes = [dst_shape] * len(im) if isinstance(im, torch.Tensor) else [x.shape[:2] for x in im]
        self.visuals = self._prompts_to_tensor(dst_shape, src_shapes)
        return imgs

    def _prompts_to_tensor(self, dst_shape, src_shapes):
        """Rasterize prompts into a batched tensor on the model device."""
        bboxes, category = self.prompts.get("bboxes", None), self.prompts["cls"]
        if not self.is_per_image(self.prompts):  # one flat set, rasterized against every image in the batch
            masks = self.prompts.get("masks", None)
            visuals = [self._process_single_image(dst_shape, src, category, bboxes, masks) for src in src_shapes]
        else:
            assert len(src_shapes) == len(category) == len(bboxes), (
                f"Expected same length for all inputs, but got {len(src_shapes)}vs{len(category)}vs{len(bboxes)}!"
            )
            visuals = [
                self._process_single_image(dst_shape, src, category[i], bboxes[i]) for i, src in enumerate(src_shapes)
            ]
        prompts = torch.nn.utils.rnn.pad_sequence(visuals, batch_first=True).to(self.device)  # (B, N, H, W)
        return prompts.half() if self.model.fp16 else prompts.float()

    def _process_single_image(self, dst_shape, src_shape, category, bboxes=None, masks=None):
        """Resize one image's prompts and generate its visuals."""
        if bboxes is not None and len(bboxes):
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.ndim == 1:
                bboxes = bboxes[None, :]
            # Calculate scaling factor and adjust bounding boxes
            gain = min(dst_shape[0] / src_shape[0], dst_shape[1] / src_shape[1])  # gain = old / new
            bboxes *= gain
            bboxes[..., 0::2] += round((dst_shape[1] - round(src_shape[1] * gain)) / 2 - 0.1)
            bboxes[..., 1::2] += round((dst_shape[0] - round(src_shape[0] * gain)) / 2 - 0.1)
        elif masks is not None:
            # Resize and process masks
            resized_masks = super().pre_transform(masks)
            masks = np.stack(resized_masks)  # (N, H, W)
            masks[masks == 114] = 0  # Reset padding values to 0
        else:
            raise ValueError("Please provide valid bboxes or masks")

        # Generate visuals using the visual prompt loader
        return LoadVisualPrompt().get_visuals(category, dst_shape, bboxes, masks)

    def inference(self, im, *args, **kwargs):
        """Run inference with visual prompts."""
        return super().inference(im, *args, vpe=self.visuals, **kwargs)

    def get_vpe(self, source):
        """Extract visual prompt embeddings from one source image."""
        self.setup_source(source)
        assert len(self.dataset) == 1, "get_vpe only supports one image!"
        for _, im0s, _ in self.dataset:
            im = self.preprocess(im0s)
            return self.model(im, vpe=self.visuals, return_vpe=True)


class YOLOEVPSegPredictor(YOLOEVPDetectPredictor, SegmentationPredictor):
    """Predictor for YOLO-EVP segmentation tasks combining detection and segmentation capabilities."""
