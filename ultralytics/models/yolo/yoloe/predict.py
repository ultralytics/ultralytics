# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np
import torch

from ultralytics.data.augment import LoadVisualPrompt
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.models.yolo.segment import SegmentationPredictor


class YOLOEVPDetectPredictor(DetectionPredictor):
    """
    A mixin class for YOLO-EVP (Enhanced Visual Prompting) predictors.

    This mixin provides common functionality for YOLO models that use visual prompting, including
    model setup, prompt handling, and preprocessing transformations.

    Attributes:
        model (torch.nn.Module): The YOLO model for inference.
        device (torch.device): Device to run the model on (CPU or CUDA).
        prompts (dict | torch.Tensor): Visual prompts containing class indices and bounding boxes or masks.

    Methods:
        setup_model: Initialize the YOLO model and set it to evaluation mode.
        set_prompts: Set the visual prompts for the model.
        pre_transform: Preprocess images and prompts before inference.
        inference: Run inference with visual prompts.
        get_vpe: Process source to get visual prompt embeddings.
    """

    def setup_model(self, model, verbose: bool = True):
        """
        Set up the model for prediction.

        Args:
            model (torch.nn.Module): Model to load or use.
            verbose (bool, optional): If True, provides detailed logging.
        """
        super().setup_model(model, verbose=verbose)
        self.done_warmup = True

    def set_prompts(self, prompts):
        """
        Set the visual prompts for the model.

        Args:
            prompts (dict): Dictionary containing class indices and bounding boxes or masks.
                Must include a 'cls' key with class indices.
        """
        self.prompts = prompts

    def pre_transform(self, im):
        """
        Preprocess images and prompts before inference.

        This method applies letterboxing to the input image and transforms the visual prompts
        (bounding boxes or masks) accordingly.

        Args:
            im (list): List containing a single input image.

        Returns:
            (list): Preprocessed image ready for model inference.

        Raises:
            ValueError: If neither valid bounding boxes nor masks are provided in the prompts.
        """
        img = super().pre_transform(im)
        bboxes = self.prompts.pop("bboxes", None)
        masks = self.prompts.pop("masks", None)
        category = self.prompts["cls"]
        if len(img) == 1:
            visuals = self._process_single_image(img[0].shape[:2], im[0].shape[:2], category, bboxes, masks)
            prompts = visuals.unsqueeze(0).to(self.device)  # (1, N, H, W)
            prompts = prompts.half() if self.model.fp16 else prompts.float()
        else:
            # NOTE: only supports bboxes as prompts for now
            assert bboxes is not None, f"Expected bboxes, but got {bboxes}!"
            # NOTE: needs list[np.ndarray]
            assert isinstance(bboxes, list) and all(isinstance(b, np.ndarray) for b in bboxes), (
                f"Expected list[np.ndarray], but got {bboxes}!"
            )
            assert isinstance(category, list) and all(isinstance(b, np.ndarray) for b in category), (
                f"Expected list[np.ndarray], but got {category}!"
            )
            assert len(im) == len(category) == len(bboxes), (
                f"Expected same length for all inputs, but got {len(im)}vs{len(category)}vs{len(bboxes)}!"
            )
            visuals = [
                self._process_single_image(img[i].shape[:2], im[i].shape[:2], category[i], bboxes[i]).to(self.device)
                for i in range(len(img))
            ]
            prompts = torch.nn.utils.rnn.pad_sequence(visuals, batch_first=True).to(self.device)  # (B, N, H, W)
        self.prompts = prompts.half() if self.model.fp16 else prompts.float()
        return img

    def _process_single_image(self, dst_shape, src_shape, category, bboxes=None, masks=None):
        """
        Process a single image by resizing bounding boxes or masks and generating visuals.

        Args:
            dst_shape (tuple): The target shape (height, width) of the image.
            src_shape (tuple): The original shape (height, width) of the image.
            category (str): The category of the image for visual prompts.
            bboxes (list | np.ndarray, optional): A list of bounding boxes in the format [x1, y1, x2, y2].
            masks (np.ndarray, optional): A list of masks corresponding to the image.

        Returns:
            (torch.Tensor): The processed visuals for the image.

        Raises:
            ValueError: If neither `bboxes` nor `masks` are provided.
        """
        if bboxes is not None and len(bboxes):
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.ndim == 1:
                bboxes = bboxes[None, :]
            # Calculate scaling factor and adjust bounding boxes
            gain = min(dst_shape[0] / src_shape[0], dst_shape[1] / src_shape[1])  # gain = old / new
            bboxes *= gain
            bboxes[..., 0::2] += round((dst_shape[1] - src_shape[1] * gain) / 2 - 0.1)
            bboxes[..., 1::2] += round((dst_shape[0] - src_shape[0] * gain) / 2 - 0.1)
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
        """
        Run inference with visual prompts.

        Args:
            im (torch.Tensor): Input image tensor.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            (torch.Tensor): Model prediction results.
        """
        return super().inference(im, vpe=self.prompts, *args, **kwargs)

    def get_vpe(self, source):
        """
        Process the source to get the visual prompt embeddings (VPE).

        Args:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | list | tuple): The source
                of the image to make predictions on. Accepts various types including file paths, URLs, PIL
                images, numpy arrays, and torch tensors.

        Returns:
            (torch.Tensor): The visual prompt embeddings (VPE) from the model.
        """
        self.setup_source(source)
        for _, im0s, _ in self.dataset:
            if len(im0s) > 1:  # multiple refer_image
                categories = self.prompts["cls"]
                prompts = self.prompts  # original prompts
                vpes = []
                for i, im in enumerate(im0s):
                    self.prompts = {k: prompts[k][i] for k in ["bboxes", "cls"]}  # extract prompt for current image
                    im = self.preprocess([im])
                    vpes += [self.model(im, vpe=self.prompts, return_vpe=True)]
                vpe_stack = []  # vpe for each class processed separately
                for cls_id in range(len(self.model.names)):  # merge embeddings classwise
                    cls_stack = []  # collect embeddings for same class
                    for vpe, cat in zip(vpes, categories):
                        if cls_id in cat:  # image may not have prompts for current class
                            cls_stack += [vpe[0, list(set(cat)).index(cls_id)]]  # index of class ID in cls list
                    vpe_stack += [torch.nn.functional.normalize(torch.stack(cls_stack).mean(0), p=2, dim=-1)]
                return torch.stack(vpe_stack)[None]
            else:
ac                im = self.preprocess(im0s)
                return self.model(im, vpe=self.prompts, return_vpe=True)


class YOLOEVPSegPredictor(YOLOEVPDetectPredictor, SegmentationPredictor):
    """Predictor for YOLO-EVP segmentation tasks combining detection and segmentation capabilities."""

    pass
