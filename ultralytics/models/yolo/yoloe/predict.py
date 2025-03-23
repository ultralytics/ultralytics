# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license


import numpy as np

from ultralytics.data.augment import LetterBox, LoadVisualPrompt
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils.torch_utils import select_device


class YOLOEVPPredictorMixin:
    """
    A mixin class for YOLO-EVP (Enhanced Visual Prompting) predictors.

    This mixin provides common functionality for YOLO models that use visual prompting, including
    model setup, prompt handling, and preprocessing transformations.

    Attributes:
        model (torch.nn.Module): The YOLO model for inference.
        device (torch.device): Device to run the model on (CPU or CUDA).
        prompts (dict): Visual prompts containing class indices and bounding boxes or masks.

    Methods:
        setup_model: Initialize the YOLO model and set it to evaluation mode.
        set_return_vpe: Set whether to return visual prompt embeddings.
        set_prompts: Set the visual prompts for the model.
        pre_transform: Preprocess images and prompts before inference.
        inference: Run inference with visual prompts.
    """

    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        device = select_device(self.args.device, verbose=verbose)
        self.model = model.to(device)

        self.device = device  # update device
        self.model.fp16 = False
        self.args.half = False
        self.model.eval()

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
        letterbox = LetterBox(
            self.imgsz,
            auto=False,
            stride=int(self.model.stride[-1].item()),
        )
        assert len(im) == 1, f"Expected 1 image, but got {len(im)} images!"

        img = letterbox(image=im[0])
        bboxes, masks = None, None
        if "bboxes" in self.prompts and len(self.prompts["bboxes"]) > 0:
            bboxes = np.array(self.prompts["bboxes"])
            if bboxes.ndim == 1:
                bboxes = bboxes[None, :]
            dst_shape, src_shape = img.shape[:2], im[0].shape[:2]
            gain = min(dst_shape[0] / src_shape[0], dst_shape[1] / src_shape[1])  # gain  = old / new
            bboxes *= gain
            bboxes[..., 0::2] += round((dst_shape[1] - src_shape[1] * gain) / 2 - 0.1)
            bboxes[..., 1::2] += round((dst_shape[0] - src_shape[0] * gain) / 2 - 0.1)
        elif "masks" in self.prompts:
            masks = self.prompts["masks"]

            resized_masks = [letterbox(image=masks[i]) for i in range(len(masks))]
            masks = np.stack(resized_masks)  # (N, H, W)
            masks[masks == 114] = 0  # Reset padding values to 0
        else:
            raise ValueError("Please provide valid bboxes or masks")

        cls = self.prompts["cls"]
        visuals = LoadVisualPrompt().get_visuals(cls, img.shape[:2], bboxes, masks)

        self.prompts = visuals.unsqueeze(0).to(self.device)  # (1, N, H, W)
        self.model.model[-1].nc = self.prompts.shape[1]
        self.model.names = [f"object{i}" for i in range(len(np.unique(cls)))]

        return [img]

    def inference(self, im, set_vpe=False, *args, **kwargs):
        """
        Run inference with visual prompts.

        Args:
            im (torch.Tensor): Input image tensor.
            set_vpe (bool): Whether to set visual prompt embeddings.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            (torch.Tensor): Model prediction results.
        """
        if set_vpe:
            vpe = self.model(im, vpe=self.prompts, return_vpe=True)
            self.model.set_classes(self.model.names, vpe)
        return super().inference(im, vpe=self.prompts, *args, **kwargs)


# TODO: Implement additional functionality
class YOLOEVPDetectPredictor(YOLOEVPPredictorMixin, DetectionPredictor):
    """Predictor for YOLOE VP detection."""

    pass


class YOLOEVPSegPredictor(YOLOEVPPredictorMixin, SegmentationPredictor):
    """Predictor for YOLOE VP segmentation."""

    pass
