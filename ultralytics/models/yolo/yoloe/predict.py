# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import deepcopy

import numpy as np
import torch

from ultralytics.data.augment import LetterBox, LoadVisualPrompt
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils.instance import Instances
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
        assert "cls" in prompts, "Please provide class index."
        self.prompts = deepcopy(prompts)

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
        assert len(im) == 1

        cls = torch.tensor(self.prompts["cls"]).unsqueeze(-1)
        if "bboxes" in self.prompts and len(self.prompts["bboxes"]) > 0:
            labels = dict(
                img=im[0],
                instances=Instances(
                    bboxes=self.prompts["bboxes"],
                    segments=np.zeros((0, 1000, 2), dtype=np.float32),
                    bbox_format="xyxy",
                    normalized=False,
                ),
                cls=cls,
            )

            labels = letterbox(labels)

            instances = labels.pop("instances")
            h, w = labels["img"].shape[:2]
            instances.normalize(w, h)
            instances.convert_bbox(format="xywh")
            labels["bboxes"] = torch.from_numpy(instances.bboxes)
        elif "masks" in self.prompts:
            masks = self.prompts["masks"]

            img = letterbox(image=im[0])
            resized_masks = []
            for i in range(len(masks)):
                resized_masks.append(letterbox(image=masks[i]))
            masks = np.stack(resized_masks)
            masks[masks == 114] = 0  # Reset padding values to 0

            labels = dict(img=img, masks=masks, cls=cls)
        else:
            raise ValueError("Please provide valid bboxes or masks")

        labels["img"] = labels["img"].transpose(2, 0, 1)

        load_vp = LoadVisualPrompt()
        labels = load_vp(labels)

        cls = np.unique(self.prompts["cls"])
        self.prompts = labels["visuals"].unsqueeze(0).to(self.device)
        self.model.model[-1].nc = self.prompts.shape[1]
        self.model.names = [f"object{cls[i]}" for i in range(self.prompts.shape[1])]

        return [labels["img"].transpose(1, 2, 0)]

    def inference(self, im, set_vpe=False, *args, **kwargs):
        """
        Run inference with visual prompts.

        Args:
            im (torch.Tensor): Input image tensor.
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
