# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
SAM model interface
"""

from ultralytics.yolo.cfg import get_cfg

from ...yolo.utils.torch_utils import model_info
from .build import build_sam
from .modules.prompt_predictor import PromptPredictor
from .predict import Predictor


class SAM:

    def __init__(self, model='sam_b.pt') -> None:
        if model and not model.endswith('.pt') and not model.endswith('.pth'):
            # Should raise AssertionError instead?
            raise NotImplementedError('Segment anything prediction requires pre-trained checkpoint')
        self.model = build_sam(model)
        self.task = 'segment'  # required
        self.predictor = None  # reuse predictor
        self.prompt_predictor = None  # predictor for prompts

    def predict(self, source, stream=False, boxes=None, points=None, labels=None, **kwargs):
        """Predicts and returns segmentation masks for given image or video source."""
        overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024)
        overrides.update(kwargs)  # prefer kwargs
        if not self.predictor:
            self.predictor = Predictor(overrides=overrides)
            self.predictor.setup_model(model=self.model)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, overrides)
        return self.predictor(source, stream=stream, boxes=boxes, points=points, labels=labels)

    def predict_box(self, source, input_bbox):
        """Predicts and returns segmentation masks for given image and box."""
        # NOTE: currently only supports np.ndarray for `source` and list for `input_bbox`.
        import cv2
        import numpy as np
        image = cv2.imread(source)
        source = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_bbox = np.array(input_bbox)
        if not self.prompt_predictor:
            self.prompt_predictor = PromptPredictor(self.model)
        self.prompt_predictor.set_image(source)
        masks, _, _ = self.prompt_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_bbox[None, :],
            multimask_output=False,
        )
        # TODO: use Results class to handle masks
        return masks

    # TODO: predict_point
    def predict_point(self, source, point):
        pass

    def train(self, **kwargs):
        """Function trains models but raises an error as SAM models do not support training."""
        raise NotImplementedError("SAM models don't support training")

    def val(self, **kwargs):
        """Run validation given dataset."""
        raise NotImplementedError("SAM models don't support validation")

    def __call__(self, source=None, stream=False, **kwargs):
        """Calls the 'predict' function with given arguments to perform object detection."""
        return self.predict(source, stream, **kwargs)

    def __getattr__(self, attr):
        """Raises error if object has no requested attribute."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    def info(self, detailed=False, verbose=True):
        """
        Logs model info.

        Args:
            detailed (bool): Show detailed information about model.
            verbose (bool): Controls verbosity.
        """
        return model_info(self.model, detailed=detailed, verbose=verbose)
