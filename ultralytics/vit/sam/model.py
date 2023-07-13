# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
SAM model interface
"""

from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils.torch_utils import model_info

from .build import build_sam
from .predict import Predictor


class SAM:

    def __init__(self, model='sam_b.pt') -> None:
        if model and not model.endswith('.pt') and not model.endswith('.pth'):
            # Should raise AssertionError instead?
            raise NotImplementedError('Segment anything prediction requires pre-trained checkpoint')
        self.model = build_sam(model)
        self.task = 'segment'  # required
        self.predictor = None  # reuse predictor

    def predict(self, source, stream=False, bboxes=None, points=None, labels=None, **kwargs):
        """Predicts and returns segmentation masks for given image or video source."""
        overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024)
        overrides.update(kwargs)  # prefer kwargs
        if not self.predictor:
            self.predictor = Predictor(overrides=overrides)
            self.predictor.setup_model(model=self.model)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, overrides)
        return self.predictor(source, stream=stream, bboxes=bboxes, points=points, labels=labels)

    def train(self, **kwargs):
        """Function trains models but raises an error as SAM models do not support training."""
        raise NotImplementedError("SAM models don't support training")

    def val(self, **kwargs):
        """Run validation given dataset."""
        raise NotImplementedError("SAM models don't support validation")

    def __call__(self, source=None, stream=False, bboxes=None, points=None, labels=None, **kwargs):
        """Calls the 'predict' function with given arguments to perform object detection."""
        return self.predict(source, stream, bboxes, points, labels, **kwargs)

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
