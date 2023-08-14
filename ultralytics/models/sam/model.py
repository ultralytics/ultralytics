# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
SAM model interface
"""

from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.utils.torch_utils import model_info

from .build import build_sam
from .predict import Predictor


class SAM(Model):
    """
    SAM model interface.
    """

    def __init__(self, model='sam_b.pt') -> None:
        if model and Path(model).suffix not in ('.pt', '.pth'):
            raise NotImplementedError('SAM prediction requires pre-trained *.pt or *.pth model.')
        super().__init__(model=model, task='segment')

    def _load(self, weights: str, task=None):
        self.model = build_sam(weights)

    def predict(self, source, stream=False, bboxes=None, points=None, labels=None, **kwargs):
        """Predicts and returns segmentation masks for given image or video source."""
        overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024)
        kwargs.update(overrides)
        prompts = dict(bboxes=bboxes, points=points, labels=labels)
        return super().predict(source, stream, prompts=prompts, **kwargs)

    def __call__(self, source=None, stream=False, bboxes=None, points=None, labels=None, **kwargs):
        """Calls the 'predict' function with given arguments to perform object detection."""
        return self.predict(source, stream, bboxes, points, labels, **kwargs)

    def info(self, detailed=False, verbose=True):
        """
        Logs model info.

        Args:
            detailed (bool): Show detailed information about model.
            verbose (bool): Controls verbosity.
        """
        return model_info(self.model, detailed=detailed, verbose=verbose)

    @property
    def task_map(self):
        return {'segment': {'predictor': Predictor}}
