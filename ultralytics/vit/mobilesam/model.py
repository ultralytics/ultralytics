# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
SAM model interface
"""
import cv2
import numpy as np

from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import ops, plt_settings
from ultralytics.yolo.v8.detect import DetectionValidator

from ...yolo.utils.torch_utils import model_info
from ..sam.build import build_sam
from .predict import Predictor


class MobileSAM(DetectionValidator):

    def __init__(self, model='mobile_sam.pt') -> None:
        if model and not model.endswith('.pt') and not model.endswith('.pth'):
            raise NotImplementedError('Mobile Segment anything prediction requires pre-trained checkpoint')
        self.model = build_sam(model)
        self.task = 'segment'
        self.predictor = None

    def init_metrics(self, model):
        super().init_metrics(model)
        self.plot_masks = []
        if self.args.save_json:
            self.process = ops.process_mask_upsample  # more accurate
        else:
            self.process = ops.process_mask  # faster

    def predict(self, source, stream=False, **kwargs):
        overrides = dict(conf=0.25, task='segment', mode='predict')
        overrides.update(kwargs)  # prefer kwargs
        if not self.predictor:
            self.predictor = Predictor(overrides=overrides)
            self.predictor.setup_model(model=self.model)
        else:
            self.predictor.args = get_cfg(self.predictor.args, overrides)
        return self.predictor(source, stream=stream)

    @plt_settings()
    def predict_point(self, source, point, label, stream=False, **kwargs):
        overrides = dict(conf=0.25, task='segment', mode='predict')
        overrides.update(kwargs)  # prefer kwargs
        image = cv2.imread(source)
        source = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_point = point
        input_label = label
        input_point = np.array(input_point)
        input_label = np.array(input_label)

        self.predictor = Predictor(overrides=overrides)
        self.predictor.predict_point(model=self.model, source=source, input_point=input_point, input_label=input_label)
        return 'Point prompt ' + str(input_point)

    @plt_settings()
    def predict_box(self, source, input_box, stream=False, **kwargs):
        overrides = dict(conf=0.25, task='segment', mode='predict')
        overrides.update(kwargs)  # prefer kwargs

        image = cv2.imread(source)
        source = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_box = np.array(input_box)
        self.predictor = Predictor(overrides=overrides)
        self.predictor.predict_box(model=self.model, source=source, input_box=input_box)
        return 'Box prompt ' + str(input_box)

    def train(self, **kwargs):
        """Function trains models but raises an error as SAM models do not support training."""
        raise NotImplementedError("MobileSAM models don't support training")

    def val(self, **kwargs):
        """Run validation given dataset."""
        raise NotImplementedError("MobileSAM models don't support validation")

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
