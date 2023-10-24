# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops


class DecathlonPredictor(BasePredictor):
    """
    A class extending the BasePredictor class.

    This class is empty, as Decathlon is only for training (although the individual prediction heads could be kept)
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        raise NotImplementedError

    def preprocess(self, img):
        raise NotImplementedError

    def postprocess(self, preds, img, orig_imgs):
        raise NotImplementedError
