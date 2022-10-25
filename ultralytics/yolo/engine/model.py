"""
Top-level YOLO model interface. First principle usage example - https://github.com/ultralytics/ultralytics/issues/13
"""
import torch
import yaml

import ultralytics.yolo as yolo
from ultralytics.yolo.utils import LOGGER
from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.yolo.utils.modeling.tasks import ClassificationModel, DetectionModel, SegmentationModel

# map head: [model, trainer]
MODEL_MAP = {
    "Classify": [ClassificationModel, 'yolo.VERSION.classify.train.ClassificationTrainer'],
    "Detect": [ClassificationModel, 'yolo.VERSION.classify.train.ClassificationTrainer'],  # temp
    "Segment": []}


class YOLO:

    def __init__(self, version=8) -> None:
        self.version = version
        self.model = None
        self.trainer = None
        self.pretrained_weights = None

    def new(self, cfg: str):
        cfg = check_yaml(cfg)  # check YAML
        self.model, self.trainer = self._get_model_and_trainer(cfg)

    def load(self, weights, autodownload=True):
        if not isinstance(self.pretrained_weights, type(None)):
            LOGGER.info("Overwriting weights")
        # TODO: weights = smart_file_loader(weights)
        if self.model:
            self.model.load(weights)
            LOGGER.info("Checkpoint loaded successfully")
        else:
            # TODO: infer model and trainer
            pass

        self.pretrained_weights = weights

    def reset(self):
        pass

    def train(self, **kwargs):
        if 'data' not in kwargs:
            raise Exception("data is required to train")
        if not self.model:
            raise Exception("model not initialized. Use .new() or .load()")
        kwargs["model"] = self.model
        trainer = self.trainer(overrides=kwargs)
        trainer.train()

    def _get_model_and_trainer(self, cfg):
        with open(cfg, encoding='ascii', errors='ignore') as f:
            cfg = yaml.safe_load(f)  # model dict
        model, trainer = MODEL_MAP[cfg["head"][-1][-2]]
        # warning: eval is unsafe. Use with caution
        trainer = eval(trainer.replace("VERSION", f"v{self.version}"))

        return model(cfg), trainer
