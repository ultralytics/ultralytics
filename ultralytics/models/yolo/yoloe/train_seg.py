# Ultralytics YOLO 🚀, AGPL-3.0 license

import itertools

from ultralytics.data import build_yolo_dataset
from ultralytics.models import yolo
from ultralytics.nn.tasks import YOLOEModel, YOLOESegModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import de_parallel
from copy import copy
from .train import YOLOETrainer
from .val import YOLOESegValidator

class YOLOESegTrainer(YOLOETrainer, yolo.segment.SegmentationTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return YOLOEModel initialized with specified config and weights."""
        # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
        # NOTE: Following the official config, nc hard-coded to 80 for now.
        model = YOLOESegModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=3,
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box", "seg", "cls", "dfl"
        return YOLOESegValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
