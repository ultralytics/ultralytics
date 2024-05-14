# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import HumanModel
from ultralytics.utils import DEFAULT_CFG, RANK


class HumanTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a human model.

    Example:
        ```python
        from ultralytics.models.yolo.human import HumanTrainer

        args = dict(model='yolov8n-human.pt', data='coco8.yaml', epochs=3)
        trainer = HumanTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a HumanTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return HumanModel initialized with specified config and weights."""
        model = HumanModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of HumanValidator for validation of YOLO model."""
        self.loss_names = (
            "box_loss",
            "seg_loss",
            "cls_loss",
            "dfl_loss",
            "w_loss",
            "h_loss",
            "g_loss",
            "a_loss",
            "r_loss",
        )
        return yolo.human.HumanValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        # TODO
        pass
