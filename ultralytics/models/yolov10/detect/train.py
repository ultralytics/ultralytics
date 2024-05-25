# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import YOLOv10DetectionModel
from ultralytics.utils import RANK

from .val import YOLOv10DetectionValidator


class YOLOv10DetectionTrainer(DetectionTrainer):
    """
    A class extending the DetectionValidator class for training based on YOLOv10 models.

    Example:
        ```python
        from ultralytics.models.yolov10.detect import YOLOv10DetectionTrainer

        args = dict(model='yolov10n.pt', data='coco8.yaml', epochs=3)
        trainer = YOLOv10DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """

    def get_validator(self):
        """Returns a DetectionValidator for YOLOv10 model validation."""
        self.loss_names = ("box", "cls", "dfl")
        return YOLOv10DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLOv10 detection model."""
        model = YOLOv10DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
