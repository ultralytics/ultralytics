import copy
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import (
    DEFAULT_CFG, LOGGER,)
from ultralytics.utils.tlc_utils import tlc_check_dataset
from ultralytics.utils.tlc_validator import TLCDetectionValidator

class TLCDetectionTrainer(DetectionTrainer):
    """A class extending the BaseTrainer class for training a detection model using the 3LC."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        LOGGER.info("Using 3LC Trainer 🌟")
        super().__init__(cfg, overrides, _callbacks)

    @staticmethod
    def get_dataset(data):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        tables = tlc_check_dataset(data["yaml_file"], get_splits=["train", "val"])
        return data["train"], data.get("val") or data.get("test")

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return TLCDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy.copy(self.args), _callbacks=self.callbacks
        )
