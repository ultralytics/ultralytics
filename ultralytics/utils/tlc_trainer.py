import copy
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import (
    DEFAULT_CFG, LOGGER,)
from ultralytics.utils.tlc_utils import tlc_check_dataset
from ultralytics.utils.tlc_validator import TLCDetectionValidator
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.data import build_yolo_dataset
class TLCDetectionTrainer(DetectionTrainer):
    """A class extending the BaseTrainer class for training a detection model using the 3LC."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        LOGGER.info("Using 3LC Trainer 🌟")
        super().__init__(cfg, overrides, _callbacks)


    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        tables = tlc_check_dataset(self.data["yaml_file"], get_splits=["train", "val"])
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)


    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return TLCDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy.copy(self.args), _callbacks=self.callbacks
        )
