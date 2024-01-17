from ultralytics.models.yolo.detect import DetectionValidator

from ultralytics.utils import LOGGER

class TLCDetectionValidator(DetectionValidator):
    """A class extending the BaseTrainer class for training a detection model using the 3LC."""

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        LOGGER.info("Using 3LC Validator 🌟")
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)