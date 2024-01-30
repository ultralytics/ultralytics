

from ultralytics.models.yolo.model import YOLO
from ultralytics.utils.tlc.detect.trainer import TLCDetectionTrainer
from ultralytics.utils.tlc.detect.validator import TLCDetectionValidator
from ultralytics.utils.tlc.detect.nn import TLCDetectionModel

class TLCYOLO(YOLO):
    """ YOLO (You Only Look Once) object detection model with 3LC integration. """
    @property
    def task_map(self):
        """ Map head to model, trainer, validator, and predictor classes. """
        task_map = super().task_map
        task_map["detect"]["model"] = TLCDetectionModel
        task_map["detect"]["trainer"] = TLCDetectionTrainer
        task_map["detect"]["validator"] = TLCDetectionValidator

        return task_map
