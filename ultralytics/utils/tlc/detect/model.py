# Ultralytics YOLO 🚀 3LC Integration, AGPL-3.0 license
import ultralytics
from ultralytics.models.yolo.model import YOLO
from ultralytics.utils.tlc.detect.nn import TLCDetectionModel
from ultralytics.utils.tlc.detect.trainer import TLCDetectionTrainer
from ultralytics.utils.tlc.detect.utils import get_names_from_yolo_table, tlc_check_dataset
from ultralytics.utils.tlc.detect.validator import TLCDetectionValidator


def check_det_dataset(data: str):
    """Check if the dataset is compatible with the 3LC."""
    tables = tlc_check_dataset(data)
    names = get_names_from_yolo_table(tables["train"])
    return {
        "train": tables["train"],
        "val": tables["val"],
        "nc": len(names),
        "names": names, }


ultralytics.engine.validator.check_det_dataset = check_det_dataset


class TLCYOLO(YOLO):
    """ YOLO (You Only Look Once) object detection model with 3LC integration. """

    @property
    def task_map(self):
        """ Map head to 3LC model, trainer, validator, and predictor classes. """
        return {
            "detect": {
                "model": TLCDetectionModel,
                "trainer": TLCDetectionTrainer,
                "validator": TLCDetectionValidator,
            }
        }
