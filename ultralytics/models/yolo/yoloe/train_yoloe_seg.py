# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.data import YOLOConcatDataset, build_grounding, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.yoloe.train_yoloe import YOLOETrainerFromScratch
from ultralytics.models.yolo.yoloe.train_seg import YOLOESegTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import de_parallel


class YOLOESegTrainerFromScratch(YOLOETrainerFromScratch, YOLOESegTrainer):
    pass
