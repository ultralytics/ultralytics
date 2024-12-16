# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo.yoloe.train_yoloe_seg import YOLOESegTrainerFromScratch
from .val import YOLOESegValidator
from copy import copy
from .train_vp import YOLOEVPTrainer

class YOLOESegVPTrainer(YOLOEVPTrainer, YOLOESegTrainerFromScratch):
    pass
