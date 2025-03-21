# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo.yoloe.train_seg import YOLOESegTrainer
from ultralytics.models.yolo.yoloe.train_yoloe import YOLOETrainerFromScratch, YOLOEVPTrainer

class YOLOESegTrainerFromScratch(YOLOETrainerFromScratch, YOLOESegTrainer):
    """Trainer for YOLOE segmentation from scratch."""
    pass

class YOLOESegVPTrainer(YOLOEVPTrainer, YOLOESegTrainerFromScratch):
    """Trainer for YOLOE segmentation with VP."""
    pass
