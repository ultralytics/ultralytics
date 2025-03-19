# Ultralytics YOLO 🚀, AGPL-3.0 license


from ultralytics.models.yolo.yoloe.train_yoloe_seg import YOLOESegTrainerFromScratch

from .train_vp import YOLOEVPTrainer


class YOLOESegVPTrainer(YOLOEVPTrainer, YOLOESegTrainerFromScratch):
    pass
