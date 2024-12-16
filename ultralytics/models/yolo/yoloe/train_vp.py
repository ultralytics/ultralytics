# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo.yoloe.train_yoloe import YOLOETrainerFromScratch
from ultralytics.models import yolo
from copy import copy

class YOLOEVPTrainer(YOLOETrainerFromScratch):
    
    def preprocess_batch(self, batch):
        """Preprocesses a batch of images for YOLOE training, adjusting formatting and dimensions as needed."""
        batch = super().preprocess_batch(batch)
        batch["visuals"] = batch["visuals"].to(self.device)
        return batch
