# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.yolo.segment import SegmentationTrainer

from .train import YOLOEPETrainer, YOLOETrainer, YOLOETrainerFromScratch, YOLOEVPTrainer


class YOLOESegTrainer(YOLOETrainer, SegmentationTrainer):
    """Trainer class for YOLOE segmentation models."""


class YOLOEPESegTrainer(SegmentationTrainer):
    """Fine-tune YOLOE segmentation models in linear probing way."""

    get_model = YOLOEPETrainer.get_model  # shared linear-probing builder; SegmentationTrainer stays the sole base


class YOLOESegTrainerFromScratch(YOLOETrainerFromScratch, YOLOESegTrainer):
    """Trainer for YOLOE segmentation models trained from scratch without pretrained weights."""


class YOLOESegVPTrainer(YOLOEVPTrainer, YOLOESegTrainerFromScratch):
    """Trainer for YOLOE segmentation models with Vision Prompt (VP) capabilities."""
