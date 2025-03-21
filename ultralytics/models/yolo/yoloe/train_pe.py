# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy, deepcopy

import torch

from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.models.yolo.yoloe.train_yoloe import YOLOETrainer, YOLOETrainerFromScratch
from ultralytics.nn.tasks import YOLOEModel, YOLOESegModel
from ultralytics.utils import RANK


class YOLOEPETrainer(DetectionTrainer):
    """Fine-tune YOLOE model in linear probing way."""
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return YOLOEModel initialized with specified config and weights."""
        # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
        # NOTE: Following the official config, nc hard-coded to 80 for now.
        model = YOLOEModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=3,
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )

        del model.model[-1].savpe

        if weights:
            model.load(weights)

        model.eval()
        # TODO: removed `train_pe_path`
        pe_state = torch.load(self.args.train_pe_path)
        model.set_classes(pe_state["names"], pe_state["pe"])
        model.model[-1].fuse(model.pe)
        model.model[-1].cv3[0][2] = deepcopy(model.model[-1].cv3[0][2]).requires_grad_(True)
        model.model[-1].cv3[1][2] = deepcopy(model.model[-1].cv3[1][2]).requires_grad_(True)
        model.model[-1].cv3[2][2] = deepcopy(model.model[-1].cv3[2][2]).requires_grad_(True)
        del model.pe
        model.train()

        return model


class YOLOEPESegTrainer(SegmentationTrainer):
    """Fine-tune YOLOESeg model in linear probing way."""
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return YOLOEModel initialized with specified config and weights."""
        # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
        # NOTE: Following the official config, nc hard-coded to 80 for now.
        model = YOLOESegModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=3,
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )

        del model.model[-1].savpe

        if weights:
            model.load(weights)

        model.eval()
        # TODO: removed `train_pe_path`
        pe_state = torch.load(self.args.train_pe_path)
        model.set_classes(pe_state["names"], pe_state["pe"])
        model.model[-1].fuse(model.pe)
        model.model[-1].cv3[0][2] = deepcopy(model.model[-1].cv3[0][2]).requires_grad_(True)
        model.model[-1].cv3[1][2] = deepcopy(model.model[-1].cv3[1][2]).requires_grad_(True)
        model.model[-1].cv3[2][2] = deepcopy(model.model[-1].cv3[2][2]).requires_grad_(True)
        del model.pe
        model.train()

        return model


class YOLOEPEFreeTrainer(YOLOEPETrainer, YOLOETrainerFromScratch):
    """Train prompt-free YOLOE model."""
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box", "cls", "dfl"
        return DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images for YOLOE training, adjusting formatting and dimensions as needed."""
        batch = super(YOLOETrainer, self).preprocess_batch(batch)
        return batch
