# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import os
import random
from copy import copy

import torch.nn as nn

from ultralytics.data import build_semantic_dataset
from ultralytics.models import yolo
from ultralytics.nn.tasks import SemanticModel
from ultralytics.utils import DEFAULT_CFG, RANK, SEMSEG_CFG
from ultralytics.utils.plotting import plot_masks, plot_results
from ultralytics.utils.torch_utils import de_parallel


class SemSegTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a segmentation model.

    Example:
        ```python
        from ultralytics.models.yolo.segment import SegmentationTrainer

        args = dict(model="yolov8n-seg.pt", data="coco8-seg.yaml", epochs=3)
        trainer = SegmentationTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=SEMSEG_CFG, overrides=None, _callbacks=None):
        """Initialize a SegmentationTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "semseg"
        super().__init__(cfg, overrides, _callbacks)
        self.loss_names = ["Loss"]

    # def plot_training_labels(self):
    #    """Create a labeled training plot of the YOLO model."""
    #    images = np.concatenate([lb["images"] for lb in self.train_loader.dataset.labels], 0)
    #   masks = np.concatenate([lb["masks"] for lb in self.train_loader.dataset.labels], 0)

    #   plot_masks(images, masks, self.data["nc"], names=self.data["names"],colors=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return SegmentationModel initialized with specified config and weights."""
        model = SemanticModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        batch["masks"] = batch["masks"].to(self.device, non_blocking=True).float()
        batch["masks"] = batch["masks"].argmax(dim=1).long()

        if self.args.multi_scale:
            imgs = batch["img"]
            msks = batch["masks"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
                msks = nn.functional.interpolate(msks, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
            batch["masks"] = msks
        return batch

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build RSI Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_semantic_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)

    def get_validator(self):
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        self.loss_names = ["loss"]
        return yolo.semseg.SemSegValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in [loss_items]]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_masks(
            images=batch["img"],
            masks=batch["masks"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            nc=self.data["nc"],
            names=self.data["names"],
            colors=self.data["colors"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            mname=self.save_dir / f"mask_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, segment=True, on_plot=self.on_plot)  # save results.png


def train(cfg=DEFAULT_CFG, use_python=False):
    """Train a YOLO segmentation model based on passed arguments."""
    model = cfg.model or "yolov11n-seg.pt"
    data = cfg.data or "coco128-seg.yaml"  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ""
    cfg.name = os.path.join(cfg.name, "train")

    args = dict(model=model, data=data, device=device, task="semseg")
    if use_python:
        from ultralytics import YOLO

        YOLO(model).train(**args)
    else:
        trainer = SemSegTrainer(cfg=cfg, overrides=args)
        trainer.train()


if __name__ == "__main__":
    train(cfg=SEMSEG_CFG)
