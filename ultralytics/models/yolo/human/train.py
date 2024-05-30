# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

import numpy as np

from ultralytics.data.dataset import HumanDataset
from ultralytics.models import yolo
from ultralytics.nn.tasks import HumanModel
from ultralytics.utils import DEFAULT_CFG, RANK, colorstr
from ultralytics.utils.plotting import plot_attributes, plot_results
from ultralytics.utils.torch_utils import de_parallel


class HumanTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a human model.

    Example:
        ```python
        from ultralytics.models.yolo.human import HumanTrainer

        args = dict(model='yolov8n-human.pt', data='coco8.yaml', epochs=3)
        trainer = HumanTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a HumanTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "human"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return HumanModel initialized with specified config and weights."""
        model = HumanModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of HumanValidator for validation of YOLO model."""
        self.loss_names = (
            "box_loss",
            "cls_loss",
            "dfl_loss",
            "w_loss",  # weight
            "h_loss",  # height
            "g_loss",  # gender
            "a_loss",  # age
            "e_loss",  # ethnicity
        )
        return yolo.human.HumanValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot, human=True)  # save results.png

    def build_dataset(self, img_path, mode="train", batch=None):
        cfg = self.args
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return HumanDataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",  # augmentation
            hyp=cfg,
            rect=cfg.rect or mode == "val",  # rectangular batches
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=int(gs),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            classes=cfg.classes,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )

    def plot_training_labels(self):
        attributes = np.concatenate([lb["attributes"] for lb in self.train_loader.dataset.labels], 0)
        plot_attributes(attributes, save_dir=self.save_dir, on_plot=self.on_plot)
        return super().plot_training_labels()
