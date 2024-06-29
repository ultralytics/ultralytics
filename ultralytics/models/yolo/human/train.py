# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.data.dataset import HumanDataset
from ultralytics.models import yolo
from ultralytics.nn.tasks import HumanModel
from ultralytics.utils import DEFAULT_CFG, RANK, colorstr


class HumanTrainer(yolo.classify.ClassificationTrainer):  # NOTE: perhaps BaseTrainer
    """
    A class extending the DetectionTrainer class for training based on a human model.

    Example:
        ```python
        from ultralytics.models.yolo.human import HumanTrainer

        args = dict(model='yolov8n-human.pt', data='coco8-human.yaml', epochs=3)
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

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images and classes."""
        batch["img"] = batch["img"].to(self.device)
        batch["attributes"] = batch["attributes"].to(self.device)
        return batch

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return HumanModel initialized with specified config and weights."""
        model = HumanModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of HumanValidator for validation of YOLO model."""
        self.loss_names = (
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
        # plot_results(file=self.csv, on_plot=self.on_plot, human=True)  # save results.png
        pass

    def build_dataset(self, img_path, mode="train", batch=None):
        return HumanDataset(
            img_path=img_path,
            args=self.args,
            augment=mode == "train",  # augmentation
            prefix=colorstr(f"{mode}: "),
        )

    def plot_training_labels(self):
        pass

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys
