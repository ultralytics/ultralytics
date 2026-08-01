# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import random
from copy import copy
from functools import partial
from typing import Any

import numpy as np
import torch
from torch import nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.modules.head import RefineDetect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


class DetectionTrainer(BaseTrainer):
    """A class extending the BaseTrainer class for training based on a detection model.

    This trainer specializes in object detection tasks, handling the specific requirements for training YOLO models for
    object detection including dataset building, data loading, preprocessing, and model configuration.

    Attributes:
        model (DetectionModel): The YOLO detection model being trained.
        data (dict): Dictionary containing dataset information including class names and number of classes.
        loss_names (tuple): Names of the loss components, derived from the loss dict returned by the criterion.

    Methods:
        build_dataset: Build YOLO dataset for training or validation.
        get_dataloader: Construct and return dataloader for the specified mode.
        preprocess_batch: Preprocess a batch of images by scaling and converting to float.
        set_model_attributes: Set model attributes based on dataset information.
        get_model: Return a YOLO detection model.
        get_validator: Return a validator for model evaluation.
        progress_string: Return a formatted string of training progress.
        plot_training_samples: Plot training samples with their annotations.
        plot_training_labels: Create a labeled training plot of the YOLO model.
        auto_batch: Calculate optimal batch size based on model memory requirements.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionTrainer
        >>> args = dict(model="yolo26n.pt", data="coco8.yaml", epochs=3)
        >>> trainer = DetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize a DetectionTrainer object for training YOLO object detection models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (dict, optional): Dictionary of callback functions to be executed during training.
        """
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for 'rect' mode.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
            device=self.device,
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """Preprocess a batch of images by scaling and converting to float.

        Args:
            batch (dict): Dictionary containing batch data with 'img' tensor.

        Returns:
            (dict): Preprocessed batch with normalized images.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type not in {"cpu", "mps"})
        batch["img"] = batch["img"].float() / 255
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            sz = (
                random.randrange(
                    max(self.stride, int(self.args.imgsz * (1.0 - self.args.multi_scale))),  # min imgsz
                    int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),  # max imgsz
                )
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        # Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        if getattr(self.model, "end2end", False):
            self.model.set_head_attr(max_det=self.args.max_det)

    def set_model_names_for_load(self, model):
        """Set target dataset names before loading weights so cls heads can remap by name."""
        if getattr(self.args, "cls_remap", True) and self.data.get("names"):
            model.names = self.data["names"]
        return model

    def get_class_counts(self):
        """Return per-class instance counts from the training dataset labels."""
        classes = np.concatenate([lb["cls"].flatten() for lb in self.train_loader.dataset.labels], 0)
        return np.bincount(classes.astype(int), minlength=self.data["nc"]).astype(np.float32)

    def compute_class_weights(self, class_counts):
        """Convert class counts to inverse-frequency weights raised to the power of cls_pw."""
        class_counts = np.where(class_counts == 0, 1.0, class_counts)
        return (1.0 / class_counts) ** self.args.cls_pw  # apply power directly

    def set_class_weights(self):
        """Compute and set class weights for handling class imbalance.

        Class weights are computed based on inverse class frequency in the training dataset,
        raised to the power of cls_pw (0 < cls_pw <= 1 dampens; values are restricted to the range [0, 1]).
        Final weights are normalized so their mean equals 1.0.
        """
        assert 0 <= self.args.cls_pw <= 1.0, "cls_pw must be in the range [0, 1]"
        if self.args.cls_pw == 0.0:
            return
        class_counts = self.get_class_counts()
        if not class_counts.any():  # nothing counted (e.g. missing/unreadable masks); keep default weights
            return
        weights = self.compute_class_weights(class_counts)
        weights = weights / weights.mean()  # normalize so mean equals 1.0
        model = self.model
        if hasattr(unwrap_model(model), "student_model"):
            model = unwrap_model(model).student_model  # distillation: the student model builds the loss criterion
        model.class_weights = torch.from_numpy(weights).to(self.device)
        LOGGER.info(f"Class weights: {model.class_weights.cpu().numpy().round(3)}")

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        model = self.set_model_names_for_load(
            DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot training samples with their annotations.

        Args:
            batch (dict[str, Any]): Dictionary containing batch data.
            ni (int): Batch index used for naming the output file.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """Get optimal batch size by calculating memory occupation of model.

        Returns:
            (int): Optimal batch size.
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
        n = len(train_dataset)
        del train_dataset  # free memory
        return super().auto_batch(max_num_obj, dataset_size=n)


class RefineDetectionTrainer(DetectionTrainer):
    """A DetectionTrainer that tunes a subset of classes while leaving the other classes untouched.

    The detection head gets a zero-initialized RefineDetect branch, and everything else is frozen except the last layer
    of the classification branches, where only the rows of the tuned classes receive updates. A pretrained model can
    therefore learn a new class or improve an existing one while the class scores of every other class stay exactly as
    they were. Boxes are shared by all classes, so the box deltas of the refinement branch move the boxes of the few
    anchors where a tuned class is confident.

    The tuned classes are selected with the `classes` argument, which also drops the labels of the other classes from
    the dataset and restricts validation to them. The dataset YAML must therefore define all classes of the pretrained
    model plus any new one, in the order the tuned model should use: classes missing from it are dropped from the head.

    Training an already tuned model stacks a second branch on it and freezes the first, so classes can be added in
    several sessions, each keeping the classes of the sessions before it. Every branch costs about 1% of inference, so
    prefer passing the classes together when they are known up front.

    Examples:
        >>> from ultralytics import YOLO
        >>> from ultralytics.models.yolo.detect import RefineDetectionTrainer
        >>> model = YOLO("yolo26n.pt")
        >>> model.train(data="coco8.yaml", epochs=10, classes=[0, 5], trainer=RefineDetectionTrainer)
    """

    @staticmethod
    def _mask_rows(grad: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Zero the gradient rows of the classes that are not tuned."""
        return grad.index_fill(0, rows, 0.0)

    @staticmethod
    def _refined_classes(head) -> list[int] | None:
        """Return the classes of the head's newest refinement branch, or None if it has none."""
        return head.refine_index[-head.refine_splits[-1] :].tolist() if isinstance(head, RefineDetect) else None

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return the pretrained model with its classification head extended to the classes of the dataset.

        The pretrained model is tuned in place instead of being rebuilt from its YAML, because the width of the
        classification branch depends on the class count, so rebuilding it for one added class discards the pretrained
        weights of every class. New classes get a zero weight and the standard Detect bias, so they predict nothing
        until they are trained, and shared classes keep their pretrained rows even when the dataset reorders them.

        Args:
            cfg (str, optional): Path to model configuration file, used when there are no pretrained weights.
            weights (torch.nn.Module, optional): Pretrained model to tune.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        if not isinstance(weights, nn.Module):
            return super().get_model(cfg, weights, verbose)
        head, names = weights.model[-1], self.data["names"]
        index = weights.cls_index_map(weights.names, names)
        assert index is not None, "the pretrained model and the dataset must both name their classes"
        if head.nc != len(names) or not torch.equal(index, torch.arange(head.nc)):
            for cv3 in (head.cv3, getattr(head, "one2one_cv3", None)):
                for i, seq in enumerate(cv3 or ()):
                    conv = nn.Conv2d(seq[-1].in_channels, len(names), 1).to(seq[-1].weight)
                    nn.init.zeros_(conv.weight)
                    conv.bias.data[:] = math.log(5 / len(names) / (640 / head.stride[i]) ** 2)  # Detect.bias_init
                    conv.weight.data[index >= 0] = seq[-1].weight.data[index[index >= 0]]
                    conv.bias.data[index >= 0] = seq[-1].bias.data[index[index >= 0]]
                    seq[-1] = conv
            if isinstance(head, RefineDetect):  # branches of earlier sessions address classes by index
                moved = torch.full((head.nc,), -1, dtype=torch.long)
                moved[index[index >= 0]] = torch.arange(len(names))[index >= 0]
                refined = moved.to(head.refine_index)[head.refine_index]
                assert (refined >= 0).all(), (
                    f"the dataset drops classes {[weights.names[c] for c in head.refine_index[refined < 0].tolist()]}, "
                    "which an earlier session refined. Keep every refined class in the dataset YAML."
                )
                head.refine_index = refined
            head.nc, head.no = len(names), len(names) + 4 * head.reg_max
            if getattr(weights, "pe", None) is not None:  # a fused YOLOE head reads its class count off these
                pe = torch.zeros(weights.pe.shape[0], len(names), weights.pe.shape[2]).to(weights.pe)
                pe[:, index >= 0] = weights.pe[:, index[index >= 0]]
                weights.pe = pe
            LOGGER.info(
                f"Extended the cls head from {len(weights.names)} to {len(names)} classes, {int((index < 0).sum())} new"
            )
        return weights

    def setup_model(self):
        """Attach the refinement branch to the detection head and freeze everything the tuned classes do not own."""
        ckpt = super().setup_model()
        assert self.args.classes is not None, (
            f"{self.__class__.__name__} requires 'classes' to select the classes to tune, e.g. classes=[0, 5]."
        )
        model = unwrap_model(self.model)
        head, i = model.model[-1], len(model.model) - 1
        classes = [self.args.classes] if isinstance(self.args.classes, int) else list(self.args.classes)
        if self.resume:  # the branch of the interrupted session is already attached
            assert self._refined_classes(head) == classes, (
                f"the run being resumed refines classes {self._refined_classes(head)}, not classes={classes}."
            )
        else:
            RefineDetect.attach(head, classes)  # a new session always gets its own branch
        freeze = [str(j) for j in range(i)]  # backbone and neck
        for name, _ in head.named_children():
            if name in {"refine", "one2one_refine"}:
                freeze += [f"{i}.{name}.{b}" for b in range(len(getattr(head, name)) - 1)]  # earlier sessions
                continue  # the branch of this session is the only fully trainable module
            # cls branches keep their last layer trainable, its untuned rows are masked in optimizer_step()
            freeze += (
                [f"{i}.{name}.{s}.{k}" for s in range(head.nl) for k in (0, 1)] if "cv3" in name else [f"{i}.{name}"]
            )
        self.args.freeze = freeze
        return ckpt

    def _setup_train(self):
        """Mask the gradients of the untuned classification rows and snapshot their weights."""
        super()._setup_train()
        tuned = set(self._refined_classes(unwrap_model(self.model).model[-1]))  # classes of this session only
        self.untuned_rows = []
        # the EMA is restored as well: it is what gets validated and saved, and the optimizer moves the rows of the
        # live model for the length of a step, which the EMA would otherwise average in
        for model in (unwrap_model(self.model), self.ema.ema):
            head = model.model[-1]
            rows = torch.tensor([c for c in range(head.nc) if c not in tuned], device=self.device)
            for cv3 in (head.cv3, getattr(head, "one2one_cv3", None)):
                for seq in cv3 or []:
                    for p in (seq[-1].weight, seq[-1].bias):
                        if p.requires_grad:  # EMA parameters carry no gradient
                            p.register_hook(partial(self._mask_rows, rows=rows))
                        self.untuned_rows.append((p, rows, p.detach()[rows].clone()))

    def optimizer_step(self):
        """Step the optimizer, then undo its weight decay and momentum on the untuned classification rows."""
        super().optimizer_step()  # also updates the EMA, restored below along with the model
        with torch.no_grad():
            for p, rows, weights in self.untuned_rows:
                p[rows] = weights
