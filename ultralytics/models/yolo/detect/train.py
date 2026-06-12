# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import random
from copy import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ultralytics.data import YOLOConcatDataset, build_dataloader, build_yolo_dataset
from ultralytics.data.sampler import (
    ProportionalBatchSampler,
    get_concat_index_pools,
    get_dataset_fractions,
    iter_dataset_labels,
)
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
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
        loss_names (tuple): Names of the loss components used in training (box_loss, cls_loss, dfl_loss).

    Methods:
        build_dataset: Build YOLO dataset for training or validation.
        get_dataloader: Construct and return dataloader for the specified mode.
        preprocess_batch: Preprocess a batch of images by scaling and converting to float.
        set_model_attributes: Set model attributes based on dataset information.
        get_model: Return a YOLO detection model.
        get_validator: Return a validator for model evaluation.
        label_loss_items: Return a loss dictionary with labeled training loss items.
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

    def _use_concat_dataset(self, paths: list[str], mode: str) -> bool:
        """Whether to build ``YOLOConcatDataset`` from multiple source paths."""
        if len(paths) <= 1:
            return False
        if mode == "train":
            return get_dataset_fractions(self.data, paths, mode) is not None
        return True

    def build_dataset(self, img_path: str | list[str], mode: str = "train", batch: int | None = None):
        """Build YOLO Dataset for training or validation.

        When multiple paths are configured, builds ``YOLOConcatDataset``. Use ``train_dataset_fractions`` /
        ``val_dataset_fractions`` with ``ProportionalBatchSampler``; without val fractions, validation
        iterates all val images once.

        Args:
            img_path (str | list[str]): Path to the folder containing images, or list of paths.
            mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for 'rect' mode.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 32), 32)
        paths = img_path if isinstance(img_path, list) else [img_path]
        if self._use_concat_dataset(paths, mode):
            datasets = [
                build_yolo_dataset(self.args, p, batch, self.data, mode=mode, rect=False, stride=gs) for p in paths
            ]
            dataset = YOLOConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
            if mode == "val" and len(paths) > 1 and get_dataset_fractions(self.data, paths, mode) is None:
                sizes = [len(d) for d in datasets]
                LOGGER.info(f"{colorstr('balanced:')} Val mixed dataset (full) | sources={sizes} total={sum(sizes)}")
            return dataset
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str | list[str], batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str | list[str]): Path to the dataset, or list of paths from data YAML.
            batch_size (int): Batch size for this run (e.g. Katib trial hyperparameter ``batch``).
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        paths = dataset_path if isinstance(dataset_path, list) else [dataset_path]
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        batch_sampler = None
        fractions = get_dataset_fractions(self.data, paths, mode)
        if fractions:
            key = "train_dataset_fractions" if mode == "train" else "val_dataset_fractions"
            if not isinstance(dataset, YOLOConcatDataset):
                raise RuntimeError(f"{key} requires multiple {mode} paths in the data YAML and YOLOConcatDataset.")
            pools = get_concat_index_pools(dataset)
            sizes = [len(p) for p in pools]
            ws = max(self.world_size, 1)
            r = max(rank, 0)
            batch_sampler = ProportionalBatchSampler(
                pools,
                fractions,
                batch_size=batch_size,
                seed=self.args.seed,
                rank=r,
                world_size=ws,
            )
            shuffle = False
            fr = [round(f, 4) for f in batch_sampler.fractions]
            LOGGER.info(
                f"{colorstr('balanced:')} ProportionalBatchSampler ({mode}) | datasets={sizes} fractions={fr} "
                f"batch={batch_size}"
            )
            n_total = sum(sizes)
            if mode == "train" and n_total > 50000 and self.args.workers > 4:
                LOGGER.warning(
                    f"{colorstr('balanced:')} Large combined dataset ({n_total} images) with workers={self.args.workers} "
                    f"may cause host RAM OOM during image decode. Try workers=2 or workers=4."
                )
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank if batch_sampler is None else -1,
            drop_last=self.args.compile and mode == "train",
            batch_sampler=batch_sampler,
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """Preprocess a batch of images by scaling and converting to float.

        Args:
            batch (dict): Dictionary containing batch data with 'img' tensor.

        Returns:
            (dict): Preprocessed batch with normalized images.
        """
        batch = self._maybe_dump_training_batch(batch)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
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

    def _maybe_dump_training_batch(self, batch: dict) -> dict:
        """Copy batch via ``ultralytics.data.batch_dump`` when ``dump_batches_dir`` is set."""
        from ultralytics.data import batch_dump

        if not batch_dump.is_enabled(getattr(self.args, "dump_batches_dir", None)) or RANK not in {-1, 0}:
            return batch
        if not getattr(self, "_dump_batches_logged", False):
            LOGGER.info(
                f"{colorstr('dump_batches:')} {batch_dump.log_message(self.args.dump_batches_dir, self.args.dump_batches_per_epoch)}"
            )
            self._dump_batches_logged = True
        epoch = getattr(self, "epoch", 0)
        if getattr(self, "_dump_last_epoch", -1) != epoch:
            self._dump_last_epoch = epoch
            self._dump_batch_in_epoch = 0
        n = getattr(self, "_dump_batch_in_epoch", 0)
        if not batch_dump.should_dump(n, self.args.dump_batches_per_epoch):
            return batch
        data_root = self.data.get("path")
        if not data_root:
            return batch
        batch_dump.dump_training_batch(
            batch,
            out_root=self.args.dump_batches_dir,
            data_root=data_root,
            batch_in_epoch=n,
            epoch=epoch,
        )
        self._dump_batch_in_epoch = n + 1
        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        self._dump_last_epoch = -1
        self._dump_batch_in_epoch = 0
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        if getattr(self.model, "end2end", False):
            self.model.set_head_attr(max_det=self.args.max_det)

    def set_class_weights(self):
        """Compute and set class weights for handling class imbalance.

        Class weights are computed based on inverse class frequency in the training dataset,
        raised to the power of cls_pw (0 < cls_pw <= 1 dampens; values are restricted to the range [0, 1]).
        Final weights are normalized so their mean equals 1.0.
        """
        assert 0 <= self.args.cls_pw <= 1.0, "cls_pw must be in the range [0, 1]"
        if self.args.cls_pw == 0.0:
            return
        labels = iter_dataset_labels(self.train_loader.dataset)
        classes = np.concatenate([lb["cls"].flatten() for lb in labels], 0)
        class_counts = np.bincount(classes.astype(int), minlength=self.data["nc"]).astype(np.float32)
        class_counts = np.where(class_counts == 0, 1.0, class_counts)

        weights = (1.0 / class_counts) ** self.args.cls_pw  # apply power directly
        weights = weights / weights.mean()  # normalize so mean equals 1.0
        self.model.class_weights = torch.from_numpy(weights).to(self.device)
        LOGGER.info(f"Class weights: {self.model.class_weights.cpu().numpy().round(3)}")

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """Return a loss dict with labeled training loss items tensor.

        Args:
            loss_items (list[float], optional): List of loss values.
            prefix (str): Prefix for keys in the returned dictionary.

        Returns:
            (dict | list): Dictionary of labeled loss items if loss_items is provided, otherwise list of keys.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

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
        labels = iter_dataset_labels(self.train_loader.dataset)
        boxes = np.concatenate([lb["bboxes"] for lb in labels], 0)
        cls = np.concatenate([lb["cls"] for lb in labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """Get optimal batch size by calculating memory occupation of model.

        Returns:
            (int): Optimal batch size.
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        labels = iter_dataset_labels(train_dataset)
        max_num_obj = max(len(label["cls"]) for label in labels) * 4  # 4 for mosaic augmentation
        n = len(train_dataset)
        del train_dataset  # free memory
        return super().auto_batch(max_num_obj, dataset_size=n)
