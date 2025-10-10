# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import torch

from ultralytics.data import build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import WorldModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model


def on_pretrain_routine_end(trainer) -> None:
    """Set up model classes and text encoder at the end of the pretrain routine."""
    if RANK in {-1, 0}:
        # Set class names for evaluation
        names = [name.split("/", 1)[0] for name in list(trainer.test_loader.dataset.data["names"].values())]
        unwrap_model(trainer.ema.ema).set_classes(names, cache_clip_model=False)


class WorldTrainer(DetectionTrainer):
    """
    A trainer class for fine-tuning YOLO World models on close-set datasets.

    This trainer extends the DetectionTrainer to support training YOLO World models, which combine visual and textual
    features for improved object detection and understanding. It handles text embedding generation and caching to
    accelerate training with multi-modal data.

    Attributes:
        text_embeddings (dict[str, torch.Tensor] | None): Cached text embeddings for category names to accelerate
            training.
        model (WorldModel): The YOLO World model being trained.
        data (dict[str, Any]): Dataset configuration containing class information.
        args (Any): Training arguments and configuration.

    Methods:
        get_model: Return WorldModel initialized with specified config and weights.
        build_dataset: Build YOLO Dataset for training or validation.
        set_text_embeddings: Set text embeddings for datasets to accelerate training.
        generate_text_embeddings: Generate text embeddings for a list of text samples.
        preprocess_batch: Preprocess a batch of images and text for YOLOWorld training.

    Examples:
        Initialize and train a YOLO World model
        >>> from ultralytics.models.yolo.world import WorldTrainer
        >>> args = dict(model="yolov8s-world.pt", data="coco8.yaml", epochs=3)
        >>> trainer = WorldTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """
        Initialize a WorldTrainer object with given arguments.

        Args:
            cfg (dict[str, Any]): Configuration for the trainer.
            overrides (dict[str, Any], optional): Configuration overrides.
            _callbacks (list[Any], optional): List of callback functions.
        """
        if overrides is None:
            overrides = {}
        assert not overrides.get("compile"), f"Training with 'model={overrides['model']}' requires 'compile=False'"
        super().__init__(cfg, overrides, _callbacks)
        self.text_embeddings = None

    def get_model(self, cfg=None, weights: str | None = None, verbose: bool = True) -> WorldModel:
        """
        Return WorldModel initialized with specified config and weights.

        Args:
            cfg (dict[str, Any] | str, optional): Model configuration.
            weights (str, optional): Path to pretrained weights.
            verbose (bool): Whether to display model info.

        Returns:
            (WorldModel): Initialized WorldModel.
        """
        # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
        # NOTE: Following the official config, nc hard-coded to 80 for now.
        model = WorldModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        self.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)

        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """
        Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (Any): YOLO dataset configured for training or validation.
        """
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        dataset = build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, multi_modal=mode == "train"
        )
        if mode == "train":
            self.set_text_embeddings([dataset], batch)  # cache text embeddings to accelerate training
        return dataset

    def set_text_embeddings(self, datasets: list[Any], batch: int | None) -> None:
        """
        Set text embeddings for datasets to accelerate training by caching category names.

        This method collects unique category names from all datasets, then generates and caches text embeddings
        for these categories to improve training efficiency.

        Args:
            datasets (list[Any]): List of datasets from which to extract category names.
            batch (int | None): Batch size used for processing.

        Notes:
            This method collects category names from datasets that have the 'category_names' attribute,
            then uses the first dataset's image path to determine where to cache the generated text embeddings.
        """
        text_embeddings = {}
        for dataset in datasets:
            if not hasattr(dataset, "category_names"):
                continue
            text_embeddings.update(
                self.generate_text_embeddings(
                    list(dataset.category_names), batch, cache_dir=Path(dataset.img_path).parent
                )
            )
        self.text_embeddings = text_embeddings

    def generate_text_embeddings(self, texts: list[str], batch: int, cache_dir: Path) -> dict[str, torch.Tensor]:
        """
        Generate text embeddings for a list of text samples.

        Args:
            texts (list[str]): List of text samples to encode.
            batch (int): Batch size for processing.
            cache_dir (Path): Directory to save/load cached embeddings.

        Returns:
            (dict[str, torch.Tensor]): Dictionary mapping text samples to their embeddings.
        """
        model = "clip:ViT-B/32"
        cache_path = cache_dir / f"text_embeddings_{model.replace(':', '_').replace('/', '_')}.pt"
        if cache_path.exists():
            LOGGER.info(f"Reading existed cache from '{cache_path}'")
            txt_map = torch.load(cache_path, map_location=self.device)
            if sorted(txt_map.keys()) == sorted(texts):
                return txt_map
        LOGGER.info(f"Caching text embeddings to '{cache_path}'")
        assert self.model is not None
        txt_feats = unwrap_model(self.model).get_text_pe(texts, batch, cache_clip_model=False)
        txt_map = dict(zip(texts, txt_feats.squeeze(0)))
        torch.save(txt_map, cache_path)
        return txt_map

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess a batch of images and text for YOLOWorld training."""
        batch = DetectionTrainer.preprocess_batch(self, batch)

        # Add text features
        texts = list(itertools.chain(*batch["texts"]))
        txt_feats = torch.stack([self.text_embeddings[text] for text in texts]).to(
            self.device, non_blocking=self.device.type == "cuda"
        )
        batch["txt_feats"] = txt_feats.reshape(len(batch["texts"]), -1, txt_feats.shape[-1])
        return batch
