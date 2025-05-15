# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import itertools
from copy import copy, deepcopy

import torch

from ultralytics.data import YOLOConcatDataset, build_yolo_dataset
from ultralytics.data.augment import LoadVisualPrompt
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.tasks import YOLOEModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import de_parallel

from ..world.train_world import WorldTrainerFromScratch
from .val import YOLOEDetectValidator


class YOLOETrainer(DetectionTrainer):
    """A base trainer for YOLOE training."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the YOLOE Trainer with specified configurations.

        This method sets up the YOLOE trainer with the provided configuration and overrides, initializing
        the training environment, model, and callbacks for YOLOE object detection training.

        Args:
            cfg (dict): Configuration dictionary with default training settings from DEFAULT_CFG.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be applied during training.
        """
        if overrides is None:
            overrides = {}
        overrides["overlap_mask"] = False
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Return a YOLOEModel initialized with the specified configuration and weights.

        Args:
            cfg (dict | str | None): Model configuration. Can be a dictionary containing a 'yaml_file' key,
                a direct path to a YAML file, or None to use default configuration.
            weights (str | Path | None): Path to pretrained weights file to load into the model.
            verbose (bool): Whether to display model information during initialization.

        Returns:
            (YOLOEModel): The initialized YOLOE model.

        Notes:
            - The number of classes (nc) is hard-coded to a maximum of 80 following the official configuration.
            - The nc parameter here represents the maximum number of different text samples in one image,
              rather than the actual number of classes.
        """
        # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
        # NOTE: Following the official config, nc hard-coded to 80 for now.
        model = YOLOEModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box", "cls", "dfl"
        return YOLOEDetectValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (Dataset): YOLO dataset configured for training or validation.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, multi_modal=mode == "train"
        )


class YOLOEPETrainer(DetectionTrainer):
    """Fine-tune YOLOE model in linear probing way."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Return YOLOEModel initialized with specified config and weights.

        Args:
            cfg (dict | str, optional): Model configuration.
            weights (str, optional): Path to pretrained weights.
            verbose (bool): Whether to display model information.

        Returns:
            (YOLOEModel): Initialized model with frozen layers except for specific projection layers.
        """
        # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
        # NOTE: Following the official config, nc hard-coded to 80 for now.
        model = YOLOEModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )

        del model.model[-1].savpe

        assert weights is not None, "Pretrained weights must be provided for linear probing."
        if weights:
            model.load(weights)

        model.eval()
        names = list(self.data["names"].values())
        # NOTE: `get_text_pe` related to text model and YOLOEDetect.reprta,
        # it'd get correct results as long as loading proper pretrained weights.
        tpe = model.get_text_pe(names)
        model.set_classes(names, tpe)
        model.model[-1].fuse(model.pe)  # fuse text embeddings to classify head
        model.model[-1].cv3[0][2] = deepcopy(model.model[-1].cv3[0][2]).requires_grad_(True)
        model.model[-1].cv3[1][2] = deepcopy(model.model[-1].cv3[1][2]).requires_grad_(True)
        model.model[-1].cv3[2][2] = deepcopy(model.model[-1].cv3[2][2]).requires_grad_(True)
        del model.pe
        model.train()

        return model


class YOLOETrainerFromScratch(YOLOETrainer, WorldTrainerFromScratch):
    """Train YOLOE models from scratch."""

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset for training or validation.

        This method constructs appropriate datasets based on the mode and input paths, handling both
        standard YOLO datasets and grounding datasets with different formats.

        Args:
            img_path (List[str] | str): Path to the folder containing images or list of paths.
            mode (str): 'train' mode or 'val' mode, allowing customized augmentations for each mode.
            batch (int, optional): Size of batches, used for rectangular training/validation.

        Returns:
            (YOLOConcatDataset | Dataset): The constructed dataset for training or validation.
        """
        return WorldTrainerFromScratch.build_dataset(self, img_path, mode, batch)

    def preprocess_batch(self, batch):
        """Process batch for training, moving text features to the appropriate device."""
        batch = DetectionTrainer.preprocess_batch(self, batch)

        texts = list(itertools.chain(*batch["texts"]))
        txt_feats = torch.stack([self.text_embeddings[text] for text in texts]).to(self.device)
        txt_feats = txt_feats.reshape(len(batch["texts"]), -1, txt_feats.shape[-1])
        batch["txt_feats"] = txt_feats
        return batch

    def generate_text_embeddings(self, texts, batch, cache_dir):
        """
        Generate text embeddings for a list of text samples.

        Args:
            texts (List[str]): List of text samples to encode.
            batch (int): Batch size for processing.
            cache_dir (Path): Directory to save/load cached embeddings.

        Returns:
            (dict): Dictionary mapping text samples to their embeddings.
        """
        model = "mobileclip:blt"
        cache_path = cache_dir / f"text_embeddings_{model.replace(':', '_').replace('/', '_')}.pt"
        if cache_path.exists():
            LOGGER.info(f"Reading existed cache from '{cache_path}'")
            txt_map = torch.load(cache_path)
            if sorted(txt_map.keys()) == sorted(texts):
                return txt_map
        LOGGER.info(f"Caching text embeddings to '{cache_path}'")
        assert self.model is not None
        txt_feats = self.model.get_text_pe(texts, batch, without_reprta=True, cache_clip_model=False)
        txt_map = dict(zip(texts, txt_feats.squeeze(0)))
        torch.save(txt_map, cache_path)
        return txt_map


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
        batch = DetectionTrainer.preprocess_batch(self, batch)
        return batch

    def set_text_embeddings(self, datasets, batch):
        """
        Set text embeddings for datasets to accelerate training by caching category names.

        This method collects unique category names from all datasets, generates text embeddings for them,
        and caches these embeddings to improve training efficiency. The embeddings are stored in a file
        in the parent directory of the first dataset's image path.

        Args:
            datasets (List[Dataset]): List of datasets containing category names to process.
            batch (int): Batch size for processing text embeddings.

        Notes:
            The method creates a dictionary mapping text samples to their embeddings and stores it
            at the path specified by 'cache_path'. If the cache file already exists, it will be loaded
            instead of regenerating the embeddings.
        """
        pass


class YOLOEVPTrainer(YOLOETrainerFromScratch):
    """Train YOLOE model with visual prompts."""

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset for training or validation with visual prompts.

        Args:
            img_path (List[str] | str): Path to the folder containing images or list of paths.
            mode (str): 'train' mode or 'val' mode, allowing customized augmentations for each mode.
            batch (int, optional): Size of batches, used for rectangular training/validation.

        Returns:
            (Dataset): YOLO dataset configured for training or validation, with visual prompts for training mode.
        """
        dataset = super().build_dataset(img_path, mode, batch)
        if isinstance(dataset, YOLOConcatDataset):
            for d in dataset.datasets:
                d.transforms.append(LoadVisualPrompt())
        else:
            dataset.transforms.append(LoadVisualPrompt())
        return dataset

    def _close_dataloader_mosaic(self):
        """Close mosaic augmentation and add visual prompt loading to the training dataset."""
        super()._close_dataloader_mosaic()
        if isinstance(self.train_loader.dataset, YOLOConcatDataset):
            for d in self.train_loader.dataset.datasets:
                d.transforms.append(LoadVisualPrompt())
        else:
            self.train_loader.dataset.transforms.append(LoadVisualPrompt())

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images for YOLOE training, moving visual prompts to the appropriate device."""
        batch = super().preprocess_batch(batch)
        batch["visuals"] = batch["visuals"].to(self.device)
        return batch
