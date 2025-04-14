# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import itertools
from copy import copy, deepcopy
from pathlib import Path

import torch

from ultralytics.data import YOLOConcatDataset, build_grounding, build_yolo_dataset
from ultralytics.data.augment import LoadVisualPrompt
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.tasks import YOLOEModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import de_parallel

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
            ch=3,
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

    def preprocess_batch(self, batch):
        """Process batch for training, moving text features to the appropriate device."""
        batch = super().preprocess_batch(batch)
        return batch


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
            ch=3,
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


class YOLOETrainerFromScratch(YOLOETrainer):
    """Train YOLOE models from scratch."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the YOLOETrainerFromScratch class.

        This class extends YOLOETrainer to train YOLOE models from scratch. It inherits all functionality from
        the parent class while providing specialized initialization for training without pre-trained weights.

        Args:
            cfg (dict, optional): Configuration dictionary with training parameters. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Dictionary of parameter overrides for configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.

        Examples:
            >>> from ultralytics.models.yoloe.train import YOLOETrainerFromScratch
            >>> trainer = YOLOETrainerFromScratch()
            >>> trainer.train()
        """
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)

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
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        if mode != "train":
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)
        datasets = [
            build_yolo_dataset(self.args, im_path, batch, self.training_data[im_path], stride=gs, multi_modal=True)
            if isinstance(im_path, str)
            else build_grounding(self.args, im_path["img_path"], im_path["json_file"], batch, stride=gs)
            for im_path in img_path
        ]
        self.set_text_embeddings(datasets, batch)  # cache text embeddings to accelerate training
        return YOLOConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    def set_text_embeddings(self, datasets, batch):
        """
        Set text embeddings for datasets to accelerate training by caching category names.

        This method collects unique category names from all datasets, then generates and caches text embeddings
        for these categories to improve training efficiency.

        Args:
            datasets (List[Dataset]): List of datasets from which to extract category names.
            batch (int | None): Batch size used for processing.

        Notes:
            This method collects category names from datasets that have the 'category_names' attribute,
            then uses the first dataset's image path to determine where to cache the generated text embeddings.
        """
        # TODO: open up an interface to determine whether to do cache
        category_names = set()
        for dataset in datasets:
            if not hasattr(dataset, "category_names"):
                continue
            category_names |= dataset.category_names

        # TODO: enable to update the path or use a more general way to get the path
        img_path = datasets[0].img_path
        self.text_embeddings = self.generate_text_embeddings(
            category_names, batch, cache_path=Path(img_path).parent / "text_embeddings.pt"
        )

    def preprocess_batch(self, batch):
        """Process batch for training, moving text features to the appropriate device."""
        batch = super().preprocess_batch(batch)

        texts = list(itertools.chain(*batch["texts"]))
        txt_feats = torch.stack([self.text_embeddings[text] for text in texts]).to(self.device)
        txt_feats = txt_feats.reshape(len(batch["texts"]), -1, txt_feats.shape[-1])
        batch["txt_feats"] = txt_feats
        return batch

    def generate_text_embeddings(self, texts, batch, cache_path="embeddings.pt"):
        """
        Generate text embeddings for a list of text samples.

        Args:
            texts (List[str]): List of text samples to encode.
            batch (int): Batch size for processing.
            cache_path (str | Path): Path to save/load cached embeddings.

        Returns:
            (dict): Dictionary mapping text samples to their embeddings.
        """
        if cache_path.exists():
            return torch.load(cache_path)
        assert self.model is not None
        txt_feats = self.model.get_text_pe(texts, batch, without_reprta=True)
        txt_map = dict(zip(texts, txt_feats.squeeze(0)))
        torch.save(txt_map, cache_path)
        return txt_map

    def get_dataset(self):
        """
        Get train and validation paths from data dictionary.

        Processes the data configuration to extract paths for training and validation datasets,
        handling both YOLO detection datasets and grounding datasets.

        Returns:
            (str): Train dataset path.
            (str): Validation dataset path.

        Raises:
            AssertionError: If train or validation datasets are not found, or if validation has multiple datasets.
        """
        final_data = {}
        data_yaml = self.args.data
        assert data_yaml.get("train", False), "train dataset not found"  # object365.yaml
        assert data_yaml.get("val", False), "validation dataset not found"  # lvis.yaml
        data = {k: [check_det_dataset(d) for d in v.get("yolo_data", [])] for k, v in data_yaml.items()}
        assert len(data["val"]) == 1, f"Only support validating on 1 dataset for now, but got {len(data['val'])}."
        val_split = "minival" if "lvis" in data["val"][0]["val"] else "val"
        for d in data["val"]:
            if d.get("minival") is None:  # for lvis dataset
                continue
            d["minival"] = str(d["path"] / d["minival"])
        for s in ["train", "val"]:
            final_data[s] = [d["train" if s == "train" else val_split] for d in data[s]]
            # save grounding data if there's one
            grounding_data = data_yaml[s].get("grounding_data")
            if grounding_data is None:
                continue
            grounding_data = grounding_data if isinstance(grounding_data, list) else [grounding_data]
            for g in grounding_data:
                assert isinstance(g, dict), f"Grounding data should be provided in dict format, but got {type(g)}"
            final_data[s] += grounding_data
        # NOTE: to make training work properly, set `nc` and `names`
        final_data["nc"] = data["val"][0]["nc"]
        final_data["names"] = data["val"][0]["names"]
        # NOTE: add path with lvis path
        final_data["path"] = data["val"][0]["path"]
        self.data = final_data
        if self.args.single_cls:  # consistent with base trainer
            LOGGER.info("Overriding class names with single class.")
            self.data["names"] = {0: "object"}
            self.data["nc"] = 1
        self.training_data = {}
        for d in data["train"]:
            if self.args.single_cls:
                d["names"] = {0: "object"}
                d["nc"] = 1
            self.training_data[d["train"]] = d
        return final_data["train"], final_data["val"][0]

    def plot_training_labels(self):
        """Do not plot labels for YOLO-World training."""
        pass

    def final_eval(self):
        """
        Perform final evaluation on the validation dataset.

        Configures the validator with the appropriate dataset and split before running evaluation.

        Returns:
            (dict): Evaluation metrics.
        """
        val = self.args.data["val"]["yolo_data"][0]
        self.validator.args.data = val
        self.validator.args.split = "minival" if isinstance(val, str) and "lvis" in val else "val"
        return super().final_eval()


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
