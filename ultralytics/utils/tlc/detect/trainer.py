# Ultralytics YOLO 🚀 3LC Integration, AGPL-3.0 license
from __future__ import annotations

import copy
from typing import Any

import tlc

import ultralytics
from ultralytics.data import build_dataloader
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, IterableSimpleNamespace
from ultralytics.utils.tlc.detect.dataset import TLCDataset, build_tlc_dataset
from ultralytics.utils.tlc.detect.model import TLCDetectionModel
from ultralytics.utils.tlc.detect.settings import Settings
from ultralytics.utils.tlc.detect.utils import check_det_dataset, get_metrics_collection_epochs
from ultralytics.utils.tlc.detect.validator import TLCDetectionValidator
from ultralytics.utils.torch_utils import de_parallel, strip_optimizer, torch_distributed_zero_first

# Patch the check_det_dataset function so 3LC parses the dataset
ultralytics.engine.trainer.check_det_dataset = check_det_dataset


class TLCDetectionTrainer(DetectionTrainer):
    """Trainer class for YOLOv8 object detection with 3LC"""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        LOGGER.info("Using 3LC Trainer 🌟")
        self._settings = Settings() if 'settings' not in overrides else overrides.pop('settings')
        self._settings.verify(training=True)
        super().__init__(cfg, overrides, _callbacks)
        self._train_validator = None
        self._collection_epochs = get_metrics_collection_epochs(self._settings.collection_epoch_start, self.args.epochs,
                                                                self._settings.collection_epoch_interval,
                                                                self._settings.collection_disable)
        
        project_name = self._settings.project_name if self._settings.project_name else self.data["train"].project_name

        self._run = tlc.init(project_name, self._settings.run_name, self._settings.run_description) if not self._settings.collection_disable else None

        self.add_callback("on_train_epoch_start", _resample_train_dataset)
        self.add_callback("on_train_end", _reduce_embeddings)
        self.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

    @property
    def train_validator(self):
        if not self._train_validator:
            if RANK in (-1, 0):
                train_val_loader = self.get_dataloader(
                    self.testset,
                    batch_size=self.batch_size if self.args.task == "obb" else self.batch_size * 2,
                    rank=-1,
                    mode="val",
                    split="train",
                )
                self._train_validator = self.get_validator(loader=train_val_loader)
        return self._train_validator

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train", split=None):
        """Construct and return dataloader."""
        assert mode in ["train", "val"]
        split = split or mode
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size, split=split)
        #shuffle = mode == "train"
        shuffle = False
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def build_dataset(self, img_path: str, mode: str = "train", batch=None, split: str = "train") -> TLCDataset:
        """
        Build YOLO Dataset.

        :param img_path: Path to the folder containing images.
        :param mode: `train` mode or `val` mode, users are able to customize different augmentations for each mode.
        :param batch: Size of batches, this is for `rect`. Defaults to None.
        :param split: Split of the dataset, defaults to `train`. Should correspond to a key in the dataset dictionary.
        :return: A YOLO dataset populated with 3LC values.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_tlc_dataset(self.args,
                                 img_path,
                                 batch,
                                 self.data,
                                 mode=mode,
                                 rect=mode == "val",
                                 stride=gs,
                                 table=self.data[split],
                                 use_sampling_weights=self._settings.sampling_weights)
    
    def get_dataset(self):
        if self.args.task != "detect":
            raise NotImplementedError("Only detection task is supported for now in the 3LC integration.")

        self.data = check_det_dataset(self.args.data)
        return self.data["train"], self.data.get("val") or self.data.get("test")

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = TLCDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self, loader=None):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        if not loader:
            loader = self.test_loader

        args_dict = dict(copy.copy(self.args))
        args_dict["settings"] = self._settings

        return TLCDetectionValidator(
            loader,
            save_dir=self.save_dir,
            args=args_dict,
            _callbacks=self.callbacks,
            run=self._run,
        )

    def validate(self) -> Any:
        # Validate on train set
        if not self._settings.collection_disable and not self._settings.collection_val_only and self.epoch in self._collection_epochs:
            self.train_validator(trainer=self)

        # Validate on val/test set
        return super().validate()

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.train_validator(trainer=self, model=f, final_validation=True)
                    self.metrics = self.validator(trainer=self, model=f, final_validation=True)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

        tlc.close()


### CALLBACKS ############################################################################################################

def _on_fit_epoch_end(trainer: TLCDetectionTrainer) -> None:
    """ Get aggregate validation metrics for the current epoch and log them to 3LC.

    :param trainer: The trainer object.
    
    """
    assert isinstance(trainer, TLCDetectionTrainer)
    if trainer._run:
        # Format metric names
        metrics = {
            metric.strip("(B)").replace("metrics", "val").replace("/", "_"): value
            for metric, value in trainer.metrics.items()}
        trainer._run.add_output_value({"epoch": trainer.epoch, **metrics})

def _resample_train_dataset(trainer: TLCDetectionTrainer) -> None:
    """ Callback to be used for resampling the training dataset using 3LC Sample Weights.

    :param trainer: The trainer object.
    :raises AssertionError: If the trainer is not an instance of TLCDetectionTrainer.
    """
    assert isinstance(trainer, TLCDetectionTrainer)
    trainer.train_loader.dataset.resample_indices()


def _reduce_embeddings(trainer: TLCDetectionTrainer) -> None:
    """ Callback to be used for reducing the image embeddings using 3LC. Should be called at the end of training.

    :param trainer: The trainer object.
    :raises AssertionError: If the trainer is not an instance of TLCDetectionTrainer.
    """
    assert isinstance(trainer, TLCDetectionTrainer)
    if trainer._settings.image_embeddings_dim > 0:
        trainer._run.reduce_embeddings_by_foreign_table_url(
            foreign_table_url=trainer.data["val"].url,
            method=trainer._settings.image_embeddings_reducer,
            n_components=trainer._settings.image_embeddings_dim
        )
