# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

import tlc

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.tlc.settings import Settings
from ultralytics.utils.tlc.constants import TLC_COLORSTR
from ultralytics.utils.tlc.utils import reduce_embeddings
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import strip_optimizer


class TLCTrainerMixin(BaseTrainer):
    """ A class extending the BaseTrainer class for training Ultralytics YOLO models with 3LC,
    which implements common 3LC-specific behavior across tasks. Use as a Mixin class for task-specific
    trainers.
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        LOGGER.info("Using 3LC Trainer 🌟")
        self._settings = Settings() if 'settings' not in overrides else overrides.pop('settings')
        self._settings.verify(training=True)

        assert 'data' in overrides or 'tables' in overrides, "You must provide either a data path or tables to train with 3LC."
        self._tables = overrides.pop('tables', None)

        # Column names
        self._image_column_name = overrides.pop("image_column_name", self._default_image_column_name)
        self._label_column_name = overrides.pop("label_column_name", self._default_label_column_name)

        super().__init__(cfg, overrides, _callbacks)

        self._train_validator = None

        if RANK in {-1, 0}:
            self._metrics_collection_epochs = set(self._settings.get_metrics_collection_epochs(self.epochs))
            
            # Create a 3LC run
            description = self._settings.run_description if self._settings.run_description else "Created with model.train()"
            project_name = self._settings.project_name if self._settings.project_name else self.data["train"].project_name
            self._run = tlc.init(
                project_name=project_name,
                description=description,
                run_name=self._settings.run_name,
            )

            LOGGER.info(f"{TLC_COLORSTR}Created run named '{self._run.url.parts[-1]}' in project {self._run.project_name}.")

            # Log parameters to 3LC
            self._log_3lc_parameters()

            self._print_metrics_collection_epochs()

        self.add_callback("on_train_epoch_start", resample_indices) # Run on all ranks
        
    def _log_3lc_parameters(self):
        """ Log various data as parameters to the tlc.Run. """
        if "val" in self.data:
            val_url = str(self.data["val"].url)
        else:
            val_url = str(self.data["test"].url)

        parameters = {
            **vars(self.args), # YOLO arguments
            "3LC/train_url": str(self.data.get("train").url), # 3LC table used for training
            "3LC/val_url": val_url, # 3LC table used for validation
            **{f"3LC/{k}": v for k, v in vars(self._settings).items()}, # 3LC settings
        }
        self._run.set_parameters(parameters)

    def _print_metrics_collection_epochs(self):
        """ Print collection epochs to the console. """

        # Special message when no collection is enabled
        if self._settings.collection_disable:
            message = "No metrics collection is enabled."
        # No collection during training
        elif not self._metrics_collection_epochs:
            message = "Metrics will be collected after training only."
        # Print collection epochs
        else:
            if len(self._metrics_collection_epochs) == 1:
                epoch = str(next(iter(self._metrics_collection_epochs)))
                message = f"Metrics will be collected after training and after epoch {epoch}."
            else:
                epochs = ", ".join(str(epoch) for epoch in sorted(self._metrics_collection_epochs))
                message = f"Metrics will be collected after training and after the following epochs: {epochs}"

        LOGGER.info(f"{TLC_COLORSTR}{message}")

    def get_dataset(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def build_dataset(self, table, mode="train", batch=None):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_validator(self, dataloader):
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def train_validator(self):
        if RANK in {-1, 0}:
            if not self._train_validator:
                train_validator_dataloader = self.get_dataloader(
                    self.trainset,
                    batch_size=self.batch_size if self.args.task == "obb" else self.batch_size * 2,
                    rank=-1, 
                    mode="val"
                )
                self._train_validator = self.get_validator(
                    dataloader=train_validator_dataloader
                )
            return self._train_validator
        else:
            return None

    def validate(self):
        """ Perform validation with 3LC metrics collection, also on the training data, if applicable."""
        # Validate on the training set
        if not self._settings.collection_disable and not self._settings.collection_val_only and self.epoch + 1 in self._metrics_collection_epochs:
            self.train_validator(trainer=self)
        
        # Validate on the validation/test set like usual
        return super().validate()
    
    def final_eval(self):
        # Set epoch on validator - required when final validation is called without prior mc during training
        if not self._settings.collection_val_only and not self._settings.collection_disable:
            self.train_validator._final_validation = True
            self.train_validator._epoch = self.epoch
            self.train_validator.data = self.data

        self.validator._final_validation = True

        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    if not self._settings.collection_val_only and not self._settings.collection_disable:
                        self.train_validator(model=f)
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

        if RANK in {-1, 0}:
            if self._settings.image_embeddings_dim > 0:
                foreign_table_url = self.trainset.url if not self._settings.collection_val_only else self.testset.url
                reduce_embeddings(
                    self._run,
                    method=self._settings.image_embeddings_reducer,
                    n_components=self._settings.image_embeddings_dim,
                    foreign_table_url=foreign_table_url,
                )
            self._run.set_status_completed()

    def save_metrics(self, metrics):
        # Log aggregate metrics after every epoch
        processed_metrics = self._process_metrics(metrics)
        self._run.add_output_value({"epoch": self.epoch+1, **processed_metrics})
        
        super().save_metrics(metrics=metrics)

    def _process_metrics(self, metrics):
        return metrics

# CALLBACKS ##############################################################################################################
def resample_indices(trainer):
    if trainer._settings.sampling_weights:
        trainer.train_loader.dataset.resample_indices()