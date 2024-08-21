# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

import tlc

from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.tlc.constants import TLC_COLORSTR
from ultralytics.utils.tlc.settings import Settings
from ultralytics.utils.tlc.utils import image_embeddings_schema, training_phase_schema

def execute_when_collecting(method):
    def wrapper(self, *args, **kwargs):
        if self._should_collect:
            return method(self, *args, **kwargs)
    return wrapper

class TLCValidatorMixin(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None, run=None, image_column_name=None, label_column_name=None, settings=None):

        # Called by trainer (Get run and settings from trainer)
        if run is not None:
            self._run = run
            self._settings = settings
            self._image_column_name = image_column_name
            self._label_column_name = label_column_name
            self._training = True

        # Called directly (Create a run and get settings directly)
        else:
            if run in args:
                self._run = args.pop("run")
            else:
                self._run = None # Create run
            self._settings = args.pop("settings", Settings())
            self._image_column_name = args.pop("image_column_name", self._default_image_column_name)
            self._label_column_name = args.pop("label_column_name", self._default_label_column_name)
            self._table = args.pop("table", None)
            self._training = False

        # State
        self._epoch = None
        self._should_collect = None
        self._seen = None
        self._final_validation = False
        self._hook_handles = []

        super().__init__(dataloader, save_dir, pbar, args, _callbacks)

        if not self._training:
            self.data = self.check_dataset(
                self.args.data,
                {self.args.split: self._table} if self._table is not None else None,
                self._image_column_name,
                self._label_column_name,
                project_name=self._settings.project_name,
            )

        # Create a run if not provided
        if self._run is None:
            if tlc.active_run() is not None:
                # TODO: Match against provided project name / run name / description
                self._run = tlc.active_run()

                LOGGER.info(f"{TLC_COLORSTR}Reusing active run named '{self._run.url.parts[-1]}' in project {self._run.project_name}.")
            else:
                first_split = list(self.data.keys())[0]
                project_name = self._settings.project_name if self._settings.project_name else self.data[first_split].project_name
                self._run = tlc.init(
                    project_name=project_name,
                    description=self._settings.run_description if self._settings.run_description else "Created with 3LC Ultralytics Integration",
                    run_name=self._settings.run_name,
                )
                
                LOGGER.info(f"{TLC_COLORSTR}Created run named '{self._run.url.parts[-1]}' in project {self._run.project_name}.")

    def __call__(self, trainer=None, model=None):
        self._epoch = trainer.epoch if trainer is not None else self._epoch
        
        if trainer:
            self._should_collect = not self._settings.collection_disable and self._epoch + 1 in trainer._metrics_collection_epochs
        else:
            self._should_collect = not self._settings.collection_disable

        # Call parent to perform the validation
        out = super().__call__(trainer, model)

        self._post_validation()

        return out
    
    def get_desc(self):
        """ Add the split name next to the validation description"""
        desc = super().get_desc()

        split = self.dataloader.dataset.display_name.split("-")[-1] # get final part
        initial_spaces = len(desc) - len(desc.lstrip())
        split_centered = split.center(initial_spaces)
        split_str = f"{colorstr(split_centered)}"
        desc = split_str + desc[len(split_centered):]

        return desc
    
    def init_metrics(self, model):
        super().init_metrics(model)

        self._verify_model_data_compatibility(model.names)
        self._pre_validation(model)

    def build_dataset(self, table):
        """ Build a dataset from a table """
        raise NotImplementedError("Subclasses must implement this method.")

    def _verify_model_data_compatibility(self, names):
        """ Verify that the model being validated is compatible with the data"""
        raise NotImplementedError("Subclasses must implement this method.")

    def _get_metrics_schemas(self) -> dict[str, tlc.Schema]:
        """ Get the metrics schemas for the 3LC metrics data """
        raise NotImplementedError("Subclasses must implement this method.")

    def _compute_3lc_metrics(self, preds, batch) -> dict[str, tlc.MetricData]:
        """ Compute 3LC metrics for a batch of predictions and targets """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def _add_embeddings_hook(self, model) -> int:
        """ Add a hook to extract embeddings from the model, and infer the activation size """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def _infer_batch_size(self, preds) -> int:
        """ Infer the batch size from the predictions """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def update_metrics(self, preds, batch):
        """ Collect 3LC metrics """
        self._update_metrics(preds, batch)

        # Let parent collect its own metrics
        super().update_metrics(preds, batch)

    @execute_when_collecting
    def _update_metrics(self, preds, batch):
        """ Update 3LC metrics with common and task-specific metrics"""
        batch_size = self._infer_batch_size(preds, batch)
        example_indices = list(range(self._seen, self._seen + batch_size))
        example_ids = [int(self.dataloader.dataset.example_ids[i]) for i in example_indices]

        batch_metrics = {
            tlc.EXAMPLE_ID: example_ids,
            **self._compute_3lc_metrics(preds, batch) # Task specific metrics
        }

        if self._settings.image_embeddings_dim > 0:
            batch_metrics["embeddings"] = self.embeddings

        if self._epoch is not None:
            batch_metrics[tlc.EPOCH] = [self._epoch + 1] * batch_size
            training_phase = 1 if self._final_validation else 0
            batch_metrics["Training Phase"] = [training_phase] * batch_size

        self._metrics_writer.add_batch(batch_metrics)
        self._seen += batch_size

    @execute_when_collecting
    def _pre_validation(self, model):
        """ Prepare the validator for metrics collection """
        column_schemas = {}
        column_schemas.update(self._get_metrics_schemas()) # Add task-specific metrics schema

        if self._settings.image_embeddings_dim > 0:
            # Add hook and get the activation size
            activation_size = self._add_embeddings_hook(model)

            column_schemas["embeddings"] = image_embeddings_schema(activation_size=activation_size)

        if self._epoch is not None:
            column_schemas["Training Phase"] = training_phase_schema()

        self._run.set_status_collecting()

        self._metrics_writer = tlc.MetricsTableWriter(
            run_url=self._run.url,
            foreign_table_url=self.dataloader.dataset.table.url,
            column_schemas=column_schemas
        )

        self._seen = 0

    @execute_when_collecting
    def _post_validation(self):
        """ Clean up the validator after one validation pass """
        # Write metrics data to 3LC run
        self._metrics_writer.finalize()
        metrics_infos = self._metrics_writer.get_written_metrics_infos()
        self._run.update_metrics(metrics_infos)

        self._run.add_input_table(self.dataloader.dataset.table.url)

        # Improve memory usage - don't cache metrics data
        for metrics_info in metrics_infos:
            tlc.ObjectRegistry._delete_object_from_caches(
                tlc.Url(metrics_info["url"]).to_absolute(self._run.url)
            )

        self._run.set_status_running()

        # Remove hook handles
        if self._settings.image_embeddings_dim > 0:
            for handle in self._hook_handles:
                handle.remove()
            self._hook_handles.clear()
        
        # Reset state
        self._seen = None
        self._training_phase = None
        self._final_validation = None

    def _verify_model_data_compatibility(self, model_class_names):
        """ Verify that the model classes match the dataset classes. For a classification model, this amounts to checking
        that the order of the class names match and that they have the same number of classes."""
        dataset_class_names=self.data["names"]
        if len(model_class_names) != len(dataset_class_names):
            raise ValueError(
                f"The model and data are incompatible. The model was trained on {len(model_class_names)} classes, but the data has {len(dataset_class_names)} classes. "
            )
        elif model_class_names != dataset_class_names:
            raise ValueError(
                "The model was trained on a different set of classes to the classes in the dataset, or the classes are in a different order."
            )