# Ultralytics YOLO 🚀 3LC Integration, AGPL-3.0 license
from __future__ import annotations

import numpy as np
import tlc
import torch
from tlc.client.torch.metrics.metrics_collectors.bounding_box_metrics_collector import _TLCPredictedBoundingBoxes

import ultralytics
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, metrics, ops
from ultralytics.utils.tlc.constants import TRAINING_PHASE
from ultralytics.utils.tlc.detect.dataset import build_tlc_dataset
from ultralytics.utils.tlc.detect.nn import TLCDetectionModel
from ultralytics.utils.tlc.detect.settings import Settings
from ultralytics.utils.tlc.detect.utils import (
    check_det_dataset,
    construct_bbox_struct,
    get_metrics_collection_epochs,
    get_names_from_yolo_table,
    infer_embeddings_size,
    training_phase_schema,
    yolo_image_embeddings_schema,
    yolo_predicted_bounding_box_schema,
)

# Patch the check_det_dataset function so 3LC parses the dataset
ultralytics.engine.validator.check_det_dataset = check_det_dataset


class TLCDetectionValidator(DetectionValidator):
    """Validator class for YOLOv8 object detection with 3LC"""

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None, run=None):
        LOGGER.info("Using 3LC Validator 🌟")

        self._settings = args.pop('settings', Settings())
        self._run = run
        self._seen = 0
        self._final_validation = True
        self._split = args.get('split', None)
        self._embedding_size = None # Set the first time the model is seen

        _callbacks['on_val_start'].append(verify_settings)
        _callbacks['on_val_start'].append(set_up_metrics_writer)
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)

        self.epoch = None

    def __call__(self, trainer=None, model=None, final_validation=False):
        self._trainer = trainer
        self._final_validation = final_validation
        if trainer:
            self.epoch = trainer.epoch

        if self._settings.image_embeddings_dim > 0 and isinstance(model, DetectionModel):
            # Add TLCDetectionModel forward method to allow for embedding collection
            model.__class__ = TLCDetectionModel

        if self._embedding_size is None and self._settings.image_embeddings_dim > 0:
            self._embedding_size = infer_embeddings_size(model if model else trainer.model)

        output = super().__call__(trainer, model)
        if self._run:
            self._run.set_status_running()
        
        return output

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build 3LC detection Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        table = self.data[self._split]
        return build_tlc_dataset(self.args,
                                 img_path,
                                 batch,
                                 self.data,
                                 mode=mode,
                                 stride=self.stride,
                                 table=table,
                                 use_sampling_weights=False)

    def _collect_metrics(self, predictions: list[torch.Tensor]) -> None:
        """Collects metrics for the current batch of predictions.

        :param predictions: The batch of predictions.
        """
        batch_size = len(predictions)
        example_index = np.arange(self._seen, self._seen + batch_size)
        example_ids = self.dataloader.dataset.irect[example_index] if hasattr(self.dataloader.dataset,
                                                                              'irect') else example_index

        metrics = {
            tlc.EXAMPLE_ID: example_ids,
            tlc.PREDICTED_BOUNDING_BOXES: self._process_batch_predictions(predictions), }
        if self.epoch is not None:
            metrics[tlc.EPOCH] = [self.epoch] * batch_size
            metrics[TRAINING_PHASE] = [1 if self._final_validation else 0] * batch_size

        if self._settings.image_embeddings_dim > 0:
            metrics["embeddings"] = TLCDetectionModel.activations.cpu()
        self.metrics_writer.add_batch(metrics_batch=metrics)

        self._seen += batch_size

        if self._seen == len(self.dataloader.dataset):
            self.metrics_writer.finalize()
            metrics_infos = self.metrics_writer.get_written_metrics_infos()
            self._run.update_metrics(metrics_infos)
            self._seen = 0
            if self.epoch:
                self.epoch += 1

    def _process_batch_predictions(self, batch_predictions: list[torch.Tensor]) -> list[_TLCPredictedBoundingBoxes]:
        """Convert a batch of predictions to a list of 3LC bounding box dicts.

        :param batch_predictions: The batch of predictions.
        :return: A list of 3LC bounding box dicts.
        """
        predicted_boxes = []
        for i, predictions in enumerate(batch_predictions):
            ori_shape = self._curr_batch['ori_shape'][i]
            resized_shape = self._curr_batch['resized_shape'][i]
            ratio_pad = self._curr_batch['ratio_pad'][i]
            height, width = ori_shape

            # Handle case with no predictions
            if len(predictions) == 0:
                predicted_boxes.append(construct_bbox_struct(
                    [],
                    image_width=width,
                    image_height=height,
                ))
                continue

            predictions = predictions.clone()
            predictions = predictions[predictions[:, 4]
                                      > self._settings.conf_thres]  # filter out low confidence predictions
            # sort by confidence and remove excess boxes
            predictions = predictions[predictions[:, 4].argsort(descending=True)[:self._settings.max_det]]

            pred_box = predictions[:, :4].clone()
            pred_scaled = ops.scale_boxes(resized_shape, pred_box, ori_shape, ratio_pad)

            # Compute IoUs
            pbatch = self._prepare_batch(i, self._curr_batch)
            if pbatch['bbox'].shape[0]:
                ious = metrics.box_iou(pbatch['bbox'], pred_scaled)  # IoU evaluated in xyxy format
                box_ious = ious.max(dim=0)[0].cpu().tolist()
            else:
                box_ious = [0.0] * pred_scaled.shape[0]  # No predictions

            pred_xywh = ops.xyxy2xywhn(pred_scaled, w=width, h=height)

            conf = predictions[:, 4].cpu().tolist()
            pred_cls = predictions[:, 5].cpu().tolist()

            annotations = []
            for pi in range(len(predictions)):
                annotations.append({
                    'score': conf[pi],
                    'category_id': int(pred_cls[pi]),
                    'bbox': pred_xywh[pi, :].cpu().tolist(),
                    'iou': box_ious[pi], })

            assert len(annotations) <= self._settings.max_det, "Should have at most MAX_DET predictions per image."

            predicted_boxes.append(construct_bbox_struct(
                annotations,
                image_width=width,
                image_height=height,
            ))

        return predicted_boxes

    def preprocess(self, batch):
        self._curr_batch = super().preprocess(batch)
        return self._curr_batch

    def postprocess(self, preds):
        postprocessed = super().postprocess(preds)

        if self._should_collect_metrics():
            self._collect_metrics(postprocessed)

        return postprocessed

    def _should_collect_metrics(self) -> bool:
        """Determines if metrics should be collected for the current batch.

        :return: True if metrics should be collected, False otherwise.
        """
        if self.epoch is None:
            return True
        if self._final_validation and not self._settings.collection_disable:
            return True
        else:
            return self._trainer and self.epoch < self._trainer.args.epochs and self.epoch in self._collection_epochs


### CALLBACKS ############################################################################################################


def verify_settings(validator: TLCDetectionValidator) -> None:
    """Sets the settings for the validator, used as a callback.

    :param validator: The validator object.
    :raises AssertionError: If the validator is not an instance of TLCDetectionValidator.
    """
    assert isinstance(validator, TLCDetectionValidator), "validator must be an instance of TLCDetectionValidator."
    if validator._trainer:
        validator._settings = validator._trainer._settings
    else:
        validator._settings.verify(training=False)


def set_up_metrics_writer(validator: TLCDetectionValidator) -> None:
    """Sets up the metrics writer for the validator, used as a callback.

    :param validator: The validator object.
    :raises AssertionError: If the validator is not an instance of TLCDetectionValidator.
    """
    assert isinstance(validator, TLCDetectionValidator), "validator must be an instance of TLCDetectionValidator."

    if validator._trainer:
        validator._collection_epochs = get_metrics_collection_epochs(validator._settings.collection_epoch_start,
                                                                     validator._trainer.args.epochs,
                                                                     validator._settings.collection_epoch_interval,
                                                                     validator._settings.collection_disable)
        names = validator.dataloader.dataset.data['names']
        dataset_url = validator.dataloader.dataset.table.url
        dataset_name = validator.dataloader.dataset.table.dataset_name
    else:
        if validator._split is None:
            raise ValueError("split must be provided when calling .val() directly.")
        project_name = validator.data[validator._split].project_name
        if not validator._run:
            # Use existing ongoing run if available
            validator._run = tlc.active_run() if tlc.active_run() else tlc.init(project_name=project_name)
        dataset_url = validator.data[validator._split].url
        dataset_name = validator.data[validator._split].dataset_name
        names = get_names_from_yolo_table(validator.data[validator._split])

    if validator._run is not None:
        metrics_column_schemas = {
            tlc.PREDICTED_BOUNDING_BOXES: yolo_predicted_bounding_box_schema(names), }
        if validator._trainer:
            metrics_column_schemas[TRAINING_PHASE] = training_phase_schema()
        if validator._settings.image_embeddings_dim > 0:
            metrics_column_schemas.update(yolo_image_embeddings_schema(activation_size=validator._embedding_size))

        validator.metrics_writer = tlc.MetricsTableWriter(
            run_url=validator._run.url,
            foreign_table_url=dataset_url,
            foreign_table_display_name=dataset_name,
            column_schemas=metrics_column_schemas
        )

        validator._run.set_status_collecting()
