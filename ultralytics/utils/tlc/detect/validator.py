import numpy as np
import tlc
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.tlc.detect.nn import TLCDetectionModel
from ultralytics.utils import LOGGER, ops, metrics
from ultralytics.utils.tlc.detect.utils import yolo_predicted_bounding_box_schema, construct_bbox_struct, parse_environment_variables, get_metrics_collection_epochs, yolo_image_embeddings_schema, training_phase_schema

class TLCDetectionValidator(DetectionValidator):
    """A class extending the BaseTrainer class for training a detection model using the 3LC."""

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None, run=None):
        LOGGER.info("Using 3LC Validator 🌟")
        self._env_vars = parse_environment_variables()
        self._run = None
        self._seen = 0
        self._final_validation = True
        if run:
            self._run = run
        else:
            if not self._env_vars['COLLECTION_DISABLE']:
                self._run = tlc.init(project_name=dataloader.dataset.table.project_name)

        if self._run is not None:
            metrics_column_schemas = {
                tlc.PREDICTED_BOUNDING_BOXES: yolo_predicted_bounding_box_schema(dataloader.dataset.data['names']),
                "Training Phase": training_phase_schema(),
            }
            metrics_column_schemas.update(yolo_image_embeddings_schema(activation_size=256))
            
            self.metrics_writer = tlc.MetricsWriter(
                run_url=self._run.url,
                dataset_url=dataloader.dataset.table.url,
                dataset_name=dataloader.dataset.table.dataset_name,
                override_column_schemas=metrics_column_schemas
            )
        self.epoch = 0
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)

    def __call__(self, trainer=None, model=None, epoch=None, final_validation=False):
        self._trainer = trainer
        if trainer:
            self._collection_epochs = get_metrics_collection_epochs(
                self._env_vars['COLLECTION_EPOCH_START'],
                trainer.args.epochs,
                self._env_vars['COLLECTION_EPOCH_INTERVAL'],
                self._env_vars['COLLECTION_DISABLE']
            )
        self._final_validation = final_validation
        if epoch:
            self.epoch = epoch
        return super().__call__(trainer, model)
    
    def _collect_metrics(self, predictions):
        batch_size = len(predictions)
        example_index = np.arange(self._seen, self._seen + batch_size)
        example_ids = self.dataloader.dataset.irect[example_index] if hasattr(self.dataloader.dataset, 'irect') else example_index

        epoch = self.epoch - 1 if self._final_validation else self.epoch
        metrics = {
            tlc.EPOCH: [epoch] * batch_size,
            "Training Phase": [1 if self._final_validation else 0] * batch_size,
            tlc.EXAMPLE_ID: example_ids,
            tlc.PREDICTED_BOUNDING_BOXES: self._process_batch_predictions(predictions),
        }
        if self._env_vars['IMAGE_EMBEDDINGS_DIM'] > 0:
            metrics["embeddings"] = TLCDetectionModel.activations.cpu()
        self.metrics_writer.add_batch(metrics_batch=metrics)

        self._seen += batch_size

        if self._seen == len(self.dataloader.dataset):
            self.metrics_writer.flush()
            metrics_infos = self.metrics_writer.get_written_metrics_infos()
            self._run.update_metrics(metrics_infos)
            self.epoch += 1
            self._seen = 0

    def _process_batch_predictions(self, batch_predictions):
        predicted_boxes = []
        for i, predictions in enumerate(batch_predictions):
            # Handle case with no predictions
            if len(predictions) == 0:
                predicted_boxes.append([])
                continue

            predictions = predictions.clone()
            predictions = predictions[predictions[:,4] > self._env_vars['CONF_THRES']] # filter out low confidence predictions
            # sort by confidence and remove excess boxes
            predictions = predictions[predictions[:, 4].argsort(descending=True)[:self._env_vars['MAX_DET']]]
            ori_shape = self._curr_batch['ori_shape'][i]
            resized_shape = self._curr_batch['resized_shape'][i]
            ratio_pad = self._curr_batch['ratio_pad'][i]
            height, width = ori_shape

            pred_box = predictions[:,:4].clone()
            pred_scaled = ops.scale_boxes(resized_shape, pred_box, ori_shape, ratio_pad)

            # Compute IoUs
            pbatch = self._prepare_batch(i, self._curr_batch)
            if pbatch['bbox'].shape[0]:
                ious = metrics.box_iou(pbatch['bbox'], pred_scaled) # IoU evaluated in xyxy format
                box_ious = ious.max(dim=0)[0].cpu().tolist()
            else:
                box_ious = [0.0] * pred_scaled.shape[0] # No predictions

            pred_xywh = ops.xyxy2xywhn(pred_scaled, w=width, h=height)

            conf = predictions[:,4].cpu().tolist()
            pred_cls = predictions[:,5].cpu().tolist()

            annotations = []
            for pi in range(len(predictions)):
                annotations.append({
                    'score': conf[pi],
                    'category_id': pred_cls[pi],
                    'bbox': pred_xywh[pi,:].cpu().tolist(),
                    'iou': box_ious[pi],
                })

            assert len(annotations) <= self._env_vars['MAX_DET'], "Should have at most MAX_DET predictions per image."

            predicted_boxes.append(
                construct_bbox_struct(
                    annotations,
                    image_width=width,
                    image_height=height,
                )
            )

        return predicted_boxes
    
    def preprocess(self, batch):
        self._curr_batch = super().preprocess(batch)
        return self._curr_batch

    def postprocess(self, preds):
        postprocessed = super().postprocess(preds)

        if self._should_collect_metrics():
            self._collect_metrics(postprocessed)

        return postprocessed
    
    def _should_collect_metrics(self):
        if self._final_validation and not self._env_vars['COLLECTION_DISABLE']:
            return True
        else:
            return self._trainer and self.epoch < self._trainer.args.epochs and self.epoch in self._collection_epochs
    