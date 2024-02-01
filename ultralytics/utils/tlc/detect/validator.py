import numpy as np
import tlc
import ultralytics

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.tlc.detect.nn import TLCDetectionModel
from ultralytics.utils import LOGGER, ops, metrics
from ultralytics.utils.tlc.detect.dataset import build_tlc_dataset
from ultralytics.utils.tlc.detect.utils import tlc_check_dataset, yolo_predicted_bounding_box_schema, construct_bbox_struct, parse_environment_variables, get_metrics_collection_epochs, yolo_image_embeddings_schema, training_phase_schema

def check_det_dataset(data: str):
    """Check if the dataset is compatible with the 3LC."""
    tables = tlc_check_dataset(data)
    names = tables["train"].get_value_map_for_column(tlc.BOUNDING_BOXES)
    return {
        "train": tables["train"],
        "val": tables["val"],
        "nc": len(names),
        "names": names,
    }

ultralytics.engine.validator.check_det_dataset = check_det_dataset

def set_up_metrics_writer(validator):
    if validator._trainer:
            validator._collection_epochs = get_metrics_collection_epochs(
                validator._env_vars['COLLECTION_EPOCH_START'],
                validator._trainer.args.epochs,
                validator._env_vars['COLLECTION_EPOCH_INTERVAL'],
                validator._env_vars['COLLECTION_DISABLE']
            )
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
        names = validator.data[validator._split].get_value_map_for_column(tlc.BOUNDING_BOXES)

    metrics_column_schemas = {
        tlc.PREDICTED_BOUNDING_BOXES: yolo_predicted_bounding_box_schema(names),
    }
    if validator._trainer:
        metrics_column_schemas["Training Phase"] = training_phase_schema()
    metrics_column_schemas.update(yolo_image_embeddings_schema(activation_size=256))
    
    validator.metrics_writer = tlc.MetricsWriter(
        run_url=validator._run.url,
        dataset_url=dataset_url,
        dataset_name=dataset_name,
        override_column_schemas=metrics_column_schemas
    )

class TLCDetectionValidator(DetectionValidator):
    """A class extending the BaseTrainer class for training a detection model using the 3LC."""

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None, run=None):
        LOGGER.info("Using 3LC Validator 🌟")
        self._env_vars = parse_environment_variables()
        self._run = run
        self._seen = 0
        self._final_validation = True
        self._split = args.get('split', None)

        _callbacks['on_val_start'].append(set_up_metrics_writer)
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)

        self.epoch = None

    def __call__(self, trainer=None, model=None, final_validation=False):
        self._trainer = trainer
        self._final_validation = final_validation
        if trainer:
            self.epoch = trainer.epoch
        return super().__call__(trainer, model)
    
    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build 3LC detection Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        table = self.data[self._split]
        return build_tlc_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride, table=table, use_sampling_weights=False)
    
    def _collect_metrics(self, predictions):
        batch_size = len(predictions)
        example_index = np.arange(self._seen, self._seen + batch_size)
        example_ids = self.dataloader.dataset.irect[example_index] if hasattr(self.dataloader.dataset, 'irect') else example_index

        metrics = {
            tlc.EXAMPLE_ID: example_ids,
            tlc.PREDICTED_BOUNDING_BOXES: self._process_batch_predictions(predictions),
        }
        if self.epoch is not None:
            metrics[tlc.EPOCH] = [self.epoch] * batch_size
            metrics["Training Phase"] = [1 if self._final_validation else 0] * batch_size

        if self._env_vars['IMAGE_EMBEDDINGS_DIM'] > 0:
            metrics["embeddings"] = TLCDetectionModel.activations.cpu()
        self.metrics_writer.add_batch(metrics_batch=metrics)

        self._seen += batch_size

        if self._seen == len(self.dataloader.dataset):
            self.metrics_writer.flush()
            metrics_infos = self.metrics_writer.get_written_metrics_infos()
            self._run.update_metrics(metrics_infos)
            self._seen = 0
            if self.epoch:
                self.epoch += 1

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
        if self.epoch is None:
            return True
        if self._final_validation and not self._env_vars['COLLECTION_DISABLE']:
            return True
        else:
            return self._trainer and self.epoch < self._trainer.args.epochs and self.epoch in self._collection_epochs
    