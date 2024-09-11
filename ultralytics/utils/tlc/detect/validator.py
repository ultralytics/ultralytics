# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

import tlc
import torch
import weakref

from ultralytics.data import build_dataloader
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import metrics, ops
from ultralytics.utils.tlc.constants import IMAGE_COLUMN_NAME, DETECTION_LABEL_COLUMN_NAME
from ultralytics.utils.tlc.detect.utils import build_tlc_yolo_dataset, yolo_predicted_bounding_box_schema, construct_bbox_struct, tlc_check_det_dataset
from ultralytics.utils.tlc.engine.validator import TLCValidatorMixin
from ultralytics.utils.tlc.utils import create_sampler

class TLCDetectionValidator(TLCValidatorMixin, DetectionValidator):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = DETECTION_LABEL_COLUMN_NAME

    def check_dataset(self, *args, **kwargs):
        return tlc_check_det_dataset(*args, **kwargs)

    def get_dataloader(self, dataset_path, batch_size):
        """Builds and returns a data loader with given parameters."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")

        sampler = create_sampler(dataset.table, mode="val", settings=self._settings)
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1, sampler=sampler)

    def build_dataset(self, table, mode="val", batch=None):
        return build_tlc_yolo_dataset(self.args, table, batch, self.data, mode=mode, stride=self.stride)

    def _get_metrics_schemas(self):
        return {
            tlc.PREDICTED_BOUNDING_BOXES: yolo_predicted_bounding_box_schema(self.data["names"]),
        }
    
    def _compute_3lc_metrics(self, preds, batch):
        processed_predictions = self._process_detection_predictions(preds, batch)
        return {
            tlc.PREDICTED_BOUNDING_BOXES: processed_predictions,
        }
    
    def _process_detection_predictions(self, preds, batch):
        predicted_boxes = []
        for i, predictions in enumerate(preds):
            ori_shape = batch['ori_shape'][i]
            resized_shape = batch['resized_shape'][i]
            ratio_pad = batch['ratio_pad'][i]
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
            pbatch = self._prepare_batch(i, batch)
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
    
    def _add_embeddings_hook(self, model) -> int:
        if hasattr(model.model, "model"):
            model = model.model

        # Find index of the SPPF layer
        sppf_index = next((i for i, m in enumerate(model.model) if "SPPF" in m.type), -1)

        if sppf_index == -1:
            raise ValueError("No SPPF layer found in model, cannot collect embeddings.")
        
        weak_self = weakref.ref(self) # Avoid circular reference (self <-> hook_fn)
        def hook_fn(module, input, output):
            # Store embeddings
            self_ref = weak_self()
            flattened_output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1)).squeeze(-1).squeeze(-1)
            embeddings = flattened_output.detach().cpu().numpy()
            self_ref.embeddings = embeddings

        # Add forward hook to collect embeddings
        for i, module in enumerate(model.model):
            if i == sppf_index:
                self._hook_handles.append(module.register_forward_hook(hook_fn))

        activation_size = model.model[sppf_index]._modules['cv2']._modules['conv'].out_channels
        return activation_size
    
    def _infer_batch_size(self, preds, batch) -> int:
        return len(batch['im_file'])
