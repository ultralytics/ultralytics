import tlc

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.tlc_utils import yolo_predicted_bounding_box_schema, construct_bbox_struct

class TLCDetectionValidator(DetectionValidator):
    """A class extending the BaseTrainer class for training a detection model using the 3LC."""

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None, run=None):
        LOGGER.info("Using 3LC Validator 🌟")
        if run:
            self._run = run
        else:
            self._run = tlc.init(project_name=dataloader.dataset.table.project_name)
        self.metrics_writer = tlc.MetricsWriter(
            run_url=self._run.url,
            dataset_url=dataloader.dataset.table.url,
            dataset_name=dataloader.dataset.table.dataset_name,
            override_column_schemas={
                tlc.PREDICTED_BOUNDING_BOXES: yolo_predicted_bounding_box_schema(dataloader.dataset.data['names'])
            },
        )
        self.epoch = 0
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)

    def __call__(self, trainer=None, model=None):
        self._trainer = trainer
        return super().__call__(trainer, model)

    def _prepare_pred(self, pred, pbatch):
        """ Call parent, but get the native space predictions first"""
        predn = super()._prepare_pred(pred, pbatch)

        if self._trainer and self.epoch < self._trainer.args.epochs:
            conf = predn[:,4]
            pred_cls = predn[:,5]
            predn_box = predn[:,:4]
            example_index = self.seen - 1
            example_id = self.dataloader.dataset.irect[example_index] if hasattr(self.dataloader.dataset, 'irect') else example_index
            height, width = pbatch['ori_shape']
            pred_xywh = ops.xyxy2xywhn(predn_box, w=width, h=height)

            annotations = []

            for i in range(len(conf)):
                annotations.append({
                    'score': conf[i].item(),
                    'category_id': pred_cls[i].item(),
                    'bbox': pred_xywh[i,:].cpu().tolist(),
                    'iou': 0.,
                })

            predicted_boxes = [
                construct_bbox_struct(
                    annotations,
                    image_width=width,
                    image_height=height,
                )
            ]

            metrics = {
                tlc.EPOCH: [self.epoch],
                tlc.EXAMPLE_ID: [example_id],
                tlc.PREDICTED_BOUNDING_BOXES: predicted_boxes,
            }

            self.metrics_writer.add_batch(metrics_batch=metrics)

            if self.seen == len(self.dataloader.dataset):
                self.metrics_writer.flush()
                metrics_infos = self.metrics_writer.get_written_metrics_infos()
                self._run.update_metrics(metrics_infos)
                self.epoch += 1
        
        return predn
