import tlc
from ultralytics.models.yolo.detect import DetectionValidator
from tlc.client.torch.metrics.metrics_collectors.bounding_box_metrics_collector import (_TLCBoundingBox,
                                                                                        _TLCBoundingBoxes)

from ultralytics.utils import LOGGER, ops

def yolo_predicted_bounding_box_schema(categories):
    """ Create a 3LC bounding box schema for YOLOv5

    :param categories: Categories for the current dataset.
    :returns: The YOLO bounding box schema for predicted boxes.
    """
    label_value_map = {float(i): tlc.MapElement(class_name) for i, class_name in categories.items()}

    bounding_box_schema = tlc.BoundingBoxListSchema(
        label_value_map=label_value_map,
        x0_number_role=tlc.NUMBER_ROLE_BB_CENTER_X,
        x1_number_role=tlc.NUMBER_ROLE_BB_SIZE_X,
        y0_number_role=tlc.NUMBER_ROLE_BB_CENTER_Y,
        y1_number_role=tlc.NUMBER_ROLE_BB_SIZE_Y,
        x0_unit=tlc.UNIT_RELATIVE,
        y0_unit=tlc.UNIT_RELATIVE,
        x1_unit=tlc.UNIT_RELATIVE,
        y1_unit=tlc.UNIT_RELATIVE,
        description='Predicted Bounding Boxes',
        writable=False,
        is_prediction=True,
        include_segmentation=False,
    )

    return bounding_box_schema

class TLCDetectionValidator(DetectionValidator):
    """A class extending the BaseTrainer class for training a detection model using the 3LC."""

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        LOGGER.info("Using 3LC Validator 🌟")
        if not hasattr(self, '_run'):
            self._run = tlc.init(project_name='yolov8-hackathon')
        self.metrics_writer = tlc.MetricsWriter(
            run_url=self._run.url,
            dataset_url=self.dataloader.dataset.table.url,
            override_column_schemas={
                tlc.PREDICTED_BOUNDING_BOXES: yolo_predicted_bounding_box_schema(dataloader.dataset.data['names'])
            },
        )
        self.epoch = 0
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)

    def _prepare_pred(self, pred, pbatch):
        """ Call parent, but get the native space predictions first"""
        predn = super()._prepare_pred(pred, pbatch)

        conf = predn[:,4]
        pred_cls = predn[:,5:]
        predn_box = predn[:,:4]
        example_id = self.seen - 1
        width, height = pbatch['ori_shape']
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
                image_width=0,
                image_height=0,
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

def construct_bbox_struct(
    predicted_annotations,
    image_width: int,
    image_height: int,
    inverse_label_mapping= None,
):
    """Construct a 3LC bounding box struct from a list of bounding boxes.

    :param predicted_annotations: A list of predicted bounding boxes.
    :param image_width: The width of the image.
    :param image_height: The height of the image.
    :param inverse_label_mapping: A mapping from predicted label to category id.
    """

    bbox_struct = _TLCBoundingBoxes(
        bb_list=[],
        image_width=image_width,
        image_height=image_height,
    )

    for pred in predicted_annotations:
        bbox, label, score, iou = pred['bbox'], pred['category_id'], pred['score'], pred['iou']
        label_val = inverse_label_mapping[label] if inverse_label_mapping is not None else label
        bbox_struct['bb_list'].append(
            _TLCBoundingBox(
                label=label_val,
                confidence=score,
                iou=iou,
                x0=bbox[0],
                y0=bbox[1],
                x1=bbox[2],
                y1=bbox[3],
            ))

    return bbox_struct