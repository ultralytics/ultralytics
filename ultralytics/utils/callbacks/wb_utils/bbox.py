from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import wandb
from tqdm.auto import tqdm

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops


def scale_bounding_box_to_original_image_shape(
    box: torch.Tensor,
    resized_image_shape: Tuple,
    original_image_shape: Tuple,
    ratio_pad: bool,
) -> List[int]:
    """
    YOLOv8 resizes images during training and the label values are normalized based on this resized shape.

    This function rescales the bounding box labels to the original
    image shape.

    Reference: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/utils/callbacks/comet.py#L105
    """
    resized_image_height, resized_image_width = resized_image_shape
    # Convert normalized xywh format predictions to xyxy in resized scale format
    box = ops.xywhn2xyxy(box, h=resized_image_height, w=resized_image_width)
    # Scale box predictions from resized image scale back to original image scale
    box = ops.scale_boxes(resized_image_shape, box, original_image_shape, ratio_pad)
    # # Convert bounding box format from xyxy to xywh for Comet logging
    box = ops.xyxy2xywh(box)
    return box.tolist()


def get_ground_truth_bbox_annotations(img_idx: int,
                                      image_path: str,
                                      batch: Dict,
                                      class_name_map: Dict = None) -> List[Dict[str, Any]]:
    """Get ground truth bounding box annotation data in the form required for `wandb.Image` overlay system."""
    indices = batch['batch_idx'] == img_idx
    bboxes = batch['bboxes'][indices]
    cls_labels = batch['cls'][indices].squeeze(1).tolist()

    class_name_map_reverse = {v: k for k, v in class_name_map.items()}

    if len(bboxes) == 0:
        wandb.termwarn(f'Image: {image_path} has no bounding boxes labels', repeat=False)
        return None

    cls_labels = batch['cls'][indices].squeeze(1).tolist()
    if class_name_map:
        cls_labels = [str(class_name_map[label]) for label in cls_labels]

    original_image_shape = batch['ori_shape'][img_idx]
    resized_image_shape = batch['resized_shape'][img_idx]
    ratio_pad = batch['ratio_pad'][img_idx]

    data = []
    for box, label in zip(bboxes, cls_labels):
        box = scale_bounding_box_to_original_image_shape(box, resized_image_shape, original_image_shape, ratio_pad)
        data.append({
            'position': {
                'middle': [int(box[0]), int(box[1])],
                'width': int(box[2]),
                'height': int(box[3]), },
            'domain': 'pixel',
            'class_id': class_name_map_reverse[label],
            'box_caption': label, })

    return data


def get_mean_confidence_map(classes: List, confidence: List, class_id_to_label: Dict) -> Dict[str, float]:
    """Get Mean-confidence map from the predictions to be logged into a `wandb.Table`."""
    confidence_map = {v: [] for _, v in class_id_to_label.items()}
    for class_idx, confidence_value in zip(classes, confidence):
        confidence_map[class_id_to_label[class_idx]].append(confidence_value)
    updated_confidence_map = {}
    for label, confidence_list in confidence_map.items():
        if len(confidence_list) > 0:
            updated_confidence_map[label] = sum(confidence_list) / len(confidence_list)
        else:
            updated_confidence_map[label] = 0
    return updated_confidence_map


def get_boxes(result: Results) -> Tuple[Dict, Dict]:
    """Convert an ultralytics prediction result into metadata for the `wandb.Image` overlay system."""
    boxes = result.boxes.xywh.long().numpy()
    classes = result.boxes.cls.long().numpy()
    confidence = result.boxes.conf.numpy()
    class_id_to_label = {int(k): str(v) for k, v in result.names.items()}
    mean_confidence_map = get_mean_confidence_map(classes, confidence, class_id_to_label)
    box_data = []
    for idx in range(len(boxes)):
        box_data.append({
            'position': {
                'middle': [int(boxes[idx][0]), int(boxes[idx][1])],
                'width': int(boxes[idx][2]),
                'height': int(boxes[idx][3]), },
            'domain': 'pixel',
            'class_id': int(classes[idx]),
            'box_caption': class_id_to_label[int(classes[idx])],
            'scores': {
                'confidence': float(confidence[idx])}, })
    boxes = {
        'predictions': {
            'box_data': box_data,
            'class_labels': class_id_to_label, }, }
    return boxes, mean_confidence_map


def plot_bbox_predictions(result: Results,
                          model_name: str,
                          table: Optional[wandb.Table] = None) -> Union[wandb.Table, Tuple[wandb.Image, Dict, Dict]]:
    """
    Plot the images with the W&B overlay system.

    The `wandb.Image` is either added to a `wandb.Table` or returned.
    """
    result = result.to('cpu')
    boxes, mean_confidence_map = get_boxes(result)
    image = wandb.Image(result.orig_img[:, :, ::-1], boxes=boxes)
    if table is not None:
        table.add_data(
            model_name,
            image,
            len(boxes['predictions']['box_data']),
            mean_confidence_map,
            result.speed,
        )
        return table
    return image, boxes['predictions'], mean_confidence_map


def plot_detection_validation_results(
    dataloader: Any,
    class_label_map: Dict,
    model_name: str,
    predictor: DetectionPredictor,
    table: wandb.Table,
    max_validation_batches: int,
    epoch: Optional[int] = None,
) -> wandb.Table:
    """Plot validation results in a table."""
    data_idx = 0
    num_dataloader_batches = len(dataloader.dataset) // dataloader.batch_size
    max_validation_batches = min(max_validation_batches, num_dataloader_batches)
    for batch_idx, batch in enumerate(dataloader):
        prediction_results = predictor(batch['im_file'])
        progress_bar_result_iterable = tqdm(
            enumerate(prediction_results),
            desc=f'Generating Visualizations for batch-{batch_idx + 1}/{max_validation_batches}')
        for img_idx, prediction_result in progress_bar_result_iterable:
            prediction_result = prediction_result.to('cpu')
            _, prediction_box_data, mean_confidence_map = plot_bbox_predictions(prediction_result, model_name)
            try:
                ground_truth_data = get_ground_truth_bbox_annotations(img_idx, batch['im_file'][img_idx], batch,
                                                                      class_label_map)
                wandb_image = wandb.Image(
                    batch['im_file'][img_idx],
                    boxes={
                        'ground-truth': {
                            'box_data': ground_truth_data,
                            'class_labels': class_label_map, },
                        'predictions': {
                            'box_data': prediction_box_data['box_data'],
                            'class_labels': class_label_map, }, },
                )
                table_rows = [
                    data_idx,
                    batch_idx,
                    wandb_image,
                    mean_confidence_map,
                    prediction_result.speed, ]
                table_rows = [epoch] + table_rows if epoch is not None else table_rows
                table_rows = [model_name] + table_rows
                table.add_data(*table_rows)
                data_idx += 1
            except TypeError:
                pass
        if batch_idx + 1 == max_validation_batches:
            break
    return table
