from typing import Any, Optional

import wandb as wb
import numpy as np
from PIL import Image
from ultralytics.engine.results import Results
from ultralytics.models.yolo.pose import PosePredictor
from ultralytics.utils.plotting import Annotator
from ultralytics.utils.callbacks.wb_utils.bbox import get_boxes, get_ground_truth_bbox_annotations


def annotate_keypoint_results(result: Results, visualize_skeleton: bool):
    annotator = Annotator(np.ascontiguousarray(result.orig_img[:, :, ::-1]))
    key_points = result.keypoints.data.numpy()
    for idx in range(key_points.shape[0]):
        annotator.kpts(key_points[idx], kpt_line=visualize_skeleton)
    return annotator.im


def annotate_keypoint_batch(image_path: str, keypoints: Any, visualize_skeleton: bool):
    original_image = None
    with Image.open(image_path) as original_image:
        original_image = np.ascontiguousarray(original_image)
        annotator = Annotator(original_image)
        annotator.kpts(keypoints.numpy(), kpt_line=visualize_skeleton)
        return annotator.im


def plot_pose_predictions(
    result: Results,
    model_name: str,
    visualize_skeleton: bool,
    table: Optional[wb.Table] = None,
):
    result = result.to("cpu")
    boxes, mean_confidence_map = get_boxes(result)
    annotated_image = annotate_keypoint_results(result, visualize_skeleton)
    prediction_image = wb.Image(annotated_image, boxes=boxes)
    table_row = [
        model_name,
        prediction_image,
        len(boxes["predictions"]["box_data"]),
        mean_confidence_map,
        result.speed,
    ]
    if table is not None:
        table.add_data(*table_row)
        return table
    return table_row
