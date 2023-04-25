import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.vit.sam import PromptPredictor, build_sam
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.engine.results import Results

def auto_annotate(data, det_model="yolov8x.pt", sam_model="sam_b.pt", device=""):
    device = select_device(device)
    det_model = YOLO(det_model)
    sam_model = build_sam(sam_model)
    det_model.to(device)
    sam_model.to(device)

    prompt_predictor = PromptPredictor(sam_model)
    det_results = det_model(data, stream=True)

    for result in det_results:
        boxes = result.boxes.xyxy  # Boxes object for bbox outputs
        class_ids = result.boxes.cls.long().tolist()
        prompt_predictor.set_image(result.orig_img)
        masks, _, _ = prompt_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=prompt_predictor.transform.apply_boxes_torch(boxes, result.orig_shape[:2]),
            multimask_output=False,
        )

        sam_result = Results(result.orig_img, path=result.path, names=det_model.names, masks=masks)
        segments = sam_result.masks.xyn
        