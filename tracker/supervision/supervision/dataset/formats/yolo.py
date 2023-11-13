import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from supervision.dataset.utils import approximate_mask_with_polygons
from supervision.detection.core import Detections
from supervision.detection.utils import polygon_to_mask, polygon_to_xyxy
from supervision.utils.file import (
    list_files_with_extensions,
    read_txt_file,
    read_yaml_file,
    save_text_file,
    save_yaml_file,
)


def _parse_box(values: List[str]) -> np.ndarray:
    x_center, y_center, width, height = values
    return np.array(
        [
            float(x_center) - float(width) / 2,
            float(y_center) - float(height) / 2,
            float(x_center) + float(width) / 2,
            float(y_center) + float(height) / 2,
        ],
        dtype=np.float32,
    )


def _box_to_polygon(box: np.ndarray) -> np.ndarray:
    return np.array(
        [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
    )


def _parse_polygon(values: List[str]) -> np.ndarray:
    return np.array(values, dtype=np.float32).reshape(-1, 2)


def _polygons_to_masks(
    polygons: List[np.ndarray], resolution_wh: Tuple[int, int]
) -> np.ndarray:
    return np.array(
        [
            polygon_to_mask(polygon=polygon, resolution_wh=resolution_wh)
            for polygon in polygons
        ],
        dtype=bool,
    )


def _with_mask(lines: List[str]) -> bool:
    return any([len(line.split()) > 5 for line in lines])


def _extract_class_names(file_path: str) -> List[str]:
    data = read_yaml_file(file_path=file_path)
    names = data["names"]
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names.keys())]
    return names


def _image_name_to_annotation_name(image_name: str) -> str:
    base_name, _ = os.path.splitext(image_name)
    return base_name + ".txt"


def yolo_annotations_to_detections(
    lines: List[str], resolution_wh: Tuple[int, int], with_masks: bool
) -> Detections:
    if len(lines) == 0:
        return Detections.empty()

    class_id, relative_xyxy, relative_polygon = [], [], []
    w, h = resolution_wh
    for line in lines:
        values = line.split()
        class_id.append(int(values[0]))
        if len(values) == 5:
            box = _parse_box(values=values[1:])
            relative_xyxy.append(box)
            if with_masks:
                relative_polygon.append(_box_to_polygon(box=box))
        elif len(values) > 5:
            polygon = _parse_polygon(values=values[1:])
            relative_xyxy.append(polygon_to_xyxy(polygon=polygon))
            if with_masks:
                relative_polygon.append(polygon)

    class_id = np.array(class_id, dtype=int)
    relative_xyxy = np.array(relative_xyxy, dtype=np.float32)
    xyxy = relative_xyxy * np.array([w, h, w, h], dtype=np.float32)

    if not with_masks:
        return Detections(class_id=class_id, xyxy=xyxy)

    polygons = [
        (polygon * np.array(resolution_wh)).astype(int) for polygon in relative_polygon
    ]
    mask = _polygons_to_masks(polygons=polygons, resolution_wh=resolution_wh)
    return Detections(class_id=class_id, xyxy=xyxy, mask=mask)


def load_yolo_annotations(
    images_directory_path: str,
    annotations_directory_path: str,
    data_yaml_path: str,
    force_masks: bool = False,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, Detections]]:
    """
    Loads YOLO annotations and returns class names, images,
        and their corresponding detections.

    Args:
        images_directory_path (str): The path to the directory containing the images.
        annotations_directory_path (str): The path to the directory
            containing the YOLO annotation files.
        data_yaml_path (str): The path to the data
            YAML file containing class information.
        force_masks (bool, optional): If True, forces masks to be loaded
            for all annotations, regardless of whether they are present.

    Returns:
        Tuple[List[str], Dict[str, np.ndarray], Dict[str, Detections]]:
            A tuple containing a list of class names, a dictionary with
            image names as keys and images as values, and a dictionary
            with image names as keys and corresponding Detections instances as values.
    """
    image_paths = list_files_with_extensions(
        directory=images_directory_path, extensions=["jpg", "jpeg", "png"]
    )

    classes = _extract_class_names(file_path=data_yaml_path)
    images = {}
    annotations = {}

    for image_path in image_paths:
        image_stem = Path(image_path).stem
        image_path = str(image_path)
        image = cv2.imread(image_path)

        annotation_path = os.path.join(annotations_directory_path, f"{image_stem}.txt")
        if not os.path.exists(annotation_path):
            images[image_path] = image
            annotations[image_path] = Detections.empty()
            continue

        lines = read_txt_file(str(annotation_path))
        h, w, _ = image.shape
        resolution_wh = (w, h)

        with_masks = _with_mask(lines=lines)
        with_masks = force_masks if force_masks else with_masks
        annotation = yolo_annotations_to_detections(
            lines=lines, resolution_wh=resolution_wh, with_masks=with_masks
        )

        images[image_path] = image
        annotations[image_path] = annotation
    return classes, images, annotations


def object_to_yolo(
    xyxy: np.ndarray,
    class_id: int,
    image_shape: Tuple[int, int, int],
    polygon: Optional[np.ndarray] = None,
) -> str:
    h, w, _ = image_shape
    if polygon is None:
        xyxy_relative = xyxy / np.array([w, h, w, h], dtype=np.float32)
        x_min, y_min, x_max, y_max = xyxy_relative
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        return f"{int(class_id)} {x_center:.5f} {y_center:.5f} {width:.5f} {height:.5f}"
    else:
        polygon_relative = polygon / np.array([w, h], dtype=np.float32)
        polygon_relative = polygon_relative.reshape(-1)
        polygon_parsed = " ".join([f"{value:.5f}" for value in polygon_relative])
        return f"{int(class_id)} {polygon_parsed}"


def detections_to_yolo_annotations(
    detections: Detections,
    image_shape: Tuple[int, int, int],
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> List[str]:
    annotation = []
    for xyxy, mask, _, class_id, _ in detections:
        if mask is not None:
            polygons = approximate_mask_with_polygons(
                mask=mask,
                min_image_area_percentage=min_image_area_percentage,
                max_image_area_percentage=max_image_area_percentage,
                approximation_percentage=approximation_percentage,
            )
            for polygon in polygons:
                xyxy = polygon_to_xyxy(polygon=polygon)
                next_object = object_to_yolo(
                    xyxy=xyxy,
                    class_id=class_id,
                    image_shape=image_shape,
                    polygon=polygon,
                )
                annotation.append(next_object)
        else:
            next_object = object_to_yolo(
                xyxy=xyxy, class_id=class_id, image_shape=image_shape
            )
            annotation.append(next_object)
    return annotation


def save_yolo_annotations(
    annotations_directory_path: str,
    images: Dict[str, np.ndarray],
    annotations: Dict[str, Detections],
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> None:
    Path(annotations_directory_path).mkdir(parents=True, exist_ok=True)
    for image_path, image in images.items():
        detections = annotations[image_path]
        image_name = Path(image_path).name
        yolo_annotations_name = _image_name_to_annotation_name(image_name=image_name)
        yolo_annotations_path = os.path.join(
            annotations_directory_path, yolo_annotations_name
        )
        lines = detections_to_yolo_annotations(
            detections=detections,
            image_shape=image.shape,
            min_image_area_percentage=min_image_area_percentage,
            max_image_area_percentage=max_image_area_percentage,
            approximation_percentage=approximation_percentage,
        )
        save_text_file(lines=lines, file_path=yolo_annotations_path)


def save_data_yaml(data_yaml_path: str, classes: List[str]) -> None:
    data = {"nc": len(classes), "names": classes}
    Path(data_yaml_path).parent.mkdir(parents=True, exist_ok=True)
    save_yaml_file(data=data, file_path=data_yaml_path)
