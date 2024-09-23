# Ultralytics YOLO 🚀 3LC Integration, AGPL-3.0 license
from __future__ import annotations

from pathlib import Path

import tlc
from tlc.client.torch.metrics.metrics_collectors.bounding_box_metrics_collector import (
    _TLCPredictedBoundingBox,
    _TLCPredictedBoundingBoxes,
)

from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import colorstr
from ultralytics.utils.tlc.detect.dataset import TLCYOLODataset
from ultralytics.utils.tlc.utils import check_tlc_dataset


def tlc_check_det_dataset(
    data: str,
    tables: dict[str, tlc.Table | tlc.Url | Path | str] | None,
    image_column_name: str,
    label_column_name: str,
    project_name: str | None = None,
) -> dict[str, tlc.Table | dict[float, str] | int]:
    return check_tlc_dataset(
        data,
        tables,
        image_column_name,
        label_column_name,
        dataset_checker=check_det_dataset,
        table_creator=get_or_create_det_table,
        table_checker=check_det_table,
        project_name=project_name,
        check_backwards_compatible_table_name=True,
    )


def get_or_create_det_table(
    key: str,
    data_dict: dict[str, object],
    image_column_name: str,
    label_column_name: str,
    project_name: str,
    dataset_name: str,
    table_name: str,
) -> tlc.Table:
    """ Get or create a detection table from a dataset dictionary.

    :param data_dict: Dictionary of dataset information
    :param project_name: Name of the project
    :param dataset_name: Name of the dataset
    :param table_name: Name of the table
    :param image_column_name: Name of the column containing image paths
    :param label_column_name: Name of the column containing labels
    :return: A tlc.Table.from_yolo() table
    """
    return tlc.Table.from_yolo(dataset_yaml_file=data_dict["yaml_file"],
                               split=key,
                               override_split_path=data_dict[key],
                               project_name=project_name,
                               dataset_name=dataset_name,
                               table_name=table_name,
                               if_exists="reuse",
                               add_weight_column=True,
                               description="Created with 3LC YOLOv8 integration")


def build_tlc_yolo_dataset(
    cfg,
    table,
    batch,
    data,
    mode="train",
    rect=False,
    stride=32,
    multi_modal=False,
    exclude_zero=False,
):
    if multi_modal:
        return ValueError("Multi-modal datasets are not supported in the 3LC YOLOv8 integration.")

    return TLCYOLODataset(
        table,
        exclude_zero=exclude_zero,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def check_det_table(table: tlc.Table, _0: str, _1: str) -> None:
    """ Check that a table is compatible with the detection task in the 3LC YOLOv8 integration.
    
    :param split: The split of the table.
    :param table: The table to check.
    :raises: ValueError if the table is not compatible with the detection task.
    """
    if not (is_yolo_table(table) or is_coco_table(table)):
        raise ValueError(
            f'Table {table.url} is not compatible with YOLOv8 object detection, needs to be a YOLO or COCO table.')


def yolo_predicted_bounding_box_schema(categories: dict[int, str]) -> tlc.Schema:
    """ Create a 3LC bounding box schema for YOLOv8

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


def yolo_loss_schemas(training: bool = False) -> dict[str, tlc.Schema]:
    """ Create a 3LC schema for YOLOv8 per-sample loss metrics.

    :param training: Whether metrics are collected during training.
    :returns: The YOLO loss schemas for each of the three components.
    """
    schemas = {}
    schemas['box_loss'] = tlc.Schema(description='Box Loss',
                                     writable=False,
                                     value=tlc.Float32Value(),
                                     display_importance=3004)
    schemas['dfl_loss'] = tlc.Schema(description='Distribution Focal Loss',
                                     writable=False,
                                     value=tlc.Float32Value(),
                                     display_importance=3005)
    schemas['cls_loss'] = tlc.Schema(description='Classification Loss',
                                     writable=False,
                                     value=tlc.Float32Value(),
                                     display_importance=3006)
    if training:
        schemas['loss'] = tlc.Schema(description='Weighted sum of box, DFL, and classification losses used in training',
                                     writable=False,
                                     value=tlc.Float32Value(),
                                     display_importance=3007)
    return schemas


def construct_bbox_struct(
    predicted_annotations: list[dict[str, int | float | dict[str, float]]],
    image_width: int,
    image_height: int,
    inverse_label_mapping: dict[int, int] | None = None,
) -> _TLCPredictedBoundingBoxes:
    """Construct a 3LC bounding box struct from a list of bounding boxes.

    :param predicted_annotations: A list of predicted bounding boxes.
    :param image_width: The width of the image.
    :param image_height: The height of the image.
    :param inverse_label_mapping: A mapping from predicted label to category id.
    """

    bbox_struct = _TLCPredictedBoundingBoxes(
        bb_list=[],
        image_width=image_width,
        image_height=image_height,
    )

    for pred in predicted_annotations:
        bbox, label, score, iou = pred['bbox'], pred['category_id'], pred['score'], pred['iou']
        label_val = inverse_label_mapping[label] if inverse_label_mapping is not None else label
        bbox_struct['bb_list'].append(
            _TLCPredictedBoundingBox(
                label=label_val,
                confidence=score,
                iou=iou,
                x0=bbox[0],
                y0=bbox[1],
                x1=bbox[2],
                y1=bbox[3],
            ))

    return bbox_struct


def is_yolo_table(table: tlc.Table) -> tuple[bool, str]:
    """Check if the table is a YOLO table.

    :param table: The table to check.
    :returns: True if the table is a YOLO table, False otherwise.
    """
    row_schema = table.row_schema.values

    try:
        assert tlc.IMAGE in row_schema
        assert tlc.WIDTH in row_schema
        assert tlc.HEIGHT in row_schema
        assert tlc.BOUNDING_BOXES in row_schema
        assert tlc.BOUNDING_BOX_LIST in row_schema[tlc.BOUNDING_BOXES].values
        assert tlc.SAMPLE_WEIGHT in row_schema
        assert tlc.LABEL in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
        assert tlc.X0 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
        assert tlc.Y0 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
        assert tlc.X1 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
        assert tlc.Y1 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values

        X0 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.X0]
        Y0 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.Y0]
        X1 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.X1]
        Y1 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.Y1]

        assert X0.value.number_role == tlc.NUMBER_ROLE_BB_CENTER_X
        assert Y0.value.number_role == tlc.NUMBER_ROLE_BB_CENTER_Y
        assert X1.value.number_role == tlc.NUMBER_ROLE_BB_SIZE_X
        assert Y1.value.number_role == tlc.NUMBER_ROLE_BB_SIZE_Y

        assert X0.value.unit == tlc.UNIT_RELATIVE
        assert Y0.value.unit == tlc.UNIT_RELATIVE
        assert X1.value.unit == tlc.UNIT_RELATIVE
        assert Y1.value.unit == tlc.UNIT_RELATIVE

    except AssertionError:
        return False

    return True


def is_coco_table(table: tlc.Table) -> bool:
    """Check if the table is a COCO table.

    :param table: The table to check.
    :returns: True if the table is a COCO table, False otherwise.
    """
    row_schema = table.row_schema.values

    try:
        assert tlc.IMAGE in row_schema
        assert tlc.WIDTH in row_schema
        assert tlc.HEIGHT in row_schema
        assert tlc.BOUNDING_BOXES in row_schema
        assert tlc.BOUNDING_BOX_LIST in row_schema[tlc.BOUNDING_BOXES].values
        assert tlc.SAMPLE_WEIGHT in row_schema
        assert tlc.LABEL in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
        assert tlc.X0 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
        assert tlc.Y0 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
        assert tlc.X1 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
        assert tlc.Y1 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values

        X0 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.X0]
        Y0 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.Y0]
        X1 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.X1]
        Y1 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.Y1]

        assert X0.value.number_role == tlc.NUMBER_ROLE_BB_MIN_X
        assert Y0.value.number_role == tlc.NUMBER_ROLE_BB_MIN_Y
        assert X1.value.number_role == tlc.NUMBER_ROLE_BB_SIZE_X
        assert Y1.value.number_role == tlc.NUMBER_ROLE_BB_SIZE_Y

    except AssertionError:
        return False

    return True
