# Ultralytics YOLO 🚀 3LC Integration, AGPL-3.0 license
from __future__ import annotations

from typing import Any

import numpy as np
import tlc

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import colorstr, ops
from ultralytics.utils.tlc.detect.utils import infer_table_format


def unpack_box(bbox: dict[str, int | float]) -> tuple[int | float]:
    return bbox[tlc.LABEL], [bbox[tlc.X0], bbox[tlc.Y0], bbox[tlc.X1], bbox[tlc.Y1]]


def unpack_boxes(bboxes: list[dict[str, int | float]]) -> tuple[np.ndarray, np.ndarray]:
    classes_list, boxes_list = [], []
    for bbox in bboxes:
        _class, box = unpack_box(bbox)
        classes_list.append(_class)
        boxes_list.append(box)

    # Convert to np array
    boxes = np.array(boxes_list, ndmin=2, dtype=np.float32)
    if len(boxes_list) == 0:
        boxes = boxes.reshape(0, 4)

    classes = np.array(classes_list, dtype=np.float32).reshape((-1, 1))
    assert classes.shape == (boxes.shape[0], 1)
    return classes, boxes


def tlc_table_row_to_yolo_label(row, table_format: str) -> dict[str, Any]:
    classes, bboxes = unpack_boxes(row[tlc.BOUNDING_BOXES][tlc.BOUNDING_BOX_LIST])
    
    if table_format == "COCO":
        # Convert from ltwh absolute to xywh relative
        bboxes_xyxy = ops.ltwh2xyxy(bboxes)
        bboxes = ops.xyxy2xywhn(bboxes_xyxy, w=row['width'], h=row['height'])

    return dict(
        im_file=tlc.Url(row[tlc.IMAGE]).to_absolute().to_str(),
        shape=(row['height'], row['width']),  # format: (height, width)
        cls=classes,
        bboxes=bboxes,
        segments=[],
        keypoints=None,
        normalized=True,
        bbox_format="xywh",
    )


def build_tlc_dataset(cfg,
                      img_path,
                      batch,
                      data,
                      mode="train",
                      rect=False,
                      stride=32,
                      table=None,
                      use_sampling_weights=False):
    """Build TLC Dataset."""
    assert table is not None
    return TLCDataset(
        img_path=img_path,
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
        table=table,
        use_sampling_weights=mode == "train" and use_sampling_weights,
    )


class TLCDataset(YOLODataset):
    """ A 3LC dataset for YOLO training and validation. Populates the dataset with samples from a 3LC table
    and supports sampling weights.

    """

    def __init__(self,
                 *args,
                 data=None,
                 task="detect",
                 table: tlc.Table = None,
                 use_sampling_weights: bool = False,
                 **kwargs) -> None:
        assert task == "detect", f"Unsupported task: {task} for TLCDataset. Only 'detect' is supported."
        assert isinstance(table, tlc.Table), f"Expected table to be a tlc.Table, got {type(table)} instead."
        self.table = table
        self._table_format = infer_table_format(table)
        if use_sampling_weights and kwargs['rect']:
            raise ValueError("Cannot use sampling weights with rect=True.")
        self._sampling_weights = self.get_sampling_weights() if use_sampling_weights else None
        self._indices = np.arange(len(self.table))
        super().__init__(*args, data=data, task=task, **kwargs)

    def resample_indices(self) -> None:
        """Resample the indices inplace using the sampling weights."""
        if self._sampling_weights is not None:
            self._indices[:] = np.random.choice(len(self.table), len(self.table), p=self._sampling_weights)

    def get_img_files(self, _: Any) -> list[str]:
        """Get the image files, converting possibly aliased 3LC Urls to absolute paths.

        :return: A list of absolute paths to the images.
        """
        return [tlc.Url(sample[tlc.IMAGE]).to_absolute().to_str() for sample in self.table.table_rows]

    def get_labels(self) -> list[dict[str, Any]]:
        """Get the labels for the dataset.

        :return: A list of YOLOv8 labels.
        """
        return [tlc_table_row_to_yolo_label(row, self._table_format) for row in self.table.table_rows]

    def get_sampling_weights(self) -> np.ndarray:
        weights = np.array([row[tlc.SAMPLE_WEIGHT] for row in self.table.table_rows])
        probabilities = weights / weights.sum()
        return probabilities

    def set_rectangle(self) -> None:
        """Save the batch shapes and inidices for the dataset. """
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]
        self.irect = irect.copy()

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def __getitem__(self, index: int):
        index = self._indices[index]  # Use potentially resampled index
        return super().__getitem__(index)
