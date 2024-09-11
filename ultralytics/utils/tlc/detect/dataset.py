# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

from multiprocessing.pool import ThreadPool
import numpy as np
import tlc
from itertools import repeat

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import verify_image
from ultralytics.utils import ops
from ultralytics.utils.tlc.engine.dataset import TLCDatasetMixin

from ultralytics.utils import LOGGER, NUM_THREADS, TQDM

from typing import Any


class TLCYOLODataset(TLCDatasetMixin, YOLODataset):

    def __init__(self, table, data=None, task="detect", **kwargs):
        assert task == "detect", f"Unsupported task: {task} for TLCYOLODataset. Only 'detect' is supported."
        self.table = table

        from ultralytics.utils.tlc.detect.utils import is_coco_table, is_yolo_table
        if is_yolo_table(self.table):
            self._table_format = "YOLO"
        elif is_coco_table(self.table):
            self._table_format = "COCO"
        else:
            raise ValueError(f"Unsupported table format for table {table.url}")

        self.example_ids = []

        super().__init__(table, data=data, task=task, **kwargs)

        self._post_init()

    def get_img_files(self, _):
        """Images are read in `get_labels` to avoid two loops, return empty list here."""
        return []

    def get_labels(self):
        self.labels = []

        for example_id, row in enumerate(self.table.table_rows):
            self.example_ids.append(example_id)
            self.im_files.append(tlc.Url(row[tlc.IMAGE]).to_absolute().to_str())
            self.labels.append(tlc_table_row_to_yolo_label(row, self._table_format))

        # Scan images if not already scanned
        if not self._is_scanned():
            self._scan_images()

        self.example_ids = np.array(self.example_ids, dtype=np.int32)

        return self.labels

    def _scan_images(self):
        desc = f"{self.prefix}Scanning images in {self.table.url.to_str()}..."

        nf, nc, msgs, im_files, labels, example_ids = 0, 0, [], [], [], []

        # We use verify_image here, but it expects (image_path, cls) since it is used for classification
        # Labels are not verified because they are verified in tlc.TableFromYolo
        samples_iterator = ((im_file, None) for im_file in self.im_files)

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image, iterable=zip(samples_iterator, repeat(self.prefix)))
            pbar = TQDM(enumerate(results), desc=desc, total=len(self.im_files))
            for i, (sample, nf_f, nc_f, msg) in pbar:
                if nf_f:
                    example_ids.append(self.example_ids[i])
                    im_files.append(self.im_files[i])
                    labels.append(self.labels[i])
                if msg:
                    msgs.append(msg)
                nf += nf_f
                nc += nc_f
                pbar.desc = f"{desc} {nf} images, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))

        if nc == 0:
            self._write_scanned_marker()

        self.example_ids = example_ids
        self.im_files = im_files
        self.labels = labels

    def set_rectangle(self):
        """Save the batch shapes and inidices for the dataset. """
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        self.example_ids = self.example_ids[irect]

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
