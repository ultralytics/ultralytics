from __future__ import annotations

import numpy as np
import tlc

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import colorstr


def unpack_box(bbox):
    return bbox[tlc.LABEL], [bbox[tlc.X0], bbox[tlc.Y0], bbox[tlc.X1], bbox[tlc.Y1]]


def unpack_boxes(bboxes):
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


def tlc_table_row_to_yolo_label(row):
    classes, bboxes = unpack_boxes(row[tlc.BOUNDING_BOXES][tlc.BOUNDING_BOX_LIST])
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

    def __init__(self,
                 *args,
                 data=None,
                 task="detect",
                 table: tlc.Table = None,
                 use_sampling_weights: bool = False,
                 **kwargs):
        assert task == "detect"
        assert isinstance(table, tlc.Table)
        self.table = table
        if use_sampling_weights and kwargs['rect']:
            raise ValueError("Cannot use sampling weights with rect=True.")
        self._sampling_weights = self.get_sampling_weights() if use_sampling_weights else None
        self._indices = np.arange(len(self.table))
        super().__init__(*args, data=data, task=task, **kwargs)

    def resample_indices(self):
        if self._sampling_weights is not None:
            self._indices[:] = np.random.choice(len(self.table), len(self.table), p=self._sampling_weights)

    def get_img_files(self, _):
        return [tlc.Url(sample[tlc.IMAGE]).to_absolute().to_str() for sample in self.table]

    def get_labels(self):
        return [tlc_table_row_to_yolo_label(row) for row in self.table]

    def get_sampling_weights(self):
        weights = np.array([row[tlc.SAMPLE_WEIGHT] for row in self.table])
        probabilities = weights / weights.sum()
        return probabilities

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
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

    def __getitem__(self, index):
        index = self._indices[index]  # Use potentially resampled index
        return super().__getitem__(index)
