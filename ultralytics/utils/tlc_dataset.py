from __future__ import annotations

from ultralytics.data.dataset import YOLODataset
from tlc import Table, Url
from tlc.core.builtins.constants.column_names import IMAGE
import numpy as np
import tlc
from ultralytics.utils import colorstr

def unpack_box(bbox):
    return [bbox[tlc.LABEL], bbox[tlc.X0], bbox[tlc.Y0], bbox[tlc.X1], bbox[tlc.Y1]]

def tlc_table_row_to_yolo_label(row):
    unpacked = [unpack_box(box) for box in row[tlc.BOUNDING_BOXES][tlc.BOUNDING_BOX_LIST]]
    arr = np.array(unpacked, ndmin=2, dtype=np.float32)
    if len(unpacked) == 0:
        arr = arr.reshape(0, 5)
    return arr

def build_tlc_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, table=None):
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
    )

class TLCDataset(YOLODataset):
    def __init__(self, *args, data=None, task="detect", table: Table, **kwargs):
        assert task == "detect"
        self.table = table
        super().__init__(*args, data=data, task=task, **kwargs)

    def get_img_files(self, _):
        return [Url(sample[IMAGE]).to_absolute().to_str() for sample in self.table]
    
    def get_labels(self):
        """
        Users can customize their own format here.

        Note:
            Ensure output is a dictionary with the following keys:
            ```python
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
            ```
        """
        raise NotImplementedError

    