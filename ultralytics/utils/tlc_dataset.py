from __future__ import annotations

from ultralytics.data.dataset import YOLODataset
from tlc import Table, Url
from tlc.core.builtins.constants.column_names import IMAGE
import numpy as np
import tlc
from ultralytics.data import build_yolo_dataset

def unpack_box(bbox):
    return [bbox[tlc.LABEL], bbox[tlc.X0], bbox[tlc.Y0], bbox[tlc.X1], bbox[tlc.Y1]]

def tlc_table_row_to_yolo_label(row):
    unpacked = [unpack_box(box) for box in row[tlc.BOUNDING_BOXES][tlc.BOUNDING_BOX_LIST]]
    arr = np.array(unpacked, ndmin=2, dtype=np.float32)
    if len(unpacked) == 0:
        arr = arr.reshape(0, 5)
    return arr

def build_tlc_dataset(table, **kwargs):
    return build_yolo_dataset(table=table, **kwargs)

class TLCDataset(YOLODataset):
    def __init__(self, *args, data=None, task="detect", table: Table, **kwargs):
        assert task == "detect"
        self.table = table
        super().__init__(*args, data=data, task=task, **kwargs)

    def get_img_files(self, _):
        return [Url(sample[IMAGE]).to_absolute().to_str() for sample in self.table]
    
    def get_labels(self, img_path):
        return [tlc_table_row_to_yolo_label(row) for row in self.table]

    