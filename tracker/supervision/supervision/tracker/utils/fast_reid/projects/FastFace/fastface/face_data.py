# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

from PIL import Image
import io
import logging
import numbers

import torch
from torch.utils.data import Dataset

from supervision.tracker.utils.fast_reid.fastreid.data.common import CommDataset

logger = logging.getLogger("fastreid.face_data")

try:
    import mxnet as mx
except ImportError:
    logger.info("Please install mxnet if you want to use .rec file")


class MXFaceDataset(Dataset):
    def __init__(self, path_imgrec, transforms):
        super().__init__()
        self.transforms = transforms

        logger.info(f"loading recordio {path_imgrec}...")
        path_imgidx = path_imgrec[0:-4] + ".idx"
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            # logger.debug(f"header0 label: {header.label}")
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = list(range(1, int(header.label[0])))
            # logger.debug(self.imgidx)
        else:
            self.imgidx = list(self.imgrec.keys)
        logger.info(f"Number of Samples: {len(self.imgidx)}, "
                    f"Number of Classes: {int(self.header0[1] - self.header0[0])}")

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)

        sample = Image.open(io.BytesIO(img))  # RGB
        if self.transforms is not None: sample = self.transforms(sample)
        return {
            "images": sample,
            "targets": label,
            "camids": 0,
        }

    def __len__(self):
        # logger.debug(f"mxface dataset length is {len(self.imgidx)}")
        return len(self.imgidx)

    @property
    def num_classes(self):
        return int(self.header0[1] - self.header0[0])


class TestFaceDataset(CommDataset):
    def __init__(self, img_items, labels):
        self.img_items = img_items
        self.labels = labels

    def __getitem__(self, index):
        img = torch.tensor(self.img_items[index]) * 127.5 + 127.5
        return {
            "images": img,
        }
