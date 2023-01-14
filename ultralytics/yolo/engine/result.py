from copy import copy
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch

from ultralytics.yolo.utils import SETTINGS, ops
from ultralytics.yolo.utils.files import increment_path


class Result:

    def __init__(self, preds, im_shape, orig_shape, args, device) -> None:
        self.im_shape = im_shape
        self.orig_shape = orig_shape
        self.device = device
        self.args = args
        task = self.args.task

        # outputs
        self.preds = None  # raw tensors
        self.boxes = []  # Bbox object. eg-> boxes.convert('xyxy')
        self.masks = []  # Instances object. eg-> segments.clip()
        self.probs = []

        if task == "detect":
            self.boxes = Boxes(preds if len(preds) else torch.tensor([], device=device), orig_shape, device)
            self.preds = self.boxes
        elif task == "segment":
            # preds, masks = preds
            self.boxes = Boxes(preds[0] if len(preds) else torch.tensor([], device=device), orig_shape, device)
            masks = preds[1] if len(preds[1]) else []
            shape = orig_shape if args.retina_masks else im_shape
            self.masks = Masks(masks, shape, orig_shape, device)
            self.preds = [self.boxes, self.masks]
        elif task == "classify":
            self.probs = preds.softmax(0)
            self.preds = self.probs

    def pandas():
        pass
        # TODO masks.pandas + boxes.pandas + cls.pandas

    def __getitem__(self, idx):
        return self.preds[idx]

    def __len__(self):
        return len(self.preds)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        repr = f'Ultralytics YOLO {self.__class__} instance\n'
        if self.boxes:
            repr = repr + self.boxes.__repr__() + '\n'
        if self.masks:
            repr = repr + self.masks.__repr__() + '\n'
        if self.probs:
            repr = repr + self.probs.__repr__()

        return repr


class Boxes:

    def __init__(self, boxes, orig_shape, device) -> None:
        self.boxes = boxes
        self.device = device
        self.orig_shape = orig_shape
        self.gn = torch.tensor(orig_shape)[[1, 0, 1, 0]]

    @property
    @lru_cache(maxsize=2)  # maxsize 1 should suffice
    def xywh(self):
        return [ops.xyxy2xywh(x) for x in self.boxes]

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self):
        return [x / g for x, g in zip(self.boxes, self.gn)]

    @property
    @lru_cache(maxsize=2)
    def xywhn(self):
        return [x / g for x, g in zip(self.xywh, self.gn)]

    def pandas(self):
        '''
        TODO: Placeholder. I don't understant this code. Need to look deeper.
        '''
        pass
        '''
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new
        '''

    @property
    def shape(self):
        return self.boxes.shape

    def __len__(self):  # override len(results)
        return len(self.boxes)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'Ultralytics YOLO {self.__class__} instance\n' + self.boxes.__repr__()

    def __getitem__(self, idx):
        return self.boxes[idx]


class Masks:

    def __init__(self, masks, im_shape, orig_shape, device) -> None:
        self.masks = masks
        self.im_shape = im_shape
        self.orig_shape = orig_shape
        self.device = device

    @property
    @lru_cache(maxsize=1)
    def segments(self):
        # TODO: convert masks to segments
        pass
        '''
        self.segments = [
        ops.scale_segments(shape, x, im0s.shape, normalize=False) for x in reversed(ops.masks2segments(mask))]
        '''

    @property
    def shape(self):
        return self.boxes.shape

    def __len__(self):  # override len(results)
        return len(self.masks)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'Ultralytics YOLO {self.__class__} instance\n' + self.masks.__repr__()

    def __getitem__(self, idx):
        return self.masks[idx]
