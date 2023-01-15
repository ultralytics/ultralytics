from copy import copy
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch

from ultralytics.yolo.utils import SETTINGS, ops
from ultralytics.yolo.utils.files import increment_path


class Result:

    def __init__(self, boxes=None, masks=None, probs=None, img_shape=None, orig_shape=None) -> None:
        self.boxes = Boxes(boxes, orig_shape) if boxes is not None else None  # native size boxes
        self.masks = Masks(masks, img_shape, orig_shape) if masks is not None else None  # native size or imgsz masks
        self.probs = probs.softmax(0) if probs is not None else None
        self.img_shape = img_shape
        self.orig_shape = orig_shape

    def pandas(self):
        pass
        # TODO masks.pandas + boxes.pandas + cls.pandas

    def __getitem__(self, idx):
        new_result = Result(img_shape=self.img_shape, orig_shape=self.orig_shape)
        for item in ["boxes", "masks", "probs"]:
            if getattr(self, item) is None:
                continue
            setattr(new_result, item, getattr(self, item)[idx])
        return new_result

    def cpu(self):
        new_result = Result(img_shape=self.img_shape, orig_shape=self.orig_shape)
        for item in ["boxes", "masks", "probs"]:
            if getattr(self, item) is None:
                continue
            setattr(new_result, item, getattr(self, item).cpu())
        return new_result

    def numpy(self):
        new_result = Result(img_shape=self.img_shape, orig_shape=self.orig_shape)
        for item in ["boxes", "masks", "probs"]:
            if getattr(self, item) is None:
                continue
            setattr(new_result, item, getattr(self, item).numpy())
        return new_result

    def cuda(self):
        new_result = Result(img_shape=self.img_shape, orig_shape=self.orig_shape)
        for item in ["boxes", "masks", "probs"]:
            if getattr(self, item) is None:
                continue
            setattr(new_result, item, getattr(self, item).cuda())
        return new_result

    def __len__(self):
        for item in ["boxes", "masks", "probs"]:
            if getattr(self, item) is None:
                continue
            return len(getattr(self, item))

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
        repr += f'imgsz: {self.img_shape}\n'
        repr += f'original size: {self.orig_shape}\n'

        return repr


class Boxes:

    def __init__(self, boxes, orig_shape) -> None:
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        assert boxes.shape[-1] == 6  # xyxy, conf, cls
        self.boxes = boxes
        self.orig_shape = orig_shape
        self.gn = torch.tensor(orig_shape, device=boxes.device)[[1, 0, 1, 0]] if orig_shape else []

    @property
    def xyxy(self):
        return self.boxes[:, :4]

    @property
    def conf(self):
        return self.boxes[:, -2]

    @property
    def cls(self):
        return self.boxes[:, -1]

    @property
    @lru_cache(maxsize=2)  # maxsize 1 should suffice
    def xywh(self):
        return [ops.xyxy2xywh(x) for x in self.xyxy]

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self):
        return [x / g for x, g in zip(self.xyxy, self.gn)]

    @property
    @lru_cache(maxsize=2)
    def xywhn(self):
        return [x / g for x, g in zip(self.xywh, self.gn)]

    def cpu(self):
        boxes = self.boxes.cpu()
        return Boxes(boxes, self.orig_shape)

    def numpy(self):
        boxes = self.boxes.numpy()
        return Boxes(boxes, self.orig_shape)

    def cuda(self):
        boxes = self.boxes.cuda()
        return Boxes(boxes, self.orig_shape)

    def to(self, *args, **kwargs):
        boxes = self.boxes.to(*args, **kwargs)
        return Boxes(boxes, self.orig_shape)

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
        # TODO: update __repr__
        return f'Ultralytics YOLO {self.__class__} instance\n' + self.boxes.__repr__()

    def __getitem__(self, idx):
        boxes = self.boxes[idx] 
        return Boxes(boxes, self.orig_shape)


class Masks:

    def __init__(self, masks, im_shape, orig_shape) -> None:
        self.masks = masks  # N, h, w
        self.im_shape = im_shape
        self.orig_shape = orig_shape

    @property
    @lru_cache(maxsize=1)
    def segments(self):
        return [ops.scale_segments(self.im_shape, x, self.orig_shape, normalize=False)
                for x in reversed(ops.masks2segments(self.masks))]

    @property
    def shape(self):
        return self.masks.shape

    def cpu(self):
        masks = self.masks.cpu()
        return Masks(masks, self.im_shape, self.orig_shape)

    def numpy(self):
        masks = self.masks.numpy()
        return Masks(masks, self.im_shape, self.orig_shape)

    def cuda(self):
        masks = self.masks.cuda()
        return Masks(masks, self.im_shape, self.orig_shape)

    def to(self, *args, **kwargs):
        masks = self.masks.to(*args, **kwargs)
        return Masks(masks, self.im_shape, self.orig_shape)

    def __len__(self):  # override len(results)
        return len(self.masks)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'Ultralytics YOLO {self.__class__} instance\n' + self.masks.__repr__()

    def __getitem__(self, idx):
        masks = self.masks[idx]
        return Masks(masks, self.im_shape, self.orig_shape)
