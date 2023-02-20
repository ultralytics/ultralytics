# Ultralytics YOLO 🚀, GPL-3.0 license
"""
Ultralytics Results, Boxes and Masks classes for handling inference results

Usage: See https://docs.ultralytics.com/predict/
"""

from copy import deepcopy
from functools import lru_cache

import numpy as np
import torch
import torchvision.transforms.functional as F

from ultralytics.yolo.utils import LOGGER, ops
from ultralytics.yolo.utils.plotting import Annotator, colors


class Results:
    """
        A class for storing and manipulating inference results.

        Args:
            boxes (Boxes, optional): A Boxes object containing the detection bounding boxes.
            masks (Masks, optional): A Masks object containing the detection masks.
            probs (torch.Tensor, optional): A tensor containing the detection class probabilities.
            orig_img (tuple, optional): Original image size.

        Attributes:
            boxes (Boxes, optional): A Boxes object containing the detection bounding boxes.
            masks (Masks, optional): A Masks object containing the detection masks.
            probs (torch.Tensor, optional): A tensor containing the detection class probabilities.
            orig_img (tuple, optional): Original image size.
            data (torch.Tensor): The raw masks tensor

        """

    def __init__(self, orig_img, path, names, boxes=None, masks=None, probs=None) -> None:
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None  # native size boxes
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None  # native size or imgsz masks
        self.probs = probs if probs is not None else None
        self.names = names
        self.path = path
        self._keys = (k for k in ('boxes', 'masks', 'probs') if getattr(self, k) is not None)

    def pandas(self):
        pass
        # TODO masks.pandas + boxes.pandas + cls.pandas

    def __getitem__(self, idx):
        r = Results(orig_img=self.orig_img, path=self.path, names=self.names)
        for k in self._keys:
            setattr(r, k, getattr(self, k)[idx])
        return r

    def update(self, boxes=None, masks=None, probs=None):
        if boxes is not None:
            self.boxes = Boxes(boxes, self.orig_shape)
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if boxes is not None:
            self.probs = probs

    def cpu(self):
        r = Results(orig_img=self.orig_img, path=self.path, names=self.names)
        for k in self._keys:
            setattr(r, k, getattr(self, k).cpu())
        return r

    def numpy(self):
        r = Results(orig_img=self.orig_img, path=self.path, names=self.names)
        for k in self._keys:
            setattr(r, k, getattr(self, k).numpy())
        return r

    def cuda(self):
        r = Results(orig_img=self.orig_img, path=self.path, names=self.names)
        for k in self._keys:
            setattr(r, k, getattr(self, k).cuda())
        return r

    def to(self, *args, **kwargs):
        r = Results(orig_img=self.orig_img, path=self.path, names=self.names)
        for k in self._keys:
            setattr(r, k, getattr(self, k).to(*args, **kwargs))
        return r

    def __len__(self):
        for k in self._keys:
            return len(getattr(self, k))

    def __str__(self):
        return ''.join(getattr(self, k).__str__() for k in self._keys)

    def __repr__(self):
        return ''.join(getattr(self, k).__repr__() for k in self._keys)

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(f"""
            '{name}' object has no attribute '{attr}'. Valid '{name}' object attributes and properties are:

            Attributes:
                boxes (Boxes, optional): A Boxes object containing the detection bounding boxes.
                masks (Masks, optional): A Masks object containing the detection masks.
                probs (torch.Tensor, optional): A tensor containing the detection class probabilities.
                orig_shape (tuple, optional): Original image size.
            """)

    def plot(self, show_conf=True, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        """
        Plots the given result on an input RGB image. Accepts cv2(numpy) or PIL Image

        Args:
            show_conf (bool): Show confidence
            line_width (Float): The line width of boxes. Automatically scaled to img size if not provided
            font_size (Float): The font size of . Automatically scaled to img size if not provided
        """
        img = deepcopy(self.orig_img)
        annotator = Annotator(img, line_width, font_size, font, pil, example)
        boxes = self.boxes
        masks = self.masks
        logits = self.probs
        names = self.names
        if boxes is not None:
            for d in reversed(boxes):
                cls, conf = d.cls.squeeze(), d.conf.squeeze()
                c = int(cls)
                label = (f'{names[c]}' if names else f'{c}') + (f'{conf:.2f}' if show_conf else '')
                annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))

        if masks is not None:
            im = torch.as_tensor(img, dtype=torch.float16, device=masks.data.device).permute(2, 0, 1).flip(0)
            im = F.resize(im.contiguous(), masks.data.shape[1:]) / 255
            annotator.masks(masks.data, colors=[colors(x, True) for x in boxes.cls], im_gpu=im)

        if logits is not None:
            top5i = logits.argsort(0, descending=True)[:5].tolist()  # top 5 indices
            text = f"{', '.join(f'{names[j] if names else j} {logits[j]:.2f}' for j in top5i)}, "
            annotator.text((32, 32), text, txt_color=(255, 255, 255))  # TODO: allow setting colors

        return img


class Boxes:
    """
    A class for storing and manipulating detection boxes.

    Args:
        boxes (torch.Tensor) or (numpy.ndarray): A tensor or numpy array containing the detection boxes,
            with shape (num_boxes, 6). The last two columns should contain confidence and class values.
        orig_shape (tuple): Original image size, in the format (height, width).

    Attributes:
        boxes (torch.Tensor) or (numpy.ndarray): A tensor or numpy array containing the detection boxes,
            with shape (num_boxes, 6).
        orig_shape (torch.Tensor) or (numpy.ndarray): Original image size, in the format (height, width).

    Properties:
        xyxy (torch.Tensor) or (numpy.ndarray): The boxes in xyxy format.
        conf (torch.Tensor) or (numpy.ndarray): The confidence values of the boxes.
        cls (torch.Tensor) or (numpy.ndarray): The class values of the boxes.
        xywh (torch.Tensor) or (numpy.ndarray): The boxes in xywh format.
        xyxyn (torch.Tensor) or (numpy.ndarray): The boxes in xyxy format normalized by original image size.
        xywhn (torch.Tensor) or (numpy.ndarray): The boxes in xywh format normalized by original image size.
        data (torch.Tensor): The raw bboxes tensor
    """

    def __init__(self, boxes, orig_shape) -> None:
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {6, 7}, f'expected `n` in [6, 7], but got {n}'  # xyxy, (track_id), conf, cls
        # TODO
        self.is_track = n == 7
        self.boxes = boxes
        self.orig_shape = torch.as_tensor(orig_shape, device=boxes.device) if isinstance(boxes, torch.Tensor) \
            else np.asarray(orig_shape)

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
    def id(self):
        return self.boxes[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)  # maxsize 1 should suffice
    def xywh(self):
        return ops.xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self):
        return self.xyxy / self.orig_shape[[1, 0, 1, 0]]

    @property
    @lru_cache(maxsize=2)
    def xywhn(self):
        return self.xywh / self.orig_shape[[1, 0, 1, 0]]

    def cpu(self):
        return Boxes(self.boxes.cpu(), self.orig_shape)

    def numpy(self):
        return Boxes(self.boxes.numpy(), self.orig_shape)

    def cuda(self):
        return Boxes(self.boxes.cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        return Boxes(self.boxes.to(*args, **kwargs), self.orig_shape)

    def pandas(self):
        LOGGER.info('results.pandas() method not yet implemented')
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

    @property
    def data(self):
        return self.boxes

    def __len__(self):  # override len(results)
        return len(self.boxes)

    def __str__(self):
        return self.boxes.__str__()

    def __repr__(self):
        return (f'Ultralytics YOLO {self.__class__} masks\n' + f'type: {type(self.boxes)}\n' +
                f'shape: {self.boxes.shape}\n' + f'dtype: {self.boxes.dtype}\n + {self.boxes.__repr__()}')

    def __getitem__(self, idx):
        return Boxes(self.boxes[idx], self.orig_shape)

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(f"""
            '{name}' object has no attribute '{attr}'. Valid '{name}' object attributes and properties are:

            Attributes:
                boxes (torch.Tensor) or (numpy.ndarray): A tensor or numpy array containing the detection boxes,
                    with shape (num_boxes, 6).
                orig_shape (torch.Tensor) or (numpy.ndarray): Original image size, in the format (height, width).

            Properties:
                xyxy (torch.Tensor) or (numpy.ndarray): The boxes in xyxy format.
                conf (torch.Tensor) or (numpy.ndarray): The confidence values of the boxes.
                cls (torch.Tensor) or (numpy.ndarray): The class values of the boxes.
                xywh (torch.Tensor) or (numpy.ndarray): The boxes in xywh format.
                xyxyn (torch.Tensor) or (numpy.ndarray): The boxes in xyxy format normalized by original image size.
                xywhn (torch.Tensor) or (numpy.ndarray): The boxes in xywh format normalized by original image size.
            """)


class Masks:
    """
    A class for storing and manipulating detection masks.

    Args:
        masks (torch.Tensor): A tensor containing the detection masks, with shape (num_masks, height, width).
        orig_shape (tuple): Original image size, in the format (height, width).

    Attributes:
        masks (torch.Tensor): A tensor containing the detection masks, with shape (num_masks, height, width).
        orig_shape (tuple): Original image size, in the format (height, width).

    Properties:
        segments (list): A list of segments which includes x,y,w,h,label,confidence, and mask of each detection masks.
    """

    def __init__(self, masks, orig_shape) -> None:
        self.masks = masks  # N, h, w
        self.orig_shape = orig_shape

    @property
    @lru_cache(maxsize=1)
    def segments(self):
        return [
            ops.scale_segments(self.masks.shape[1:], x, self.orig_shape, normalize=True)
            for x in ops.masks2segments(self.masks)]

    @property
    def shape(self):
        return self.masks.shape

    @property
    def data(self):
        return self.masks

    def cpu(self):
        return Masks(self.masks.cpu(), self.orig_shape)

    def numpy(self):
        return Masks(self.masks.numpy(), self.orig_shape)

    def cuda(self):
        return Masks(self.masks.cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        return Masks(self.masks.to(*args, **kwargs), self.orig_shape)

    def __len__(self):  # override len(results)
        return len(self.masks)

    def __str__(self):
        return self.masks.__str__()

    def __repr__(self):
        return (f'Ultralytics YOLO {self.__class__} masks\n' + f'type: {type(self.masks)}\n' +
                f'shape: {self.masks.shape}\n' + f'dtype: {self.masks.dtype}\n + {self.masks.__repr__()}')

    def __getitem__(self, idx):
        return Masks(self.masks[idx], self.orig_shape)

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(f"""
            '{name}' object has no attribute '{attr}'. Valid '{name}' object attributes and properties are:

            Attributes:
                masks (torch.Tensor): A tensor containing the detection masks, with shape (num_masks, height, width).
                orig_shape (tuple): Original image size, in the format (height, width).

            Properties:
                segments (list): A list of segments which includes x,y,w,h,label,confidence, and mask of each detection masks.
            """)
