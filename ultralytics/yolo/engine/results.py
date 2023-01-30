from functools import lru_cache

import numpy as np
import torch

from ultralytics.yolo.utils import LOGGER, ops


class Results:
    """
        A class for storing and manipulating inference results.

        Args:
            boxes (Boxes, optional): A Boxes object containing the detection bounding boxes.
            masks (Masks, optional): A Masks object containing the detection masks.
            probs (torch.Tensor, optional): A tensor containing the detection class probabilities.
            orig_shape (tuple, optional): Original image size.

        Attributes:
            boxes (Boxes, optional): A Boxes object containing the detection bounding boxes.
            masks (Masks, optional): A Masks object containing the detection masks.
            probs (torch.Tensor, optional): A tensor containing the detection class probabilities.
            orig_shape (tuple, optional): Original image size.
            data (torch.Tensor): The raw masks tensor

        """

    def __init__(self, boxes=None, masks=None, probs=None, orig_shape=None) -> None:
        self.boxes = Boxes(boxes, orig_shape) if boxes is not None else None  # native size boxes
        self.masks = Masks(masks, orig_shape) if masks is not None else None  # native size or imgsz masks
        self.probs = probs.softmax(0) if probs is not None else None
        self.orig_shape = orig_shape
        self.comp = ["boxes", "masks", "probs"]

    def pandas(self):
        pass
        # TODO masks.pandas + boxes.pandas + cls.pandas

    def __getitem__(self, idx):
        r = Results(orig_shape=self.orig_shape)
        for item in self.comp:
            if getattr(self, item) is None:
                continue
            setattr(r, item, getattr(self, item)[idx])
        return r

    def cpu(self):
        r = Results(orig_shape=self.orig_shape)
        for item in self.comp:
            if getattr(self, item) is None:
                continue
            setattr(r, item, getattr(self, item).cpu())
        return r

    def numpy(self):
        r = Results(orig_shape=self.orig_shape)
        for item in self.comp:
            if getattr(self, item) is None:
                continue
            setattr(r, item, getattr(self, item).numpy())
        return r

    def cuda(self):
        r = Results(orig_shape=self.orig_shape)
        for item in self.comp:
            if getattr(self, item) is None:
                continue
            setattr(r, item, getattr(self, item).cuda())
        return r

    def to(self, *args, **kwargs):
        r = Results(orig_shape=self.orig_shape)
        for item in self.comp:
            if getattr(self, item) is None:
                continue
            setattr(r, item, getattr(self, item).to(*args, **kwargs))
        return r

    def __len__(self):
        for item in self.comp:
            if getattr(self, item) is None:
                continue
            return len(getattr(self, item))

    def __str__(self):
        str_out = ""
        for item in self.comp:
            if getattr(self, item) is None:
                continue
            str_out = str_out + getattr(self, item).__str__()
        return str_out

    def __repr__(self):
        str_out = ""
        for item in self.comp:
            if getattr(self, item) is None:
                continue
            str_out = str_out + getattr(self, item).__repr__()
        return str_out

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
        assert boxes.shape[-1] == 6  # xyxy, conf, cls
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
        return (f"Ultralytics YOLO {self.__class__} masks\n" + f"type: {type(self.boxes)}\n" +
                f"shape: {self.boxes.shape}\n" + f"dtype: {self.boxes.dtype}\n + {self.boxes.__repr__()}")

    def __getitem__(self, idx):
        boxes = self.boxes[idx]
        return Boxes(boxes, self.orig_shape)

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
        masks = self.masks.cpu()
        return Masks(masks, self.orig_shape)

    def numpy(self):
        masks = self.masks.numpy()
        return Masks(masks, self.orig_shape)

    def cuda(self):
        masks = self.masks.cuda()
        return Masks(masks, self.orig_shape)

    def to(self, *args, **kwargs):
        masks = self.masks.to(*args, **kwargs)
        return Masks(masks, self.orig_shape)

    def __len__(self):  # override len(results)
        return len(self.masks)

    def __str__(self):
        return self.masks.__str__()

    def __repr__(self):
        return (f"Ultralytics YOLO {self.__class__} masks\n" + f"type: {type(self.masks)}\n" +
                f"shape: {self.masks.shape}\n" + f"dtype: {self.masks.dtype}\n + {self.masks.__repr__()}")

    def __getitem__(self, idx):
        masks = self.masks[idx]
        return Masks(masks, self.im_shape, self.orig_shape)

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


if __name__ == "__main__":
    # test examples
    results = Results(boxes=torch.randn((2, 6)), masks=torch.randn((2, 160, 160)), orig_shape=[640, 640])
    results = results.cuda()
    print("--cuda--pass--")
    results = results.cpu()
    print("--cpu--pass--")
    results = results.to("cuda:0")
    print("--to-cuda--pass--")
    results = results.to("cpu")
    print("--to-cpu--pass--")
    results = results.numpy()
    print("--numpy--pass--")
    # box = Boxes(boxes=torch.randn((2, 6)), orig_shape=[5, 5])
    # box = box.cuda()
    # box = box.cpu()
    # box = box.numpy()
    # for b in box:
    #     print(b)
