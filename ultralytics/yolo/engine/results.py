# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics Results, Boxes and Masks classes for handling inference results

Usage: See https://docs.ultralytics.com/modes/predict/
"""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.utils import LOGGER, SimpleClass, deprecation_warn, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box


class BaseTensor(SimpleClass):
    """
    Base tensor class with additional methods for easy manipulation and device handling.
    """

    def __init__(self, data, orig_shape) -> None:
        """Initialize BaseTensor with data and original shape.

        Args:
            data (torch.Tensor | np.ndarray): Predictions, such as bboxes, masks and keypoints.
            orig_shape (tuple): Original shape of image.
        """
        assert isinstance(data, (torch.Tensor, np.ndarray))
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self):
        """Return the shape of the data tensor."""
        return self.data.shape

    def cpu(self):
        """Return a copy of the tensor on CPU memory."""
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def numpy(self):
        """Return a copy of the tensor as a numpy array."""
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def cuda(self):
        """Return a copy of the tensor on GPU memory."""
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        """Return a copy of the tensor with the specified device and dtype."""
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def __len__(self):  # override len(results)
        """Return the length of the data tensor."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a BaseTensor with the specified index of the data tensor."""
        return self.__class__(self.data[idx], self.orig_shape)


class Results(SimpleClass):
    """
    A class for storing and manipulating inference results.

    Args:
        orig_img (numpy.ndarray): The original image as a numpy array.
        path (str): The path to the image file.
        names (dict): A dictionary of class names.
        boxes (torch.tensor, optional): A 2D tensor of bounding box coordinates for each detection.
        masks (torch.tensor, optional): A 3D tensor of detection masks, where each mask is a binary image.
        probs (torch.tensor, optional): A 1D tensor of probabilities of each class for classification task.
        keypoints (List[List[float]], optional): A list of detected keypoints for each object.


    Attributes:
        orig_img (numpy.ndarray): The original image as a numpy array.
        orig_shape (tuple): The original image shape in (height, width) format.
        boxes (Boxes, optional): A Boxes object containing the detection bounding boxes.
        masks (Masks, optional): A Masks object containing the detection masks.
        probs (Probs, optional): A Probs object containing probabilities of each class for classification task.
        names (dict): A dictionary of class names.
        path (str): The path to the image file.
        keypoints (Keypoints, optional): A Keypoints object containing detected keypoints for each object.
        speed (dict): A dictionary of preprocess, inference and postprocess speeds in milliseconds per image.
        _keys (tuple): A tuple of attribute names for non-empty attributes.
    """

    def __init__(self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None) -> None:
        """Initialize the Results class."""
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None  # native size boxes
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None  # native size or imgsz masks
        self.probs = Probs(probs) if probs is not None else None
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self.speed = {'preprocess': None, 'inference': None, 'postprocess': None}  # milliseconds per image
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = ('boxes', 'masks', 'probs', 'keypoints')

    def __getitem__(self, idx):
        """Return a Results object for the specified index."""
        r = self.new()
        for k in self.keys:
            setattr(r, k, getattr(self, k)[idx])
        return r

    def update(self, boxes=None, masks=None, probs=None):
        """Update the boxes, masks, and probs attributes of the Results object."""
        if boxes is not None:
            self.boxes = Boxes(boxes, self.orig_shape)
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if probs is not None:
            self.probs = probs

    def cpu(self):
        """Return a copy of the Results object with all tensors on CPU memory."""
        r = self.new()
        for k in self.keys:
            setattr(r, k, getattr(self, k).cpu())
        return r

    def numpy(self):
        """Return a copy of the Results object with all tensors as numpy arrays."""
        r = self.new()
        for k in self.keys:
            setattr(r, k, getattr(self, k).numpy())
        return r

    def cuda(self):
        """Return a copy of the Results object with all tensors on GPU memory."""
        r = self.new()
        for k in self.keys:
            setattr(r, k, getattr(self, k).cuda())
        return r

    def to(self, *args, **kwargs):
        """Return a copy of the Results object with tensors on the specified device and dtype."""
        r = self.new()
        for k in self.keys:
            setattr(r, k, getattr(self, k).to(*args, **kwargs))
        return r

    def __len__(self):
        """Return the number of detections in the Results object."""
        for k in self.keys:
            return len(getattr(self, k))

    def new(self):
        """Return a new Results object with the same image, path, and names."""
        return Results(orig_img=self.orig_img, path=self.path, names=self.names)

    @property
    def keys(self):
        """Return a list of non-empty attribute names."""
        return [k for k in self._keys if getattr(self, k) is not None]

    def plot(
            self,
            conf=True,
            line_width=None,
            font_size=None,
            font='Arial.ttf',
            pil=False,
            img=None,
            img_gpu=None,
            kpt_line=True,
            labels=True,
            boxes=True,
            masks=True,
            probs=True,
            **kwargs  # deprecated args TODO: remove support in 8.2
    ):
        """
        Plots the detection results on an input RGB image. Accepts a numpy array (cv2) or a PIL Image.

        Args:
            conf (bool): Whether to plot the detection confidence score.
            line_width (float, optional): The line width of the bounding boxes. If None, it is scaled to the image size.
            font_size (float, optional): The font size of the text. If None, it is scaled to the image size.
            font (str): The font to use for the text.
            pil (bool): Whether to return the image as a PIL Image.
            img (numpy.ndarray): Plot to another image. if not, plot to original image.
            img_gpu (torch.Tensor): Normalized image in gpu with shape (1, 3, 640, 640), for faster mask plotting.
            kpt_line (bool): Whether to draw lines connecting keypoints.
            labels (bool): Whether to plot the label of bounding boxes.
            boxes (bool): Whether to plot the bounding boxes.
            masks (bool): Whether to plot the masks.
            probs (bool): Whether to plot classification probability

        Returns:
            (numpy.ndarray): A numpy array of the annotated image.
        """
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = np.ascontiguousarray(self.orig_img[0].permute(1, 2, 0).cpu().detach().numpy()) * 255

        # Deprecation warn TODO: remove in 8.2
        if 'show_conf' in kwargs:
            deprecation_warn('show_conf', 'conf')
            conf = kwargs['show_conf']
            assert type(conf) == bool, '`show_conf` should be of boolean type, i.e, show_conf=True/False'

        if 'line_thickness' in kwargs:
            deprecation_warn('line_thickness', 'line_width')
            line_width = kwargs['line_thickness']
            assert type(line_width) == int, '`line_width` should be of int type, i.e, line_width=3'

        names = self.names
        annotator = Annotator(deepcopy(self.orig_img if img is None else img),
                              line_width,
                              font_size,
                              font,
                              pil,
                              example=names)
        pred_boxes, show_boxes = self.boxes, boxes
        pred_masks, show_masks = self.masks, masks
        pred_probs, show_probs = self.probs, probs
        keypoints = self.keypoints
        if pred_masks and show_masks:
            if img_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                img_gpu = torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device).permute(
                    2, 0, 1).flip(0).contiguous() / 255
            idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
            annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=img_gpu)

        if pred_boxes and show_boxes:
            for d in reversed(pred_boxes):
                c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
                name = ('' if id is None else f'id:{id} ') + names[c]
                label = (f'{name} {conf:.2f}' if conf else name) if labels else None
                annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))

        if pred_probs is not None and show_probs:
            text = f"{', '.join(f'{names[j] if names else j} {pred_probs.data[j]:.2f}' for j in pred_probs.top5)}, "
            annotator.text((32, 32), text, txt_color=(255, 255, 255))  # TODO: allow setting colors

        if keypoints is not None:
            for k in reversed(keypoints.data):
                annotator.kpts(k, self.orig_shape, kpt_line=kpt_line)

        return annotator.result()

    def verbose(self):
        """
        Return log string for each task.
        """
        log_string = ''
        probs = self.probs
        boxes = self.boxes
        if len(self) == 0:
            return log_string if probs is not None else f'{log_string}(no detections), '
        if probs is not None:
            log_string += f"{', '.join(f'{self.names[j]} {probs.data[j]:.2f}' for j in probs.top5)}, "
        if boxes:
            for c in boxes.cls.unique():
                n = (boxes.cls == c).sum()  # detections per class
                log_string += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
        return log_string

    def save_txt(self, txt_file, save_conf=False):
        """
        Save predictions into txt file.

        Args:
            txt_file (str): txt file path.
            save_conf (bool): save confidence score or not.
        """
        boxes = self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        texts = []
        if probs is not None:
            # Classify
            [texts.append(f'{probs.data[j]:.2f} {self.names[j]}') for j in probs.top5]
        elif boxes:
            # Detect/segment/pose
            for j, d in enumerate(boxes):
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                line = (c, *d.xywhn.view(-1))
                if masks:
                    seg = masks[j].xyn[0].copy().reshape(-1)  # reversed mask.xyn, (n,2) to (n*2)
                    line = (c, *seg)
                if kpts is not None:
                    kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                    line += (*kpt.reshape(-1).tolist(), )
                line += (conf, ) * save_conf + (() if id is None else (id, ))
                texts.append(('%g ' * len(line)).rstrip() % line)

        if texts:
            with open(txt_file, 'a') as f:
                f.writelines(text + '\n' for text in texts)

    def save_crop(self, save_dir, file_name=Path('im.jpg')):
        """
        Save cropped predictions to `save_dir/cls/file_name.jpg`.

        Args:
            save_dir (str | pathlib.Path): Save path.
            file_name (str | pathlib.Path): File name.
        """
        if self.probs is not None:
            LOGGER.warning('WARNING ⚠️ Classify task do not support `save_crop`.')
            return
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        if isinstance(file_name, str):
            file_name = Path(file_name)
        for d in self.boxes:
            save_one_box(d.xyxy,
                         self.orig_img.copy(),
                         file=save_dir / self.names[int(d.cls)] / f'{file_name.stem}.jpg',
                         BGR=True)

    def pandas(self):
        """Convert the object to a pandas DataFrame (not yet implemented)."""
        LOGGER.warning("WARNING ⚠️ 'Results.pandas' method is not yet implemented.")

    def tojson(self, normalize=False):
        """Convert the object to JSON format."""
        if self.probs is not None:
            LOGGER.warning('Warning: Classify task do not support `tojson` yet.')
            return

        import json

        # Create list of detection dictionaries
        results = []
        data = self.boxes.data.cpu().tolist()
        h, w = self.orig_shape if normalize else (1, 1)
        for i, row in enumerate(data):
            box = {'x1': row[0] / w, 'y1': row[1] / h, 'x2': row[2] / w, 'y2': row[3] / h}
            conf = row[4]
            id = int(row[5])
            name = self.names[id]
            result = {'name': name, 'class': id, 'confidence': conf, 'box': box}
            if self.masks:
                x, y = self.masks.xy[i][:, 0], self.masks.xy[i][:, 1]  # numpy array
                result['segments'] = {'x': (x / w).tolist(), 'y': (y / h).tolist()}
            if self.keypoints is not None:
                x, y, visible = self.keypoints[i].data[0].cpu().unbind(dim=1)  # torch Tensor
                result['keypoints'] = {'x': (x / w).tolist(), 'y': (y / h).tolist(), 'visible': visible.tolist()}
            results.append(result)

        # Convert detections to JSON
        return json.dumps(results, indent=2)


class Boxes(BaseTensor):
    """
    A class for storing and manipulating detection boxes.

    Args:
        boxes (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the detection boxes,
            with shape (num_boxes, 6). The last two columns should contain confidence and class values.
        orig_shape (tuple): Original image size, in the format (height, width).

    Attributes:
        boxes (torch.Tensor | numpy.ndarray): The detection boxes with shape (num_boxes, 6).
        orig_shape (torch.Tensor | numpy.ndarray): Original image size, in the format (height, width).
        is_track (bool): True if the boxes also include track IDs, False otherwise.

    Properties:
        xyxy (torch.Tensor | numpy.ndarray): The boxes in xyxy format.
        conf (torch.Tensor | numpy.ndarray): The confidence values of the boxes.
        cls (torch.Tensor | numpy.ndarray): The class values of the boxes.
        id (torch.Tensor | numpy.ndarray): The track IDs of the boxes (if available).
        xywh (torch.Tensor | numpy.ndarray): The boxes in xywh format.
        xyxyn (torch.Tensor | numpy.ndarray): The boxes in xyxy format normalized by original image size.
        xywhn (torch.Tensor | numpy.ndarray): The boxes in xywh format normalized by original image size.
        data (torch.Tensor): The raw bboxes tensor

    Methods:
        cpu(): Move the object to CPU memory.
        numpy(): Convert the object to a numpy array.
        cuda(): Move the object to CUDA memory.
        to(*args, **kwargs): Move the object to the specified device.
        pandas(): Convert the object to a pandas DataFrame (not yet implemented).
    """

    def __init__(self, boxes, orig_shape) -> None:
        """Initialize the Boxes class."""
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in (6, 7), f'expected `n` in [6, 7], but got {n}'  # xyxy, (track_id), conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape

    @property
    def xyxy(self):
        """Return the boxes in xyxy format."""
        return self.data[:, :4]

    @property
    def conf(self):
        """Return the confidence values of the boxes."""
        return self.data[:, -2]

    @property
    def cls(self):
        """Return the class values of the boxes."""
        return self.data[:, -1]

    @property
    def id(self):
        """Return the track IDs of the boxes (if available)."""
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)  # maxsize 1 should suffice
    def xywh(self):
        """Return the boxes in xywh format."""
        return ops.xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self):
        """Return the boxes in xyxy format normalized by original image size."""
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self):
        """Return the boxes in xywh format normalized by original image size."""
        xywh = ops.xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh

    @property
    def boxes(self):
        """Return the raw bboxes tensor (deprecated)."""
        LOGGER.warning("WARNING ⚠️ 'Boxes.boxes' is deprecated. Use 'Boxes.data' instead.")
        return self.data


class Masks(BaseTensor):
    """
    A class for storing and manipulating detection masks.

    Args:
        masks (torch.Tensor | np.ndarray): A tensor containing the detection masks, with shape (num_masks, height, width).
        orig_shape (tuple): Original image size, in the format (height, width).

    Attributes:
        masks (torch.Tensor | np.ndarray): A tensor containing the detection masks, with shape (num_masks, height, width).
        orig_shape (tuple): Original image size, in the format (height, width).

    Properties:
        xy (list): A list of segments (pixels) which includes x, y segments of each detection.
        xyn (list): A list of segments (normalized) which includes x, y segments of each detection.

    Methods:
        cpu(): Returns a copy of the masks tensor on CPU memory.
        numpy(): Returns a copy of the masks tensor as a numpy array.
        cuda(): Returns a copy of the masks tensor on GPU memory.
        to(): Returns a copy of the masks tensor with the specified device and dtype.
    """

    def __init__(self, masks, orig_shape) -> None:
        """Initialize the Masks class."""
        if masks.ndim == 2:
            masks = masks[None, :]
        super().__init__(masks, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def segments(self):
        """Return segments (deprecated; normalized)."""
        LOGGER.warning("WARNING ⚠️ 'Masks.segments' is deprecated. Use 'Masks.xyn' for segments (normalized) and "
                       "'Masks.xy' for segments (pixels) instead.")
        return self.xyn

    @property
    @lru_cache(maxsize=1)
    def xyn(self):
        """Return segments (normalized)."""
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=True)
            for x in ops.masks2segments(self.data)]

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        """Return segments (pixels)."""
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=False)
            for x in ops.masks2segments(self.data)]

    @property
    def masks(self):
        """Return the raw masks tensor (deprecated)."""
        LOGGER.warning("WARNING ⚠️ 'Masks.masks' is deprecated. Use 'Masks.data' instead.")
        return self.data

    def pandas(self):
        """Convert the object to a pandas DataFrame (not yet implemented)."""
        LOGGER.warning("WARNING ⚠️ 'Masks.pandas' method is not yet implemented.")


class Keypoints(BaseTensor):
    """
    A class for storing and manipulating detection keypoints.

    Args:
        keypoints (torch.Tensor | np.ndarray): A tensor containing the detection keypoints, with shape (num_dets, num_kpts, 2/3).
        orig_shape (tuple): Original image size, in the format (height, width).

    Attributes:
        keypoints (torch.Tensor | np.ndarray): A tensor containing the detection keypoints, with shape (num_dets, num_kpts, 2/3).
        orig_shape (tuple): Original image size, in the format (height, width).

    Properties:
        xy (list): A list of keypoints (pixels) which includes x, y keypoints of each detection.
        xyn (list): A list of keypoints (normalized) which includes x, y keypoints of each detection.

    Methods:
        cpu(): Returns a copy of the keypoints tensor on CPU memory.
        numpy(): Returns a copy of the keypoints tensor as a numpy array.
        cuda(): Returns a copy of the keypoints tensor on GPU memory.
        to(): Returns a copy of the keypoints tensor with the specified device and dtype.
    """

    def __init__(self, keypoints, orig_shape) -> None:
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
        super().__init__(keypoints, orig_shape)
        self.has_visible = self.data.shape[-1] == 3

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        return self.data[..., :2]

    @property
    @lru_cache(maxsize=1)
    def xyn(self):
        xy = self.xy.clone() if isinstance(self.xy, torch.Tensor) else np.copy(self.xy)
        xy[..., 0] /= self.orig_shape[1]
        xy[..., 1] /= self.orig_shape[0]
        return xy

    @property
    @lru_cache(maxsize=1)
    def conf(self):
        return self.data[..., 2] if self.has_visible else None


class Probs(BaseTensor):
    """
    A class for storing and manipulating classify predictions.

    Args:
        probs (torch.Tensor | np.ndarray): A tensor containing the detection keypoints, with shape (num_class, ).

    Attributes:
        probs (torch.Tensor | np.ndarray): A tensor containing the detection keypoints, with shape (num_class).

    Properties:
        top5 (list[int]): Top 1 indice.
        top1 (int): Top 5 indices.

    Methods:
        cpu(): Returns a copy of the probs tensor on CPU memory.
        numpy(): Returns a copy of the probs tensor as a numpy array.
        cuda(): Returns a copy of the probs tensor on GPU memory.
        to(): Returns a copy of the probs tensor with the specified device and dtype.
    """

    def __init__(self, probs, orig_shape=None) -> None:
        super().__init__(probs, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def top5(self):
        """Return the indices of top 5."""
        return (-self.data).argsort(0)[:5].tolist()  # this way works with both torch and numpy.

    @property
    @lru_cache(maxsize=1)
    def top1(self):
        """Return the indices of top 1."""
        return int(self.data.argmax())

    @property
    @lru_cache(maxsize=1)
    def top5conf(self):
        """Return the confidences of top 5."""
        return self.data[self.top5]

    @property
    @lru_cache(maxsize=1)
    def top1conf(self):
        """Return the confidences of top 1."""
        return self.data[self.top1]
