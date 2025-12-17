# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import abc
from itertools import repeat
from numbers import Number
from typing import List

import numpy as np
import cv2

from .ops import ltwh2xywh, ltwh2xyxy, resample_segments, xywh2ltwh, xywh2xyxy, xyxy2ltwh, xyxy2xywh, xyxyxyxy2xywhr


def _ntuple(n):
    """From PyTorch internals."""

    def parse(x):
        """Parse input to return n-tuple by repeating singleton values n times."""
        return x if isinstance(x, abc.Iterable) else tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)
to_4tuple = _ntuple(4)

# `xyxy` means left top and right bottom
# `xywh` means center x, center y and width, height(YOLO format)
# `ltwh` means left top and width, height(COCO format)
_formats = ["xyxy", "xywh", "ltwh"]

__all__ = ("Bboxes", "Instances")  # tuple or list


class Bboxes:
    """
    A class for handling bounding boxes.

    The class supports various bounding box formats like 'xyxy', 'xywh', and 'ltwh'.
    Bounding box data should be provided in numpy arrays.

    Attributes:
        bboxes (np.ndarray): The bounding boxes stored in a 2D numpy array with shape (N, 4).
        format (str): The format of the bounding boxes ('xyxy', 'xywh', or 'ltwh').

    Note:
        This class does not handle normalization or denormalization of bounding boxes.
    """

    def __init__(self, bboxes, format="xyxy") -> None:
        """
        Initialize the Bboxes class with bounding box data in a specified format.

        Args:
            bboxes (np.ndarray): Array of bounding boxes with shape (N, 4) or (4,).
            format (str): Format of the bounding boxes, one of 'xyxy', 'xywh', or 'ltwh'.
        """
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4
        self.bboxes = bboxes
        self.format = format
        # self.normalized = normalized

    def convert(self, format):
        """
        Convert bounding box format from one type to another.

        Args:
            format (str): Target format for conversion, one of 'xyxy', 'xywh', or 'ltwh'.
        """
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        if self.format == format:
            return
        elif self.format == "xyxy":
            func = xyxy2xywh if format == "xywh" else xyxy2ltwh
        elif self.format == "xywh":
            func = xywh2xyxy if format == "xyxy" else xywh2ltwh
        else:
            func = ltwh2xyxy if format == "xyxy" else ltwh2xywh
        self.bboxes = func(self.bboxes)
        self.format = format

    def areas(self):
        """Return box areas."""
        return (
            (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])  # format xyxy
            if self.format == "xyxy"
            else self.bboxes[:, 3] * self.bboxes[:, 2]  # format xywh or ltwh
        )

    # def denormalize(self, w, h):
    #    if not self.normalized:
    #         return
    #     assert (self.bboxes <= 1.0).all()
    #     self.bboxes[:, 0::2] *= w
    #     self.bboxes[:, 1::2] *= h
    #     self.normalized = False
    #
    # def normalize(self, w, h):
    #     if self.normalized:
    #         return
    #     assert (self.bboxes > 1.0).any()
    #     self.bboxes[:, 0::2] /= w
    #     self.bboxes[:, 1::2] /= h
    #     self.normalized = True

    def mul(self, scale):
        """
        Multiply bounding box coordinates by scale factor(s).

        Args:
            scale (int | tuple | list): Scale factor(s) for four coordinates.
                If int, the same scale is applied to all coordinates.
        """
        if isinstance(scale, Number):
            scale = to_4tuple(scale)
        assert isinstance(scale, (tuple, list))
        assert len(scale) == 4
        self.bboxes[:, 0] *= scale[0]
        self.bboxes[:, 1] *= scale[1]
        self.bboxes[:, 2] *= scale[2]
        self.bboxes[:, 3] *= scale[3]

    def add(self, offset):
        """
        Add offset to bounding box coordinates.

        Args:
            offset (int | tuple | list): Offset(s) for four coordinates.
                If int, the same offset is applied to all coordinates.
        """
        if isinstance(offset, Number):
            offset = to_4tuple(offset)
        assert isinstance(offset, (tuple, list))
        assert len(offset) == 4
        self.bboxes[:, 0] += offset[0]
        self.bboxes[:, 1] += offset[1]
        self.bboxes[:, 2] += offset[2]
        self.bboxes[:, 3] += offset[3]

    def __len__(self):
        """Return the number of boxes."""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, boxes_list: List["Bboxes"], axis=0) -> "Bboxes":
        """
        Concatenate a list of Bboxes objects into a single Bboxes object.

        Args:
            boxes_list (List[Bboxes]): A list of Bboxes objects to concatenate.
            axis (int, optional): The axis along which to concatenate the bounding boxes.

        Returns:
            (Bboxes): A new Bboxes object containing the concatenated bounding boxes.

        Note:
            The input should be a list or tuple of Bboxes objects.
        """
        assert isinstance(boxes_list, (list, tuple))
        if not boxes_list:
            return cls(np.empty(0))
        assert all(isinstance(box, Bboxes) for box in boxes_list)

        if len(boxes_list) == 1:
            return boxes_list[0]
        return cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis))

    def __getitem__(self, index) -> "Bboxes":
        """
        Retrieve a specific bounding box or a set of bounding boxes using indexing.

        Args:
            index (int | slice | np.ndarray): The index, slice, or boolean array to select
                                              the desired bounding boxes.

        Returns:
            (Bboxes): A new Bboxes object containing the selected bounding boxes.

        Raises:
            AssertionError: If the indexed bounding boxes do not form a 2-dimensional matrix.

        Note:
            When using boolean indexing, make sure to provide a boolean array with the same
            length as the number of bounding boxes.
        """
        if isinstance(index, int):
            return Bboxes(self.bboxes[index].reshape(1, -1))
        b = self.bboxes[index]
        assert b.ndim == 2, f"Indexing on Bboxes with {index} failed to return a matrix!"
        return Bboxes(b)


class Instances:
    """
    Container for bounding boxes, segments, and keypoints of detected objects in an image.

    Attributes:
        _bboxes (Bboxes): Internal object for handling bounding box operations.
        keypoints (np.ndarray): Keypoints with shape (N, 17, 3) in format (x, y, visible).
        normalized (bool): Flag indicating whether the bounding box coordinates are normalized.
        segments (np.ndarray): Segments array with shape (N, M, 2) after resampling.

    Methods:
        convert_bbox: Convert bounding box format.
        scale: Scale coordinates by given factors.
        denormalize: Convert normalized coordinates to absolute coordinates.
        normalize: Convert absolute coordinates to normalized coordinates.
        add_padding: Add padding to coordinates.
        flipud: Flip coordinates vertically.
        fliplr: Flip coordinates horizontally.
        clip: Clip coordinates to stay within image boundaries.
        remove_zero_area_boxes: Remove boxes with zero area.
        update: Update instance variables.
        concatenate: Concatenate multiple Instances objects.

    Examples:
        >>> instances = Instances(
        ...     bboxes=np.array([[10, 10, 30, 30], [20, 20, 40, 40]]),
        ...     segments=[np.array([[5, 5], [10, 10]]), np.array([[15, 15], [20, 20]])],
        ...     keypoints=np.array([[[5, 5, 1], [10, 10, 1]], [[15, 15, 1], [20, 20, 1]]]),
        ... )
    """

    def __init__(self, bboxes, segments=None, keypoints=None, bbox_format="xywh", normalized=True, obbData=None) -> None:
        """
        Initialize the object with bounding boxes, segments, and keypoints.

        Args:
            bboxes (np.ndarray): Bounding boxes, shape (N, 4).
            segments (List | np.ndarray, optional): Segmentation masks.
            keypoints (np.ndarray, optional): Keypoints, shape (N, 17, 3) in format (x, y, visible).
            bbox_format (str, optional): Format of bboxes.
            normalized (bool, optional): Whether the coordinates are normalized.
        """
        self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format)
        self.keypoints = keypoints
        self.normalized = normalized
        self.segments = segments
        self.obbData = obbData

    def initOrientedBoundingBoxes(self, w, h):
        if not len(self.segments):
            self.obbData = np.zeros((0, 5), dtype=np.float32)
            return
        # get the segment data in denormalized data
        denormSegments = self.segments * np.array([w, h], dtype=np.float32).reshape((1, 1, 2))
        self.obbData = xyxyxyxy2xywhr(denormSegments)
        # Convert back to degrees
        self.obbData[:, 4] = self.obbData[:, 4]*180.0/np.pi

    def convert_bbox(self, format):
        """
        Convert bounding box format.

        Args:
            format (str): Target format for conversion, one of 'xyxy', 'xywh', or 'ltwh'.
        """
        self._bboxes.convert(format=format)

    @property
    def bbox_areas(self):
        """Calculate the area of bounding boxes."""
        return self._bboxes.areas()

    def scale(self, scale_w, scale_h, bbox_only=False):
        """
        Scale coordinates by given factors.

        Args:
            scale_w (float): Scale factor for width.
            scale_h (float): Scale factor for height.
            bbox_only (bool, optional): Whether to scale only bounding boxes.
        """
        self._bboxes.mul(scale=(scale_w, scale_h, scale_w, scale_h))
        if bbox_only:
            return
        self.segments[..., 0] *= scale_w
        self.segments[..., 1] *= scale_h
        if self.keypoints is not None:
            self.keypoints[..., 0] *= scale_w
            self.keypoints[..., 1] *= scale_h

    def denormalize(self, w, h):
        """
        Convert normalized coordinates to absolute coordinates.

        Args:
            w (int): Image width.
            h (int): Image height.
        """
        if not self.normalized:
            return
        self._bboxes.mul(scale=(w, h, w, h))
        self.segments[..., 0] *= w
        self.segments[..., 1] *= h
        if self.keypoints is not None:
            self.keypoints[..., 0] *= w
            self.keypoints[..., 1] *= h
        self.normalized = False

    def normalize(self, w, h):
        """
        Convert absolute coordinates to normalized coordinates.

        Args:
            w (int): Image width.
            h (int): Image height.
        """
        if self.normalized:
            return
        self._bboxes.mul(scale=(1 / w, 1 / h, 1 / w, 1 / h))
        self.segments[..., 0] /= w
        self.segments[..., 1] /= h
        if self.keypoints is not None:
            self.keypoints[..., 0] /= w
            self.keypoints[..., 1] /= h
        self.normalized = True

    def add_padding(self, padw, padh):
        """
        Add padding to coordinates.

        Args:
            padw (int): Padding width.
            padh (int): Padding height.
        """
        assert not self.normalized, "you should add padding with absolute coordinates."
        self._bboxes.add(offset=(padw, padh, padw, padh))
        self.segments[..., 0] += padw
        self.segments[..., 1] += padh
        if self.keypoints is not None:
            self.keypoints[..., 0] += padw
            self.keypoints[..., 1] += padh

    def add_padding_obb(self, padw, padh):
        """
        Add padding to coordinates.

        Args:
            padw (int): Padding width.
            padh (int): Padding height.
        """
        if self.obbData is not None and len(self.obbData):
            self.obbData[:, 0:2] = self.obbData[:, 0:2] + np.array([padw, padh], dtype=np.float32).reshape(1, 2)
        """
        Add padding to coordinates.

        Args:
            padw (int): Padding width.
            padh (int): Padding height.
        """
        if len(self.obbData):
            self.obbData[:, 0:2] = self.obbData[:, 0:2]+np.array([padw, padh], dtype=np.float32).reshape(1, 2)

    def get_obb_boxes(self, imgw, imgh):
        if not len(self.obbData):
            return np.zeros((0, 5), dtype=np.float32)
        boxes = []
        for idx, obb_box_data in enumerate(self.obbData):
            M = self.translate(obb_box_data[0], obb_box_data[1])
            M = M @ self.rotationMatrix(obb_box_data[4]*np.pi/180.0)
            M = M @ self.translate(-obb_box_data[2]/2.0, -obb_box_data[3]/2.0)
            Minv = np.linalg.inv(M)
            pointsSegment = self.segments[idx]
            pointsSegment = np.concatenate((pointsSegment, np.ones((pointsSegment.shape[0], 1), dtype=np.float32)), axis=-1)
            #print(pointsSegment)
            alignedPoints = pointsSegment @ Minv.T
            #print(alignedPoints)
            (xmin, ymin, xmax, ymax) = np.min(alignedPoints[:, 0]), np.min(alignedPoints[:, 1]), np.max(alignedPoints[:, 0]), np.max(alignedPoints[:, 1])
            # get the 4 corners
            P1 = np.array([xmin, ymin, 1.0], dtype=np.float32)
            P2 = np.array([xmax, ymin, 1.0], dtype=np.float32)
            P3 = np.array([xmax, ymax, 1.0], dtype=np.float32)
            P4 = np.array([xmin, ymax, 1.0], dtype=np.float32)
            #print(P1)
            #print(P2)
            #print(P3)
            #print(P4)

            # get the points back in image coordinates
            pointsInImage = np.array([P1, P2, P3, P4], dtype=np.float32) @ M.T
            #print(pointsInImage)
            #print(pointsInImage.dtype)
            #print(pointsInImage[:, 0:2])
            pointsInImage = np.ascontiguousarray(pointsInImage[:, 0:2])
            # get the OBB data
            (cx, cy), (w, h), angle = cv2.minAreaRect(pointsInImage)
            boxes.append([cx, cy, w, h, angle / 180 * np.pi])
            #boxes.append([cx, cy, w, h, angle])
        #print(self.obbData)
        #print(boxes)
        boxesarray = np.array(boxes)
        np.clip(boxesarray[:, 0:1], a_min=0, a_max=imgw, out=boxesarray[:, 0:1])
        np.clip(boxesarray[:, 1:2], a_min=0, a_max=imgh, out=boxesarray[:, 1:2])

        return boxesarray

    def apply_affine_obb(self, M, scale, rotation):
        if self.obbData is None or not len(self.obbData):
            return
        if not len(self.obbData):
            return
        #print()
        #print(M)
        #print(scale)
        #print(rotation)
        centers = self.obbData[:, 0:2]
        #print(centers)
        centers = np.concatenate((centers, np.ones((centers.shape[0], 1), dtype=np.float32)), axis=-1)
        #print(centers)

        self.obbData[:, 0:2] = (centers @ M.T)[:, 0:2]
        #print(self.obbData)
        self.obbData[:, 2:4] *= scale
        self.obbData[:, 4] -= rotation
        # regularize the OBB again
        for box in self.obbData:
            angle = box[4]
            angle = angle % 180
            if (angle > 90):
                w,h = box[2], box[3]
                box[2] = h
                box[3] = w
                box[4] = angle % 90
            if (angle == 0.0):
                w,h = box[2], box[3]
                box[2] = h
                box[3] = w
                box[4] = 90

            

    def rotationMatrix(self, angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    def translate(self, tx, ty):
        return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float32)




    def __getitem__(self, index) -> "Instances":
        """
        Retrieve a specific instance or a set of instances using indexing.

        Args:
            index (int | slice | np.ndarray): The index, slice, or boolean array to select the desired instances.

        Returns:
            (Instances): A new Instances object containing the selected boxes, segments, and keypoints if present.

        Note:
            When using boolean indexing, make sure to provide a boolean array with the same
            length as the number of instances.
        """
        segments = self.segments[index] if len(self.segments) else self.segments
        keypoints = self.keypoints[index] if self.keypoints is not None else None
        bboxes = self.bboxes[index]
        bbox_format = self._bboxes.format
        obbData = None
        if not self.obbData is None:
            if len(self.obbData):
                obbData = self.obbData[index]
            else:
                obbData = self.obbData

        return Instances(
            bboxes=bboxes,
            segments=segments,
            keypoints=keypoints,
            bbox_format=bbox_format,
            normalized=self.normalized,
            obbData=obbData
        )

    def flipud(self, h):
        """
        Flip coordinates vertically.

        Args:
            h (int): Image height.
        """
        if self._bboxes.format == "xyxy":
            y1 = self.bboxes[:, 1].copy()
            y2 = self.bboxes[:, 3].copy()
            self.bboxes[:, 1] = h - y2
            self.bboxes[:, 3] = h - y1
        else:
            self.bboxes[:, 1] = h - self.bboxes[:, 1]
        self.segments[..., 1] = h - self.segments[..., 1]
        if self.keypoints is not None:
            self.keypoints[..., 1] = h - self.keypoints[..., 1]

    def flipud_obb(self, h):
        if self.obbData is None:
            return
        if len(self.obbData) == 0:
            return
        self.obbData[:, 1] = h-self.obbData[:, 1]
        for box in self.obbData:
            if box[4] == 90:
                continue
            # change angle
            box[4] = 90 - box[4]
            # flip wh
            boxw, boxh = box[2], box[3]
            box[2] = boxh
            box[3] = boxw




    def fliplr(self, w):
        """
        Flip coordinates horizontally.

        Args:
            w (int): Image width.
        """
        if self._bboxes.format == "xyxy":
            x1 = self.bboxes[:, 0].copy()
            x2 = self.bboxes[:, 2].copy()
            self.bboxes[:, 0] = w - x2
            self.bboxes[:, 2] = w - x1
        else:
            self.bboxes[:, 0] = w - self.bboxes[:, 0]
        self.segments[..., 0] = w - self.segments[..., 0]
        if self.keypoints is not None:
            self.keypoints[..., 0] = w - self.keypoints[..., 0]

    def fliplr_obb(self, w):
        if self.obbData is None:
            return
        if len(self.obbData) == 0:
            return
        self.obbData[:, 0] = w - self.obbData[:, 0]
        for box in self.obbData:
            if box[4] == 90:
                continue
            # change angle
            box[4] = 90 - box[4]
            # flip wh
            boxw, boxh = box[2], box[3]
            box[2] = boxh
            box[3] = boxw



    def clip(self, w, h):
        """
        Clip coordinates to stay within image boundaries.

        Args:
            w (int): Image width.
            h (int): Image height.
        """
        ori_format = self._bboxes.format
        self.convert_bbox(format="xyxy")
        self.bboxes[:, [0, 2]] = self.bboxes[:, [0, 2]].clip(0, w)
        self.bboxes[:, [1, 3]] = self.bboxes[:, [1, 3]].clip(0, h)
        if ori_format != "xyxy":
            self.convert_bbox(format=ori_format)
        self.segments[..., 0] = self.segments[..., 0].clip(0, w)
        self.segments[..., 1] = self.segments[..., 1].clip(0, h)
        if self.keypoints is not None:
            # Set out of bounds visibility to zero
            self.keypoints[..., 2][
                (self.keypoints[..., 0] < 0)
                | (self.keypoints[..., 0] > w)
                | (self.keypoints[..., 1] < 0)
                | (self.keypoints[..., 1] > h)
            ] = 0.0

    def remove_zero_area_boxes(self):
        """
        Remove zero-area boxes, i.e. after clipping some boxes may have zero width or height.

        Returns:
            (np.ndarray): Boolean array indicating which boxes were kept.
        """
        good = self.bbox_areas > 0
        if not all(good):
            self._bboxes = self._bboxes[good]
            if len(self.segments):
                self.segments = self.segments[good]
            if self.keypoints is not None:
                self.keypoints = self.keypoints[good]
            if self.obbData is not None:
                self.obbData = self.obbData[good]

        return good

    def update(self, bboxes, segments=None, keypoints=None):
        """
        Update instance variables.

        Args:
            bboxes (np.ndarray): New bounding boxes.
            segments (np.ndarray, optional): New segments.
            keypoints (np.ndarray, optional): New keypoints.
        """
        self._bboxes = Bboxes(bboxes, format=self._bboxes.format)
        if segments is not None:
            self.segments = segments
        if keypoints is not None:
            self.keypoints = keypoints

    def __len__(self):
        """Return the length of the instance list."""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, instances_list: List["Instances"], axis=0) -> "Instances":
        """
        Concatenate a list of Instances objects into a single Instances object.

        Args:
            instances_list (List[Instances]): A list of Instances objects to concatenate.
            axis (int, optional): The axis along which the arrays will be concatenated.

        Returns:
            (Instances): A new Instances object containing the concatenated bounding boxes,
                       segments, and keypoints if present.

        Note:
            The `Instances` objects in the list should have the same properties, such as
            the format of the bounding boxes, whether keypoints are present, and if the
            coordinates are normalized.
        """
        assert isinstance(instances_list, (list, tuple))
        if not instances_list:
            return cls(np.empty(0))
        assert all(isinstance(instance, Instances) for instance in instances_list)

        if len(instances_list) == 1:
            return instances_list[0]

        use_keypoint = instances_list[0].keypoints is not None
        bbox_format = instances_list[0]._bboxes.format
        normalized = instances_list[0].normalized

        cat_boxes = np.concatenate([ins.bboxes for ins in instances_list], axis=axis)
        seg_len = [b.segments.shape[1] for b in instances_list]
        if len(frozenset(seg_len)) > 1:  # resample segments if there's different length
            max_len = max(seg_len)
            cat_segments = np.concatenate(
                [
                    resample_segments(list(b.segments), max_len)
                    if len(b.segments)
                    else np.zeros((0, max_len, 2), dtype=np.float32)  # re-generating empty segments
                    for b in instances_list
                ],
                axis=axis,
            )
        else:
            cat_segments = np.concatenate([b.segments for b in instances_list], axis=axis)
        cat_keypoints = np.concatenate([b.keypoints for b in instances_list], axis=axis) if use_keypoint else None
        obbBoxes = None
        if not instances_list[0].obbData is None:
            obbBoxes = np.concatenate([b.obbData for b in instances_list],axis=axis)
        return cls(cat_boxes, cat_segments, cat_keypoints, bbox_format, normalized, obbBoxes)


    @property
    def bboxes(self):
        """Return bounding boxes."""
        return self._bboxes.bboxes
