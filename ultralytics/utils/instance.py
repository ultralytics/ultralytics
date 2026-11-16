# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from collections import abc
from itertools import repeat
from numbers import Number

import cv2
import numpy as np

from .ops import (
    ltwh2xywh,
    ltwh2xyxy,
    resample_segments,
    segment2box,
    union_boxes_by_owner,
    xywh2ltwh,
    xywh2xyxy,
    xyxy2ltwh,
    xyxy2xywh,
)


def _ntuple(n):
    """Create a function that converts input to n-tuple by repeating singleton values."""

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
    """A class for handling bounding boxes in multiple formats.

    The class supports various bounding box formats like 'xyxy', 'xywh', and 'ltwh' and provides methods for format
    conversion, scaling, and area calculation. Bounding box data should be provided as numpy arrays.

    Attributes:
        bboxes (np.ndarray): The bounding boxes stored in a 2D numpy array with shape (N, 4).
        format (str): The format of the bounding boxes ('xyxy', 'xywh', or 'ltwh').

    Methods:
        convert: Convert bounding box format from one type to another.
        areas: Calculate the area of bounding boxes.
        mul: Multiply bounding box coordinates by scale factor(s).
        add: Add offset to bounding box coordinates.
        concatenate: Concatenate multiple Bboxes objects.

    Examples:
        Create bounding boxes in YOLO format
        >>> bboxes = Bboxes(np.array([[100, 50, 150, 100]]), format="xywh")
        >>> bboxes.convert("xyxy")
        >>> print(bboxes.areas())
        [15000]

    Notes:
        This class does not handle normalization or denormalization of bounding boxes.
    """

    def __init__(self, bboxes: np.ndarray, format: str = "xyxy") -> None:
        """Initialize the Bboxes class with bounding box data in a specified format.

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

    def convert(self, format: str) -> None:
        """Convert bounding box format from one type to another.

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

    def areas(self) -> np.ndarray:
        """Calculate the area of bounding boxes."""
        return (
            (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])  # format xyxy
            if self.format == "xyxy"
            else self.bboxes[:, 3] * self.bboxes[:, 2]  # format xywh or ltwh
        )

    def mul(self, scale: float | tuple | list) -> None:
        """Multiply bounding box coordinates by scale factor(s).

        Args:
            scale (int | float | tuple | list): Scale factor(s) for four coordinates. If a single number, the same scale
                is applied to all coordinates.
        """
        if isinstance(scale, Number):
            scale = to_4tuple(scale)
        assert isinstance(scale, (tuple, list))
        assert len(scale) == 4
        self.bboxes[:, 0] *= scale[0]
        self.bboxes[:, 1] *= scale[1]
        self.bboxes[:, 2] *= scale[2]
        self.bboxes[:, 3] *= scale[3]

    def add(self, offset: float | tuple | list) -> None:
        """Add offset to bounding box coordinates.

        Args:
            offset (int | float | tuple | list): Offset(s) for four coordinates. If a single number, the same offset is
                applied to all coordinates.
        """
        if isinstance(offset, Number):
            offset = to_4tuple(offset)
        assert isinstance(offset, (tuple, list))
        assert len(offset) == 4
        self.bboxes[:, 0] += offset[0]
        self.bboxes[:, 1] += offset[1]
        self.bboxes[:, 2] += offset[2]
        self.bboxes[:, 3] += offset[3]

    def __len__(self) -> int:
        """Return the number of bounding boxes."""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, boxes_list: list[Bboxes], axis: int = 0) -> Bboxes:
        """Concatenate a list of Bboxes objects into a single Bboxes object.

        Args:
            boxes_list (list[Bboxes]): A list of Bboxes objects to concatenate.
            axis (int, optional): The axis along which to concatenate the bounding boxes.

        Returns:
            (Bboxes): A new Bboxes object containing the concatenated bounding boxes, or the input object itself if the
                list has a single element.

        Notes:
            The input should be a list or tuple of Bboxes objects sharing the same format, which the result keeps.
        """
        assert isinstance(boxes_list, (list, tuple))
        if not boxes_list:
            return cls(np.empty((0, 4)))
        assert all(isinstance(box, Bboxes) for box in boxes_list)

        if len(boxes_list) == 1:
            return boxes_list[0]
        return cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis), format=boxes_list[0].format)

    def __getitem__(self, index: int | np.ndarray | slice) -> Bboxes:
        """Retrieve a specific bounding box or a set of bounding boxes using indexing.

        Args:
            index (int | slice | np.ndarray): The index, slice, or boolean array to select the desired bounding boxes.

        Returns:
            (Bboxes): A new Bboxes object containing the selected bounding boxes.

        Notes:
            When using boolean indexing, make sure to provide a boolean array with the same length as the number of
            bounding boxes.
        """
        b = self.bboxes[index]
        if b.ndim == 1:
            b = b.reshape(1, -1)
        assert b.ndim == 2, f"Indexing on Bboxes with {index} failed to return a matrix!"
        return Bboxes(b, format=self.format)


class Instances:
    """Container for bounding boxes, segments, and keypoints of detected objects in an image.

    This class provides a unified interface for handling different types of object annotations including bounding boxes,
    segmentation masks, and keypoints. It supports various operations like scaling, normalization, clipping, and format
    conversion.

    Attributes:
        _bboxes (Bboxes): Internal object for handling bounding box operations.
        keypoints (np.ndarray | None): Keypoints with shape (N, K, 3) in format (x, y, visible), e.g. K=17 for COCO.
        normalized (bool): Flag indicating whether the coordinates are normalized.
        segments (np.ndarray): Segments array with shape (K, M, 2) after resampling, one row per contour.
        segments_owner (np.ndarray): Instance index of each contour, with shape (K,).

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
        Create instances with bounding boxes and segments
        >>> instances = Instances(
        ...     bboxes=np.array([[10, 10, 30, 30], [20, 20, 40, 40]]),
        ...     segments=np.array([[[5, 5], [10, 10]], [[15, 15], [20, 20]]], dtype=np.float32),
        ...     keypoints=np.array([[[5, 5, 1], [10, 10, 1]], [[15, 15, 1], [20, 20, 1]]], dtype=np.float32),
        ...     bbox_format="xyxy",
        ...     normalized=False,
        ... )
    """

    def __init__(
        self,
        bboxes: np.ndarray,
        segments: np.ndarray | None = None,
        keypoints: np.ndarray | None = None,
        bbox_format: str = "xywh",
        normalized: bool = True,
        segments_owner: np.ndarray | None = None,
    ) -> None:
        """Initialize the Instances object with bounding boxes, segments, and keypoints.

        Args:
            bboxes (np.ndarray): Bounding boxes with shape (N, 4).
            segments (np.ndarray, optional): Segment polygons with shape (N, M, 2), where M is the number of points.
            keypoints (np.ndarray, optional): Keypoints with shape (N, K, 3) in format (x, y, visible).
            bbox_format (str): Format of bboxes, one of 'xyxy', 'xywh', or 'ltwh'.
            normalized (bool): Whether the coordinates are normalized.
            segments_owner (np.ndarray, optional): Instance index of each contour, parallel to `segments`; defaults to one contour per instance.
        """
        self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format)
        self.keypoints = keypoints
        self.normalized = normalized
        self.segments = segments if segments is not None else np.zeros((0, 0, 2), dtype=np.float32)
        # segments_owner[c] is the instance index that contour c belongs to. Defaults to one contour per
        # instance (the legacy layout); multipart instances have repeated owner ids.
        if segments_owner is None:
            segments_owner = np.arange(len(self.segments), dtype=np.int64)
        self.segments_owner = np.asarray(segments_owner, dtype=np.int64)
        assert len(self.segments_owner) == len(self.segments), (
            f"segments_owner has {len(self.segments_owner)} entries for {len(self.segments)} contours"
        )
        assert len(self.segments_owner) == 0 or self.segments_owner.max() < len(self._bboxes), (
            "segments_owner refers to an instance index beyond the bounding boxes"
        )

    def convert_bbox(self, format: str) -> None:
        """Convert bounding box format.

        Args:
            format (str): Target format for conversion, one of 'xyxy', 'xywh', or 'ltwh'.
        """
        self._bboxes.convert(format=format)

    @property
    def bbox_areas(self) -> np.ndarray:
        """Return the areas of the bounding boxes."""
        return self._bboxes.areas()

    @property
    def is_multipart(self) -> bool:
        """Whether any instance owns more than one contour."""
        return len(self.segments_owner) != len(np.unique(self.segments_owner))

    def scale(self, scale_w: float, scale_h: float, bbox_only: bool = False):
        """Scale coordinates by given factors.

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

    def denormalize(self, w: int, h: int) -> None:
        """Convert normalized coordinates to absolute coordinates.

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

    def normalize(self, w: int, h: int) -> None:
        """Convert absolute coordinates to normalized coordinates.

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

    def add_padding(self, padw: int, padh: int) -> None:
        """Add padding to coordinates.

        Args:
            padw (int): Horizontal padding added to x coordinates.
            padh (int): Vertical padding added to y coordinates.
        """
        assert not self.normalized, "you should add padding with absolute coordinates."
        self._bboxes.add(offset=(padw, padh, padw, padh))
        self.segments[..., 0] += padw
        self.segments[..., 1] += padh
        if self.keypoints is not None:
            self.keypoints[..., 0] += padw
            self.keypoints[..., 1] += padh

    def __getitem__(self, index: int | np.ndarray | slice) -> Instances:
        """Retrieve a specific instance or a set of instances using indexing.

        Args:
            index (int | slice | np.ndarray): The index, slice, or boolean array to select the desired instances.

        Returns:
            (Instances): A new Instances object containing the selected boxes, segments, and keypoints if present.
        """
        index = [index] if isinstance(index, (int, np.integer)) else index
        if isinstance(index, slice):
            index = np.arange(len(self))[index]
        index = np.atleast_1d(np.asarray(index))
        if index.dtype != bool and len(index):
            index = np.where(index < 0, index + len(self), index)
        if index.dtype == bool:
            keep = np.flatnonzero(index)
            seg_mask = index[self.segments_owner] if len(self.segments) else np.zeros(0, dtype=bool)
        else:
            keep = index
            seg_mask = np.isin(self.segments_owner, index) if len(self.segments) else np.zeros(0, dtype=bool)
        segments = self.segments[seg_mask] if len(self.segments) else self.segments
        # Remap contour owners to the new instance positions (first occurrence wins for repeated indices)
        pos = {}
        for new, old in enumerate(keep.tolist()):
            pos.setdefault(old, new)
        segments_owner = (
            np.array([pos[o] for o in self.segments_owner[seg_mask]], dtype=np.int64)
            if len(self.segments)
            else self.segments_owner
        )
        keypoints = self.keypoints[index] if self.keypoints is not None else None
        bboxes = self.bboxes[index]
        bbox_format = self._bboxes.format
        return Instances(
            bboxes=bboxes,
            segments=segments,
            keypoints=keypoints,
            bbox_format=bbox_format,
            normalized=self.normalized,
            segments_owner=segments_owner,
        )

    def flipud(self, h: int) -> None:
        """Flip coordinates vertically.

        Args:
            h (int): Image height.
        """
        if self._bboxes.format == "xyxy":
            y1 = self.bboxes[:, 1].copy()
            y2 = self.bboxes[:, 3].copy()
            self.bboxes[:, 1] = h - y2
            self.bboxes[:, 3] = h - y1
        elif self._bboxes.format == "ltwh":
            self.bboxes[:, 1] = h - self.bboxes[:, 1] - self.bboxes[:, 3]
        else:
            self.bboxes[:, 1] = h - self.bboxes[:, 1]
        self.segments[..., 1] = h - self.segments[..., 1]
        if self.keypoints is not None:
            self.keypoints[..., 1] = h - self.keypoints[..., 1]

    def fliplr(self, w: int) -> None:
        """Flip coordinates horizontally.

        Args:
            w (int): Image width.
        """
        if self._bboxes.format == "xyxy":
            x1 = self.bboxes[:, 0].copy()
            x2 = self.bboxes[:, 2].copy()
            self.bboxes[:, 0] = w - x2
            self.bboxes[:, 2] = w - x1
        elif self._bboxes.format == "ltwh":
            self.bboxes[:, 0] = w - self.bboxes[:, 0] - self.bboxes[:, 2]
        else:
            self.bboxes[:, 0] = w - self.bboxes[:, 0]
        self.segments[..., 0] = w - self.segments[..., 0]
        if self.keypoints is not None:
            self.keypoints[..., 0] = w - self.keypoints[..., 0]

    def clip(self, w: int, h: int, preserve_obb: bool = False) -> None:
        """Clip coordinates to stay within image boundaries.

        Keypoints outside the image get zero visibility before being clipped.

        Args:
            w (int): Image width.
            h (int): Image height.
            preserve_obb (bool): Preserve oriented-box direction while clipping segments: each segment extending outside
                the image is replaced by the rectangle, aligned with its original orientation, that bounds its visible
                part, and its box by the axis-aligned bounds of that visible part.
        """
        ori_format = self._bboxes.format
        self.convert_bbox(format="xyxy")
        if preserve_obb and len(self.segments) and not self.is_multipart:
            segment_mins, segment_maxs = self.segments.min(1), self.segments.max(1)
            inside = ((segment_mins >= 0) & (segment_maxs <= (w, h))).all(1)
            if not inside.all():
                canvas = np.array(((0, 0), (w, 0), (w, h), (0, h)), dtype=np.float32)
                bboxes, segments = self.bboxes.copy(), self.segments.copy()
                for i in np.flatnonzero(~inside):
                    segment = self.segments[i]
                    visible_area, visible = cv2.intersectConvexConvex(
                        cv2.convexHull(segment.astype(np.float32)), canvas
                    )
                    if visible is None or visible_area <= 0:
                        bboxes[i], segments[i] = 0, 0
                        continue
                    visible = visible.reshape(-1, 2)
                    bboxes[i] = np.array((*visible.min(0), *visible.max(0)), dtype=segment.dtype)

                    # Fit the visible polygon in the full box's coordinates so edge crops cannot change its angle.
                    (_, _), (box_w, box_h), angle = cv2.minAreaRect(segment.astype(np.float32))
                    if box_w < box_h:
                        angle += 90
                    angle = np.deg2rad(angle)
                    cos, sin = np.cos(angle), np.sin(angle)
                    basis = np.array(((cos, -sin), (sin, cos)), dtype=np.float32)
                    aligned = visible @ basis
                    (u1, v1), (u2, v2) = aligned.min(0), aligned.max(0)
                    corners = np.array(((u2, v2), (u2, v1), (u1, v1), (u1, v2)), dtype=np.float32)
                    segments[i] = resample_segments([corners @ basis.T], n=len(segment))[0]
                self.update(bboxes, segments)
        else:
            self.bboxes[:, [0, 2]] = self.bboxes[:, [0, 2]].clip(0, w)
            self.bboxes[:, [1, 3]] = self.bboxes[:, [1, 3]].clip(0, h)
            self.segments[..., 0] = self.segments[..., 0].clip(0, w)
            self.segments[..., 1] = self.segments[..., 1].clip(0, h)
            if len(self.segments) and self.is_multipart:
                # A multipart instance survives only through its visible contours: recompute each box as
                # the union of its contours' visible extents, so an instance whose contours all fell
                # outside the canvas gets a zero box and is dropped by remove_zero_area_boxes().
                self._bboxes = Bboxes(
                    union_boxes_by_owner(
                        np.stack([segment2box(s, w, h) for s in self.segments], 0),
                        self.segments_owner,
                        len(self),
                    ),
                    format="xyxy",
                )
        if ori_format != "xyxy":
            self.convert_bbox(format=ori_format)
        if self.keypoints is not None:
            # Set out of bounds visibility to zero
            self.keypoints[..., 2][
                (self.keypoints[..., 0] < 0)
                | (self.keypoints[..., 0] > w)
                | (self.keypoints[..., 1] < 0)
                | (self.keypoints[..., 1] > h)
            ] = 0.0
            self.keypoints[..., 0] = self.keypoints[..., 0].clip(0, w)
            self.keypoints[..., 1] = self.keypoints[..., 1].clip(0, h)

    def crop(self, window: tuple[int, int, int, int], preserve_obb: bool = False) -> np.ndarray:
        """Crop pixel-space instances and return the mask retaining at least 10% of each original box area."""
        x1, y1, x2, y2 = window
        w, h = x2 - x1, y2 - y1
        areas = self.bbox_areas.copy()
        self.convert_bbox("xyxy")
        self.add_padding(-x1, -y1)
        if len(self.segments) and not preserve_obb:
            outside = ((self.segments < 0) | (self.segments > (w, h))).any((1, 2))
            if self.is_multipart:
                for c in np.flatnonzero(outside):
                    # Clip this contour to its own visible extent; the instance box is recomputed
                    # below from all of its contours, so a contour that fell fully outside cannot
                    # drag the box (segment2box returns zeros for a fully-outside contour).
                    box = segment2box(self.segments[c], w, h)
                    self.segments[c] = self.segments[c].clip(box[:2], box[2:])
                self._bboxes = Bboxes(
                    union_boxes_by_owner(
                        np.stack([segment2box(s, w, h) for s in self.segments], 0),
                        self.segments_owner,
                        len(self),
                    ),
                    format="xyxy",
                )
            else:
                for i in np.flatnonzero(outside):
                    self.bboxes[i] = box = segment2box(self.segments[i], w, h)
                    self.segments[i] = self.segments[i].clip(box[:2], box[2:])
        self.clip(w, h, preserve_obb=preserve_obb)
        return (self.bbox_areas > 0) & (self.bbox_areas >= areas * 0.1)

    def remove_zero_area_boxes(self) -> np.ndarray:
        """Remove zero-area boxes, i.e. after clipping some boxes may have zero width or height.

        Returns:
            (np.ndarray): Boolean array indicating which boxes were kept.
        """
        good = self.bbox_areas > 0
        if not all(good):
            self._bboxes = self._bboxes[good]
            if self.segments is not None and len(self.segments):
                # good is per instance; keep the contours whose owner instance survived, then remap
                # the surviving owners to the new instance positions (boxes were compacted above,
                # so stale owner ids would point at the wrong instance — or past the end).
                seg_keep = good[self.segments_owner]
                self.segments = self.segments[seg_keep]
                pos = {old: new for new, old in enumerate(np.flatnonzero(good).tolist())}
                self.segments_owner = np.array([pos[o] for o in self.segments_owner[seg_keep]], dtype=np.int64)
            if self.keypoints is not None:
                self.keypoints = self.keypoints[good]
        return good

    def update(
        self,
        bboxes: np.ndarray,
        segments: np.ndarray | None = None,
        keypoints: np.ndarray | None = None,
        segments_owner: np.ndarray | None = None,
    ):
        """Update instance variables.

        Args:
            bboxes (np.ndarray): New bounding boxes.
            segments (np.ndarray, optional): New segments.
            keypoints (np.ndarray, optional): New keypoints.
            segments_owner (np.ndarray, optional): Instance index of each new segment; defaults to one contour per
                instance.
        """
        self._bboxes = Bboxes(bboxes, format=self._bboxes.format)
        if segments is not None:
            self.segments = segments
            self.segments_owner = (
                np.asarray(segments_owner, dtype=np.int64)
                if segments_owner is not None
                else np.arange(len(segments), dtype=np.int64)
            )
        if keypoints is not None:
            self.keypoints = keypoints

    def __len__(self) -> int:
        """Return the number of instances."""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, instances_list: list[Instances], axis: int = 0) -> Instances:
        """Concatenate a list of Instances objects into a single Instances object.

        Args:
            instances_list (list[Instances]): A list of Instances objects to concatenate.
            axis (int, optional): The axis along which the arrays will be concatenated.

        Returns:
            (Instances): A new Instances object containing the concatenated bounding boxes, segments, and keypoints if
                present.

        Notes:
            The `Instances` objects in the list should have the same properties, such as the format of the bounding
            boxes, whether keypoints are present, and if the coordinates are normalized.
        """
        assert isinstance(instances_list, (list, tuple))
        if not instances_list:
            return cls(np.empty((0, 4)))
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
        # Shift each part's owner ids past the instances that came before it, so ids from different
        # images (mosaic / mixup / cutmix) can never collide; resampling above preserves contour order.
        offsets = np.cumsum([0, *[len(ins) for ins in instances_list[:-1]]])
        cat_owner = np.concatenate([ins.segments_owner + o for ins, o in zip(instances_list, offsets)])
        return cls(cat_boxes, cat_segments, cat_keypoints, bbox_format, normalized, segments_owner=cat_owner)

    @property
    def bboxes(self) -> np.ndarray:
        """Return bounding boxes."""
        return self._bboxes.bboxes

    def __repr__(self) -> str:
        """Return a string representation of the Instances object."""
        # Map private to public names and include direct attributes
        attr_map = {"_bboxes": "bboxes"}
        parts = []
        for key, value in self.__dict__.items():
            name = attr_map.get(key, key)
            if name == "bboxes":
                value = self.bboxes  # Use the property
            if value is not None:
                parts.append(f"{name}={value!r}")
        return "Instances({})".format("\n".join(parts))
