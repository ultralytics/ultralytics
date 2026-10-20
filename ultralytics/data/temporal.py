# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Temporal (frame-difference) inputs for detecting fast-moving objects, e.g. a tennis ball.

A temporal sample is the current frame plus motion channels built from the frames before it. Everything is stacked into
one HxWxC uint8 image inside `load_image`, so every spatial augmentation (mosaic, affine, flip, ...) is applied
identically to the appearance and motion channels and the rest of the YOLO pipeline needs no changes.

Supported `temporal.mode` values (data YAML), with `k = temporal.frames`:
    - `rgb`: current BGR frame only (3 channels), the no-motion ablation.
    - `diff`: BGR(t) + |gray(t-i) - gray(t-i-1)| for i in 0..k-2 (3 + k - 1 channels).
    - `stack`: BGR of frames t, t-1, ..., t-k+1 (3 * k channels), a TrackNet-style input.
"""

from __future__ import annotations

import math
import os
import re
from collections import deque

import cv2
import numpy as np

from ultralytics.utils.patches import imread

from .dataset import YOLODataset

FRAME_RE = re.compile(r"^(?P<stem>.*_frame_)(?P<idx>\d+)(?P<suffix>\.\w+)$")
TEMPORAL_MODES = {"rgb", "diff", "stack"}


def temporal_channels(mode: str, frames: int) -> int:
    """Return the number of input channels produced by a temporal mode."""
    if mode not in TEMPORAL_MODES:
        raise ValueError(f"temporal mode '{mode}' is not one of {sorted(TEMPORAL_MODES)}")
    return {"rgb": 3, "diff": 3 + frames - 1, "stack": 3 * frames}[mode]


def stack_temporal(frames: list[np.ndarray | None], mode: str) -> np.ndarray:
    """Stack a newest-first list of equally sized BGR frames into one HxWxC uint8 temporal image.

    Missing history (`None`, e.g. the first frames of a clip or a gap in the frame numbering) contributes zero motion
    for `diff` and a copy of the newest available frame for `stack`.

    Args:
        frames (list[np.ndarray | None]): Frames ordered t, t-1, ..., t-k+1; frames[0] must not be None.
        mode (str): One of 'rgb', 'diff', 'stack'.

    Returns:
        (np.ndarray): Stacked image with temporal_channels(mode, len(frames)) channels.
    """
    cur = frames[0]
    if mode == "rgb":
        return cur
    if mode == "stack":
        out, last = [], cur
        for f in frames:
            last = f if f is not None else last
            out.append(last)
        return np.concatenate(out, axis=2)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f is not None else None for f in frames]
    diffs = []
    for a, b in zip(grays[:-1], grays[1:]):
        diffs.append(cv2.absdiff(a, b) if a is not None and b is not None else np.zeros_like(grays[0]))
    return np.concatenate([cur, np.stack(diffs, axis=2)], axis=2)


class TemporalStream:
    """Stateful per-video frame buffer that produces the same temporal images as TemporalYOLODataset.

    Examples:
        >>> stream = TemporalStream(frames=3, mode="diff", size=(1280, 720))
        >>> for bgr in video_frames:
        ...     x = stream.push(bgr)  # HxWx5 uint8
    """

    def __init__(self, frames: int = 3, mode: str = "diff", size: tuple[int, int] | None = None):
        """Initialize with history length, stacking mode and optional (w, h) resize applied before differencing."""
        self.frames, self.mode, self.size = frames, mode, size
        self.buf = deque(maxlen=frames)

    def reset(self):
        """Clear history, required at a video boundary or seek."""
        self.buf.clear()

    def push(self, bgr: np.ndarray) -> np.ndarray:
        """Add a new frame and return the stacked temporal image for it."""
        if self.size is not None and bgr.shape[1::-1] != tuple(self.size):
            bgr = cv2.resize(bgr, self.size, interpolation=cv2.INTER_LINEAR)
        self.buf.appendleft(bgr)
        hist = list(self.buf) + [None] * (self.frames - len(self.buf))
        return stack_temporal(hist, self.mode)


class TemporalYOLODataset(YOLODataset):
    """YOLODataset whose images are the current frame stacked with motion channels from preceding frames.

    Previous frames are located by file name: `<stem>_frame_<idx>.<ext>` looks for `<stem>_frame_<idx-1>.<ext>` in the
    same directory. Previous frames need no labels and need not be in the image list. A missing previous frame is
    treated as the start of a clip (zero motion).

    Enabled by a `temporal` section in the data YAML:
        temporal:
          frames: 3
          mode: diff
        channels: 5  # must equal temporal_channels(mode, frames)
    """

    def __init__(self, *args, data: dict, **kwargs):
        """Validate the temporal config, then build the dataset as a YOLODataset."""
        cfg = data.get("temporal") or {}
        self.t_frames = int(cfg.get("frames", 3))
        self.t_mode = cfg.get("mode", "diff")
        ch = temporal_channels(self.t_mode, self.t_frames)
        if data.get("channels", 3) != ch:
            raise ValueError(f"temporal mode '{self.t_mode}' with {self.t_frames} frames needs channels: {ch}")
        self._dir_files: dict[str, set[str]] = {}
        super().__init__(*args, data=data, **kwargs)

    def _exists(self, path: str) -> bool:
        """Check file existence with one directory listing per directory."""
        d, name = os.path.split(path)
        if d not in self._dir_files:
            self._dir_files[d] = set(os.listdir(d)) if os.path.isdir(d) else set()
        return name in self._dir_files[d]

    def history_paths(self, f: str) -> list[str | None]:
        """Return newest-first paths [t, t-1, ...]; history stops at the first missing frame."""
        m = FRAME_RE.match(f)
        paths: list[str | None] = [f]
        if m is None:
            return paths + [None] * (self.t_frames - 1)
        idx, width = int(m["idx"]), len(m["idx"])
        for k in range(1, self.t_frames):
            p = f"{m['stem']}{idx - k:0{width}d}{m['suffix']}"
            if idx - k < 0 or paths[-1] is None or not self._exists(p):
                p = None
            paths.append(p)
        return paths

    def load_image(self, i: int, rect_mode: bool = True, resize_short: bool = False):
        """Load frame i and its history, resize each like BaseDataset.load_image, and stack them."""
        if self.ims[i] is not None:
            return self.ims[i], self.im_hw0[i], self.im_hw[i]
        f = self.im_files[i]
        cur = imread(f, flags=cv2.IMREAD_COLOR)
        if cur is None:
            raise FileNotFoundError(f"Image Not Found {f}")
        h0, w0 = cur.shape[:2]
        shape = tuple(self.imgsz) if isinstance(self.imgsz, (tuple, list)) else (self.imgsz, self.imgsz)
        imgsz, size = max(shape), (w0, h0)
        if rect_mode and resize_short:
            r = imgsz / min(h0, w0)
            if r != 1:
                size = (math.ceil(w0 * r), imgsz) if h0 < w0 else (imgsz, math.ceil(h0 * r))
        elif rect_mode:
            r = imgsz / max(h0, w0)
            if r != 1:
                size = (min(math.ceil(w0 * r), imgsz), min(math.ceil(h0 * r), imgsz))
        else:
            size = shape[::-1]
        frames = [cur]
        for p in self.history_paths(f)[1:]:
            im = imread(p, flags=cv2.IMREAD_COLOR) if p is not None else None
            frames.append(im if im is None or im.shape[:2] == (h0, w0) else None)
        if size != (w0, h0):
            frames = [cv2.resize(x, size, interpolation=cv2.INTER_LINEAR) if x is not None else x for x in frames]
        im = stack_temporal(frames, self.t_mode)

        if self.augment and self.cache != "ram":
            self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
            self.buffer.append(i)
            if 1 < len(self.buffer) >= self.max_buffer_length:
                j = self.buffer.pop(0)
                self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
        return im, (h0, w0), im.shape[:2]


__all__ = ("TemporalStream", "TemporalYOLODataset", "stack_temporal", "temporal_channels")
