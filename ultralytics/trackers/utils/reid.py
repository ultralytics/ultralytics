# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Shared ReID encoder used by BoT-SORT, Deep OC-SORT, and TrackTrack.

* `.pt` reid-task checkpoints (`task == "reid"`) — loaded via `YOLO()` and run normally; the
trained ReID head's L2-normalized embedding is read from `Results.embeddings`, and the crop
size is taken from the model's training `imgsz`.
* Other `.pt` checkpoints (e.g. classification) — loaded via `YOLO()`; embeddings are pulled
from the second-to-last layer through the predictor's `embed=[...]` argument.
* Any other extension (`.torchscript`, `.onnx`, `.engine`, `.openvino`, …) — loaded via
`AutoBackend`; the model is expected to output the embedding tensor directly.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.plotting import save_one_box

# Occlusion-aware appearance gating. Crops that are tiny or heavily overlapped by other detections
# yield embeddings that match strangers better than themselves (diagnosed on MOT17: same-vs-other
# margin goes negative below ~0.6 visibility), so we suppress their embeddings to a zero vector. The
# tracker's smooth_feature() then treats a zero-norm feature as "no appearance" and falls back to
# motion for that detection, instead of feeding it a contaminated, mis-associating embedding.
APP_MIN_H = 64.0  # min crop height (px); shorter crops give unreliable embeddings
APP_MAX_OVERLAP = 0.45  # max IoU with any other detection; above this the crop is likely occluded


def _pairwise_max_iou(xyxy: np.ndarray) -> np.ndarray:
    """For each box, the maximum IoU with any other box (0 if alone). xyxy: (N,4)."""
    n = len(xyxy)
    if n < 2:
        return np.zeros(n, dtype=np.float32)
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ix1 = np.maximum(x1[:, None], x1[None]); iy1 = np.maximum(y1[:, None], y1[None])
    ix2 = np.minimum(x2[:, None], x2[None]); iy2 = np.minimum(y2[:, None], y2[None])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    union = area[:, None] + area[None] - inter
    iou = inter / np.maximum(union, 1e-9)
    np.fill_diagonal(iou, 0.0)
    return iou.max(1)


def appearance_reliable(dets: np.ndarray, min_h: float = APP_MIN_H, max_overlap: float = APP_MAX_OVERLAP) -> np.ndarray:
    """Boolean mask: True where a detection's crop is large enough and not heavily occluded.

    Args:
        dets (np.ndarray): Detections in center-xywh format (first 4 cols used).

    Returns:
        (np.ndarray): (N,) bool mask of detections whose appearance embedding should be trusted.
    """
    if dets is None or len(dets) == 0:
        return np.ones(0, dtype=bool)
    cx, cy, w, h = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return (h >= min_h) & (_pairwise_max_iou(xyxy) <= max_overlap)


class ReID:
    """ReID encoder. Routes `.pt` to the YOLO predictor path; everything else to `AutoBackend`."""

    def __init__(self, model: str, imgsz: int = 224, device: str | torch.device | None = None, fp16: bool = False):
        """Initialize encoder for re-identification.

        Args:
            model (str): Path to a ReID model. `.pt` runs through the YOLO predictor (embed-layer extraction); other
                extensions go through `AutoBackend`.
            imgsz (int): Square input size used for crop preprocessing on the AutoBackend path. Overridden by the
                model's own static input size when one is detected.
            device (str | torch.device | None): Inference device; defaults to CUDA if available.
            fp16 (bool): Use half precision when the backend supports it.
        """
        self.imgsz = imgsz
        self.batch_size = None
        self.device = (
            torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.is_pt = str(model).endswith(".pt")

        if self.is_pt:
            from ultralytics import YOLO

            self.model = YOLO(model)
            # ReID-task checkpoints have a trained head whose inference output IS the
            # L2-normalized embedding; run them normally and read Results.embeddings.
            # Everything else (e.g. yolo*-cls.pt) uses the generic second-to-last-layer tap.
            self.is_reid = getattr(self.model, "task", None) == "reid"
            if self.is_reid:
                margs = getattr(self.model.model, "args", None)
                self.imgsz = int(
                    margs.get("imgsz", self.imgsz) if isinstance(margs, dict) else getattr(margs, "imgsz", self.imgsz)
                )
                self.model(embed=None, imgsz=self.imgsz, verbose=False, save=False)  # warm up ReidPredictor
            else:
                self.model(embed=[len(self.model.model.model) - 2], verbose=False, save=False)
            self.fp16 = False
        else:
            self.is_reid = False
            self.model = AutoBackend(str(model), device=self.device, fp16=fp16, verbose=False)
            self.fp16 = self.model.fp16

            # Get model's input size for a fixed batch and crop size or detect dynamic batch and crop sizes.
            session = getattr(self.model, "session", None)
            shape = session.get_inputs()[0].shape if session is not None else ()
            if len(shape) == 4:
                if isinstance(shape[0], int) and shape[0] > 0:
                    self.batch_size = shape[0]
                if isinstance(shape[2], int) and shape[2] > 0:
                    self.imgsz = shape[2]

    @staticmethod
    def _crop_detections(img: np.ndarray, dets: np.ndarray) -> list[np.ndarray]:
        """Crop detection regions from image, converting xywh to xyxy first.

        Args:
            img (np.ndarray): BGR image.
            dets (np.ndarray): Detections in xywh format (first 4 columns used).

        Returns:
            (list[np.ndarray]): Cropped image patches.
        """
        return [save_one_box(det, img, save=False) for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))]

    def _crops_to_tensor(self, img: np.ndarray, dets: np.ndarray) -> torch.Tensor:
        """Crop detections from img and stack into a normalized BCHW float tensor at self.imgsz."""
        crops = self._crop_detections(img, dets)
        batch = torch.empty(len(crops), 3, self.imgsz, self.imgsz, dtype=torch.float32)
        for i, c in enumerate(crops):
            t = torch.from_numpy(np.ascontiguousarray(c[..., ::-1])).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            batch[i] = torch.nn.functional.interpolate(
                t, size=(self.imgsz, self.imgsz), mode="bilinear", align_corners=False
            )[0]
        batch = batch.to(self.device)
        return batch.half() if self.fp16 else batch

    @torch.no_grad()
    def __call__(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray]:
        """Extract embeddings for detected objects, suppressing unreliable (tiny/occluded) crops."""
        feats = self._extract(img, dets)
        if feats and os.environ.get("REID_APP_GATE", "1") != "0":  # gating on by default; REID_APP_GATE=0 disables
            reliable = appearance_reliable(dets)
            feats = [f if reliable[i] else np.zeros_like(f) for i, f in enumerate(feats)]
        return feats

    def _extract(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray]:
        """Raw per-detection embeddings (no gating)."""
        if self.is_pt:
            crops = self._crop_detections(img, dets)
            if self.is_reid:  # ReidPredictor returns Results; pull the head's normalized embedding
                results = self.model.predictor(crops)
                return [r.embeddings.data.cpu().numpy() for r in results]
            feats = self.model.predictor(crops)
            if len(feats) != dets.shape[0] and feats[0].shape[0] == dets.shape[0]:
                feats = feats[0]  # batched prediction with non-PyTorch backend
            return [f.cpu().numpy() for f in feats]
        batch = self._crops_to_tensor(img, dets)
        bs, n = self.batch_size, batch.shape[0]
        if bs is None or n == bs:
            feats = self.model(batch)
        else:  # fixed-batch model (e.g. static ONNX): run in chunks of bs, padding the last partial chunk
            outs = []
            for s in range(0, n, bs):
                chunk = batch[s : s + bs]
                if chunk.shape[0] < bs:
                    chunk = torch.cat([chunk, chunk[-1:].expand(bs - chunk.shape[0], *chunk.shape[1:])], 0)
                outs.append(self.model(chunk))
            feats = torch.cat(outs, 0)[:n]
        return [f.cpu().numpy() for f in feats]


def build_encoder(with_reid: bool, model: str | None):
    """Return a ReID encoder, the native-features pass-through, or None.

    Args:
        with_reid (bool): Whether ReID is enabled at all.
        model (str | None): `"auto"` returns a callable that converts pre-extracted backbone features to numpy arrays;
            any other value loads a `ReID` model from that path. Ignored when `with_reid` is False.

    Returns:
        (Callable | None): A `(img, dets) -> list[np.ndarray]` encoder, or None when ReID is disabled.
    """
    if not with_reid:
        return None
    if model == "auto":

        def _auto_encoder(feats, _dets):
            if isinstance(feats, np.ndarray):
                return [f for f in feats]
            return [f.cpu().numpy() for f in feats]

        return _auto_encoder
    return ReID(model)


def smooth_feature(
    feat: np.ndarray, smooth: np.ndarray | None, alpha: float
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """L2-normalize `feat` and blend it into `smooth` via exponential moving average.

    Args:
        feat (np.ndarray): New (un-normalized) appearance feature.
        smooth (np.ndarray | None): Current smoothed feature, or None on the first update.
        alpha (float): EMA weight on the existing `smooth` (``1.0`` keeps it unchanged).

    Returns:
        curr (np.ndarray | None): The normalized current feature, or None when `feat` is zero-norm (carries no
            appearance information, so the caller should leave its features unchanged).
        smooth (np.ndarray | None): The updated, renormalized smoothed feature.
    """
    norm = np.linalg.norm(feat)
    if norm < 1e-12:  # zero-norm feature has no appearance info; signal the caller to keep its current features
        return None, smooth
    feat = feat / norm
    if smooth is None:
        return feat, feat.copy()
    smooth = alpha * smooth + (1 - alpha) * feat
    return feat, smooth / np.linalg.norm(smooth)
