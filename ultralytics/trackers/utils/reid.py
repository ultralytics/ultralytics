# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Shared ReID encoder used by BoT-SORT, Deep OC-SORT, and TrackTrack.

* `.pt` YOLO checkpoints — loaded via `YOLO()`; embeddings are pulled from the second-to-last
layer through the predictor's `embed=[...]` argument (works with classification and ReID
backbones).
* Any other extension (`.torchscript`, `.onnx`, `.engine`, `.openvino`, …) — loaded via
`AutoBackend`; the model is expected to output the embedding tensor directly.
"""

from __future__ import annotations

import numpy as np
import torch

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.plotting import save_one_box

REID_ASSETS = frozenset(f"yolo26{k}-reid.onnx" for k in "nsmlx")


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
            # Initialize predictor with embed=[idx] so subsequent calls return embeddings.
            self.model(embed=[len(self.model.model.model) - 2], device=self.device, verbose=False, save=False)
            self.fp16 = False
        else:
            from pathlib import Path

            if Path(str(model)).name in REID_ASSETS:
                from ultralytics.utils.downloads import attempt_download_asset

                model = attempt_download_asset(str(model))
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

    def _crops_to_tensor(self, crops: list[np.ndarray]) -> torch.Tensor:
        """Stack a list of valid image crops into a normalized BCHW float tensor at self.imgsz."""
        batch = torch.empty(len(crops), 3, self.imgsz, self.imgsz, dtype=torch.float32)
        for i, c in enumerate(crops):
            t = torch.from_numpy(np.ascontiguousarray(c[..., ::-1])).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            batch[i] = torch.nn.functional.interpolate(
                t, size=(self.imgsz, self.imgsz), mode="bilinear", align_corners=False
            )[0]
        batch = batch.to(self.device)
        return batch.half() if self.fp16 else batch

    @torch.no_grad()
    def __call__(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray | None]:
        """Extract embeddings for detected objects."""
        crops = self._crop_detections(img, dets)
        valid = [bool(c.size) for c in crops]
        valid_crops = [crop for crop, keep in zip(crops, valid) if keep]
        if not valid_crops:
            return [None] * len(crops)

        if self.is_pt:
            feats = self.model.predictor(valid_crops)
            if len(feats) != len(valid_crops) and feats[0].shape[0] == len(valid_crops):
                feats = feats[0]  # batched prediction with non-PyTorch backend
            valid_feats = [f.cpu().numpy() for f in feats]
        else:
            batch = self._crops_to_tensor(valid_crops)
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
            valid_feats = [f.cpu().numpy() for f in feats]

        valid_feats = iter(valid_feats)
        return [next(valid_feats) if keep else None for keep in valid]


def build_encoder(with_reid: bool, model: str | None, device: str | torch.device | None = None):
    """Return a ReID encoder, the native-features pass-through, or None.

    Args:
        with_reid (bool): Whether ReID is enabled at all.
        model (str | None): `"auto"` returns a callable that converts pre-extracted backbone features to numpy arrays;
            any other value loads a `ReID` model from that path. Ignored when `with_reid` is False.
        device (str | torch.device | None): Inference device for the ReID model; defaults to CUDA if available.

    Returns:
        (Callable | None): A `(img, dets) -> list[np.ndarray | None]` encoder, or None when ReID is disabled.
    """
    if not with_reid:
        return None
    if model == "auto":

        def _auto_encoder(feats, _dets):
            if isinstance(feats, np.ndarray):
                return [f for f in feats]
            return [f.cpu().numpy() for f in feats]

        return _auto_encoder
    return ReID(model, device=device)


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
