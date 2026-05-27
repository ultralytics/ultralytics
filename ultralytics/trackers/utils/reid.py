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


class ReID:
    """ReID encoder. Routes `.pt` to the YOLO predictor path; everything else to `AutoBackend`."""

    def __init__(self, model: str, imgsz: int = 224, device: str | torch.device | None = None, fp16: bool = False):
        """Initialize encoder for re-identification.

        Args:
            model (str): Path to a ReID model. `.pt` runs through the YOLO predictor (embed-layer extraction); other
                extensions go through `AutoBackend`.
            imgsz (int): Square input size used for crop preprocessing on the AutoBackend path.
            device (str | torch.device | None): Inference device; defaults to CUDA if available.
            fp16 (bool): Use half precision when the backend supports it.
        """
        self.imgsz = imgsz
        self.device = (
            torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.is_pt = str(model).endswith(".pt")

        if self.is_pt:
            from ultralytics import YOLO

            self.model = YOLO(model)
            # Initialize predictor with embed=[idx] so subsequent calls return embeddings.
            self.model(embed=[len(self.model.model.model) - 2], verbose=False, save=False)
            self.fp16 = False
        else:
            self.model = AutoBackend(str(model), device=self.device, fp16=fp16, verbose=False)
            self.fp16 = self.model.fp16

    def _crops_to_tensor(self, img: np.ndarray, dets: np.ndarray) -> torch.Tensor:
        """Crop detections from img and stack into a normalized BCHW float tensor at self.imgsz."""
        crops = [save_one_box(det, img, save=False) for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))]
        size = self.imgsz
        batch = torch.empty(len(crops), 3, size, size, dtype=torch.float32)
        for i, c in enumerate(crops):
            t = torch.from_numpy(np.ascontiguousarray(c[..., ::-1])).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            batch[i] = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0]
        batch = batch.to(self.device)
        return batch.half() if self.fp16 else batch

    @torch.no_grad()
    def __call__(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray]:
        """Extract embeddings for detected objects."""
        if self.is_pt:
            crops = [save_one_box(det, img, save=False) for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))]
            feats = self.model.predictor(crops)
            if len(feats) != dets.shape[0] and feats[0].shape[0] == dets.shape[0]:
                feats = feats[0]  # batched prediction with non-PyTorch backend
            return [f.cpu().numpy() for f in feats]
        feats = self.model(self._crops_to_tensor(img, dets))
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
