# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements, is_rockchip

from .base import BaseBackend


class RKNNBackend(BaseBackend):
    """Rockchip RKNN inference backend for Rockchip NPU hardware.

    Loads and runs inference with RKNN models (.rknn files) using the RKNN-Toolkit-Lite2 runtime. Only supported on
    Rockchip devices with NPU hardware (e.g., RK3588, RK3566).
    """

    def load_model(self, weight: str | Path) -> None:
        """Load a Rockchip RKNN model from a .rknn file or model directory.

        Args:
            weight (str | Path): Path to the .rknn file or directory containing the model.

        Raises:
            OSError: If not running on a Rockchip device.
            RuntimeError: If model loading or runtime initialization fails.
        """
        if not is_rockchip():
            raise OSError("RKNN inference is only supported on Rockchip devices.")

        LOGGER.info(f"Loading {weight} for RKNN inference...")
        check_requirements("rknn-toolkit-lite2")
        from rknnlite.api import RKNNLite

        w = Path(weight)
        if not w.is_file():
            w = next(w.rglob("*.rknn"))

        self.model = RKNNLite()
        ret = self.model.load_rknn(str(w))
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")

        ret = self.model.init_runtime()
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime: {ret}")

        # Load metadata
        metadata_file = w.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> list | torch.Tensor:
        """Run inference on the Rockchip NPU.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (list | torch.Tensor): Decoded detections of shape (1, 4 + nc, anchors) for detection models exported with
                raw head maps, otherwise the raw list of output arrays.
        """
        im = (im.cpu().numpy() * 255).astype("uint8")
        im = im if isinstance(im, (list, tuple)) else [im]
        y = self.model.inference(inputs=im)
        # only the INT8 raw-head path emits all-4D NCHW maps in (reg, cls) pairs
        # FP16 seg outputs (3D preds + 4D protos) must pass through
        if isinstance(y, (list, tuple)) and len(y) > 1 and len(y) % 2 == 0 and all(o.ndim == 4 for o in y):
            return self._decode(y)  # raw per-scale reg/cls maps -> (1, 4 + nc, anchors)
        return y

    def _decode(self, outputs: list) -> torch.Tensor:
        """Decode raw RKNN reg/cls head maps into (1, 4 + nc, anchors), applying DFL and box decode in float.

        The INT8 export forward (rknn_detect_forward) emits per-scale regression (4 * reg_max channels) and
        already-sigmoided classification (nc channels) maps in export order (reg, cls per scale, large to small).
        Doing DFL and box decode here on CPU keeps the wide-range decode off the NPU so INT8 models stay accurate.
        The predictor then runs NMS.

        Args:
            outputs (list): Raw NCHW arrays from RKNNLite.inference, reg and cls per scale, large to small.

        Returns:
            (torch.Tensor): Predictions of shape (1, 4 + nc, anchors) with boxes in xywh pixel units.
        """
        import numpy as np

        from ultralytics.utils.tal import dist2bbox, make_anchors

        nc = len(self.names)
        feats = [torch.as_tensor(np.ascontiguousarray(o), dtype=torch.float32) for o in outputs]
        regs, clss = feats[0::2], feats[1::2]  # reg/cls per scale, large -> small feature maps
        reg_max = regs[0].shape[1] // 4  # reg has 4 * reg_max channels; reg_max == 1 means no DFL

        strides = [self.imgsz[0] // r.shape[2] for r in regs]
        anchors, strides_t = (a.transpose(0, 1) for a in make_anchors(regs, strides, 0.5))

        boxes = torch.cat([r.reshape(1, 4 * reg_max, -1) for r in regs], 2)
        scores = torch.cat([c.reshape(1, nc, -1) for c in clss], 2)

        b, _, a = boxes.shape
        if reg_max > 1:  # DFL integral; reg_max == 1 means boxes are raw ltrb distances (DFL is Identity)
            proj = torch.arange(reg_max, dtype=boxes.dtype)
            boxes = (boxes.view(b, 4, reg_max, a).softmax(2) * proj.view(1, 1, reg_max, 1)).sum(2)
        dbox = dist2bbox(boxes, anchors.unsqueeze(0), xywh=True, dim=1) * strides_t
        return torch.cat((dbox, scores), 1)  # cls already sigmoided in-graph (see rknn_detect_forward)
