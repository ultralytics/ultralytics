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
            (list | torch.Tensor): Decoded detections of shape (1, 4 + nc, anchors) for detection models exported
                with raw head maps, otherwise the raw list of output arrays.
        """
        im = (im.cpu().numpy() * 255).astype("uint8")
        im = im if isinstance(im, (list, tuple)) else [im]
        y = self.model.inference(inputs=im)
        if isinstance(y, (list, tuple)) and len(y) > 1:
            return self._decode(y)  # raw per-scale reg/cls maps -> (1, 4 + nc, anchors)
        return y

    def _decode(self, outputs: list) -> torch.Tensor:
        """Decode raw RKNN head maps into the standard detection tensor on CPU.

        The RKNN graph emits per-scale regression (``4 * reg_max`` channels) and classification (``nc`` channels)
        maps without DFL, box decoding or sigmoid (see ``Detect.forward``). Those quantization-sensitive ops are
        applied here in float so INT8 RKNN models retain accuracy. The predictor then runs NMS on the result.

        Args:
            outputs (list): Raw NCHW output arrays from ``RKNNLite.inference``, one regression and one
                classification map per detection scale, in any order.

        Returns:
            (torch.Tensor): Predictions of shape (1, 4 + nc, anchors) with boxes in xywh pixel units.
        """
        import numpy as np

        from ultralytics.utils.tal import dist2bbox, make_anchors

        nc = len(self.names)
        reg_max = int(getattr(self, "reg_max", 16))
        tensors = [torch.as_tensor(np.ascontiguousarray(o), dtype=torch.float32) for o in outputs]

        # RKNN keeps the ONNX framework layout (NCHW), so channel is dim 1: cls maps have nc channels,
        # reg maps have 4 * reg_max. Group the two maps of each scale by their (H, W) and split them by
        # channel count. When nc == 4 * reg_max the counts collide, so fall back to export order within the
        # scale (Detect.forward emits reg before cls), which RKNN preserves.
        groups: dict[tuple[int, int], list[torch.Tensor]] = {}
        for t in tensors:
            groups.setdefault((t.shape[2], t.shape[3]), []).append(t)
        regs, clss = {}, {}
        for hw, (m0, m1) in groups.items():
            reg, cls = (m1, m0) if nc != 4 * reg_max and m0.shape[1] == nc else (m0, m1)
            regs[hw], clss[hw] = reg, cls

        keys = sorted(regs, key=lambda k: k[0] * k[1], reverse=True)  # large -> small feature maps
        feats = [regs[k] for k in keys]
        strides = [self.imgsz[0] // k[0] for k in keys]
        anchors, strides_t = (a.transpose(0, 1) for a in make_anchors(feats, strides, 0.5))

        boxes = torch.cat([regs[k].reshape(1, 4 * reg_max, -1) for k in keys], dim=2)
        scores = torch.cat([clss[k].reshape(1, nc, -1) for k in keys], dim=2)

        b, _, a = boxes.shape
        if reg_max > 1:  # DFL integral; reg_max == 1 means DFL is Identity and boxes are raw ltrb distances
            proj = torch.arange(reg_max, dtype=boxes.dtype)
            boxes = (boxes.view(b, 4, reg_max, a).softmax(2) * proj.view(1, 1, reg_max, 1)).sum(2)
        dbox = dist2bbox(boxes, anchors.unsqueeze(0), xywh=True, dim=1) * strides_t
        return torch.cat((dbox, scores.sigmoid()), 1)
