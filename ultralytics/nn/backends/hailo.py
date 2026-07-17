# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER

from .base import BaseBackend


class HailoBackend(BaseBackend):
    """HailoRT inference backend for Ultralytics Hailo HEF models."""

    def load_model(self, weight: str | Path) -> None:
        """Load a Hailo export directory and its Ultralytics metadata."""
        try:
            from hailo_platform import (
                HEF,
                ConfigureParams,
                FormatType,
                HailoStreamInterface,
                InferVStreams,
                InputVStreamParams,
                OutputVStreamParams,
                VDevice,
            )
        except ImportError as e:
            raise ImportError(
                "Hailo inference requires HailoRT. "
                "See https://docs.ultralytics.com/integrations/hailo/#run-hailo-inference"
            ) from e

        w = Path(weight)
        hef_file = next(w.rglob("*.hef"), None)
        if hef_file is None or not hef_file.is_file():
            raise FileNotFoundError(f"No .hef file found in: {w}")

        LOGGER.info(f"Loading {hef_file} for Hailo inference...")
        metadata_file = hef_file.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))
        if self.task and self.task not in {"detect", "segment"}:
            raise ValueError(
                f"Hailo inference only supports detection and segmentation models, not task='{self.task}'."
            )

        self.hef = HEF(str(hef_file))
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.output_infos = self.hef.get_output_vstream_infos()
        with ExitStack() as stack:
            target = stack.enter_context(VDevice())
            configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
            network_group = target.configure(self.hef, configure_params)[0]
            stack.enter_context(network_group.activate(network_group.create_params()))
            input_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
            output_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
            self.model = stack.enter_context(InferVStreams(network_group, input_params, output_params))
            self._stack = stack.pop_all()
        self._anchors = None
        self.end2end = self.task != "segment"  # segmentation returns dense output for the predictor's NMS

    def __del__(self):
        """Release the Hailo pipeline and device."""
        if stack := getattr(self, "_stack", None):
            stack.close()

    def forward(self, im: torch.Tensor) -> np.ndarray | list[torch.Tensor]:
        """Run Hailo inference and return decoded detections, or dense outputs and prototypes for segmentation."""
        im = np.ascontiguousarray(np.clip(im.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8))
        results = self.model.infer({self.input_info.name: im})
        outputs = [results[x.name] for x in self.output_infos]
        if self.task == "segment":
            return self._decode_segment(outputs)
        return self._decode_raw(outputs) if not self.metadata.get("nms", False) else self._decode_nms(outputs[0])

    def _decode_nms(self, output: list) -> np.ndarray:
        """Convert Hailo per-class NMS output from normalized ``yxyx`` to pixel ``xyxy`` coordinates."""
        height, width = self.input_info.shape[:2]
        scale = np.array([width, height, width, height], dtype=np.float32)
        frames = []
        for detections in output:
            rows = []
            for class_id, class_detections in enumerate(detections):
                if len(class_detections):
                    class_detections = np.asarray(class_detections)
                    boxes = class_detections[:, [1, 0, 3, 2]] * scale
                    classes = np.full((len(boxes), 1), class_id, dtype=np.float32)
                    rows.append(np.concatenate((boxes, class_detections[:, 4:5], classes), axis=1))
            frame = np.concatenate(rows) if rows else np.empty((0, 6), dtype=np.float32)
            frames.append(frame[np.argsort(-frame[:, 4])[:300]])
        count = max(map(len, frames), default=0)
        predictions = np.zeros((len(frames), count, 6), dtype=np.float32)
        for i, frame in enumerate(frames):
            predictions[i, : len(frame)] = frame
        return predictions

    def _decode_segment(self, outputs: list[np.ndarray]) -> list[torch.Tensor]:
        """Decode raw segmentation tensors (reg, cls, coeff per scale + prototypes) for the predictor's NMS."""
        from ultralytics.utils.tal import dist2bbox, make_anchors

        proto = torch.from_numpy(outputs[-1]).permute(0, 3, 1, 2)
        k = len(outputs) - 1
        reg_maps = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in outputs[0:k:3]]
        cls_maps = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in outputs[1:k:3]]
        cof_maps = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in outputs[2:k:3]]
        if self._anchors is None:
            strides = [self.input_info.shape[0] / x.shape[2] for x in reg_maps]
            self._anchors = make_anchors(reg_maps, strides)
        anchors, stride_tensor = self._anchors
        dist = torch.cat([x.flatten(2) for x in reg_maps], 2)
        b, _, a = dist.shape
        proj = torch.arange(16, dtype=torch.float32)
        dist = (dist.view(b, 4, 16, a).permute(0, 3, 1, 2).softmax(3) * proj).sum(3)  # DFL
        boxes = dist2bbox(dist, anchors, xywh=True) * stride_tensor
        cls = torch.cat([x.flatten(2) for x in cls_maps], 2).transpose(1, 2)  # sigmoid baked in at export
        cof = torch.cat([x.flatten(2) for x in cof_maps], 2).transpose(1, 2)
        return [torch.cat((boxes, cls, cof), 2).transpose(1, 2), proto]

    def _decode_raw(self, outputs: list[np.ndarray]) -> np.ndarray:
        """Decode branch-first YOLO26 regression and class outputs."""
        from ultralytics.utils.tal import dist2bbox, make_anchors

        split = len(outputs) // 2
        box_maps = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in outputs[:split]]
        cls_maps = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in outputs[split:]]
        if self._anchors is None:
            strides = [self.input_info.shape[0] / x.shape[2] for x in box_maps]
            self._anchors = make_anchors(box_maps, strides)
        anchors, stride_tensor = self._anchors
        boxes = torch.cat([x.flatten(2) for x in box_maps], 2).transpose(1, 2)
        boxes = dist2bbox(boxes, anchors, xywh=False) * stride_tensor
        scores = torch.cat([x.flatten(2) for x in cls_maps], 2).transpose(1, 2).sigmoid()
        classes = scores.shape[2]
        anchor_index = scores.amax(-1).topk(min(300, scores.shape[1]), dim=1).indices[..., None]
        boxes = boxes.gather(1, anchor_index.repeat(1, 1, 4))
        scores = scores.gather(1, anchor_index.repeat(1, 1, classes))
        scores, index = scores.flatten(1).topk(min(300, scores.shape[1] * classes), dim=1)
        boxes = boxes.gather(1, (index // classes)[..., None].repeat(1, 1, 4))
        return torch.cat((boxes, scores[..., None], (index % classes)[..., None].float()), 2).numpy()
