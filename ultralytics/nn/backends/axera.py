# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER

from .base import BaseBackend


class AxeraBackend(BaseBackend):
    """AXERA inference backend for compiled ``.axmodel`` artifacts.

    Runs on the AXERA NPU through ``axengine`` when the device runtime is installed, and falls back to the
    ``axnn.simulator`` x86 reference execution otherwise. Both paths share the same artifact contract: uint8 RGB NHWC
    input and float32 outputs in metadata order.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load an AXERA model from a ``_axera_model`` directory or an ``.axmodel`` file.

        Args:
            weight (str | Path): Path to the AXERA artifact directory or ``.axmodel`` file.
        """
        w = Path(weight)
        if w.is_file():
            model_file = w
        else:
            model_file = w / "compiled.axmodel"
            if not model_file.is_file():
                model_file = next(w.rglob("*.axmodel"))
        metadata = self.read_metadata(w)

        try:  # real NPU runtime first
            import axengine
        except ImportError:
            axengine = None

        if axengine is not None:
            LOGGER.info(f"Loading {weight} for AXERA NPU inference...")
            self.model = axengine.InferenceSession(str(model_file))
            inputs = [x.name for x in self.model.get_inputs()]
            self.output_names = [x.name for x in self.model.get_outputs()]
            self.real_npu = True
        else:
            LOGGER.info(f"Loading {weight} for AXERA simulator inference...")
            from axnn.simulator.sim import InferenceSession, RunMode
            from axnn.yamain.command.build import TargetPlatform

            target = str(metadata.get("axera", {}).get("target", "AX650")).upper()
            try:
                members = TargetPlatform.__members__
                target_platform = TargetPlatform[target] if target in members else TargetPlatform.from_backend(target)
            except (KeyError, ValueError) as e:
                raise ValueError(f"AXERA simulator does not support target '{target}'.") from e

            self.model = InferenceSession(str(model_file), run_mode=RunMode.Reference, target_hardware=target_platform)
            inputs = list(self.model.input_names)
            self.output_names = list(self.model.output_names)
            self.real_npu = False

        if len(inputs) != 1:
            raise RuntimeError(f"AXERA backend expects one input, but the runtime exposes {len(inputs)}.")
        self.input_name = inputs[0]

        self.tail = None
        artifact_dir = w if w.is_dir() else w.parent
        if (artifact_dir / "tail.onnx").is_file():  # INT8 stops at the head convs; the decode runs here in FP32
            import onnxruntime

            self.tail = onnxruntime.InferenceSession(
                str(artifact_dir / "tail.onnx"), providers=["CPUExecutionProvider"]
            )
        self.apply_metadata(metadata)
        self._warmup()

    def _infer(self, im: np.ndarray) -> list[np.ndarray]:
        """Run one inference and return the outputs in metadata order."""
        if self.real_npu:
            out = self.model.run(None, {self.input_name: im})
            return list(out) if isinstance(out, (list, tuple)) else [out[name] for name in self.output_names]
        result = self.model.run({self.input_name: im})
        return [result[name] for name in self.output_names]

    def _warmup(self) -> None:
        """Run one dummy inference so the first real prediction excludes runtime initialization."""
        imgsz = getattr(self, "imgsz", None)
        if not imgsz:
            return
        layout = self.metadata.get("axera", {}).get("inputs", [{}])[0].get("layout", "NHWC").upper()
        shape = (1, *tuple(imgsz), self.channels) if layout == "NHWC" else (1, self.channels, *tuple(imgsz))
        self._infer(np.zeros(shape, dtype=np.uint8))

    def forward(self, im: torch.Tensor) -> list[np.ndarray]:
        """Run inference and return outputs in YOLO order.

        Args:
            im (torch.Tensor): Input tensor in BHWC format, RGB and normalized to [0, 1].

        Returns:
            (list[np.ndarray]): Raw model outputs, with the FP32 ``tail.onnx`` decode applied when present.
        """
        inp = self.metadata.get("axera", {}).get("inputs", [{}])[0]
        mean = np.asarray(inp.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32)
        std = np.asarray(inp.get("std", [1.0, 1.0, 1.0]), dtype=np.float32)
        im = im.detach().cpu().numpy() * std + mean  # undo the predictor normalization back to [0, 1] pixels
        im = (im * 255).round().clip(0, 255).astype(np.uint8)
        im = np.ascontiguousarray(im)  # the compiled graph expects RGB; it embeds no color conversion
        outputs = self._infer(im)
        if self.tail is None:
            return outputs
        named = dict(zip(self.output_names, outputs))
        feeds = {
            tensor.name: np.asarray(named.get(tensor.name, outputs[i]), dtype=np.float32)
            for i, tensor in enumerate(self.tail.get_inputs())
        }
        return list(self.tail.run(None, feeds))
