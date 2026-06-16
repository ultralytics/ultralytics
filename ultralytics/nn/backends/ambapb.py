# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER

from .base import BaseBackend


class AmbaPBBackend(BaseBackend):
    """Ambarella PB inference backend using CVFlow."""

    def __init__(
        self,
        weight: str | Path,
        device: torch.device | str,
        fp16: bool = False,
        cvb_infer_mode: str = "acinf",
    ):
        """Initialize the Ambarella PB backend.

        Args:
            weight (str | Path): Path to the Ambarella PB model.
            device (torch.device | str): CUDA device to run inference on.
            fp16 (bool): Whether FP16 inference was requested. Ambarella PB uses model-defined precision.
            cvb_infer_mode (str): CVFlow backend inference mode, e.g. "acinf" or "ades".
        """
        self.cvb_infer_mode = cvb_infer_mode
        self.debug = os.getenv("ULTRALYTICS_AMBAPB_DEBUG", "").lower() in {"1", "true", "yes", "on"}
        self._debug_calls = 0
        super().__init__(weight, device, fp16)
        # Precision is baked into the compiled PB model, so ignore the requested fp16 and
        # prevent AutoBackend from half()-ing inputs before forward().
        self.fp16 = False

    def load_model(self, weight: str | Path) -> None:
        """Load an Ambarella PB model with CVFlow.

        Args:
            weight (str | Path): Path to the Ambarella PB model file.
        """
        LOGGER.info(f"Loading {weight} for Ambarella PB inference...")
        self._import_cvflowbackend()

        InferenceSession = importlib.import_module("cvflowbackend.evaluation_stage").InferenceSession

        self.session = InferenceSession(
            str(weight),
            mode=self.cvb_infer_mode,
            cuda_devices=self._cuda_device_id(),
        )

        metadata_file = Path(weight).parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> np.ndarray:
        """Run Ambarella PB inference.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (np.ndarray): Model predictions reshaped to batch-first format.
        """
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        input_info = inputs[0]

        if im.is_cuda:
            im = im.cpu()
        im_np = np.multiply(im.numpy(), 255, dtype=np.float32).astype(input_info.dtype, copy=False)
        # CVFlow expects one array per batch item with leading batch dim.
        im_list = list(np.split(im_np, im_np.shape[0], axis=0))

        self.session.set_inputs({input_info.name: im_list}, batch_size=im_np.shape[0])
        output = self.session.run(fwd_quantized=True, dequantize=True)
        output = [self._as_batch_output(output[x.name], im_np.shape[0]) for x in outputs]
        formatted = self._format_outputs(output)
        self._log_output_shapes(im_np.shape, output, formatted)
        return formatted

    @staticmethod
    def _as_batch_output(output: list | np.ndarray, batch: int) -> np.ndarray:
        """Convert a CVFlow output to a batch-first NumPy array."""
        output_dtype = output[0].dtype if len(output) else np.float32
        output = np.asarray(output, dtype=output_dtype)
        if output.shape[0] == batch:
            # CVFlow may emit a singleton sequence dimension: [B, 1, C, N] -> [B, C, N].
            if output.ndim == 4 and output.shape[1] == 1:
                return output[:, 0]
            return output
        return np.reshape(output, [batch, -1])

    @staticmethod
    def _format_outputs(outputs: list[np.ndarray]) -> np.ndarray | list[np.ndarray]:
        """Format CVFlow outputs for Ultralytics postprocessing."""
        if len(outputs) == 1:
            return AmbaPBBackend._ensure_bcn(outputs[0])

        if all(x.ndim == 3 for x in outputs):
            shapes = {x.shape[0] for x in outputs}
            if len(shapes) == 1 and len({x.shape[-1] for x in outputs}) == 1:
                return AmbaPBBackend._ensure_bcn(np.concatenate(outputs, axis=1))
            if len(shapes) == 1 and len({x.shape[1] for x in outputs}) == 1:
                return AmbaPBBackend._ensure_bcn(np.concatenate(outputs, axis=2))

        return [AmbaPBBackend._ensure_bcn(x) for x in outputs]

    @staticmethod
    def _ensure_bcn(output: np.ndarray) -> np.ndarray:
        """Return standard detection outputs as BCN for Ultralytics NMS."""
        if output.ndim == 3 and output.shape[-1] != 6 and output.shape[1] > output.shape[2]:
            return output.transpose(0, 2, 1)
        return output

    def _log_output_shapes(
        self,
        input_shape: tuple[int, ...],
        raw_outputs: list[np.ndarray],
        formatted: np.ndarray | list[np.ndarray],
    ) -> None:
        """Log raw/formatted output shapes to debug NMS layout issues."""
        raw_shapes = [tuple(x.shape) for x in raw_outputs]
        formatted_shapes = [tuple(x.shape) for x in formatted] if isinstance(formatted, list) else [tuple(formatted.shape)]

        if self.debug and self._debug_calls < 10:
            LOGGER.info(
                f"Ambapb debug: input={input_shape} raw_outputs={raw_shapes} formatted={formatted_shapes}"
            )
            self._debug_calls += 1

        if len(formatted_shapes) == 1:
            shape = formatted_shapes[0]
            if len(shape) == 3 and shape[1] <= 4:
                LOGGER.warning(
                    f"Ambapb output channels look invalid for NMS: shape={shape} (expected BCN, e.g. [B,84,N])."
                )

    @staticmethod
    def _import_cvflowbackend() -> None:
        """Add the CVFlow backend library path reported by tv2 to sys.path."""
        try:
            libroot = subprocess.check_output(["tv2", "-libpath", "cvflowbackend"], stderr=subprocess.STDOUT)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise ImportError("Unable to resolve cvflowbackend with 'tv2 -libpath cvflowbackend'.") from e

        libroot = libroot.decode().strip()
        if libroot and libroot not in sys.path:
            sys.path.insert(0, libroot)

    def _cuda_device_id(self) -> int:
        """Return the CUDA device index expected by CVFlow."""
        if isinstance(self.device, torch.device):
            return self.device.index or 0
        if isinstance(self.device, str):
            if ":" in self.device:
                return int(self.device.rsplit(":", 1)[1])
            if self.device.isdigit():
                return int(self.device)
            return 0
        return int(self.device)
