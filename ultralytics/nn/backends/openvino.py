# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import ARM64, LINUX, LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class OpenVINOBackend(BaseBackend):
    """Intel OpenVINO inference backend for Intel hardware acceleration.

    Loads and runs inference with Intel OpenVINO IR models (*_openvino_model/ directories). Supports automatic device
    selection, Intel-specific device targeting, and async inference for throughput optimization.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load an Intel OpenVINO IR model from a .xml/.bin file pair or model directory.

        Args:
            weight (str | Path): Path to the .xml file or directory containing OpenVINO model files.
        """
        LOGGER.info(f"Loading {weight} for OpenVINO inference...")
        check_requirements("openvino>=2024.0.0")
        import openvino as ov

        core = ov.Core()
        fallback_device = "CPU" if core.available_devices == ["CPU"] else "AUTO"
        device_name = fallback_device

        if isinstance(self.device, str) and self.device.startswith("intel"):
            device_name = self.device.split(":")[1].upper()
            self.device = torch.device("cpu")
            if device_name not in core.available_devices:
                LOGGER.warning(f"OpenVINO device '{device_name}' not available. Using '{fallback_device}' instead.")
                device_name = fallback_device

        w = Path(weight)
        if not w.is_file():
            w = next(w.glob("*.xml"))

        ov_model = core.read_model(model=str(w), weights=w.with_suffix(".bin"))
        if ov_model.get_parameters()[0].get_layout().empty:
            ov_model.get_parameters()[0].set_layout(ov.Layout("NCHW"))

        # Load metadata
        metadata_file = w.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

        # Set inference mode
        self.inference_mode = "CUMULATIVE_THROUGHPUT" if self.dynamic and self.batch > 1 else "LATENCY"
        config = {"PERFORMANCE_HINT": self.inference_mode}
        if LINUX and ARM64 and device_name == "CPU":
            config["EXECUTION_MODE_HINT"] = ov.properties.hint.ExecutionMode.ACCURACY
            config["INFERENCE_PRECISION_HINT"] = ov.Type.f32

        self.ov_compiled_model = core.compile_model(
            ov_model,
            device_name=device_name,
            config=config,
        )
        LOGGER.info(
            f"Using OpenVINO {self.inference_mode} mode for batch={self.batch} inference on "
            f"{', '.join(self.ov_compiled_model.get_property('EXECUTION_DEVICES'))}..."
        )
        self.input_name = self.ov_compiled_model.input().get_any_name()
        self.ov = ov

    def forward(self, im: torch.Tensor) -> list[np.ndarray]:
        """Run Intel OpenVINO inference with sync or async execution based on inference mode.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (list[np.ndarray]): Model predictions as a list of numpy arrays, one per output layer.
        """
        im = im.cpu().numpy().astype(np.float32)

        if self.inference_mode in {"THROUGHPUT", "CUMULATIVE_THROUGHPUT"}:
            # Async inference for larger batch sizes
            n = im.shape[0]
            results = [None] * n

            def callback(request, userdata):
                """Store async inference result in the preallocated results list at the given index."""
                results[userdata] = request.results

            async_queue = self.ov.AsyncInferQueue(self.ov_compiled_model)
            async_queue.set_callback(callback)

            for i in range(n):
                async_queue.start_async(inputs={self.input_name: im[i : i + 1]}, userdata=i)
            async_queue.wait_all()

            y = [list(r.values()) for r in results]
            y = [np.concatenate(x) for x in zip(*y)]
        else:
            # Sync inference for LATENCY mode
            y = list(self.ov_compiled_model(im).values())
        return y
