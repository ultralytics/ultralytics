# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER, WINDOWS, YAML
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class QNNBackend(BaseBackend):
    """Qualcomm QNN inference backend for Snapdragon hardware.

    Loads and runs the QNN context binary produced by the Ultralytics QNN export (an ``*_qnn.onnx`` file inside a
    ``_qnn_model`` directory) using ONNX Runtime with the QNN Execution Provider. Inference runs on Qualcomm Snapdragon
    devices (Android, Windows on Snapdragon, or Qualcomm Linux boards) via the HTP (NPU), GPU, or CPU backend.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load a QNN context-binary model with ONNX Runtime's QNN Execution Provider.

        Args:
            weight (str | Path): Path to the ``*_qnn.onnx`` file or the ``_qnn_model`` directory containing it.

        Raises:
            OSError: If the QNN Execution Provider is not available in the active ONNX Runtime build.
            RuntimeError: If model loading fails.
        """
        check_requirements("onnxruntime-qnn")
        import onnxruntime

        available = onnxruntime.get_available_providers()
        if "QNNExecutionProvider" not in available:
            raise OSError(
                "QNNExecutionProvider is not available. Install 'onnxruntime-qnn' in an environment without a standard "
                "'onnxruntime' build and run on a Qualcomm Snapdragon device. Available providers: "
                f"{', '.join(available)}"
            )

        w = Path(weight)
        onnx_file = w if w.is_file() else next(w.rglob("*_qnn.onnx"))
        LOGGER.info(f"Loading {onnx_file} for Qualcomm QNN inference...")

        backend_lib = "QnnHtp.dll" if WINDOWS else "libQnnHtp.so"
        self.session = onnxruntime.InferenceSession(
            str(onnx_file),
            providers=["QNNExecutionProvider"],
            provider_options=[{"backend_path": backend_lib, "enable_htp_fp16_precision": "1"}],
        )
        self.output_names = [x.name for x in self.session.get_outputs()]

        # Load metadata saved alongside the model during export
        metadata_file = onnx_file.parent / "metadata.yaml"
        if metadata_file.exists():
            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> list:
        """Run inference on the Qualcomm QNN runtime.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (list): Model predictions as a list of output arrays.
        """
        return self.session.run(self.output_names, {self.session.get_inputs()[0].name: im.cpu().numpy()})
