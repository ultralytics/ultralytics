# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class QNNBackend(BaseBackend):
    """Qualcomm QNN inference backend for Snapdragon hardware.

    Loads and runs the QNN context binary produced by the Ultralytics QNN export (`*_qnn.onnx`) using ONNX Runtime with
    the QNN Execution Provider plugin (`onnxruntime-qnn`). Inference runs on Qualcomm Snapdragon devices (Android,
    Windows on Snapdragon, or Qualcomm Linux boards) via the HTP (NPU) backend.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load a QNN context-binary model with ONNX Runtime's QNN Execution Provider plugin.

        Args:
            weight (str | Path): Path to the `*_qnn.onnx` file.

        Raises:
            OSError: If the QNN Execution Provider cannot be registered (e.g. not on Snapdragon hardware).
        """
        check_requirements("onnxruntime-qnn")
        import onnxruntime

        from ultralytics.utils.export.qnn import qnn_library_paths

        onnx_file = Path(weight)
        LOGGER.info(f"Loading {onnx_file} for Qualcomm QNN inference...")

        # Register the QNN EP (libraries resolved from the plugin helper or the onnxruntime/capi bundle) and select
        # it; ep_library is None when QNN is already built into ONNX Runtime and needs no plugin registration
        ep_name = "QNNExecutionProvider"
        ep_library, htp_backend = qnn_library_paths()
        ep_options = {"backend_path": htp_backend}
        options = onnxruntime.SessionOptions()
        if ep_library:
            onnxruntime.register_execution_provider_library(ep_name, ep_library)
            devices = [d for d in onnxruntime.get_ep_devices() if d.ep_name == ep_name]
            if not devices:
                raise OSError(
                    "QNN Execution Provider registered but no QNN devices were found. Run on a Qualcomm Snapdragon "
                    "device with 'onnxruntime-qnn' installed."
                )
            options.add_provider_for_devices(devices, ep_options)
            self.session = onnxruntime.InferenceSession(str(onnx_file), sess_options=options)
        else:
            self.session = onnxruntime.InferenceSession(
                str(onnx_file), sess_options=options, providers=[ep_name], provider_options=[ep_options]
            )
        self.output_names = [x.name for x in self.session.get_outputs()]

        metadata_map = self.session.get_modelmeta().custom_metadata_map
        if metadata_map:
            self.apply_metadata(dict(metadata_map))

    def forward(self, im: torch.Tensor) -> list:
        """Run inference on the Qualcomm QNN runtime.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (list): Model predictions as a list of output arrays.
        """
        return self.session.run(self.output_names, {self.session.get_inputs()[0].name: im.cpu().numpy()})
