# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import ARM64, LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class PaddleBackend(BaseBackend):
    """Baidu PaddlePaddle inference backend.

    Loads and runs inference with Baidu PaddlePaddle models (*_paddle_model/ directories). Supports both CPU and GPU
    execution with automatic device configuration and memory pool initialization.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load a Baidu PaddlePaddle model from a directory containing .json and .pdiparams files.

        Args:
            weight (str | Path): Path to the model directory or .pdiparams file.
        """
        cuda = isinstance(self.device, torch.device) and torch.cuda.is_available() and self.device.type != "cpu"
        LOGGER.info(f"Loading {weight} for PaddlePaddle inference...")
        if cuda:
            check_requirements("paddlepaddle-gpu>=3.0.0,!=3.3.0")
        elif ARM64:
            check_requirements("paddlepaddle==3.0.0")
        else:
            check_requirements("paddlepaddle>=3.0.0,!=3.3.0")

        import paddle.inference as pdi

        w = Path(weight)
        model_file, params_file = None, None

        if w.is_dir():
            model_file = next(w.rglob("*.json"), None)
            params_file = next(w.rglob("*.pdiparams"), None)
        elif w.suffix == ".pdiparams":
            model_file = w.with_name("model.json")
            params_file = w

        if not (model_file and params_file and model_file.is_file() and params_file.is_file()):
            raise FileNotFoundError(f"Paddle model not found in {w}. Both .json and .pdiparams files are required.")

        config = pdi.Config(str(model_file), str(params_file))
        if cuda:
            config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=self.device.index or 0)

        self.predictor = pdi.create_predictor(config)
        self.input_handle = self.predictor.get_input_handle(self.predictor.get_input_names()[0])
        self.output_names = self.predictor.get_output_names()

        # Load metadata
        metadata_file = (w if w.is_dir() else w.parent) / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> list[np.ndarray]:
        """Run Baidu PaddlePaddle inference.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (list[np.ndarray]): Model predictions as a list of numpy arrays, one per output handle.
        """
        self.input_handle.copy_from_cpu(im.cpu().numpy().astype(np.float32))
        self.predictor.run()
        return [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
