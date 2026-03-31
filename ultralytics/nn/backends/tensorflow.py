# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import ast
import json
import platform
import zipfile
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER

from .base import BaseBackend


class TensorFlowBackend(BaseBackend):
    """Google TensorFlow inference backend supporting multiple serialization formats.

    Loads and runs inference with Google TensorFlow models in SavedModel, GraphDef (.pb), TFLite (.tflite), and Edge TPU
    formats. Handles quantized model dequantization and task-specific output formatting.
    """

    def __init__(self, weight: str | Path, device: torch.device, fp16: bool = False, format: str = "saved_model"):
        """Initialize the Google TensorFlow backend.

        Args:
            weight (str | Path): Path to the SavedModel directory, .pb file, or .tflite file.
            device (torch.device): Device to run inference on.
            fp16 (bool): Whether to use FP16 half-precision inference.
            format (str): Model format, one of "saved_model", "pb", "tflite", or "edgetpu".
        """
        assert format in {"saved_model", "pb", "tflite", "edgetpu"}, f"Unsupported TensorFlow format: {format}."
        self.format = format
        super().__init__(weight, device, fp16)

    def load_model(self, weight: str | Path) -> None:
        """Load a Google TensorFlow model in SavedModel, GraphDef, TFLite, or Edge TPU format.

        Args:
            weight (str | Path): Path to the model file or directory.
        """
        if self.format in {"saved_model", "pb"}:
            import tensorflow as tf

        if self.format == "saved_model":
            LOGGER.info(f"Loading {weight} for TensorFlow SavedModel inference...")
            self.model = tf.saved_model.load(weight)
            # Load metadata
            metadata_file = Path(weight) / "metadata.yaml"
            if metadata_file.exists():
                from ultralytics.utils import YAML

                self.apply_metadata(YAML.load(metadata_file))
        elif self.format == "pb":
            LOGGER.info(f"Loading {weight} for TensorFlow GraphDef inference...")
            from ultralytics.utils.export.tensorflow import gd_outputs

            def wrap_frozen_graph(gd, inputs, outputs):
                """Wrap a TensorFlow frozen graph for inference by pruning to specified input/output nodes."""
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            gd = tf.Graph().as_graph_def()
            with open(weight, "rb") as f:
                gd.ParseFromString(f.read())
            self.frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))

            # Try to find metadata
            try:
                metadata_file = next(
                    Path(weight).resolve().parent.rglob(f"{Path(weight).stem}_saved_model*/metadata.yaml")
                )
                from ultralytics.utils import YAML

                self.apply_metadata(YAML.load(metadata_file))
            except StopIteration:
                pass
        else:  # tflite and edgetpu
            try:
                from tflite_runtime.interpreter import Interpreter, load_delegate

                self.tf = None
            except ImportError:
                import tensorflow as tf

                self.tf = tf
                Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate

            if self.format == "edgetpu":
                device = self.device[3:] if str(self.device).startswith("tpu") else ":0"
                LOGGER.info(f"Loading {weight} on device {device[1:]} for TensorFlow Lite Edge TPU inference...")
                delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                    platform.system()
                ]
                self.interpreter = Interpreter(
                    model_path=str(weight),
                    experimental_delegates=[load_delegate(delegate, options={"device": device})],
                )
                self.device = torch.device("cpu")  # Edge TPU runs on CPU from PyTorch's perspective
            else:
                LOGGER.info(f"Loading {weight} for TensorFlow Lite inference...")
                self.interpreter = Interpreter(model_path=weight)

            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # Load metadata
            try:
                with zipfile.ZipFile(weight, "r") as zf:
                    name = zf.namelist()[0]
                    contents = zf.read(name).decode("utf-8")
                    if name == "metadata.json":
                        self.apply_metadata(json.loads(contents))
                    else:
                        self.apply_metadata(ast.literal_eval(contents))
            except (zipfile.BadZipFile, SyntaxError, ValueError, json.JSONDecodeError):
                pass

    def forward(self, im: torch.Tensor) -> list[np.ndarray]:
        """Run Google TensorFlow inference with format-specific execution and output post-processing.

        Args:
            im (torch.Tensor): Input image tensor in BHWC format (converted from BCHW by AutoBackend).

        Returns:
            (list[np.ndarray]): Model predictions as a list of numpy arrays.
        """
        im = im.cpu().numpy()
        if self.format == "saved_model":
            y = self.model.serving_default(im)
            if not isinstance(y, list):
                y = [y]
        elif self.format == "pb":
            import tensorflow as tf

            y = self.frozen_func(x=tf.constant(im))
        else:
            h, w = im.shape[1:3]

            details = self.input_details[0]
            is_int = details["dtype"] in {np.int8, np.int16}

            if is_int:
                scale, zero_point = details["quantization"]
                im = (im / scale + zero_point).astype(details["dtype"])

            self.interpreter.set_tensor(details["index"], im)
            self.interpreter.invoke()

            y = []
            for output in self.output_details:
                x = self.interpreter.get_tensor(output["index"])
                if is_int:
                    scale, zero_point = output["quantization"]
                    x = (x.astype(np.float32) - zero_point) * scale
                if x.ndim == 3:
                    # Denormalize xywh by image size
                    if x.shape[-1] == 6 or self.end2end:
                        x[:, :, [0, 2]] *= w
                        x[:, :, [1, 3]] *= h
                        if self.task == "pose":
                            x[:, :, 6::3] *= w
                            x[:, :, 7::3] *= h
                    else:
                        x[:, [0, 2]] *= w
                        x[:, [1, 3]] *= h
                        if self.task == "pose":
                            x[:, 5::3] *= w
                            x[:, 6::3] *= h
                y.append(x)

        if self.task == "segment":  # segment with (det, proto) output order reversed
            if len(y[1].shape) != 4:
                y = list(reversed(y))  # should be y = (1, 116, 8400), (1, 160, 160, 32)
            if y[1].shape[-1] == 6:  # end-to-end model
                y = [y[1]]
            else:
                y[1] = np.transpose(y[1], (0, 3, 1, 2))  # should be y = (1, 116, 8400), (1, 32, 160, 160)
        return [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
