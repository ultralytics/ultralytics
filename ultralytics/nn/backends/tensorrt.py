# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from collections import OrderedDict, namedtuple
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import IS_JETSON, LOGGER, PYTHON_VERSION
from ultralytics.utils.checks import check_requirements, check_tensorrt, check_version

from .base import BaseBackend


class TensorRTBackend(BaseBackend):
    """NVIDIA TensorRT inference backend for GPU-accelerated deployment.

    Loads and runs inference with NVIDIA TensorRT serialized engines (.engine files). Supports both TensorRT 7-9 and
    TensorRT 10+ APIs, dynamic input shapes, FP16 precision, and DLA core offloading.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load an NVIDIA TensorRT engine from a serialized .engine file.

        Args:
            weight (str | Path): Path to the .engine file with optional embedded metadata.
        """
        LOGGER.info(f"Loading {weight} for TensorRT inference...")

        if IS_JETSON and check_version(PYTHON_VERSION, "<=3.8.10"):
            check_requirements("numpy==1.23.5")

        try:
            import tensorrt as trt
        except ImportError:
            check_tensorrt()
            import tensorrt as trt

        check_version(trt.__version__, ">=7.0.0", hard=True)
        check_version(trt.__version__, "!=10.2.0", msg="https://github.com/ultralytics/ultralytics/pull/24367")

        if self.device.type == "cpu":
            self.device = torch.device("cuda:0")

        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        logger = trt.Logger(trt.Logger.INFO)

        # Read engine file
        with open(weight, "rb") as f, trt.Runtime(logger) as runtime:
            try:
                meta_len = int.from_bytes(f.read(4), byteorder="little")
                metadata = json.loads(f.read(meta_len).decode("utf-8"))
                dla = metadata.get("dla", None)
                if dla is not None:
                    runtime.DLA_core = int(dla)
            except UnicodeDecodeError:
                f.seek(0)
                metadata = None
            engine = runtime.deserialize_cuda_engine(f.read())
            self.apply_metadata(metadata)
        try:
            self.context = engine.create_execution_context()
        except Exception as e:
            LOGGER.error("TensorRT model exported with a different version than expected\n")
            raise e

        # Setup bindings
        self.bindings = OrderedDict()
        self.output_names = []
        self.fp16 = False
        self.dynamic = False
        self.is_trt10 = not hasattr(engine, "num_bindings")
        num = range(engine.num_io_tensors) if self.is_trt10 else range(engine.num_bindings)

        for i in num:
            if self.is_trt10:
                name = engine.get_tensor_name(i)
                dtype = trt.nptype(engine.get_tensor_dtype(name))
                is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                shape = tuple(engine.get_tensor_shape(name))
                profile_shape = tuple(engine.get_tensor_profile_shape(name, 0)[2]) if is_input else None
            else:
                name = engine.get_binding_name(i)
                dtype = trt.nptype(engine.get_binding_dtype(i))
                is_input = engine.binding_is_input(i)
                shape = tuple(engine.get_binding_shape(i))
                profile_shape = tuple(engine.get_profile_shape(0, i)[1]) if is_input else None

            if is_input:
                if -1 in shape:
                    self.dynamic = True
                    if self.is_trt10:
                        self.context.set_input_shape(name, profile_shape)
                    else:
                        self.context.set_binding_shape(i, profile_shape)
                if dtype == np.float16:
                    self.fp16 = True
            else:
                self.output_names.append(name)

            shape = (
                tuple(self.context.get_tensor_shape(name))
                if self.is_trt10
                else tuple(self.context.get_binding_shape(i))
            )
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))

        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.model = engine

    def forward(self, im: torch.Tensor) -> list[torch.Tensor]:
        """Run NVIDIA TensorRT inference with dynamic shape handling.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format on the CUDA device.

        Returns:
            (list[torch.Tensor]): Model predictions as a list of tensors on the CUDA device.
        """
        if self.dynamic and im.shape != self.bindings["images"].shape:
            if self.is_trt10:
                self.context.set_input_shape("images", im.shape)
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name in self.output_names:
                    self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(name)))
            else:
                i = self.model.get_binding_index("images")
                self.context.set_binding_shape(i, im.shape)
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))

        s = self.bindings["images"].shape
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"

        self.binding_addrs["images"] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        return [self.bindings[x].data for x in sorted(self.output_names)]
