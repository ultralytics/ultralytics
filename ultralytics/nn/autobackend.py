# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import ast
import json
import zipfile
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ultralytics.utils import IS_JETSON, LINUX, LOGGER, PYTHON_VERSION, ROOT, YAML, is_jetson
from ultralytics.utils.checks import check_requirements, check_suffix, check_version, check_yaml
from ultralytics.utils.downloads import attempt_download_asset, is_url
from ultralytics.utils.nms import non_max_suppression


def check_class_names(names: list | dict) -> dict[int, str]:
    """
    Check class names and convert to dict format if needed.

    Args:
        names (list | dict): Class names as list or dict format.

    Returns:
        (dict): Class names in dict format with integer keys and string values.

    Raises:
        KeyError: If class indices are invalid for the dataset size.
    """
    if isinstance(names, list):  # names is a list
        names = dict(enumerate(names))  # convert to dict
    if isinstance(names, dict):
        # Convert 1) string keys to int, i.e. '0' to 0, and non-string values to strings, i.e. True to 'True'
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(
                f"{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices "
                f"{min(names.keys())}-{max(names.keys())} defined in your dataset YAML."
            )
        if isinstance(names[0], str) and names[0].startswith("n0"):  # imagenet class codes, i.e. 'n01440764'
            names_map = YAML.load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]  # human-readable names
            names = {k: names_map[v] for k, v in names.items()}
    return names


def default_class_names(data: str | Path | None = None) -> dict[int, str]:
    """
    Apply default class names to an input YAML file or return numerical class names.

    Args:
        data (str | Path, optional): Path to YAML file containing class names.

    Returns:
        (dict): Dictionary mapping class indices to class names.
    """
    if data:
        try:
            return YAML.load(check_yaml(data))["names"]
        except Exception:
            pass
    return {i: f"class{i}" for i in range(999)}  # return default if above errors


class AutoBackend(nn.Module):
    """
    Handle dynamic backend selection for running inference using Ultralytics YOLO models.

    The AutoBackend class abstracts supported inference engines.

        Supported Formats and Naming Conventions:
            | Format       | File Suffix       |
            | ------------ | ----------------- |
            | PyTorch      | *.pt              |
            | TorchScript  | *.torchscript     |
            | ONNX Runtime | *.onnx            |
            | TensorRT     | *.engine          |

    Attributes:
        model (torch.nn.Module): The loaded YOLO model.
        device (torch.device): The device (CPU or GPU) on which the model is loaded.
        task (str): The type of task the model performs (detect, segment, classify, pose).
        names (dict): A dictionary of class names that the model can detect.
        stride (int): The model stride, typically 32 for YOLO models.
        fp16 (bool): Whether the model uses half-precision (FP16) inference.
        nhwc (bool): Whether the model expects NHWC input format instead of NCHW.
        pt (bool): Whether the model is a PyTorch model.
        jit (bool): Whether the model is a TorchScript model.
        onnx (bool): Whether the model is an ONNX model.
        xml (bool): Whether the model is an OpenVINO model.
        engine (bool): Whether the model is a TensorRT engine.

    Methods:
        forward: Run inference on an input image.
        from_numpy: Convert numpy array to tensor.
        warmup: Warm up the model with a dummy input.
        _model_type: Determine the model type from file path.

    Examples:
        >>> model = AutoBackend(model="yolo11n.pt", device="cuda")
        >>> results = model(img)
    """

    @torch.no_grad()
    def __init__(
        self,
        model: str | torch.nn.Module = "yolo11n.pt",
        device: torch.device = torch.device("cpu"),
        data: str | Path | None = None,
        fp16: bool = False,
        fuse: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize the AutoBackend for inference.

        Args:
            model (str | torch.nn.Module): Path to the model weights file or a module instance.
            device (torch.device): Device to run the model on.
            dnn (bool): Use OpenCV DNN module for ONNX inference.
            data (str | Path, optional): Path to the additional data.yaml file containing class names.
            fp16 (bool): Enable half-precision inference. Supported only on specific backends.
            fuse (bool): Fuse Conv2D + BatchNorm layers for optimization.
            verbose (bool): Enable verbose logging.
        """
        super().__init__()
        nn_module = isinstance(model, torch.nn.Module)
        pt, jit, onnx, engine = self._model_type("" if nn_module else model)
        fp16 &= pt or jit or onnx or engine or nn_module  # FP16
        nhwc = False  # Only kept backends use NCHW
        stride, ch = 32, 3  # default stride and channels
        end2end, dynamic = False, False
        metadata, task = None, None

        # Set device
        cuda = isinstance(device, torch.device) and torch.cuda.is_available() and device.type != "cpu"  # use CUDA
        if cuda and not any([nn_module, pt, jit, engine, onnx]):  # GPU dataloader formats
            device = torch.device("cpu")
            cuda = False

        # Download if not local
        w = attempt_download_asset(model) if pt else model  # weights path

        # PyTorch (in-memory or file)
        if nn_module or pt:
            if nn_module:
                pt = True
                if fuse:
                    if IS_JETSON and is_jetson(jetpack=5):
                        # Jetson Jetpack5 requires device before fuse https://github.com/ultralytics/ultralytics/pull/21028
                        model = model.to(device)
                    model = model.fuse(verbose=verbose)
                model = model.to(device)
            else:  # pt file
                from ultralytics.nn.tasks import load_checkpoint

                model, _ = load_checkpoint(model, device=device, fuse=fuse)  # load model, ckpt

            # Common PyTorch model processing
            if hasattr(model, "kpt_shape"):
                kpt_shape = model.kpt_shape  # pose-only
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            ch = model.yaml.get("channels", 3)
            for p in model.parameters():
                p.requires_grad = False
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

        # TorchScript
        elif jit:
            import torchvision  # noqa - https://github.com/ultralytics/ultralytics/pull/19747

            LOGGER.info(f"Loading {w} for TorchScript inference...")
            extra_files = {"config.txt": ""}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files["config.txt"]:  # load metadata dict
                metadata = json.loads(extra_files["config.txt"], object_hook=lambda x: dict(x.items()))

        # ONNX Runtime
        elif onnx:
            LOGGER.info(f"Loading {w} for ONNX Runtime inference...")
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
            import onnxruntime

            providers = ["CPUExecutionProvider"]
            if cuda:
                if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                    providers.insert(0, "CUDAExecutionProvider")
                else:  # Only log warning if CUDA was requested but unavailable
                    LOGGER.warning("Failed to start ONNX Runtime with CUDA. Using CPU...")
                    device = torch.device("cpu")
                    cuda = False
            LOGGER.info(f"Using ONNX Runtime {onnxruntime.__version__} {providers[0]}")
            session = onnxruntime.InferenceSession(w, providers=providers)

            output_names = [x.name for x in session.get_outputs()]
            metadata = session.get_modelmeta().custom_metadata_map
            dynamic = isinstance(session.get_outputs()[0].shape[0], str)
            fp16 = "float16" in session.get_inputs()[0].type
            if not dynamic:
                io = session.io_binding()
                bindings = []
                for output in session.get_outputs():
                    out_fp16 = "float16" in output.type
                    y_tensor = torch.empty(output.shape, dtype=torch.float16 if out_fp16 else torch.float32).to(device)
                    io.bind_output(
                        name=output.name,
                        device_type=device.type,
                        device_id=device.index if cuda else 0,
                        element_type=np.float16 if out_fp16 else np.float32,
                        shape=tuple(y_tensor.shape),
                        buffer_ptr=y_tensor.data_ptr(),
                    )
                    bindings.append(y_tensor)

        # TensorRT
        elif engine:
            LOGGER.info(f"Loading {w} for TensorRT inference...")

            if IS_JETSON and check_version(PYTHON_VERSION, "<=3.8.10"):
                # fix error: `np.bool` was a deprecated alias for the builtin `bool` for JetPack 4 and JetPack 5 with Python <= 3.8.10
                check_requirements("numpy==1.23.5")

            try:  # https://developer.nvidia.com/nvidia-tensorrt-download
                import tensorrt as trt
            except ImportError:
                if LINUX:
                    check_requirements("tensorrt>7.0.0,!=10.1.0")
                import tensorrt as trt
            check_version(trt.__version__, ">=7.0.0", hard=True)
            check_version(trt.__version__, "!=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")
            if device.type == "cpu":
                device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            # Read file
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                try:
                    meta_len = int.from_bytes(f.read(4), byteorder="little")  # read metadata length
                    metadata = json.loads(f.read(meta_len).decode("utf-8"))  # read metadata
                    dla = metadata.get("dla", None)
                    if dla is not None:
                        runtime.DLA_core = int(dla)
                except UnicodeDecodeError:
                    f.seek(0)  # engine file may lack embedded Ultralytics metadata
                model = runtime.deserialize_cuda_engine(f.read())  # read engine

            # Model context
            try:
                context = model.create_execution_context()
            except Exception as e:  # model is None
                LOGGER.error(f"TensorRT model exported with a different version than {trt.__version__}\n")
                raise e

            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            is_trt10 = not hasattr(model, "num_bindings")
            num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
            for i in num:
                if is_trt10:
                    name = model.get_tensor_name(i)
                    dtype = trt.nptype(model.get_tensor_dtype(name))
                    is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                    if is_input:
                        if -1 in tuple(model.get_tensor_shape(name)):
                            dynamic = True
                            context.set_input_shape(name, tuple(model.get_tensor_profile_shape(name, 0)[1]))
                        if dtype == np.float16:
                            fp16 = True
                    else:
                        output_names.append(name)
                    shape = tuple(context.get_tensor_shape(name))
                else:  # TensorRT < 10.0
                    name = model.get_binding_name(i)
                    dtype = trt.nptype(model.get_binding_dtype(i))
                    is_input = model.binding_is_input(i)
                    if model.binding_is_input(i):
                        if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                            dynamic = True
                            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[1]))
                        if dtype == np.float16:
                            fp16 = True
                    else:
                        output_names.append(name)
                    shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())

        # Any other format (unsupported)
        else:
            from ultralytics.engine.exporter import export_formats

            raise TypeError(
                f"model='{w}' is not a supported model format. Ultralytics supports: {export_formats()['Format']}\n"
                f"See https://docs.ultralytics.com/modes/predict for help."
            )

        # Load external metadata YAML
        if isinstance(metadata, (str, Path)) and Path(metadata).exists():
            metadata = YAML.load(metadata)
        if metadata and isinstance(metadata, dict):
            for k, v in metadata.items():
                if k in {"stride", "batch", "channels"}:
                    metadata[k] = int(v)
                elif k in {"imgsz", "names", "kpt_shape", "kpt_names", "args"} and isinstance(v, str):
                    metadata[k] = eval(v)
            stride = metadata["stride"]
            task = metadata["task"]
            batch = metadata["batch"]
            imgsz = metadata["imgsz"]
            names = metadata["names"]
            kpt_shape = metadata.get("kpt_shape")
            kpt_names = metadata.get("kpt_names")
            end2end = metadata.get("args", {}).get("nms", False)
            dynamic = metadata.get("args", {}).get("dynamic", dynamic)
            ch = metadata.get("channels", 3)
        elif not (pt or nn_module):
            LOGGER.warning(f"Metadata not found for 'model={w}'")

        # Check names
        if "names" not in locals():  # names missing
            names = default_class_names(data)
        names = check_class_names(names)

        self.__dict__.update(locals())  # assign all variables to self

    def forward(
        self,
        im: torch.Tensor,
        augment: bool = False,
        visualize: bool = False,
        embed: list | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Run inference on an AutoBackend model.

        Args:
            im (torch.Tensor): The image tensor to perform inference on.
            augment (bool): Whether to perform data augmentation during inference.
            visualize (bool): Whether to visualize the output predictions.
            embed (list, optional): A list of feature vectors/embeddings to return.
            **kwargs (Any): Additional keyword arguments for model configuration.

        Returns:
            (torch.Tensor | list[torch.Tensor]): The raw output tensor(s) from the model.
        """
        _b, _ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        # PyTorch
        if self.pt or self.nn_module:
            y = self.model(im, augment=augment, visualize=visualize, embed=embed, **kwargs)

        # TorchScript
        elif self.jit:
            y = self.model(im)

        # ONNX Runtime
        elif self.onnx:
            if self.dynamic:
                im = im.cpu().numpy()  # torch to numpy
                y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
            else:
                if not self.cuda:
                    im = im.cpu()
                self.io.bind_input(
                    name="images",
                    device_type=im.device.type,
                    device_id=im.device.index if im.device.type == "cuda" else 0,
                    element_type=np.float16 if self.fp16 else np.float32,
                    shape=tuple(im.shape),
                    buffer_ptr=im.data_ptr(),
                )
                self.session.run_with_iobinding(self.io)
                y = self.bindings

        # TensorRT
        elif self.engine:
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
            y = [self.bindings[x].data for x in sorted(self.output_names)]

        else:
            raise TypeError("Unsupported backend in forward path.")

        # for x in y:
        #     print(type(x), len(x)) if isinstance(x, (list, tuple)) else print(type(x), x.shape)  # debug shapes
        if isinstance(y, (list, tuple)):
            if len(self.names) == 999 and (self.task == "segment" or len(y) == 2):  # segments and names not defined
                nc = y[0].shape[1] - y[1].shape[1] - 4  # y = (1, 32, 160, 160), (1, 116, 8400)
                self.names = {i: f"class{i}" for i in range(nc)}
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x: np.ndarray) -> torch.Tensor:
        """
        Convert a numpy array to a tensor.

        Args:
            x (np.ndarray): The array to be converted.

        Returns:
            (torch.Tensor): The converted tensor
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz: tuple[int, int, int, int] = (1, 3, 640, 640)) -> None:
        """
        Warm up the model by running one forward pass with a dummy input.

        Args:
            imgsz (tuple): The shape of the dummy input tensor in the format (batch_size, channels, height, width)
        """
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.nn_module
        if any(warmup_types) and (self.device.type != "cpu"):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):
                self.forward(im)  # warmup model
                warmup_boxes = torch.rand(1, 84, 16, device=self.device)  # 16 boxes works best empirically
                warmup_boxes[:, :4] *= imgsz[-1]
                non_max_suppression(warmup_boxes)  # warmup NMS

    @staticmethod
    def _model_type(p: str = "path/to/model.pt") -> tuple[bool, bool, bool, bool]:
        """
        Take a path to a model file and return the model type.

        Args:
            p (str): Path to the model file.

        Returns:
            (tuple[bool, bool, bool, bool]): (pt, jit, onnx, engine)

        Examples:
            >>> model = AutoBackend(model="path/to/model.onnx")
            >>> model_type = model._model_type()  # returns "onnx"
        """
        name = Path(p).name if isinstance(p, str) else Path(p).name
        pt = name.endswith(".pt")
        jit = name.endswith(".torchscript")
        onnx = name.endswith(".onnx")
        engine = name.endswith(".engine")
        return pt, jit, onnx, engine
