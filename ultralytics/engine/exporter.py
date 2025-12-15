# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Export a YOLO PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit.

Format                  | `format=argument`         | Model
---                     | ---                       | ---
PyTorch                 | -                         | yolo11n.pt
TorchScript             | `torchscript`             | yolo11n.torchscript
ONNX                    | `onnx`                    | yolo11n.onnx
OpenVINO                | `openvino`                | yolo11n_openvino_model/
TensorRT                | `engine`                  | yolo11n.engine
CoreML                  | `coreml`                  | yolo11n.mlpackage
TensorFlow SavedModel   | `saved_model`             | yolo11n_saved_model/
TensorFlow GraphDef     | `pb`                      | yolo11n.pb
TensorFlow Lite         | `tflite`                  | yolo11n.tflite
TensorFlow Edge TPU     | `edgetpu`                 | yolo11n_edgetpu.tflite
TensorFlow.js           | `tfjs`                    | yolo11n_web_model/
PaddlePaddle            | `paddle`                  | yolo11n_paddle_model/
MNN                     | `mnn`                     | yolo11n.mnn
NCNN                    | `ncnn`                    | yolo11n_ncnn_model/
IMX                     | `imx`                     | yolo11n_imx_model/
RKNN                    | `rknn`                    | yolo11n_rknn_model/
ExecuTorch              | `executorch`              | yolo11n_executorch_model/
Axelera                 | `axelera`                 | yolo11n_axelera_model/

Requirements:
    $ pip install "ultralytics[export]"

Python:
    from ultralytics import YOLO
    model = YOLO('yolo11n.pt')
    results = model.export(format='onnx')

CLI:
    $ yolo mode=export model=yolo11n.pt format=onnx

Inference:
    $ yolo predict model=yolo11n.pt                 # PyTorch
                         yolo11n.torchscript        # TorchScript
                         yolo11n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                         yolo11n_openvino_model     # OpenVINO
                         yolo11n.engine             # TensorRT
                         yolo11n.mlpackage          # CoreML (macOS-only)
                         yolo11n_saved_model        # TensorFlow SavedModel
                         yolo11n.pb                 # TensorFlow GraphDef
                         yolo11n.tflite             # TensorFlow Lite
                         yolo11n_edgetpu.tflite     # TensorFlow Edge TPU
                         yolo11n_paddle_model       # PaddlePaddle
                         yolo11n.mnn                # MNN
                         yolo11n_ncnn_model         # NCNN
                         yolo11n_imx_model          # IMX
                         yolo11n_rknn_model         # RKNN
                         yolo11n_executorch_model   # ExecuTorch
                         yolo11n_axelera_model      # Axelera

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolo11n_web_model public/yolo11n_web_model
    $ npm start
"""

import json
import os
import re
import shutil
import subprocess
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ultralytics import __version__
from ultralytics.cfg import TASK2DATA, get_cfg
from ultralytics.data import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import check_class_names, default_class_names
from ultralytics.nn.modules import C2f, Classify, Detect, RTDETRDecoder
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, SegmentationModel, WorldModel
from ultralytics.utils import (
    ARM64,
    DEFAULT_CFG,
    IS_COLAB,
    IS_DEBIAN_BOOKWORM,
    IS_DEBIAN_TRIXIE,
    IS_JETSON,
    IS_RASPBERRYPI,
    IS_UBUNTU,
    LINUX,
    LOGGER,
    MACOS,
    MACOS_VERSION,
    RKNN_CHIPS,
    SETTINGS,
    TORCH_VERSION,
    WINDOWS,
    YAML,
    callbacks,
    colorstr,
    get_default_args,
)
from ultralytics.utils.checks import (
    IS_PYTHON_3_10,
    IS_PYTHON_MINIMUM_3_9,
    check_apt_requirements,
    check_imgsz,
    check_requirements,
    check_version,
    is_intel,
    is_sudo_available,
)
from ultralytics.utils.export import (
    keras2pb,
    onnx2engine,
    onnx2saved_model,
    pb2tfjs,
    tflite2edgetpu,
    torch2imx,
    torch2onnx,
)
from ultralytics.utils.files import file_size
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.nms import TorchNMS
from ultralytics.utils.ops import Profile
from ultralytics.utils.patches import arange_patch
from ultralytics.utils.torch_utils import (
    TORCH_1_10,
    TORCH_1_11,
    TORCH_1_13,
    TORCH_2_1,
    TORCH_2_4,
    TORCH_2_9,
    select_device,
)


def export_formats():
    """Return a dictionary of Ultralytics YOLO export formats."""
    x = [
        ["PyTorch", "-", ".pt", True, True, []],
        ["TorchScript", "torchscript", ".torchscript", True, True, ["batch", "optimize", "half", "nms", "dynamic"]],
        ["ONNX", "onnx", ".onnx", True, True, ["batch", "dynamic", "half", "opset", "simplify", "nms"]],
        [
            "OpenVINO",
            "openvino",
            "_openvino_model",
            True,
            False,
            ["batch", "dynamic", "half", "int8", "nms", "fraction"],
        ],
        [
            "TensorRT",
            "engine",
            ".engine",
            False,
            True,
            ["batch", "dynamic", "half", "int8", "simplify", "nms", "fraction"],
        ],
        ["CoreML", "coreml", ".mlpackage", True, False, ["batch", "dynamic", "half", "int8", "nms"]],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True, ["batch", "int8", "keras", "nms"]],
        ["TensorFlow GraphDef", "pb", ".pb", True, True, ["batch"]],
        ["TensorFlow Lite", "tflite", ".tflite", True, False, ["batch", "half", "int8", "nms", "fraction"]],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", True, False, []],
        ["TensorFlow.js", "tfjs", "_web_model", True, False, ["batch", "half", "int8", "nms"]],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True, ["batch"]],
        ["MNN", "mnn", ".mnn", True, True, ["batch", "half", "int8"]],
        ["NCNN", "ncnn", "_ncnn_model", True, True, ["batch", "half"]],
        ["IMX", "imx", "_imx_model", True, True, ["int8", "fraction", "nms"]],
        ["RKNN", "rknn", "_rknn_model", False, False, ["batch", "name"]],
        ["ExecuTorch", "executorch", "_executorch_model", True, False, ["batch"]],
        ["Axelera", "axelera", "_axelera_model", False, False, ["batch", "int8"]],
    ]
    return dict(zip(["Format", "Argument", "Suffix", "CPU", "GPU", "Arguments"], zip(*x)))


def best_onnx_opset(onnx, cuda=False) -> int:
    """Return max ONNX opset for this torch version with ONNX fallback."""
    if TORCH_2_4:  # _constants.ONNX_MAX_OPSET first defined in torch 1.13
        opset = torch.onnx.utils._constants.ONNX_MAX_OPSET - 1  # use second-latest version for safety
        if cuda:
            opset -= 2  # fix CUDA ONNXRuntime NMS squeeze op errors
    else:
        version = ".".join(TORCH_VERSION.split(".")[:2])
        opset = {
            "1.8": 12,
            "1.9": 12,
            "1.10": 13,
            "1.11": 14,
            "1.12": 15,
            "1.13": 17,
            "2.0": 17,  # reduced from 18 to fix ONNX errors
            "2.1": 17,  # reduced from 19
            "2.2": 17,  # reduced from 19
            "2.3": 17,  # reduced from 19
            "2.4": 20,
            "2.5": 20,
            "2.6": 20,
            "2.7": 20,
            "2.8": 23,
        }.get(version, 12)
    return min(opset, onnx.defs.onnx_opset_version())


def validate_args(format, passed_args, valid_args):
    """Validate arguments based on the export format.

    Args:
        format (str): The export format.
        passed_args (Namespace): The arguments used during export.
        valid_args (list): List of valid arguments for the format.

    Raises:
        AssertionError: If an unsupported argument is used, or if the format lacks supported argument listings.
    """
    export_args = ["half", "int8", "dynamic", "keras", "nms", "batch", "fraction"]

    assert valid_args is not None, f"ERROR ❌️ valid arguments for '{format}' not listed."
    custom = {"batch": 1, "data": None, "device": None}  # exporter defaults
    default_args = get_cfg(DEFAULT_CFG, custom)
    for arg in export_args:
        not_default = getattr(passed_args, arg, None) != getattr(default_args, arg, None)
        if not_default:
            assert arg in valid_args, f"ERROR ❌️ argument '{arg}' is not supported for format='{format}'"


def try_export(inner_func):
    """YOLO export decorator, i.e. @try_export."""
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        """Export a model."""
        prefix = inner_args["prefix"]
        dt = 0.0
        try:
            with Profile() as dt:
                f = inner_func(*args, **kwargs)  # exported file/dir or tuple of (file/dir, *)
            path = f if isinstance(f, (str, Path)) else f[0]
            mb = file_size(path)
            assert mb > 0.0, "0.0 MB output model size"
            LOGGER.info(f"{prefix} export success ✅ {dt.t:.1f}s, saved as '{path}' ({mb:.1f} MB)")
            return f
        except Exception as e:
            LOGGER.error(f"{prefix} export failure {dt.t:.1f}s: {e}")
            raise e

    return outer_func


class Exporter:
    """A class for exporting YOLO models to various formats.

    This class provides functionality to export YOLO models to different formats including ONNX, TensorRT, CoreML,
    TensorFlow, and others. It handles format validation, device selection, model preparation, and the actual export
    process for each supported format.

    Attributes:
        args (SimpleNamespace): Configuration arguments for the exporter.
        callbacks (dict): Dictionary of callback functions for different export events.
        im (torch.Tensor): Input tensor for model inference during export.
        model (torch.nn.Module): The YOLO model to be exported.
        file (Path): Path to the model file being exported.
        output_shape (tuple): Shape of the model output tensor(s).
        pretty_name (str): Formatted model name for display purposes.
        metadata (dict): Model metadata including description, author, version, etc.
        device (torch.device): Device on which the model is loaded.
        imgsz (tuple): Input image size for the model.

    Methods:
        __call__: Main export method that handles the export process.
        get_int8_calibration_dataloader: Build dataloader for INT8 calibration.
        export_torchscript: Export model to TorchScript format.
        export_onnx: Export model to ONNX format.
        export_openvino: Export model to OpenVINO format.
        export_paddle: Export model to PaddlePaddle format.
        export_mnn: Export model to MNN format.
        export_ncnn: Export model to NCNN format.
        export_coreml: Export model to CoreML format.
        export_engine: Export model to TensorRT format.
        export_saved_model: Export model to TensorFlow SavedModel format.
        export_pb: Export model to TensorFlow GraphDef format.
        export_tflite: Export model to TensorFlow Lite format.
        export_edgetpu: Export model to Edge TPU format.
        export_tfjs: Export model to TensorFlow.js format.
        export_rknn: Export model to RKNN format.
        export_imx: Export model to IMX format.

    Examples:
        Export a YOLOv8 model to ONNX format
        >>> from ultralytics.engine.exporter import Exporter
        >>> exporter = Exporter()
        >>> exporter(model="yolov8n.pt")  # exports to yolov8n.onnx

        Export with specific arguments
        >>> args = {"format": "onnx", "dynamic": True, "half": True}
        >>> exporter = Exporter(overrides=args)
        >>> exporter(model="yolov8n.pt")
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the Exporter class.

        Args:
            cfg (str, optional): Path to a configuration file.
            overrides (dict, optional): Configuration overrides.
            _callbacks (dict, optional): Dictionary of callback functions.
        """
        self.args = get_cfg(cfg, overrides)
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    def __call__(self, model=None) -> str:
        """Export a model and return the final exported path as a string.

        Returns:
            (str): Path to the exported file or directory (the last export artifact).
        """
        t = time.time()
        fmt = self.args.format.lower()  # to lowercase
        if fmt in {"tensorrt", "trt"}:  # 'engine' aliases
            fmt = "engine"
        if fmt in {"mlmodel", "mlpackage", "mlprogram", "apple", "ios", "coreml"}:  # 'coreml' aliases
            fmt = "coreml"
        fmts_dict = export_formats()
        fmts = tuple(fmts_dict["Argument"][1:])  # available export formats
        if fmt not in fmts:
            import difflib

            # Get the closest match if format is invalid
            matches = difflib.get_close_matches(fmt, fmts, n=1, cutoff=0.6)  # 60% similarity required to match
            if not matches:
                msg = "Model is already in PyTorch format." if fmt == "pt" else f"Invalid export format='{fmt}'."
                raise ValueError(f"{msg} Valid formats are {fmts}")
            LOGGER.warning(f"Invalid export format='{fmt}', updating to format='{matches[0]}'")
            fmt = matches[0]
        flags = [x == fmt for x in fmts]
        if sum(flags) != 1:
            raise ValueError(f"Invalid export format='{fmt}'. Valid formats are {fmts}")
        (
            jit,
            onnx,
            xml,
            engine,
            coreml,
            saved_model,
            pb,
            tflite,
            edgetpu,
            tfjs,
            paddle,
            mnn,
            ncnn,
            imx,
            rknn,
            executorch,
            axelera,
        ) = flags  # export booleans

        is_tf_format = any((saved_model, pb, tflite, edgetpu, tfjs))

        # Device
        dla = None
        if engine and self.args.device is None:
            LOGGER.warning("TensorRT requires GPU export, automatically assigning device=0")
            self.args.device = "0"
        if engine and "dla" in str(self.args.device):  # convert int/list to str first
            device_str = str(self.args.device)
            dla = device_str.rsplit(":", 1)[-1]
            self.args.device = "0"  # update device to "0"
            assert dla in {"0", "1"}, f"Expected device 'dla:0' or 'dla:1', but got {device_str}."
        if imx and self.args.device is None and torch.cuda.is_available():
            LOGGER.warning("Exporting on CPU while CUDA is available, setting device=0 for faster export on GPU.")
            self.args.device = "0"  # update device to "0"
        self.device = select_device("cpu" if self.args.device is None else self.args.device)

        # Argument compatibility checks
        fmt_keys = fmts_dict["Arguments"][flags.index(True) + 1]
        validate_args(fmt, self.args, fmt_keys)
        if axelera:
            if not IS_PYTHON_3_10:
                raise SystemError("Axelera export only supported on Python 3.10.")
            if not self.args.int8:
                LOGGER.warning("Setting int8=True for Axelera mixed-precision export.")
                self.args.int8 = True
            if model.task not in {"detect"}:
                raise ValueError("Axelera export only supported for detection models.")
        if imx:
            if not self.args.int8:
                LOGGER.warning("IMX export requires int8=True, setting int8=True.")
                self.args.int8 = True
            if not self.args.nms and model.task in {"detect", "pose", "segment"}:
                LOGGER.warning("IMX export requires nms=True, setting nms=True.")
                self.args.nms = True
            if model.task not in {"detect", "pose", "classify", "segment"}:
                raise ValueError(
                    "IMX export only supported for detection, pose estimation, classification, and segmentation models."
                )
        if not hasattr(model, "names"):
            model.names = default_class_names()
        model.names = check_class_names(model.names)
        if self.args.half and self.args.int8:
            LOGGER.warning("half=True and int8=True are mutually exclusive, setting half=False.")
            self.args.half = False
        if self.args.half and jit and self.device.type == "cpu":
            LOGGER.warning(
                "half=True only compatible with GPU export for TorchScript, i.e. use device=0, setting half=False."
            )
            self.args.half = False
        self.imgsz = check_imgsz(self.args.imgsz, stride=model.stride, min_dim=2)  # check image size
        if self.args.optimize:
            assert not ncnn, "optimize=True not compatible with format='ncnn', i.e. use optimize=False"
            assert self.device.type == "cpu", "optimize=True not compatible with cuda devices, i.e. use device='cpu'"
        if rknn:
            if not self.args.name:
                LOGGER.warning(
                    "Rockchip RKNN export requires a missing 'name' arg for processor type. "
                    "Using default name='rk3588'."
                )
                self.args.name = "rk3588"
            self.args.name = self.args.name.lower()
            assert self.args.name in RKNN_CHIPS, (
                f"Invalid processor name '{self.args.name}' for Rockchip RKNN export. Valid names are {RKNN_CHIPS}."
            )
        if self.args.nms:
            assert not isinstance(model, ClassificationModel), "'nms=True' is not valid for classification models."
            assert not tflite or not ARM64 or not LINUX, "TFLite export with NMS unsupported on ARM64 Linux"
            assert not is_tf_format or TORCH_1_13, "TensorFlow exports with NMS require torch>=1.13"
            assert not onnx or TORCH_1_13, "ONNX export with NMS requires torch>=1.13"
            if getattr(model, "end2end", False) or isinstance(model.model[-1], RTDETRDecoder):
                LOGGER.warning("'nms=True' is not available for end2end models. Forcing 'nms=False'.")
                self.args.nms = False
            self.args.conf = self.args.conf or 0.25  # set conf default value for nms export
        if (engine or coreml or self.args.nms) and self.args.dynamic and self.args.batch == 1:
            LOGGER.warning(
                f"'dynamic=True' model with '{'nms=True' if self.args.nms else f'format={self.args.format}'}' requires max batch size, i.e. 'batch=16'"
            )
        if edgetpu:
            if not LINUX or ARM64:
                raise SystemError(
                    "Edge TPU export only supported on non-aarch64 Linux. See https://coral.ai/docs/edgetpu/compiler"
                )
            elif self.args.batch != 1:  # see github.com/ultralytics/ultralytics/pull/13420
                LOGGER.warning("Edge TPU export requires batch size 1, setting batch=1.")
                self.args.batch = 1
        if isinstance(model, WorldModel):
            LOGGER.warning(
                "YOLOWorld (original version) export is not supported to any format. "
                "YOLOWorldv2 models (i.e. 'yolov8s-worldv2.pt') only support export to "
                "(torchscript, onnx, openvino, engine, coreml) formats. "
                "See https://docs.ultralytics.com/models/yolo-world for details."
            )
            model.clip_model = None  # openvino int8 export error: https://github.com/ultralytics/ultralytics/pull/18445
        if self.args.int8 and not self.args.data:
            if axelera:
                self.args.data = "coco128.yaml"  # Axelera default to coco128.yaml
            else:
                self.args.data = DEFAULT_CFG.data or TASK2DATA[getattr(model, "task", "detect")]  # assign default data
            LOGGER.warning(
                f"INT8 export requires a missing 'data' arg for calibration. Using default 'data={self.args.data}'."
            )
        if tfjs and (ARM64 and LINUX):
            raise SystemError("TF.js exports are not currently supported on ARM64 Linux")
        # Recommend OpenVINO if export and Intel CPU
        if SETTINGS.get("openvino_msg"):
            if is_intel():
                LOGGER.info(
                    "💡 ProTip: Export to OpenVINO format for best performance on Intel hardware."
                    " Learn more at https://docs.ultralytics.com/integrations/openvino/"
                )
            SETTINGS["openvino_msg"] = False

        # Input
        im = torch.zeros(self.args.batch, model.yaml.get("channels", 3), *self.imgsz).to(self.device)
        file = Path(
            getattr(model, "pt_path", None) or getattr(model, "yaml_file", None) or model.yaml.get("yaml_file", "")
        )
        if file.suffix in {".yaml", ".yml"}:
            file = Path(file.name)

        # Update model
        model = deepcopy(model).to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()

        if imx:
            from ultralytics.utils.export.imx import FXModel

            model = FXModel(model, self.imgsz)
        if tflite or edgetpu:
            from ultralytics.utils.export.tensorflow import tf_wrapper

            model = tf_wrapper(model)
        for m in model.modules():
            if isinstance(m, Classify):
                m.export = True
            if isinstance(m, (Detect, RTDETRDecoder)):  # includes all Detect subclasses like Segment, Pose, OBB
                m.dynamic = self.args.dynamic
                m.export = True
                m.format = self.args.format
                m.max_det = self.args.max_det
                m.xyxy = self.args.nms and not coreml
                if hasattr(model, "pe") and hasattr(m, "fuse"):  # for YOLOE models
                    m.fuse(model.pe.to(self.device))
            elif isinstance(m, C2f) and not is_tf_format:
                # EdgeTPU does not support FlexSplitV while split provides cleaner ONNX graph
                m.forward = m.forward_split

        y = None
        for _ in range(2):  # dry runs
            y = NMSModel(model, self.args)(im) if self.args.nms and not coreml and not imx else model(im)
        if self.args.half and (onnx or jit) and self.device.type != "cpu":
            im, model = im.half(), model.half()  # to FP16

        # Assign
        self.im = im
        self.model = model
        self.file = file
        self.output_shape = (
            tuple(y.shape)
            if isinstance(y, torch.Tensor)
            else tuple(tuple(x.shape if isinstance(x, torch.Tensor) else []) for x in y)
        )
        self.pretty_name = Path(self.model.yaml.get("yaml_file", self.file)).stem.replace("yolo", "YOLO")
        data = model.args["data"] if hasattr(model, "args") and isinstance(model.args, dict) else ""
        description = f"Ultralytics {self.pretty_name} model {f'trained on {data}' if data else ''}"
        self.metadata = {
            "description": description,
            "author": "Ultralytics",
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
            "stride": int(max(model.stride)),
            "task": model.task,
            "batch": self.args.batch,
            "imgsz": self.imgsz,
            "names": model.names,
            "args": {k: v for k, v in self.args if k in fmt_keys},
            "channels": model.yaml.get("channels", 3),
        }  # model metadata
        if dla is not None:
            self.metadata["dla"] = dla  # make sure `AutoBackend` uses correct dla device if it has one
        if model.task == "pose":
            self.metadata["kpt_shape"] = model.model[-1].kpt_shape
            if hasattr(model, "kpt_names"):
                self.metadata["kpt_names"] = model.kpt_names

        LOGGER.info(
            f"\n{colorstr('PyTorch:')} starting from '{file}' with input shape {tuple(im.shape)} BCHW and "
            f"output shape(s) {self.output_shape} ({file_size(file):.1f} MB)"
        )
        self.run_callbacks("on_export_start")
        # Exports
        f = [""] * len(fmts)  # exported filenames
        if jit:  # TorchScript
            f[0] = self.export_torchscript()
        if engine:  # TensorRT required before ONNX
            f[1] = self.export_engine(dla=dla)
        if onnx:  # ONNX
            f[2] = self.export_onnx()
        if xml:  # OpenVINO
            f[3] = self.export_openvino()
        if coreml:  # CoreML
            f[4] = self.export_coreml()
        if is_tf_format:  # TensorFlow formats
            self.args.int8 |= edgetpu
            f[5], keras_model = self.export_saved_model()
            if pb or tfjs:  # pb prerequisite to tfjs
                f[6] = self.export_pb(keras_model=keras_model)
            if tflite:
                f[7] = self.export_tflite()
            if edgetpu:
                f[8] = self.export_edgetpu(tflite_model=Path(f[5]) / f"{self.file.stem}_full_integer_quant.tflite")
            if tfjs:
                f[9] = self.export_tfjs()
        if paddle:  # PaddlePaddle
            f[10] = self.export_paddle()
        if mnn:  # MNN
            f[11] = self.export_mnn()
        if ncnn:  # NCNN
            f[12] = self.export_ncnn()
        if imx:
            f[13] = self.export_imx()
        if rknn:
            f[14] = self.export_rknn()
        if executorch:
            f[15] = self.export_executorch()
        if axelera:
            f[16] = self.export_axelera()

        # Finish
        f = [str(x) for x in f if x]  # filter out '' and None
        if any(f):
            f = str(Path(f[-1]))
            square = self.imgsz[0] == self.imgsz[1]
            s = (
                ""
                if square
                else f"WARNING ⚠️ non-PyTorch val requires square images, 'imgsz={self.imgsz}' will not "
                f"work. Use export 'imgsz={max(self.imgsz)}' if val is required."
            )
            imgsz = self.imgsz[0] if square else str(self.imgsz)[1:-1].replace(" ", "")
            predict_data = f"data={data}" if model.task == "segment" and pb else ""
            q = "int8" if self.args.int8 else "half" if self.args.half else ""  # quantization
            LOGGER.info(
                f"\nExport complete ({time.time() - t:.1f}s)"
                f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                f"\nPredict:         yolo predict task={model.task} model={f} imgsz={imgsz} {q} {predict_data}"
                f"\nValidate:        yolo val task={model.task} model={f} imgsz={imgsz} data={data} {q} {s}"
                f"\nVisualize:       https://netron.app"
            )

        self.run_callbacks("on_export_end")
        return f  # path to final export artifact

    def get_int8_calibration_dataloader(self, prefix=""):
        """Build and return a dataloader for calibration of INT8 models."""
        LOGGER.info(f"{prefix} collecting INT8 calibration images from 'data={self.args.data}'")
        data = (check_cls_dataset if self.model.task == "classify" else check_det_dataset)(self.args.data)
        dataset = YOLODataset(
            data[self.args.split or "val"],
            data=data,
            fraction=self.args.fraction,
            task=self.model.task,
            imgsz=self.imgsz[0],
            augment=False,
            batch_size=self.args.batch,
        )
        n = len(dataset)
        if n < self.args.batch:
            raise ValueError(
                f"The calibration dataset ({n} images) must have at least as many images as the batch size "
                f"('batch={self.args.batch}')."
            )
        elif self.args.format == "axelera" and n < 100:
            LOGGER.warning(f"{prefix} >100 images required for Axelera calibration, found {n} images.")
        elif self.args.format != "axelera" and n < 300:
            LOGGER.warning(f"{prefix} >300 images recommended for INT8 calibration, found {n} images.")
        return build_dataloader(dataset, batch=self.args.batch, workers=0, drop_last=True)  # required for batch loading

    @try_export
    def export_torchscript(self, prefix=colorstr("TorchScript:")):
        """Export YOLO model to TorchScript format."""
        LOGGER.info(f"\n{prefix} starting export with torch {TORCH_VERSION}...")
        f = self.file.with_suffix(".torchscript")

        ts = torch.jit.trace(NMSModel(self.model, self.args) if self.args.nms else self.model, self.im, strict=False)
        extra_files = {"config.txt": json.dumps(self.metadata)}  # torch._C.ExtraFilesMap()
        if self.args.optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
            LOGGER.info(f"{prefix} optimizing for mobile...")
            from torch.utils.mobile_optimizer import optimize_for_mobile

            optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
        else:
            ts.save(str(f), _extra_files=extra_files)
        return f

    @try_export
    def export_onnx(self, prefix=colorstr("ONNX:")):
        """Export YOLO model to ONNX format."""
        requirements = ["onnx>=1.12.0,<2.0.0"]
        if self.args.simplify:
            requirements += ["onnxslim>=0.1.71", "onnxruntime" + ("-gpu" if torch.cuda.is_available() else "")]
        check_requirements(requirements)
        import onnx

        opset = self.args.opset or best_onnx_opset(onnx, cuda="cuda" in self.device.type)
        LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__} opset {opset}...")
        if self.args.nms:
            assert TORCH_1_13, f"'nms=True' ONNX export requires torch>=1.13 (found torch=={TORCH_VERSION})"

        f = str(self.file.with_suffix(".onnx"))
        output_names = ["output0", "output1"] if self.model.task == "segment" else ["output0"]
        dynamic = self.args.dynamic
        if dynamic:
            dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}  # shape(1,3,640,640)
            if isinstance(self.model, SegmentationModel):
                dynamic["output0"] = {0: "batch", 2: "anchors"}  # shape(1, 116, 8400)
                dynamic["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}  # shape(1,32,160,160)
            elif isinstance(self.model, DetectionModel):
                dynamic["output0"] = {0: "batch", 2: "anchors"}  # shape(1, 84, 8400)
            if self.args.nms:  # only batch size is dynamic with NMS
                dynamic["output0"].pop(2)
        if self.args.nms and self.model.task == "obb":
            self.args.opset = opset  # for NMSModel

        with arange_patch(self.args):
            torch2onnx(
                NMSModel(self.model, self.args) if self.args.nms else self.model,
                self.im,
                f,
                opset=opset,
                input_names=["images"],
                output_names=output_names,
                dynamic=dynamic or None,
            )

        # Checks
        model_onnx = onnx.load(f)  # load onnx model

        # Simplify
        if self.args.simplify:
            try:
                import onnxslim

                LOGGER.info(f"{prefix} slimming with onnxslim {onnxslim.__version__}...")
                model_onnx = onnxslim.slim(model_onnx)

            except Exception as e:
                LOGGER.warning(f"{prefix} simplifier failure: {e}")

        # Metadata
        for k, v in self.metadata.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)

        # IR version
        if getattr(model_onnx, "ir_version", 0) > 10:
            LOGGER.info(f"{prefix} limiting IR version {model_onnx.ir_version} to 10 for ONNXRuntime compatibility...")
            model_onnx.ir_version = 10

        # FP16 conversion for CPU export (GPU exports are already FP16 from model.half() during tracing)
        if self.args.half and self.args.format == "onnx" and self.device.type == "cpu":
            try:
                from onnxruntime.transformers import float16

                LOGGER.info(f"{prefix} converting to FP16...")
                model_onnx = float16.convert_float_to_float16(model_onnx, keep_io_types=True)
            except Exception as e:
                LOGGER.warning(f"{prefix} FP16 conversion failure: {e}")

        onnx.save(model_onnx, f)
        return f

    @try_export
    def export_openvino(self, prefix=colorstr("OpenVINO:")):
        """Export YOLO model to OpenVINO format."""
        # OpenVINO <= 2025.1.0 error on macOS 15.4+: https://github.com/openvinotoolkit/openvino/issues/30023"
        check_requirements("openvino>=2025.2.0" if MACOS and MACOS_VERSION >= "15.4" else "openvino>=2024.0.0")
        import openvino as ov

        LOGGER.info(f"\n{prefix} starting export with openvino {ov.__version__}...")
        assert TORCH_2_1, f"OpenVINO export requires torch>=2.1 but torch=={TORCH_VERSION} is installed"
        ov_model = ov.convert_model(
            NMSModel(self.model, self.args) if self.args.nms else self.model,
            input=None if self.args.dynamic else [self.im.shape],
            example_input=self.im,
        )

        def serialize(ov_model, file):
            """Set RT info, serialize, and save metadata YAML."""
            ov_model.set_rt_info("YOLO", ["model_info", "model_type"])
            ov_model.set_rt_info(True, ["model_info", "reverse_input_channels"])
            ov_model.set_rt_info(114, ["model_info", "pad_value"])
            ov_model.set_rt_info([255.0], ["model_info", "scale_values"])
            ov_model.set_rt_info(self.args.iou, ["model_info", "iou_threshold"])
            ov_model.set_rt_info([v.replace(" ", "_") for v in self.model.names.values()], ["model_info", "labels"])
            if self.model.task != "classify":
                ov_model.set_rt_info("fit_to_window_letterbox", ["model_info", "resize_type"])

            ov.save_model(ov_model, file, compress_to_fp16=self.args.half)
            YAML.save(Path(file).parent / "metadata.yaml", self.metadata)  # add metadata.yaml

        if self.args.int8:
            fq = str(self.file).replace(self.file.suffix, f"_int8_openvino_model{os.sep}")
            fq_ov = str(Path(fq) / self.file.with_suffix(".xml").name)
            # INT8 requires nncf, nncf requires packaging>=23.2 https://github.com/openvinotoolkit/nncf/issues/3463
            check_requirements("packaging>=23.2")  # must be installed first to build nncf wheel
            check_requirements("nncf>=2.14.0")
            import nncf

            def transform_fn(data_item) -> np.ndarray:
                """Quantization transform function."""
                data_item: torch.Tensor = data_item["img"] if isinstance(data_item, dict) else data_item
                assert data_item.dtype == torch.uint8, "Input image must be uint8 for the quantization preprocessing"
                im = data_item.numpy().astype(np.float32) / 255.0  # uint8 to fp16/32 and 0-255 to 0.0-1.0
                return np.expand_dims(im, 0) if im.ndim == 3 else im

            # Generate calibration data for integer quantization
            ignored_scope = None
            if isinstance(self.model.model[-1], Detect):
                # Includes all Detect subclasses like Segment, Pose, OBB, WorldDetect, YOLOEDetect
                head_module_name = ".".join(list(self.model.named_modules())[-1][0].split(".")[:2])
                ignored_scope = nncf.IgnoredScope(  # ignore operations
                    patterns=[
                        f".*{head_module_name}/.*/Add",
                        f".*{head_module_name}/.*/Sub*",
                        f".*{head_module_name}/.*/Mul*",
                        f".*{head_module_name}/.*/Div*",
                        f".*{head_module_name}\\.dfl.*",
                    ],
                    types=["Sigmoid"],
                )

            quantized_ov_model = nncf.quantize(
                model=ov_model,
                calibration_dataset=nncf.Dataset(self.get_int8_calibration_dataloader(prefix), transform_fn),
                preset=nncf.QuantizationPreset.MIXED,
                ignored_scope=ignored_scope,
            )
            serialize(quantized_ov_model, fq_ov)
            return fq

        f = str(self.file).replace(self.file.suffix, f"_openvino_model{os.sep}")
        f_ov = str(Path(f) / self.file.with_suffix(".xml").name)

        serialize(ov_model, f_ov)
        return f

    @try_export
    def export_paddle(self, prefix=colorstr("PaddlePaddle:")):
        """Export YOLO model to PaddlePaddle format."""
        assert not IS_JETSON, "Jetson Paddle exports not supported yet"
        check_requirements(
            (
                "paddlepaddle-gpu"
                if torch.cuda.is_available()
                else "paddlepaddle==3.0.0"  # pin 3.0.0 for ARM64
                if ARM64
                else "paddlepaddle>=3.0.0",
                "x2paddle",
            )
        )
        import x2paddle
        from x2paddle.convert import pytorch2paddle

        LOGGER.info(f"\n{prefix} starting export with X2Paddle {x2paddle.__version__}...")
        f = str(self.file).replace(self.file.suffix, f"_paddle_model{os.sep}")

        pytorch2paddle(module=self.model, save_dir=f, jit_type="trace", input_examples=[self.im])  # export
        YAML.save(Path(f) / "metadata.yaml", self.metadata)  # add metadata.yaml
        return f

    @try_export
    def export_mnn(self, prefix=colorstr("MNN:")):
        """Export YOLO model to MNN format using MNN https://github.com/alibaba/MNN."""
        assert TORCH_1_10, "MNN export requires torch>=1.10.0 to avoid segmentation faults"
        f_onnx = self.export_onnx()  # get onnx model first

        check_requirements("MNN>=2.9.6")
        import MNN
        from MNN.tools import mnnconvert

        # Setup and checks
        LOGGER.info(f"\n{prefix} starting export with MNN {MNN.version()}...")
        assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"
        f = str(self.file.with_suffix(".mnn"))  # MNN model file
        args = ["", "-f", "ONNX", "--modelFile", f_onnx, "--MNNModel", f, "--bizCode", json.dumps(self.metadata)]
        if self.args.int8:
            args.extend(("--weightQuantBits", "8"))
        if self.args.half:
            args.append("--fp16")
        mnnconvert.convert(args)
        # remove scratch file for model convert optimize
        convert_scratch = Path(self.file.parent / ".__convert_external_data.bin")
        if convert_scratch.exists():
            convert_scratch.unlink()
        return f

    @try_export
    def export_ncnn(self, prefix=colorstr("NCNN:")):
        """Export YOLO model to NCNN format using PNNX https://github.com/pnnx/pnnx."""
        check_requirements("ncnn", cmds="--no-deps")  # no deps to avoid installing opencv-python
        check_requirements("pnnx")
        import ncnn
        import pnnx

        LOGGER.info(f"\n{prefix} starting export with NCNN {ncnn.__version__} and PNNX {pnnx.__version__}...")
        f = Path(str(self.file).replace(self.file.suffix, f"_ncnn_model{os.sep}"))

        ncnn_args = dict(
            ncnnparam=(f / "model.ncnn.param").as_posix(),
            ncnnbin=(f / "model.ncnn.bin").as_posix(),
            ncnnpy=(f / "model_ncnn.py").as_posix(),
        )

        pnnx_args = dict(
            ptpath=(f / "model.pt").as_posix(),
            pnnxparam=(f / "model.pnnx.param").as_posix(),
            pnnxbin=(f / "model.pnnx.bin").as_posix(),
            pnnxpy=(f / "model_pnnx.py").as_posix(),
            pnnxonnx=(f / "model.pnnx.onnx").as_posix(),
        )

        f.mkdir(exist_ok=True)  # make ncnn_model directory
        pnnx.export(self.model, inputs=self.im, **ncnn_args, **pnnx_args, fp16=self.args.half, device=self.device.type)

        for f_debug in ("debug.bin", "debug.param", "debug2.bin", "debug2.param", *pnnx_args.values()):
            Path(f_debug).unlink(missing_ok=True)

        YAML.save(f / "metadata.yaml", self.metadata)  # add metadata.yaml
        return str(f)

    @try_export
    def export_coreml(self, prefix=colorstr("CoreML:")):
        """Export YOLO model to CoreML format."""
        mlmodel = self.args.format.lower() == "mlmodel"  # legacy *.mlmodel export format requested
        check_requirements(
            ["coremltools>=9.0", "numpy>=1.14.5,<=2.3.5"]
        )  # latest numpy 2.4.0rc1 breaks coremltools exports
        import coremltools as ct

        LOGGER.info(f"\n{prefix} starting export with coremltools {ct.__version__}...")
        assert not WINDOWS, "CoreML export is not supported on Windows, please run on macOS or Linux."
        assert TORCH_1_11, "CoreML export requires torch>=1.11"
        if self.args.batch > 1:
            assert self.args.dynamic, (
                "batch sizes > 1 are not supported without 'dynamic=True' for CoreML export. Please retry at 'dynamic=True'."
            )
        if self.args.dynamic:
            assert not self.args.nms, (
                "'nms=True' cannot be used together with 'dynamic=True' for CoreML export. Please disable one of them."
            )
            assert self.model.task != "classify", "'dynamic=True' is not supported for CoreML classification models."
        f = self.file.with_suffix(".mlmodel" if mlmodel else ".mlpackage")
        if f.is_dir():
            shutil.rmtree(f)

        classifier_config = None
        if self.model.task == "classify":
            classifier_config = ct.ClassifierConfig(list(self.model.names.values()))
            model = self.model
        elif self.model.task == "detect":
            model = IOSDetectModel(self.model, self.im, mlprogram=not mlmodel) if self.args.nms else self.model
        else:
            if self.args.nms:
                LOGGER.warning(f"{prefix} 'nms=True' is only available for Detect models like 'yolo11n.pt'.")
                # TODO CoreML Segment and Pose model pipelining
            model = self.model
        ts = torch.jit.trace(model.eval(), self.im, strict=False)  # TorchScript model

        if self.args.dynamic:
            input_shape = ct.Shape(
                shape=(
                    ct.RangeDim(lower_bound=1, upper_bound=self.args.batch, default=1),
                    self.im.shape[1],
                    ct.RangeDim(lower_bound=32, upper_bound=self.imgsz[0] * 2, default=self.imgsz[0]),
                    ct.RangeDim(lower_bound=32, upper_bound=self.imgsz[1] * 2, default=self.imgsz[1]),
                )
            )
            inputs = [ct.TensorType("image", shape=input_shape)]
        else:
            inputs = [ct.ImageType("image", shape=self.im.shape, scale=1 / 255, bias=[0.0, 0.0, 0.0])]

        # Based on apple's documentation it is better to leave out the minimum_deployment target and let that get set
        # Internally based on the model conversion and output type.
        # Setting minimum_deployment_target >= iOS16 will require setting compute_precision=ct.precision.FLOAT32.
        # iOS16 adds in better support for FP16, but none of the CoreML NMS specifications handle FP16 as input.
        ct_model = ct.convert(
            ts,
            inputs=inputs,
            classifier_config=classifier_config,
            convert_to="neuralnetwork" if mlmodel else "mlprogram",
        )
        bits, mode = (8, "kmeans") if self.args.int8 else (16, "linear") if self.args.half else (32, None)
        if bits < 32:
            if "kmeans" in mode:
                check_requirements("scikit-learn")  # scikit-learn package required for k-means quantization
            if mlmodel:
                ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
            elif bits == 8:  # mlprogram already quantized to FP16
                import coremltools.optimize.coreml as cto

                op_config = cto.OpPalettizerConfig(mode="kmeans", nbits=bits, weight_threshold=512)
                config = cto.OptimizationConfig(global_config=op_config)
                ct_model = cto.palettize_weights(ct_model, config=config)
        if self.args.nms and self.model.task == "detect":
            ct_model = self._pipeline_coreml(ct_model, weights_dir=None if mlmodel else ct_model.weights_dir)

        m = self.metadata  # metadata dict
        ct_model.short_description = m.pop("description")
        ct_model.author = m.pop("author")
        ct_model.license = m.pop("license")
        ct_model.version = m.pop("version")
        ct_model.user_defined_metadata.update({k: str(v) for k, v in m.items()})
        if self.model.task == "classify":
            ct_model.user_defined_metadata.update({"com.apple.coreml.model.preview.type": "imageClassifier"})

        try:
            ct_model.save(str(f))  # save *.mlpackage
        except Exception as e:
            LOGGER.warning(
                f"{prefix} CoreML export to *.mlpackage failed ({e}), reverting to *.mlmodel export. "
                f"Known coremltools Python 3.11 and Windows bugs https://github.com/apple/coremltools/issues/1928."
            )
            f = f.with_suffix(".mlmodel")
            ct_model.save(str(f))
        return f

    @try_export
    def export_engine(self, dla=None, prefix=colorstr("TensorRT:")):
        """Export YOLO model to TensorRT format https://developer.nvidia.com/tensorrt."""
        assert self.im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. use 'device=0'"
        f_onnx = self.export_onnx()  # run before TRT import https://github.com/ultralytics/ultralytics/issues/7016

        try:
            import tensorrt as trt
        except ImportError:
            if LINUX:
                cuda_version = torch.version.cuda.split(".")[0]
                check_requirements(f"tensorrt-cu{cuda_version}>7.0.0,!=10.1.0")
            import tensorrt as trt
        check_version(trt.__version__, ">=7.0.0", hard=True)
        check_version(trt.__version__, "!=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")

        # Setup and checks
        LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
        assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"
        f = self.file.with_suffix(".engine")  # TensorRT engine file
        onnx2engine(
            f_onnx,
            f,
            self.args.workspace,
            self.args.half,
            self.args.int8,
            self.args.dynamic,
            self.im.shape,
            dla=dla,
            dataset=self.get_int8_calibration_dataloader(prefix) if self.args.int8 else None,
            metadata=self.metadata,
            verbose=self.args.verbose,
            prefix=prefix,
        )

        return f

    @try_export
    def export_saved_model(self, prefix=colorstr("TensorFlow SavedModel:")):
        """Export YOLO model to TensorFlow SavedModel format."""
        cuda = torch.cuda.is_available()
        try:
            import tensorflow as tf
        except ImportError:
            check_requirements("tensorflow>=2.0.0,<=2.19.0")
            import tensorflow as tf
        check_requirements(
            (
                "tf_keras<=2.19.0",  # required by 'onnx2tf' package
                "sng4onnx>=1.0.1",  # required by 'onnx2tf' package
                "onnx_graphsurgeon>=0.3.26",  # required by 'onnx2tf' package
                "ai-edge-litert>=1.2.0" + (",<1.4.0" if MACOS else ""),  # required by 'onnx2tf' package
                "onnx>=1.12.0,<2.0.0",
                "onnx2tf>=1.26.3",
                "onnxslim>=0.1.71",
                "onnxruntime-gpu" if cuda else "onnxruntime",
                "protobuf>=5",
            ),
            cmds="--extra-index-url https://pypi.ngc.nvidia.com",  # onnx_graphsurgeon only on NVIDIA
        )

        LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
        check_version(
            tf.__version__,
            ">=2.0.0",
            name="tensorflow",
            verbose=True,
            msg="https://github.com/ultralytics/ultralytics/issues/5161",
        )
        f = Path(str(self.file).replace(self.file.suffix, "_saved_model"))
        if f.is_dir():
            shutil.rmtree(f)  # delete output folder

        # Export to TF
        images = None
        if self.args.int8 and self.args.data:
            images = [batch["img"] for batch in self.get_int8_calibration_dataloader(prefix)]
            images = (
                torch.nn.functional.interpolate(torch.cat(images, 0).float(), size=self.imgsz)
                .permute(0, 2, 3, 1)
                .numpy()
                .astype(np.float32)
            )

        # Export to ONNX
        if isinstance(self.model.model[-1], RTDETRDecoder):
            self.args.opset = self.args.opset or 19
            assert 16 <= self.args.opset <= 19, "RTDETR export requires opset>=16;<=19"
        self.args.simplify = True
        f_onnx = self.export_onnx()  # ensure ONNX is available
        keras_model = onnx2saved_model(
            f_onnx,
            f,
            int8=self.args.int8,
            images=images,
            disable_group_convolution=self.args.format in {"tfjs", "edgetpu"},
            prefix=prefix,
        )
        YAML.save(f / "metadata.yaml", self.metadata)  # add metadata.yaml
        # Add TFLite metadata
        for file in f.rglob("*.tflite"):
            file.unlink() if "quant_with_int16_act.tflite" in str(file) else self._add_tflite_metadata(file)

        return str(f), keras_model  # or keras_model = tf.saved_model.load(f, tags=None, options=None)

    @try_export
    def export_pb(self, keras_model, prefix=colorstr("TensorFlow GraphDef:")):
        """Export YOLO model to TensorFlow GraphDef *.pb format https://github.com/leimao/Frozen-Graph-TensorFlow."""
        f = self.file.with_suffix(".pb")
        keras2pb(keras_model, f, prefix)
        return f

    @try_export
    def export_tflite(self, prefix=colorstr("TensorFlow Lite:")):
        """Export YOLO model to TensorFlow Lite format."""
        # BUG https://github.com/ultralytics/ultralytics/issues/13436
        import tensorflow as tf

        LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
        saved_model = Path(str(self.file).replace(self.file.suffix, "_saved_model"))
        if self.args.int8:
            f = saved_model / f"{self.file.stem}_int8.tflite"  # fp32 in/out
        elif self.args.half:
            f = saved_model / f"{self.file.stem}_float16.tflite"  # fp32 in/out
        else:
            f = saved_model / f"{self.file.stem}_float32.tflite"
        return str(f)

    @try_export
    def export_axelera(self, prefix=colorstr("Axelera:")):
        """YOLO Axelera export."""
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        try:
            from axelera import compiler
        except ImportError:
            check_apt_requirements(
                ["libllvm14", "libgirepository1.0-dev", "pkg-config", "libcairo2-dev", "build-essential", "cmake"]
            )

            check_requirements(
                "axelera-voyager-sdk==1.5.2",
                cmds="--extra-index-url https://software.axelera.ai/artifactory/axelera-runtime-pypi "
                "--extra-index-url https://software.axelera.ai/artifactory/axelera-dev-pypi",
            )

        from axelera import compiler
        from axelera.compiler import CompilerConfig

        self.args.opset = 17
        onnx_path = self.export_onnx()
        model_name = f"{Path(onnx_path).stem}"
        export_path = Path(f"{model_name}_axelera_model")
        export_path.mkdir(exist_ok=True)

        def transform_fn(data_item) -> np.ndarray:
            data_item: torch.Tensor = data_item["img"] if isinstance(data_item, dict) else data_item
            assert data_item.dtype == torch.uint8, "Input image must be uint8 for the quantization preprocessing"
            im = data_item.numpy().astype(np.float32) / 255.0  # uint8 to fp16/32 and 0 - 255 to 0.0 - 1.0
            return np.expand_dims(im, 0) if im.ndim == 3 else im

        if "C2PSA" in self.model.__str__():  # YOLO11
            config = CompilerConfig(
                quantization_scheme="per_tensor_min_max",
                ignore_weight_buffers=False,
                resources_used=0.25,
                aipu_cores_used=1,
                multicore_mode="batch",
                output_axm_format=True,
                model_name=model_name,
            )
        else:  # YOLOv8
            config = CompilerConfig(
                tiling_depth=6,
                split_buffer_promotion=True,
                resources_used=0.25,
                aipu_cores_used=1,
                multicore_mode="batch",
                output_axm_format=True,
                model_name=model_name,
            )

        qmodel = compiler.quantize(
            model=onnx_path,
            calibration_dataset=self.get_int8_calibration_dataloader(prefix),
            config=config,
            transform_fn=transform_fn,
        )

        compiler.compile(model=qmodel, config=config, output_dir=export_path)

        axm_name = f"{model_name}.axm"
        axm_src = Path(axm_name)
        axm_dst = export_path / axm_name

        if axm_src.exists():
            axm_src.replace(axm_dst)

        YAML.save(export_path / "metadata.yaml", self.metadata)

        return export_path

    @try_export
    def export_executorch(self, prefix=colorstr("ExecuTorch:")):
        """Exports a model to ExecuTorch (.pte) format into a dedicated directory and saves the required metadata,
        following Ultralytics conventions.
        """
        LOGGER.info(f"\n{prefix} starting export with ExecuTorch...")
        assert TORCH_2_9, f"ExecuTorch export requires torch>=2.9.0 but torch=={TORCH_VERSION} is installed"
        # TorchAO release compatibility table bug https://github.com/pytorch/ao/issues/2919
        # Setuptools bug: https://github.com/pypa/setuptools/issues/4483
        check_requirements("setuptools<71.0.0")  # Setuptools bug: https://github.com/pypa/setuptools/issues/4483
        check_requirements(("executorch==1.0.1", "flatbuffers"))

        import torch
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.exir import to_edge_transform_and_lower

        file_directory = Path(str(self.file).replace(self.file.suffix, "_executorch_model"))
        file_directory.mkdir(parents=True, exist_ok=True)

        file_pte = file_directory / self.file.with_suffix(".pte").name
        sample_inputs = (self.im,)

        et_program = to_edge_transform_and_lower(
            torch.export.export(self.model, sample_inputs), partitioner=[XnnpackPartitioner()]
        ).to_executorch()

        with open(file_pte, "wb") as file:
            file.write(et_program.buffer)

        YAML.save(file_directory / "metadata.yaml", self.metadata)

        return str(file_directory)

    @try_export
    def export_edgetpu(self, tflite_model="", prefix=colorstr("Edge TPU:")):
        """Export YOLO model to Edge TPU format https://coral.ai/docs/edgetpu/models-intro/."""
        cmd = "edgetpu_compiler --version"
        help_url = "https://coral.ai/docs/edgetpu/compiler/"
        assert LINUX, f"export only supported on Linux. See {help_url}"
        if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True).returncode != 0:
            LOGGER.info(f"\n{prefix} export requires Edge TPU compiler. Attempting install from {help_url}")
            sudo = "sudo " if is_sudo_available() else ""
            for c in (
                f"{sudo}mkdir -p /etc/apt/keyrings",
                f"curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | {sudo}gpg --dearmor -o /etc/apt/keyrings/google.gpg",
                f'echo "deb [signed-by=/etc/apt/keyrings/google.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | {sudo}tee /etc/apt/sources.list.d/coral-edgetpu.list',
            ):
                subprocess.run(c, shell=True, check=True)
            check_apt_requirements(["edgetpu-compiler"])

        ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().rsplit(maxsplit=1)[-1]
        LOGGER.info(f"\n{prefix} starting export with Edge TPU compiler {ver}...")
        tflite2edgetpu(tflite_file=tflite_model, output_dir=tflite_model.parent, prefix=prefix)
        f = str(tflite_model).replace(".tflite", "_edgetpu.tflite")  # Edge TPU model
        self._add_tflite_metadata(f)
        return f

    @try_export
    def export_tfjs(self, prefix=colorstr("TensorFlow.js:")):
        """Export YOLO model to TensorFlow.js format."""
        check_requirements("tensorflowjs")

        f = str(self.file).replace(self.file.suffix, "_web_model")  # js dir
        f_pb = str(self.file.with_suffix(".pb"))  # *.pb path
        pb2tfjs(pb_file=f_pb, output_dir=f, half=self.args.half, int8=self.args.int8, prefix=prefix)
        # Add metadata
        YAML.save(Path(f) / "metadata.yaml", self.metadata)  # add metadata.yaml
        return f

    @try_export
    def export_rknn(self, prefix=colorstr("RKNN:")):
        """Export YOLO model to RKNN format."""
        LOGGER.info(f"\n{prefix} starting export with rknn-toolkit2...")

        check_requirements("rknn-toolkit2")
        if IS_COLAB:
            # Prevent 'exit' from closing the notebook https://github.com/airockchip/rknn-toolkit2/issues/259
            import builtins

            builtins.exit = lambda: None

        from rknn.api import RKNN

        f = self.export_onnx()
        export_path = Path(f"{Path(f).stem}_rknn_model")
        export_path.mkdir(exist_ok=True)

        rknn = RKNN(verbose=False)
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=self.args.name)
        rknn.load_onnx(model=f)
        rknn.build(do_quantization=False)  # TODO: Add quantization support
        f = f.replace(".onnx", f"-{self.args.name}.rknn")
        rknn.export_rknn(f"{export_path / f}")
        YAML.save(export_path / "metadata.yaml", self.metadata)
        return export_path

    @try_export
    def export_imx(self, prefix=colorstr("IMX:")):
        """Export YOLO model to IMX format."""
        assert LINUX, (
            "Export only supported on Linux."
            "See https://developer.aitrios.sony-semicon.com/en/docs/raspberry-pi-ai-camera/imx500-converter?version=3.17.3&progLang="
        )
        assert not ARM64, "IMX export is not supported on ARM64 architectures."
        assert IS_PYTHON_MINIMUM_3_9, "IMX export is only supported on Python 3.9 or above."

        if getattr(self.model, "end2end", False):
            raise ValueError("IMX export is not supported for end2end models.")
        check_requirements(
            (
                "model-compression-toolkit>=2.4.1",
                "edge-mdt-cl<1.1.0",
                "edge-mdt-tpc>=1.2.0",
                "pydantic<=2.11.7",
            )
        )

        check_requirements("imx500-converter[pt]>=3.17.3")

        # Install Java>=17
        try:
            java_output = subprocess.run(["java", "--version"], check=True, capture_output=True).stdout.decode()
            version_match = re.search(r"(?:openjdk|java) (\d+)", java_output)
            java_version = int(version_match.group(1)) if version_match else 0
            assert java_version >= 17, "Java version too old"
        except (FileNotFoundError, subprocess.CalledProcessError, AssertionError):
            if IS_UBUNTU or IS_DEBIAN_TRIXIE:
                LOGGER.info(f"\n{prefix} installing Java 21 for Ubuntu...")
                check_apt_requirements(["openjdk-21-jre"])
            elif IS_RASPBERRYPI or IS_DEBIAN_BOOKWORM:
                LOGGER.info(f"\n{prefix} installing Java 17 for Raspberry Pi or Debian ...")
                check_apt_requirements(["openjdk-17-jre"])

        return torch2imx(
            self.model,
            self.file,
            self.args.conf,
            self.args.iou,
            self.args.max_det,
            metadata=self.metadata,
            dataset=self.get_int8_calibration_dataloader(prefix),
            prefix=prefix,
        )

    def _add_tflite_metadata(self, file):
        """Add metadata to *.tflite models per https://ai.google.dev/edge/litert/models/metadata."""
        import zipfile

        with zipfile.ZipFile(file, "a", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metadata.json", json.dumps(self.metadata, indent=2))

    def _pipeline_coreml(self, model, weights_dir=None, prefix=colorstr("CoreML Pipeline:")):
        """Create CoreML pipeline with NMS for YOLO detection models."""
        import coremltools as ct

        LOGGER.info(f"{prefix} starting pipeline with coremltools {ct.__version__}...")

        # Output shapes
        spec = model.get_spec()
        outs = list(iter(spec.description.output))
        if self.args.format == "mlmodel":  # mlmodel doesn't infer shapes automatically
            outs[0].type.multiArrayType.shape[:] = self.output_shape[2], self.output_shape[1] - 4
            outs[1].type.multiArrayType.shape[:] = self.output_shape[2], 4

        # Checks
        names = self.metadata["names"]
        nx, ny = spec.description.input[0].type.imageType.width, spec.description.input[0].type.imageType.height
        nc = outs[0].type.multiArrayType.shape[-1]
        if len(names) != nc:  # Hack fix for MLProgram NMS bug https://github.com/ultralytics/ultralytics/issues/22309
            names = {**names, **{i: str(i) for i in range(len(names), nc)}}

        # Model from spec
        model = ct.models.MLModel(spec, weights_dir=weights_dir)

        # Create NMS protobuf
        nms_spec = ct.proto.Model_pb2.Model()
        nms_spec.specificationVersion = spec.specificationVersion
        for i in range(len(outs)):
            decoder_output = model._spec.description.output[i].SerializeToString()
            nms_spec.description.input.add()
            nms_spec.description.input[i].ParseFromString(decoder_output)
            nms_spec.description.output.add()
            nms_spec.description.output[i].ParseFromString(decoder_output)

        output_names = ["confidence", "coordinates"]
        for i, name in enumerate(output_names):
            nms_spec.description.output[i].name = name

        for i, out in enumerate(outs):
            ma_type = nms_spec.description.output[i].type.multiArrayType
            ma_type.shapeRange.sizeRanges.add()
            ma_type.shapeRange.sizeRanges[0].lowerBound = 0
            ma_type.shapeRange.sizeRanges[0].upperBound = -1
            ma_type.shapeRange.sizeRanges.add()
            ma_type.shapeRange.sizeRanges[1].lowerBound = out.type.multiArrayType.shape[-1]
            ma_type.shapeRange.sizeRanges[1].upperBound = out.type.multiArrayType.shape[-1]
            del ma_type.shape[:]

        nms = nms_spec.nonMaximumSuppression
        nms.confidenceInputFeatureName = outs[0].name  # 1x507x80
        nms.coordinatesInputFeatureName = outs[1].name  # 1x507x4
        nms.confidenceOutputFeatureName = output_names[0]
        nms.coordinatesOutputFeatureName = output_names[1]
        nms.iouThresholdInputFeatureName = "iouThreshold"
        nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
        nms.iouThreshold = self.args.iou
        nms.confidenceThreshold = self.args.conf
        nms.pickTop.perClass = True
        nms.stringClassLabels.vector.extend(names.values())
        nms_model = ct.models.MLModel(nms_spec)

        # Pipeline models together
        pipeline = ct.models.pipeline.Pipeline(
            input_features=[
                ("image", ct.models.datatypes.Array(3, ny, nx)),
                ("iouThreshold", ct.models.datatypes.Double()),
                ("confidenceThreshold", ct.models.datatypes.Double()),
            ],
            output_features=output_names,
        )
        pipeline.add_model(model)
        pipeline.add_model(nms_model)

        # Correct datatypes
        pipeline.spec.description.input[0].ParseFromString(model._spec.description.input[0].SerializeToString())
        pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
        pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

        # Update metadata
        pipeline.spec.specificationVersion = spec.specificationVersion
        pipeline.spec.description.metadata.userDefined.update(
            {"IoU threshold": str(nms.iouThreshold), "Confidence threshold": str(nms.confidenceThreshold)}
        )

        # Save the model
        model = ct.models.MLModel(pipeline.spec, weights_dir=weights_dir)
        model.input_description["image"] = "Input image"
        model.input_description["iouThreshold"] = f"(optional) IoU threshold override (default: {nms.iouThreshold})"
        model.input_description["confidenceThreshold"] = (
            f"(optional) Confidence threshold override (default: {nms.confidenceThreshold})"
        )
        model.output_description["confidence"] = 'Boxes × Class confidence (see user-defined metadata "classes")'
        model.output_description["coordinates"] = "Boxes × [x, y, width, height] (relative to image size)"
        LOGGER.info(f"{prefix} pipeline success")
        return model

    def add_callback(self, event: str, callback):
        """Append the given callback to the specified event."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Execute all callbacks for a given event."""
        for callback in self.callbacks.get(event, []):
            callback(self)


class IOSDetectModel(torch.nn.Module):
    """Wrap an Ultralytics YOLO model for Apple iOS CoreML export."""

    def __init__(self, model, im, mlprogram=True):
        """Initialize the IOSDetectModel class with a YOLO model and example image.

        Args:
            model (torch.nn.Module): The YOLO model to wrap.
            im (torch.Tensor): Example input tensor with shape (B, C, H, W).
            mlprogram (bool): Whether exporting to MLProgram format to fix NMS bug.
        """
        super().__init__()
        _, _, h, w = im.shape  # batch, channel, height, width
        self.model = model
        self.nc = len(model.names)  # number of classes
        self.mlprogram = mlprogram
        if w == h:
            self.normalize = 1.0 / w  # scalar
        else:
            self.normalize = torch.tensor(
                [1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h],  # broadcast (slower, smaller)
                device=next(model.parameters()).device,
            )

    def forward(self, x):
        """Normalize predictions of object detection model with input size-dependent factors."""
        xywh, cls = self.model(x)[0].transpose(0, 1).split((4, self.nc), 1)
        if self.mlprogram and self.nc % 80 != 0:  # NMS bug https://github.com/ultralytics/ultralytics/issues/22309
            pad_length = int(((self.nc + 79) // 80) * 80) - self.nc  # pad class length to multiple of 80
            cls = torch.nn.functional.pad(cls, (0, pad_length, 0, 0), "constant", 0)

        return cls, xywh * self.normalize


class NMSModel(torch.nn.Module):
    """Model wrapper with embedded NMS for Detect, Segment, Pose and OBB."""

    def __init__(self, model, args):
        """Initialize the NMSModel.

        Args:
            model (torch.nn.Module): The model to wrap with NMS postprocessing.
            args (Namespace): The export arguments.
        """
        super().__init__()
        self.model = model
        self.args = args
        self.obb = model.task == "obb"
        self.is_tf = self.args.format in frozenset({"saved_model", "tflite", "tfjs"})

    def forward(self, x):
        """Perform inference with NMS post-processing. Supports Detect, Segment, OBB and Pose.

        Args:
            x (torch.Tensor): The preprocessed tensor with shape (N, 3, H, W).

        Returns:
            (torch.Tensor): List of detections, each an (N, max_det, 4 + 2 + extra_shape) Tensor where N is the number
                of detections after NMS.
        """
        from functools import partial

        from torchvision.ops import nms

        preds = self.model(x)
        pred = preds[0] if isinstance(preds, tuple) else preds
        kwargs = dict(device=pred.device, dtype=pred.dtype)
        bs = pred.shape[0]
        pred = pred.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
        extra_shape = pred.shape[-1] - (4 + len(self.model.names))  # extras from Segment, OBB, Pose
        if self.args.dynamic and self.args.batch > 1:  # batch size needs to always be same due to loop unroll
            pad = torch.zeros(torch.max(torch.tensor(self.args.batch - bs), torch.tensor(0)), *pred.shape[1:], **kwargs)
            pred = torch.cat((pred, pad))
        boxes, scores, extras = pred.split([4, len(self.model.names), extra_shape], dim=2)
        scores, classes = scores.max(dim=-1)
        self.args.max_det = min(pred.shape[1], self.args.max_det)  # in case num_anchors < max_det
        # (N, max_det, 4 coords + 1 class score + 1 class label + extra_shape).
        out = torch.zeros(pred.shape[0], self.args.max_det, boxes.shape[-1] + 2 + extra_shape, **kwargs)
        for i in range(bs):
            box, cls, score, extra = boxes[i], classes[i], scores[i], extras[i]
            mask = score > self.args.conf
            if self.is_tf or (self.args.format == "onnx" and self.obb):
                # TFLite GatherND error if mask is empty
                score *= mask
                # Explicit length otherwise reshape error, hardcoded to `self.args.max_det * 5`
                mask = score.topk(min(self.args.max_det * 5, score.shape[0])).indices
            box, score, cls, extra = box[mask], score[mask], cls[mask], extra[mask]
            nmsbox = box.clone()
            # `8` is the minimum value experimented to get correct NMS results for obb
            multiplier = 8 if self.obb else 1 / max(len(self.model.names), 1)
            # Normalize boxes for NMS since large values for class offset causes issue with int8 quantization
            if self.args.format == "tflite":  # TFLite is already normalized
                nmsbox *= multiplier
            else:
                nmsbox = multiplier * (nmsbox / torch.tensor(x.shape[2:], **kwargs).max())
            if not self.args.agnostic_nms:  # class-wise NMS
                end = 2 if self.obb else 4
                # fully explicit expansion otherwise reshape error
                cls_offset = cls.view(cls.shape[0], 1).expand(cls.shape[0], end)
                offbox = nmsbox[:, :end] + cls_offset * multiplier
                nmsbox = torch.cat((offbox, nmsbox[:, end:]), dim=-1)
            nms_fn = (
                partial(
                    TorchNMS.fast_nms,
                    use_triu=not (
                        self.is_tf
                        or (self.args.opset or 14) < 14
                        or (self.args.format == "openvino" and self.args.int8)  # OpenVINO int8 error with triu
                    ),
                    iou_func=batch_probiou,
                    exit_early=False,
                )
                if self.obb
                else nms
            )
            keep = nms_fn(
                torch.cat([nmsbox, extra], dim=-1) if self.obb else nmsbox,
                score,
                self.args.iou,
            )[: self.args.max_det]
            dets = torch.cat(
                [box[keep], score[keep].view(-1, 1), cls[keep].view(-1, 1).to(out.dtype), extra[keep]], dim=-1
            )
            # Zero-pad to max_det size to avoid reshape error
            pad = (0, 0, 0, self.args.max_det - dets.shape[0])
            out[i] = torch.nn.functional.pad(dets, pad)
        return (out[:bs], preds[1]) if self.model.task == "segment" else out[:bs]
