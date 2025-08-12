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
import warnings
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
    IS_JETSON,
    LINUX,
    LOGGER,
    MACOS,
    MACOS_VERSION,
    RKNN_CHIPS,
    ROOT,
    SETTINGS,
    WINDOWS,
    YAML,
    callbacks,
    colorstr,
    get_default_args,
)
from ultralytics.utils.checks import (
    check_imgsz,
    check_is_path_safe,
    check_requirements,
    check_version,
    is_intel,
    is_sudo_available,
)
from ultralytics.utils.downloads import attempt_download_asset, get_github_assets, safe_download
from ultralytics.utils.export import export_engine, export_onnx
from ultralytics.utils.files import file_size, spaces_in_path
from ultralytics.utils.ops import Profile, nms_rotated
from ultralytics.utils.patches import arange_patch
from ultralytics.utils.torch_utils import TORCH_1_13, get_latest_opset, select_device


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
        ["CoreML", "coreml", ".mlpackage", True, False, ["batch", "half", "int8", "nms"]],
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
    ]
    return dict(zip(["Format", "Argument", "Suffix", "CPU", "GPU", "Arguments"], zip(*x)))


def validate_args(format, passed_args, valid_args):
    """
    Validate arguments based on the export format.

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


def gd_outputs(gd):
    """Return TensorFlow GraphDef model output node names."""
    name_list, input_list = [], []
    for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
        name_list.append(node.name)
        input_list.extend(node.input)
    return sorted(f"{x}:0" for x in list(set(name_list) - set(input_list)) if not x.startswith("NoOp"))


def try_export(inner_func):
    """YOLO export decorator, i.e. @try_export."""
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        """Export a model."""
        prefix = inner_args["prefix"]
        dt = 0.0
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f"{prefix} export success ✅ {dt.t:.1f}s, saved as '{f}' ({file_size(f):.1f} MB)")
            return f, model
        except Exception as e:
            LOGGER.error(f"{prefix} export failure {dt.t:.1f}s: {e}")
            raise e

    return outer_func


class Exporter:
    """
    A class for exporting YOLO models to various formats.

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
        """
        Initialize the Exporter class.

        Args:
            cfg (str, optional): Path to a configuration file.
            overrides (dict, optional): Configuration overrides.
            _callbacks (dict, optional): Dictionary of callback functions.
        """
        self.args = get_cfg(cfg, overrides)
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    def __call__(self, model=None) -> str:
        """Return list of exported files/dirs after running callbacks."""
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
                raise ValueError(f"Invalid export format='{fmt}'. Valid formats are {fmts}")
            LOGGER.warning(f"Invalid export format='{fmt}', updating to format='{matches[0]}'")
            fmt = matches[0]
        flags = [x == fmt for x in fmts]
        if sum(flags) != 1:
            raise ValueError(f"Invalid export format='{fmt}'. Valid formats are {fmts}")
        (jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, mnn, ncnn, imx, rknn) = (
            flags  # export booleans
        )

        is_tf_format = any((saved_model, pb, tflite, edgetpu, tfjs))

        # Device
        dla = None
        if engine and self.args.device is None:
            LOGGER.warning("TensorRT requires GPU export, automatically assigning device=0")
            self.args.device = "0"
        if engine and "dla" in str(self.args.device):  # convert int/list to str first
            dla = self.args.device.rsplit(":", 1)[-1]
            self.args.device = "0"  # update device to "0"
            assert dla in {"0", "1"}, f"Expected self.args.device='dla:0' or 'dla:1, but got {self.args.device}."
        if imx and self.args.device is None and torch.cuda.is_available():
            LOGGER.warning("Exporting on CPU while CUDA is available, setting device=0 for faster export on GPU.")
            self.args.device = "0"  # update device to "0"
        self.device = select_device("cpu" if self.args.device is None else self.args.device)

        # Argument compatibility checks
        fmt_keys = fmts_dict["Arguments"][flags.index(True) + 1]
        validate_args(fmt, self.args, fmt_keys)
        if imx:
            if not self.args.int8:
                LOGGER.warning("IMX export requires int8=True, setting int8=True.")
                self.args.int8 = True
            if not self.args.nms:
                LOGGER.warning("IMX export requires nms=True, setting nms=True.")
                self.args.nms = True
            if model.task not in {"detect", "pose"}:
                raise ValueError("IMX export only supported for detection and pose estimation models.")
        if not hasattr(model, "names"):
            model.names = default_class_names()
        model.names = check_class_names(model.names)
        if self.args.half and self.args.int8:
            LOGGER.warning("half=True and int8=True are mutually exclusive, setting half=False.")
            self.args.half = False
        if self.args.half and onnx and self.device.type == "cpu":
            LOGGER.warning("half=True only compatible with GPU export, i.e. use device=0")
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
        if self.args.int8 and tflite:
            assert not getattr(model, "end2end", False), "TFLite INT8 export not supported for end2end models."
        if self.args.nms:
            assert not isinstance(model, ClassificationModel), "'nms=True' is not valid for classification models."
            assert not (tflite and ARM64 and LINUX), "TFLite export with NMS unsupported on ARM64 Linux"
            if getattr(model, "end2end", False):
                LOGGER.warning("'nms=True' is not available for end2end models. Forcing 'nms=False'.")
                self.args.nms = False
            self.args.conf = self.args.conf or 0.25  # set conf default value for nms export
        if (engine or self.args.nms) and self.args.dynamic and self.args.batch == 1:
            LOGGER.warning(
                f"'dynamic=True' model with '{'nms=True' if self.args.nms else 'format=engine'}' requires max batch size, i.e. 'batch=16'"
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
            from ultralytics.utils.torch_utils import FXModel

            model = FXModel(model)
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
            if isinstance(m, Detect) and imx:
                from ultralytics.utils.tal import make_anchors

                m.anchors, m.strides = (
                    x.transpose(0, 1)
                    for x in make_anchors(
                        torch.cat([s / m.stride.unsqueeze(-1) for s in self.imgsz], dim=1), m.stride, 0.5
                    )
                )

        y = None
        for _ in range(2):  # dry runs
            y = NMSModel(model, self.args)(im) if self.args.nms and not (coreml or imx) else model(im)
        if self.args.half and onnx and self.device.type != "cpu":
            im, model = im.half(), model.half()  # to FP16

        # Filter warnings
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)  # suppress TracerWarning
        warnings.filterwarnings("ignore", category=UserWarning)  # suppress shape prim::Constant missing ONNX warning
        warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress CoreML np.bool deprecation warning

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

        LOGGER.info(
            f"\n{colorstr('PyTorch:')} starting from '{file}' with input shape {tuple(im.shape)} BCHW and "
            f"output shape(s) {self.output_shape} ({file_size(file):.1f} MB)"
        )
        self.run_callbacks("on_export_start")
        # Exports
        f = [""] * len(fmts)  # exported filenames
        if jit or ncnn:  # TorchScript
            f[0], _ = self.export_torchscript()
        if engine:  # TensorRT required before ONNX
            f[1], _ = self.export_engine(dla=dla)
        if onnx:  # ONNX
            f[2], _ = self.export_onnx()
        if xml:  # OpenVINO
            f[3], _ = self.export_openvino()
        if coreml:  # CoreML
            f[4], _ = self.export_coreml()
        if is_tf_format:  # TensorFlow formats
            self.args.int8 |= edgetpu
            f[5], keras_model = self.export_saved_model()
            if pb or tfjs:  # pb prerequisite to tfjs
                f[6], _ = self.export_pb(keras_model=keras_model)
            if tflite:
                f[7], _ = self.export_tflite()
            if edgetpu:
                f[8], _ = self.export_edgetpu(tflite_model=Path(f[5]) / f"{self.file.stem}_full_integer_quant.tflite")
            if tfjs:
                f[9], _ = self.export_tfjs()
        if paddle:  # PaddlePaddle
            f[10], _ = self.export_paddle()
        if mnn:  # MNN
            f[11], _ = self.export_mnn()
        if ncnn:  # NCNN
            f[12], _ = self.export_ncnn()
        if imx:
            f[13], _ = self.export_imx()
        if rknn:
            f[14], _ = self.export_rknn()

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
        return f  # return list of exported files/dirs

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
        elif n < 300:
            LOGGER.warning(f"{prefix} >300 images recommended for INT8 calibration, found {n} images.")
        return build_dataloader(dataset, batch=self.args.batch, workers=0, drop_last=True)  # required for batch loading

    @try_export
    def export_torchscript(self, prefix=colorstr("TorchScript:")):
        """Export YOLO model to TorchScript format."""
        LOGGER.info(f"\n{prefix} starting export with torch {torch.__version__}...")
        f = self.file.with_suffix(".torchscript")

        ts = torch.jit.trace(NMSModel(self.model, self.args) if self.args.nms else self.model, self.im, strict=False)
        extra_files = {"config.txt": json.dumps(self.metadata)}  # torch._C.ExtraFilesMap()
        if self.args.optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
            LOGGER.info(f"{prefix} optimizing for mobile...")
            from torch.utils.mobile_optimizer import optimize_for_mobile

            optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
        else:
            ts.save(str(f), _extra_files=extra_files)
        return f, None

    @try_export
    def export_onnx(self, prefix=colorstr("ONNX:")):
        """Export YOLO model to ONNX format."""
        requirements = ["onnx>=1.12.0,<1.18.0"]
        if self.args.simplify:
            requirements += ["onnxslim>=0.1.59", "onnxruntime" + ("-gpu" if torch.cuda.is_available() else "")]
        check_requirements(requirements)
        import onnx  # noqa

        opset_version = self.args.opset or get_latest_opset()
        LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__} opset {opset_version}...")
        f = str(self.file.with_suffix(".onnx"))
        output_names = ["output0", "output1"] if isinstance(self.model, SegmentationModel) else ["output0"]
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
            self.args.opset = opset_version  # for NMSModel

        with arange_patch(self.args):
            export_onnx(
                NMSModel(self.model, self.args) if self.args.nms else self.model,
                self.im,
                f,
                opset=opset_version,
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

        onnx.save(model_onnx, f)
        return f, model_onnx

    @try_export
    def export_openvino(self, prefix=colorstr("OpenVINO:")):
        """Export YOLO model to OpenVINO format."""
        # OpenVINO <= 2025.1.0 error on macOS 15.4+: https://github.com/openvinotoolkit/openvino/issues/30023"
        check_requirements("openvino>=2025.2.0" if MACOS and MACOS_VERSION >= "15.4" else "openvino>=2024.0.0")
        import openvino as ov

        LOGGER.info(f"\n{prefix} starting export with openvino {ov.__version__}...")
        assert TORCH_1_13, f"OpenVINO export requires torch>=1.13.0 but torch=={torch.__version__} is installed"
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
            return fq, None

        f = str(self.file).replace(self.file.suffix, f"_openvino_model{os.sep}")
        f_ov = str(Path(f) / self.file.with_suffix(".xml").name)

        serialize(ov_model, f_ov)
        return f, None

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
        import x2paddle  # noqa
        from x2paddle.convert import pytorch2paddle  # noqa

        LOGGER.info(f"\n{prefix} starting export with X2Paddle {x2paddle.__version__}...")
        f = str(self.file).replace(self.file.suffix, f"_paddle_model{os.sep}")

        pytorch2paddle(module=self.model, save_dir=f, jit_type="trace", input_examples=[self.im])  # export
        YAML.save(Path(f) / "metadata.yaml", self.metadata)  # add metadata.yaml
        return f, None

    @try_export
    def export_mnn(self, prefix=colorstr("MNN:")):
        """Export YOLO model to MNN format using MNN https://github.com/alibaba/MNN."""
        f_onnx, _ = self.export_onnx()  # get onnx model first

        check_requirements("MNN>=2.9.6")
        import MNN  # noqa
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
        return f, None

    @try_export
    def export_ncnn(self, prefix=colorstr("NCNN:")):
        """Export YOLO model to NCNN format using PNNX https://github.com/pnnx/pnnx."""
        check_requirements("ncnn", cmds="--no-deps")  # no deps to avoid installing opencv-python
        import ncnn  # noqa

        LOGGER.info(f"\n{prefix} starting export with NCNN {ncnn.__version__}...")
        f = Path(str(self.file).replace(self.file.suffix, f"_ncnn_model{os.sep}"))
        f_ts = self.file.with_suffix(".torchscript")

        name = Path("pnnx.exe" if WINDOWS else "pnnx")  # PNNX filename
        pnnx = name if name.is_file() else (ROOT / name)
        if not pnnx.is_file():
            LOGGER.warning(
                f"{prefix} PNNX not found. Attempting to download binary file from "
                "https://github.com/pnnx/pnnx/.\nNote PNNX Binary file must be placed in current working directory "
                f"or in {ROOT}. See PNNX repo for full installation instructions."
            )
            system = "macos" if MACOS else "windows" if WINDOWS else "linux-aarch64" if ARM64 else "linux"
            try:
                release, assets = get_github_assets(repo="pnnx/pnnx")
                asset = [x for x in assets if f"{system}.zip" in x][0]
                assert isinstance(asset, str), "Unable to retrieve PNNX repo assets"  # i.e. pnnx-20240410-macos.zip
                LOGGER.info(f"{prefix} successfully found latest PNNX asset file {asset}")
            except Exception as e:
                release = "20240410"
                asset = f"pnnx-{release}-{system}.zip"
                LOGGER.warning(f"{prefix} PNNX GitHub assets not found: {e}, using default {asset}")
            unzip_dir = safe_download(f"https://github.com/pnnx/pnnx/releases/download/{release}/{asset}", delete=True)
            if check_is_path_safe(Path.cwd(), unzip_dir):  # avoid path traversal security vulnerability
                shutil.move(src=unzip_dir / name, dst=pnnx)  # move binary to ROOT
                pnnx.chmod(0o777)  # set read, write, and execute permissions for everyone
                shutil.rmtree(unzip_dir)  # delete unzip dir

        ncnn_args = [
            f"ncnnparam={f / 'model.ncnn.param'}",
            f"ncnnbin={f / 'model.ncnn.bin'}",
            f"ncnnpy={f / 'model_ncnn.py'}",
        ]

        pnnx_args = [
            f"pnnxparam={f / 'model.pnnx.param'}",
            f"pnnxbin={f / 'model.pnnx.bin'}",
            f"pnnxpy={f / 'model_pnnx.py'}",
            f"pnnxonnx={f / 'model.pnnx.onnx'}",
        ]

        cmd = [
            str(pnnx),
            str(f_ts),
            *ncnn_args,
            *pnnx_args,
            f"fp16={int(self.args.half)}",
            f"device={self.device.type}",
            f'inputshape="{[self.args.batch, 3, *self.imgsz]}"',
        ]
        f.mkdir(exist_ok=True)  # make ncnn_model directory
        LOGGER.info(f"{prefix} running '{' '.join(cmd)}'")
        subprocess.run(cmd, check=True)

        # Remove debug files
        pnnx_files = [x.rsplit("=", 1)[-1] for x in pnnx_args]
        for f_debug in ("debug.bin", "debug.param", "debug2.bin", "debug2.param", *pnnx_files):
            Path(f_debug).unlink(missing_ok=True)

        YAML.save(f / "metadata.yaml", self.metadata)  # add metadata.yaml
        return str(f), None

    @try_export
    def export_coreml(self, prefix=colorstr("CoreML:")):
        """Export YOLO model to CoreML format."""
        mlmodel = self.args.format.lower() == "mlmodel"  # legacy *.mlmodel export format requested
        check_requirements("coremltools>=8.0")
        import coremltools as ct  # noqa

        LOGGER.info(f"\n{prefix} starting export with coremltools {ct.__version__}...")
        assert not WINDOWS, "CoreML export is not supported on Windows, please run on macOS or Linux."
        assert self.args.batch == 1, "CoreML batch sizes > 1 are not supported. Please retry at 'batch=1'."
        f = self.file.with_suffix(".mlmodel" if mlmodel else ".mlpackage")
        if f.is_dir():
            shutil.rmtree(f)

        bias = [0.0, 0.0, 0.0]
        scale = 1 / 255
        classifier_config = None
        if self.model.task == "classify":
            classifier_config = ct.ClassifierConfig(list(self.model.names.values()))
            model = self.model
        elif self.model.task == "detect":
            model = IOSDetectModel(self.model, self.im) if self.args.nms else self.model
        else:
            if self.args.nms:
                LOGGER.warning(f"{prefix} 'nms=True' is only available for Detect models like 'yolo11n.pt'.")
                # TODO CoreML Segment and Pose model pipelining
            model = self.model
        ts = torch.jit.trace(model.eval(), self.im, strict=False)  # TorchScript model

        # Based on apple's documentation it is better to leave out the minimum_deployment target and let that get set
        # Internally based on the model conversion and output type.
        # Setting minimum_depoloyment_target >= iOS16 will require setting compute_precision=ct.precision.FLOAT32.
        # iOS16 adds in better support for FP16, but none of the CoreML NMS specifications handle FP16 as input.
        ct_model = ct.convert(
            ts,
            inputs=[ct.ImageType("image", shape=self.im.shape, scale=scale, bias=bias)],  # expects ct.TensorType
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
            if mlmodel:
                weights_dir = None
            else:
                ct_model.save(str(f))  # save otherwise weights_dir does not exist
                weights_dir = str(f / "Data/com.apple.CoreML/weights")
            ct_model = self._pipeline_coreml(ct_model, weights_dir=weights_dir)

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
        return f, ct_model

    @try_export
    def export_engine(self, dla=None, prefix=colorstr("TensorRT:")):
        """Export YOLO model to TensorRT format https://developer.nvidia.com/tensorrt."""
        assert self.im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. use 'device=0'"
        f_onnx, _ = self.export_onnx()  # run before TRT import https://github.com/ultralytics/ultralytics/issues/7016

        try:
            import tensorrt as trt  # noqa
        except ImportError:
            if LINUX:
                check_requirements("tensorrt>7.0.0,!=10.1.0")
            import tensorrt as trt  # noqa
        check_version(trt.__version__, ">=7.0.0", hard=True)
        check_version(trt.__version__, "!=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")

        # Setup and checks
        LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
        assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"
        f = self.file.with_suffix(".engine")  # TensorRT engine file
        export_engine(
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

        return f, None

    @try_export
    def export_saved_model(self, prefix=colorstr("TensorFlow SavedModel:")):
        """Export YOLO model to TensorFlow SavedModel format."""
        cuda = torch.cuda.is_available()
        try:
            import tensorflow as tf  # noqa
        except ImportError:
            check_requirements("tensorflow>=2.0.0")
            import tensorflow as tf  # noqa
        check_requirements(
            (
                "tf_keras",  # required by 'onnx2tf' package
                "sng4onnx>=1.0.1",  # required by 'onnx2tf' package
                "onnx_graphsurgeon>=0.3.26",  # required by 'onnx2tf' package
                "ai-edge-litert>=1.2.0,<1.4.0",  # required by 'onnx2tf' package
                "onnx>=1.12.0,<1.18.0",
                "onnx2tf>=1.26.3",
                "onnxslim>=0.1.59",
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

        # Pre-download calibration file to fix https://github.com/PINTO0309/onnx2tf/issues/545
        onnx2tf_file = Path("calibration_image_sample_data_20x128x128x3_float32.npy")
        if not onnx2tf_file.exists():
            attempt_download_asset(f"{onnx2tf_file}.zip", unzip=True, delete=True)

        # Export to ONNX
        self.args.simplify = True
        f_onnx, _ = self.export_onnx()

        # Export to TF
        np_data = None
        if self.args.int8:
            tmp_file = f / "tmp_tflite_int8_calibration_images.npy"  # int8 calibration images file
            if self.args.data:
                f.mkdir()
                images = [batch["img"] for batch in self.get_int8_calibration_dataloader(prefix)]
                images = torch.nn.functional.interpolate(torch.cat(images, 0).float(), size=self.imgsz).permute(
                    0, 2, 3, 1
                )
                np.save(str(tmp_file), images.numpy().astype(np.float32))  # BHWC
                np_data = [["images", tmp_file, [[[[0, 0, 0]]]], [[[[255, 255, 255]]]]]]

        import onnx2tf  # scoped for after ONNX export for reduced conflict during import

        LOGGER.info(f"{prefix} starting TFLite export with onnx2tf {onnx2tf.__version__}...")
        keras_model = onnx2tf.convert(
            input_onnx_file_path=f_onnx,
            output_folder_path=str(f),
            not_use_onnxsim=True,
            verbosity="error",  # note INT8-FP16 activation bug https://github.com/ultralytics/ultralytics/issues/15873
            output_integer_quantized_tflite=self.args.int8,
            quant_type="per-tensor",  # "per-tensor" (faster) or "per-channel" (slower but more accurate)
            custom_input_op_name_np_data_path=np_data,
            enable_batchmatmul_unfold=True,  # fix lower no. of detected objects on GPU delegate
            output_signaturedefs=True,  # fix error with Attention block group convolution
            disable_group_convolution=self.args.format in {"tfjs", "edgetpu"},  # fix error with group convolution
        )
        YAML.save(f / "metadata.yaml", self.metadata)  # add metadata.yaml

        # Remove/rename TFLite models
        if self.args.int8:
            tmp_file.unlink(missing_ok=True)
            for file in f.rglob("*_dynamic_range_quant.tflite"):
                file.rename(file.with_name(file.stem.replace("_dynamic_range_quant", "_int8") + file.suffix))
            for file in f.rglob("*_integer_quant_with_int16_act.tflite"):
                file.unlink()  # delete extra fp16 activation TFLite files

        # Add TFLite metadata
        for file in f.rglob("*.tflite"):
            f.unlink() if "quant_with_int16_act.tflite" in str(f) else self._add_tflite_metadata(file)

        return str(f), keras_model  # or keras_model = tf.saved_model.load(f, tags=None, options=None)

    @try_export
    def export_pb(self, keras_model, prefix=colorstr("TensorFlow GraphDef:")):
        """Export YOLO model to TensorFlow GraphDef *.pb format https://github.com/leimao/Frozen-Graph-TensorFlow."""
        import tensorflow as tf  # noqa
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2  # noqa

        LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
        f = self.file.with_suffix(".pb")

        m = tf.function(lambda x: keras_model(x))  # full model
        m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
        frozen_func = convert_variables_to_constants_v2(m)
        frozen_func.graph.as_graph_def()
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(f.parent), name=f.name, as_text=False)
        return f, None

    @try_export
    def export_tflite(self, prefix=colorstr("TensorFlow Lite:")):
        """Export YOLO model to TensorFlow Lite format."""
        # BUG https://github.com/ultralytics/ultralytics/issues/13436
        import tensorflow as tf  # noqa

        LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
        saved_model = Path(str(self.file).replace(self.file.suffix, "_saved_model"))
        if self.args.int8:
            f = saved_model / f"{self.file.stem}_int8.tflite"  # fp32 in/out
        elif self.args.half:
            f = saved_model / f"{self.file.stem}_float16.tflite"  # fp32 in/out
        else:
            f = saved_model / f"{self.file.stem}_float32.tflite"
        return str(f), None

    @try_export
    def export_edgetpu(self, tflite_model="", prefix=colorstr("Edge TPU:")):
        """Export YOLO model to Edge TPU format https://coral.ai/docs/edgetpu/models-intro/."""
        cmd = "edgetpu_compiler --version"
        help_url = "https://coral.ai/docs/edgetpu/compiler/"
        assert LINUX, f"export only supported on Linux. See {help_url}"
        if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True).returncode != 0:
            LOGGER.info(f"\n{prefix} export requires Edge TPU compiler. Attempting install from {help_url}")
            for c in (
                "curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -",
                'echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | '
                "sudo tee /etc/apt/sources.list.d/coral-edgetpu.list",
                "sudo apt-get update",
                "sudo apt-get install edgetpu-compiler",
            ):
                subprocess.run(c if is_sudo_available() else c.replace("sudo ", ""), shell=True, check=True)
        ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().rsplit(maxsplit=1)[-1]

        LOGGER.info(f"\n{prefix} starting export with Edge TPU compiler {ver}...")
        f = str(tflite_model).replace(".tflite", "_edgetpu.tflite")  # Edge TPU model

        cmd = (
            "edgetpu_compiler "
            f'--out_dir "{Path(f).parent}" '
            "--show_operations "
            "--search_delegate "
            "--delegate_search_step 30 "
            "--timeout_sec 180 "
            f'"{tflite_model}"'
        )
        LOGGER.info(f"{prefix} running '{cmd}'")
        subprocess.run(cmd, shell=True)
        self._add_tflite_metadata(f)
        return f, None

    @try_export
    def export_tfjs(self, prefix=colorstr("TensorFlow.js:")):
        """Export YOLO model to TensorFlow.js format."""
        check_requirements("tensorflowjs")
        import tensorflow as tf
        import tensorflowjs as tfjs  # noqa

        LOGGER.info(f"\n{prefix} starting export with tensorflowjs {tfjs.__version__}...")
        f = str(self.file).replace(self.file.suffix, "_web_model")  # js dir
        f_pb = str(self.file.with_suffix(".pb"))  # *.pb path

        gd = tf.Graph().as_graph_def()  # TF GraphDef
        with open(f_pb, "rb") as file:
            gd.ParseFromString(file.read())
        outputs = ",".join(gd_outputs(gd))
        LOGGER.info(f"\n{prefix} output node names: {outputs}")

        quantization = "--quantize_float16" if self.args.half else "--quantize_uint8" if self.args.int8 else ""
        with spaces_in_path(f_pb) as fpb_, spaces_in_path(f) as f_:  # exporter can not handle spaces in path
            cmd = (
                "tensorflowjs_converter "
                f'--input_format=tf_frozen_model {quantization} --output_node_names={outputs} "{fpb_}" "{f_}"'
            )
            LOGGER.info(f"{prefix} running '{cmd}'")
            subprocess.run(cmd, shell=True)

        if " " in f:
            LOGGER.warning(f"{prefix} your model may not work correctly with spaces in path '{f}'.")

        # Add metadata
        YAML.save(Path(f) / "metadata.yaml", self.metadata)  # add metadata.yaml
        return f, None

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

        f, _ = self.export_onnx()
        export_path = Path(f"{Path(f).stem}_rknn_model")
        export_path.mkdir(exist_ok=True)

        rknn = RKNN(verbose=False)
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=self.args.name)
        rknn.load_onnx(model=f)
        rknn.build(do_quantization=False)  # TODO: Add quantization support
        f = f.replace(".onnx", f"-{self.args.name}.rknn")
        rknn.export_rknn(f"{export_path / f}")
        YAML.save(export_path / "metadata.yaml", self.metadata)
        return export_path, None

    @try_export
    def export_imx(self, prefix=colorstr("IMX:")):
        """Export YOLO model to IMX format."""
        gptq = False
        assert LINUX, (
            "export only supported on Linux. "
            "See https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera/documentation/imx500-converter"
        )
        if getattr(self.model, "end2end", False):
            raise ValueError("IMX export is not supported for end2end models.")
        check_requirements(
            ("model-compression-toolkit>=2.4.1", "sony-custom-layers>=0.3.0", "edge-mdt-tpc>=1.1.0", "pydantic<=2.11.7")
        )
        check_requirements("imx500-converter[pt]>=3.16.1")  # Separate requirements for imx500-converter
        check_requirements("mct-quantizers>=1.6.0")  # Separate for compatibility with model-compression-toolkit

        import model_compression_toolkit as mct
        import onnx
        from edgemdt_tpc import get_target_platform_capabilities
        from sony_custom_layers.pytorch import multiclass_nms_with_indices

        LOGGER.info(f"\n{prefix} starting export with model_compression_toolkit {mct.__version__}...")

        # Install Java>=17
        try:
            java_output = subprocess.run(["java", "--version"], check=True, capture_output=True).stdout.decode()
            version_match = re.search(r"(?:openjdk|java) (\d+)", java_output)
            java_version = int(version_match.group(1)) if version_match else 0
            assert java_version >= 17, "Java version too old"
        except (FileNotFoundError, subprocess.CalledProcessError, AssertionError):
            cmd = (["sudo"] if is_sudo_available() else []) + ["apt", "install", "-y", "openjdk-21-jre"]
            subprocess.run(cmd, check=True)

        def representative_dataset_gen(dataloader=self.get_int8_calibration_dataloader(prefix)):
            for batch in dataloader:
                img = batch["img"]
                img = img / 255.0
                yield [img]

        tpc = get_target_platform_capabilities(tpc_version="4.0", device_type="imx500")

        bit_cfg = mct.core.BitWidthConfig()
        if "C2PSA" in self.model.__str__():  # YOLO11
            if self.model.task == "detect":
                layer_names = ["sub", "mul_2", "add_14", "cat_21"]
                weights_memory = 2585350.2439
                n_layers = 238  # 238 layers for fused YOLO11n
            elif self.model.task == "pose":
                layer_names = ["sub", "mul_2", "add_14", "cat_22", "cat_23", "mul_4", "add_15"]
                weights_memory = 2437771.67
                n_layers = 257  # 257 layers for fused YOLO11n-pose
        else:  # YOLOv8
            if self.model.task == "detect":
                layer_names = ["sub", "mul", "add_6", "cat_17"]
                weights_memory = 2550540.8
                n_layers = 168  # 168 layers for fused YOLOv8n
            elif self.model.task == "pose":
                layer_names = ["add_7", "mul_2", "cat_19", "mul", "sub", "add_6", "cat_18"]
                weights_memory = 2482451.85
                n_layers = 187  # 187 layers for fused YOLO11n-pose

        # Check if the model has the expected number of layers
        if len(list(self.model.modules())) != n_layers:
            raise ValueError("IMX export only supported for YOLOv8n and YOLO11n models.")

        for layer_name in layer_names:
            bit_cfg.set_manual_activation_bit_width([mct.core.common.network_editors.NodeNameFilter(layer_name)], 16)

        config = mct.core.CoreConfig(
            mixed_precision_config=mct.core.MixedPrecisionQuantizationConfig(num_of_images=10),
            quantization_config=mct.core.QuantizationConfig(concat_threshold_update=True),
            bit_width_config=bit_cfg,
        )

        resource_utilization = mct.core.ResourceUtilization(weights_memory=weights_memory)

        quant_model = (
            mct.gptq.pytorch_gradient_post_training_quantization(  # Perform Gradient-Based Post Training Quantization
                model=self.model,
                representative_data_gen=representative_dataset_gen,
                target_resource_utilization=resource_utilization,
                gptq_config=mct.gptq.get_pytorch_gptq_config(
                    n_epochs=1000, use_hessian_based_weights=False, use_hessian_sample_attention=False
                ),
                core_config=config,
                target_platform_capabilities=tpc,
            )[0]
            if gptq
            else mct.ptq.pytorch_post_training_quantization(  # Perform post training quantization
                in_module=self.model,
                representative_data_gen=representative_dataset_gen,
                target_resource_utilization=resource_utilization,
                core_config=config,
                target_platform_capabilities=tpc,
            )[0]
        )

        class NMSWrapper(torch.nn.Module):
            """Wrap PyTorch Module with multiclass_nms layer from sony_custom_layers."""

            def __init__(
                self,
                model: torch.nn.Module,
                score_threshold: float = 0.001,
                iou_threshold: float = 0.7,
                max_detections: int = 300,
                task: str = "detect",
            ):
                """
                Initialize NMSWrapper with PyTorch Module and NMS parameters.

                Args:
                    model (torch.nn.Module): Model instance.
                    score_threshold (float): Score threshold for non-maximum suppression.
                    iou_threshold (float): Intersection over union threshold for non-maximum suppression.
                    max_detections (int): The number of detections to return.
                    task (str): Task type, either 'detect' or 'pose'.
                """
                super().__init__()
                self.model = model
                self.score_threshold = score_threshold
                self.iou_threshold = iou_threshold
                self.max_detections = max_detections
                self.task = task

            def forward(self, images):
                """Forward pass with model inference and NMS post-processing."""
                # model inference
                outputs = self.model(images)

                boxes, scores = outputs[0], outputs[1]
                nms_outputs = multiclass_nms_with_indices(
                    boxes=boxes,
                    scores=scores,
                    score_threshold=self.score_threshold,
                    iou_threshold=self.iou_threshold,
                    max_detections=self.max_detections,
                )
                if self.task == "pose":
                    kpts = outputs[2]  # (bs, max_detections, kpts 17*3)
                    out_kpts = torch.gather(kpts, 1, nms_outputs.indices.unsqueeze(-1).expand(-1, -1, kpts.size(-1)))
                    return nms_outputs.boxes, nms_outputs.scores, nms_outputs.labels, out_kpts
                return nms_outputs

        quant_model = NMSWrapper(
            model=quant_model,
            score_threshold=self.args.conf or 0.001,
            iou_threshold=self.args.iou,
            max_detections=self.args.max_det,
            task=self.model.task,
        ).to(self.device)

        f = Path(str(self.file).replace(self.file.suffix, "_imx_model"))
        f.mkdir(exist_ok=True)
        onnx_model = f / Path(str(self.file.name).replace(self.file.suffix, "_imx.onnx"))  # js dir
        mct.exporter.pytorch_export_model(
            model=quant_model, save_model_path=onnx_model, repr_dataset=representative_dataset_gen
        )

        model_onnx = onnx.load(onnx_model)  # load onnx model
        for k, v in self.metadata.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)

        onnx.save(model_onnx, onnx_model)

        subprocess.run(
            ["imxconv-pt", "-i", str(onnx_model), "-o", str(f), "--no-input-persistency", "--overwrite-output"],
            check=True,
        )

        # Needed for imx models.
        with open(f / "labels.txt", "w", encoding="utf-8") as file:
            file.writelines([f"{name}\n" for _, name in self.model.names.items()])

        return f, None

    def _add_tflite_metadata(self, file):
        """Add metadata to *.tflite models per https://ai.google.dev/edge/litert/models/metadata."""
        import zipfile

        with zipfile.ZipFile(file, "a", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metadata.json", json.dumps(self.metadata, indent=2))

    def _pipeline_coreml(self, model, weights_dir=None, prefix=colorstr("CoreML Pipeline:")):
        """Create CoreML pipeline with NMS for YOLO detection models."""
        import coremltools as ct  # noqa

        LOGGER.info(f"{prefix} starting pipeline with coremltools {ct.__version__}...")
        _, _, h, w = list(self.im.shape)  # BCHW

        # Output shapes
        spec = model.get_spec()
        out0, out1 = iter(spec.description.output)
        if MACOS:
            from PIL import Image

            img = Image.new("RGB", (w, h))  # w=192, h=320
            out = model.predict({"image": img})
            out0_shape = out[out0.name].shape  # (3780, 80)
            out1_shape = out[out1.name].shape  # (3780, 4)
        else:  # linux and windows can not run model.predict(), get sizes from PyTorch model output y
            out0_shape = self.output_shape[2], self.output_shape[1] - 4  # (3780, 80)
            out1_shape = self.output_shape[2], 4  # (3780, 4)

        # Checks
        names = self.metadata["names"]
        nx, ny = spec.description.input[0].type.imageType.width, spec.description.input[0].type.imageType.height
        _, nc = out0_shape  # number of anchors, number of classes
        assert len(names) == nc, f"{len(names)} names found for nc={nc}"  # check

        # Define output shapes (missing)
        out0.type.multiArrayType.shape[:] = out0_shape  # (3780, 80)
        out1.type.multiArrayType.shape[:] = out1_shape  # (3780, 4)

        # Model from spec
        model = ct.models.MLModel(spec, weights_dir=weights_dir)

        # 3. Create NMS protobuf
        nms_spec = ct.proto.Model_pb2.Model()
        nms_spec.specificationVersion = spec.specificationVersion
        for i in range(2):
            decoder_output = model._spec.description.output[i].SerializeToString()
            nms_spec.description.input.add()
            nms_spec.description.input[i].ParseFromString(decoder_output)
            nms_spec.description.output.add()
            nms_spec.description.output[i].ParseFromString(decoder_output)

        nms_spec.description.output[0].name = "confidence"
        nms_spec.description.output[1].name = "coordinates"

        output_sizes = [nc, 4]
        for i in range(2):
            ma_type = nms_spec.description.output[i].type.multiArrayType
            ma_type.shapeRange.sizeRanges.add()
            ma_type.shapeRange.sizeRanges[0].lowerBound = 0
            ma_type.shapeRange.sizeRanges[0].upperBound = -1
            ma_type.shapeRange.sizeRanges.add()
            ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]
            ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i]
            del ma_type.shape[:]

        nms = nms_spec.nonMaximumSuppression
        nms.confidenceInputFeatureName = out0.name  # 1x507x80
        nms.coordinatesInputFeatureName = out1.name  # 1x507x4
        nms.confidenceOutputFeatureName = "confidence"
        nms.coordinatesOutputFeatureName = "coordinates"
        nms.iouThresholdInputFeatureName = "iouThreshold"
        nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
        nms.iouThreshold = self.args.iou
        nms.confidenceThreshold = self.args.conf
        nms.pickTop.perClass = True
        nms.stringClassLabels.vector.extend(names.values())
        nms_model = ct.models.MLModel(nms_spec)

        # 4. Pipeline models together
        pipeline = ct.models.pipeline.Pipeline(
            input_features=[
                ("image", ct.models.datatypes.Array(3, ny, nx)),
                ("iouThreshold", ct.models.datatypes.Double()),
                ("confidenceThreshold", ct.models.datatypes.Double()),
            ],
            output_features=["confidence", "coordinates"],
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

    def __init__(self, model, im):
        """
        Initialize the IOSDetectModel class with a YOLO model and example image.

        Args:
            model (torch.nn.Module): The YOLO model to wrap.
            im (torch.Tensor): Example input tensor with shape (B, C, H, W).
        """
        super().__init__()
        _, _, h, w = im.shape  # batch, channel, height, width
        self.model = model
        self.nc = len(model.names)  # number of classes
        if w == h:
            self.normalize = 1.0 / w  # scalar
        else:
            self.normalize = torch.tensor([1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h])  # broadcast (slower, smaller)

    def forward(self, x):
        """Normalize predictions of object detection model with input size-dependent factors."""
        xywh, cls = self.model(x)[0].transpose(0, 1).split((4, self.nc), 1)
        return cls, xywh * self.normalize  # confidence (3780, 80), coordinates (3780, 4)


class NMSModel(torch.nn.Module):
    """Model wrapper with embedded NMS for Detect, Segment, Pose and OBB."""

    def __init__(self, model, args):
        """
        Initialize the NMSModel.

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
        """
        Perform inference with NMS post-processing. Supports Detect, Segment, OBB and Pose.

        Args:
            x (torch.Tensor): The preprocessed tensor with shape (N, 3, H, W).

        Returns:
            (torch.Tensor): List of detections, each an (N, max_det, 4 + 2 + extra_shape) Tensor where N is the
                number of detections after NMS.
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
            if self.is_tf:
                # TFLite GatherND error if mask is empty
                score *= mask
                # Explicit length otherwise reshape error, hardcoded to `self.args.max_det * 5`
                mask = score.topk(min(self.args.max_det * 5, score.shape[0])).indices
            box, score, cls, extra = box[mask], score[mask], cls[mask], extra[mask]
            nmsbox = box.clone()
            # `8` is the minimum value experimented to get correct NMS results for obb
            multiplier = 8 if self.obb else 1
            # Normalize boxes for NMS since large values for class offset causes issue with int8 quantization
            if self.args.format == "tflite":  # TFLite is already normalized
                nmsbox *= multiplier
            else:
                nmsbox = multiplier * nmsbox / torch.tensor(x.shape[2:], **kwargs).max()
            if not self.args.agnostic_nms:  # class-specific NMS
                end = 2 if self.obb else 4
                # fully explicit expansion otherwise reshape error
                # large max_wh causes issues when quantizing
                cls_offset = cls.reshape(-1, 1).expand(nmsbox.shape[0], end)
                offbox = nmsbox[:, :end] + cls_offset * multiplier
                nmsbox = torch.cat((offbox, nmsbox[:, end:]), dim=-1)
            nms_fn = (
                partial(
                    nms_rotated,
                    use_triu=not (
                        self.is_tf
                        or (self.args.opset or 14) < 14
                        or (self.args.format == "openvino" and self.args.int8)  # OpenVINO int8 error with triu
                    ),
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
