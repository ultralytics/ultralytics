# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Export a YOLO PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit.

Format                  | `format=argument`         | Model
---                     | ---                       | ---
PyTorch                 | -                         | yolo11n.pt
TorchScript             | `torchscript`             | yolo11n.torchscript
ONNX                    | `onnx`                    | yolo11n.onnx
TensorRT                | `engine`                  | yolo11n.engine

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
                         yolo11n.onnx               # ONNX Runtime
                         yolo11n.engine             # TensorRT

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
    ROOT,
    SETTINGS,
    TORCH_VERSION,
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
from ultralytics.utils.export import onnx2engine, torch2imx, torch2onnx
from ultralytics.utils.files import file_size, spaces_in_path
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.nms import TorchNMS
from ultralytics.utils.ops import Profile
from ultralytics.utils.patches import arange_patch
from ultralytics.utils.torch_utils import TORCH_1_11, TORCH_1_13, TORCH_2_1, TORCH_2_4, TORCH_2_9, select_device


def export_formats():
    """Return a dictionary of Ultralytics YOLO export formats."""
    x = [
        ["PyTorch", "-", ".pt", True, True, []],
        [
            "TorchScript",
            "torchscript",
            ".torchscript",
            True,
            True,
            ["batch", "optimize", "half", "nms", "dynamic"],
        ],
        ["ONNX", "onnx", ".onnx", True, True, ["batch", "dynamic", "half", "opset", "simplify", "nms"]],
        [
            "TensorRT",
            "engine",
            ".engine",
            False,
            True,
            ["batch", "dynamic", "half", "int8", "simplify", "nms", "fraction"],
        ],
    ]
    return dict(zip(["Format", "Argument", "Suffix", "CPU", "GPU", "Arguments"], zip(*x)))


def best_onnx_opset(onnx, cuda=False) -> int:
    """Return max ONNX opset for this torch version with ONNX fallback."""
    version = ".".join(TORCH_VERSION.split(".")[:2])
    if TORCH_2_4:  # _constants.ONNX_MAX_OPSET first defined in torch 1.13
        opset = torch.onnx.utils._constants.ONNX_MAX_OPSET - 1  # use second-latest version for safety
        if cuda:
            opset -= 2  # fix CUDA ONNXRuntime NMS squeeze op errors
    else:
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
    """
    A class for exporting YOLO models to various formats.

    This class provides functionality to export YOLO models to different formats including ONNX, TensorRT,
    and others. It handles format validation, device selection, model preparation, and the actual export
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
        export_engine: Export model to TensorRT format.

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
            engine,
        ) = flags  # export booleans

        # Device
        dla = None
        if engine and self.args.device is None:
            LOGGER.warning("TensorRT requires GPU export, automatically assigning device=0")
            self.args.device = "0"
        if engine and "dla" in str(self.args.device):  # convert int/list to str first
            dla = self.args.device.rsplit(":", 1)[-1]
            self.args.device = "0"  # update device to "0"
            assert dla in {"0", "1"}, f"Expected self.args.device='dla:0' or 'dla:1, but got {self.args.device}."
        self.device = select_device("cpu" if self.args.device is None else self.args.device)

        # Argument compatibility checks
        fmt_keys = fmts_dict["Arguments"][flags.index(True) + 1]
        validate_args(fmt, self.args, fmt_keys)
        if not hasattr(model, "names"):
            model.names = default_class_names()
        model.names = check_class_names(model.names)
        if self.args.half and self.args.int8:
            LOGGER.warning("half=True and int8=True are mutually exclusive, setting half=False.")
            self.args.half = False
        if self.args.half and (onnx or jit) and self.device.type == "cpu":
            LOGGER.warning("half=True only compatible with GPU export, i.e. use device=0, setting half=False.")
            self.args.half = False
        self.imgsz = check_imgsz(self.args.imgsz, stride=model.stride, min_dim=2)  # check image size
        if self.args.optimize:
            assert self.device.type == "cpu", "optimize=True not compatible with cuda devices, i.e. use device='cpu'"
        if self.args.nms:
            assert not isinstance(model, ClassificationModel), "'nms=True' is not valid for classification models."
            assert not onnx or TORCH_1_13, "ONNX export with NMS requires torch>=1.13"
            if getattr(model, "end2end", False) or isinstance(model.model[-1], RTDETRDecoder):
                LOGGER.warning("'nms=True' is not available for end2end models. Forcing 'nms=False'.")
                self.args.nms = False
            self.args.conf = self.args.conf or 0.25  # set conf default value for nms export
        if (engine or self.args.nms) and self.args.dynamic and self.args.batch == 1:
            LOGGER.warning(
                f"'dynamic=True' model with '{'nms=True' if self.args.nms else f'format={self.args.format}'}' requires max batch size, i.e. 'batch=16'"
            )
        if isinstance(model, WorldModel):
            LOGGER.warning(
                "YOLOWorld (original version) export is not supported to any format. "
                "YOLOWorldv2 models (i.e. 'yolov8s-worldv2.pt') only support export to "
                "(torchscript, onnx, engine) formats. "
                "See https://docs.ultralytics.com/models/yolo-world for details."
            )
            model.clip_model = None  # openvino int8 export error: https://github.com/ultralytics/ultralytics/pull/18445
        if self.args.int8 and not self.args.data:
            self.args.data = DEFAULT_CFG.data or TASK2DATA[getattr(model, "task", "detect")]  # assign default data
            LOGGER.warning(
                f"INT8 export requires a missing 'data' arg for calibration. Using default 'data={self.args.data}'."
            )
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

        for m in model.modules():
            if isinstance(m, Classify):
                m.export = True
            if isinstance(m, (Detect, RTDETRDecoder)):  # includes all Detect subclasses like Segment, Pose, OBB
                m.dynamic = self.args.dynamic
                m.export = True
                m.format = self.args.format
                m.max_det = self.args.max_det
                m.xyxy = self.args.nms
                if hasattr(model, "pe") and hasattr(m, "fuse"):  # for YOLOE models
                    m.fuse(model.pe.to(self.device))
            elif isinstance(m, C2f):
                # EdgeTPU does not support FlexSplitV while split provides cleaner ONNX graph
                m.forward = m.forward_split

        y = None
        for _ in range(2):  # dry runs
            y = NMSModel(model, self.args)(im) if self.args.nms else model(im)
        if self.args.half and (onnx or jit) and self.device.type != "cpu":
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
            predict_data = ""
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
        requirements = ["onnx>=1.12.0"]
        if self.args.simplify:
            requirements += ["onnxslim>=0.1.71", "onnxruntime" + ("-gpu" if torch.cuda.is_available() else "")]
        check_requirements(requirements)
        import onnx

        opset = self.args.opset or best_onnx_opset(onnx, cuda="cuda" in self.device.type)
        LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__} opset {opset}...")
        if self.args.nms:
            assert TORCH_1_13, f"'nms=True' ONNX export requires torch>=1.13 (found torch=={TORCH_VERSION})"

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

        onnx.save(model_onnx, f)
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

    def add_callback(self, event: str, callback):
        """Append the given callback to the specified event."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Execute all callbacks for a given event."""
        for callback in self.callbacks.get(event, []):
            callback(self)


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
        self.is_tf = False # No TensorFlow support

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
            if self.is_tf or (self.args.format == "onnx" and self.obb):
                # TFLite GatherND error if mask is empty
                score *= mask
                # Explicit length otherwise reshape error, hardcoded to `self.args.max_det * 5`
                mask = score.topk(min(self.args.max_det * 5, score.shape[0])).indices
            box, score, cls, extra = box[mask], score[mask], cls[mask], extra[mask]
            nmsbox = box.clone()
            # `8` is the minimum value experimented to get correct NMS results for obb
            multiplier = (8 if self.obb else 1) / max(len(self.model.names), 1)
            # Normalize boxes for NMS since large values for class offset causes issue with int8 quantization
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
                        (self.args.opset or 14) < 14
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
