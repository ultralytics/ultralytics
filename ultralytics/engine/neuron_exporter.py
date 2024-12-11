# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Export a YOLOv8 PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit.

Format                  | `format=argument`         | Model
---                     | ---                       | ---
PyTorch                 | -                         | yolov8n.pt
TorchScript             | `torchscript`             | yolov8n.torchscript
AWS NeuronX             | `neuronx`                 | yolov8n.neuronx
ONNX                    | `onnx`                    | yolov8n.onnx
OpenVINO                | `openvino`                | yolov8n_openvino_model/
TensorRT                | `engine`                  | yolov8n.engine
CoreML                  | `coreml`                  | yolov8n.mlpackage
TensorFlow SavedModel   | `saved_model`             | yolov8n_saved_model/
TensorFlow GraphDef     | `pb`                      | yolov8n.pb
TensorFlow Lite         | `tflite`                  | yolov8n.tflite
TensorFlow Edge TPU     | `edgetpu`                 | yolov8n_edgetpu.tflite
TensorFlow.js           | `tfjs`                    | yolov8n_web_model/
PaddlePaddle            | `paddle`                  | yolov8n_paddle_model/
NCNN                    | `ncnn`                    | yolov8n_ncnn_model/

Requirements:
    $ pip install "ultralytics[export]"

Python:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    results = model.export(format='onnx')

CLI:
    $ yolo mode=export model=yolov8n.pt format=onnx

Inference:
    $ yolo predict model=yolov8n.pt                 # PyTorch
                         yolov8n.torchscript        # TorchScript
                         yolov8.neuronx             # AWS NeuronX
                         yolov8n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                         yolov8n_openvino_model     # OpenVINO
                         yolov8n.engine             # TensorRT
                         yolov8n.mlpackage          # CoreML (macOS-only)
                         yolov8n_saved_model        # TensorFlow SavedModel
                         yolov8n.pb                 # TensorFlow GraphDef
                         yolov8n.tflite             # TensorFlow Lite
                         yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                         yolov8n_paddle_model       # PaddlePaddle
                         yolov8n_ncnn_model         # NCNN

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolov5/yolov8n_web_model public/yolov8n_web_model
    $ npm start
"""

import json
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch

from ultralytics.cfg import TASK2DATA
from ultralytics.engine.exporter import Exporter
from ultralytics.nn.autobackend import check_class_names, default_class_names
from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder
from ultralytics.nn.tasks import WorldModel
from ultralytics.utils import (
    DEFAULT_CFG,
    LINUX,
    LOGGER,
    __version__,
    colorstr,
    get_default_args,
)
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.files import file_size
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import select_device, smart_inference_mode


def export_formats():
    """YOLOv8 export formats."""
    import pandas  # scope for faster 'import ultralytics'

    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["AWS NeuronX", "neuronx", ".neuronx", True, True],
        ["AWS Neuron", "neuron", ".neuron", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorRT", "engine", ".engine", False, True],
        ["CoreML", "coreml", ".mlpackage", True, False],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
        ["TensorFlow GraphDef", "pb", ".pb", True, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", True, False],
        ["TensorFlow.js", "tfjs", "_web_model", True, False],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True],
        ["NCNN", "ncnn", "_ncnn_model", True, True],
    ]
    return pandas.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])


def gd_outputs(gd):
    """TensorFlow GraphDef model output node names."""
    name_list, input_list = [], []
    for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
        name_list.append(node.name)
        input_list.extend(node.input)
    return sorted(f"{x}:0" for x in list(set(name_list) - set(input_list)) if not x.startswith("NoOp"))


def try_export(inner_func):
    """YOLOv8 export decorator, i.e. @try_export."""
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        """Export a model."""
        prefix = inner_args["prefix"]
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f"{prefix} export success ✅ {dt.t:.1f}s, saved as '{f}' ({file_size(f):.1f} MB)")
            return f, model
        except Exception as e:
            LOGGER.info(f"{prefix} export failure ❌ {dt.t:.1f}s: {e}")
            raise e

    return outer_func


class NeuronExporter(Exporter):
    """
    A class for exporting a model.

    Attributes:
        args (SimpleNamespace): Configuration for the exporter.
        callbacks (list, optional): List of callback functions. Defaults to None.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)

    @smart_inference_mode()
    def __call__(self, model=None) -> str:
        """Returns list of exported files/dirs after running callbacks."""
        self.run_callbacks("on_export_start")
        t = time.time()
        fmt = self.args.format.lower()  # to lowercase
        if fmt in {"tensorrt", "trt"}:  # 'engine' aliases
            fmt = "engine"
        if fmt in {"mlmodel", "mlpackage", "mlprogram", "apple", "ios", "coreml"}:  # 'coreml' aliases
            fmt = "coreml"
        fmts = tuple(export_formats()["Argument"][1:])  # available export formats
        flags = [x == fmt for x in fmts]
        if sum(flags) != 1:
            raise ValueError(f"Invalid export format='{fmt}'. Valid formats are {fmts}")
        (
            jit,
            neuronx,
            neuron,
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
            ncnn,
        ) = flags  # export booleans
        is_tf_format = any((saved_model, pb, tflite, edgetpu, tfjs))

        # Device
        if fmt == "engine" and self.args.device is None:
            LOGGER.warning("WARNING ⚠️ TensorRT requires GPU export, automatically assigning device=0")
            self.args.device = "0"
        self.device = select_device("cpu" if self.args.device is None else self.args.device)

        # Checks
        if not hasattr(model, "names"):
            model.names = default_class_names()
        model.names = check_class_names(model.names)
        if self.args.half and self.args.int8:
            LOGGER.warning("WARNING ⚠️ half=True and int8=True are mutually exclusive, setting half=False.")
            self.args.half = False
        if self.args.half and onnx and self.device.type == "cpu":
            LOGGER.warning("WARNING ⚠️ half=True only compatible with GPU export, i.e. use device=0")
            self.args.half = False
            assert not self.args.dynamic, "half=True not compatible with dynamic=True, i.e. use only one."
        self.imgsz = check_imgsz(self.args.imgsz, stride=model.stride, min_dim=2)  # check image size
        if self.args.int8 and (engine or xml):
            self.args.dynamic = True  # enforce dynamic to export TensorRT INT8; ensures ONNX is dynamic
        if self.args.optimize:
            assert not ncnn, "optimize=True not compatible with format='ncnn', i.e. use optimize=False"
            assert self.device.type == "cpu", "optimize=True not compatible with cuda devices, i.e. use device='cpu'"
        if edgetpu:
            if not LINUX:
                raise SystemError("Edge TPU export only supported on Linux. See https://coral.ai/docs/edgetpu/compiler")
            elif self.args.batch != 1:  # see github.com/ultralytics/ultralytics/pull/13420
                LOGGER.warning("WARNING ⚠️ Edge TPU export requires batch size 1, setting batch=1.")
                self.args.batch = 1
        if isinstance(model, WorldModel):
            LOGGER.warning(
                "WARNING ⚠️ YOLOWorld (original version) export is not supported to any format.\n"
                "WARNING ⚠️ YOLOWorldv2 models (i.e. 'yolov8s-worldv2.pt') only support export to "
                "(torchscript, onnx, openvino, engine, coreml) formats. "
                "See https://docs.ultralytics.com/models/yolo-world for details."
            )
        if self.args.int8 and not self.args.data:
            self.args.data = DEFAULT_CFG.data or TASK2DATA[getattr(model, "task", "detect")]  # assign default data
            LOGGER.warning(
                "WARNING ⚠️ INT8 export requires a missing 'data' arg for calibration. "
                f"Using default 'data={self.args.data}'."
            )
        # Input
        im = torch.zeros(self.args.batch, 3, *self.imgsz).to(self.device)
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
            if isinstance(m, (Detect, RTDETRDecoder)):  # includes all Detect subclasses like Segment, Pose, OBB
                m.dynamic = self.args.dynamic
                m.export = True
                m.format = self.args.format
            elif isinstance(m, C2f) and not is_tf_format:
                # EdgeTPU does not support FlexSplitV while split provides cleaner ONNX graph
                m.forward = m.forward_split

        y = None
        for _ in range(2):
            y = model(im)  # dry runs
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
        description = f'Ultralytics {self.pretty_name} model {f"trained on {data}" if data else ""}'
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
        }  # model metadata
        if model.task == "pose":
            self.metadata["kpt_shape"] = model.model[-1].kpt_shape

        LOGGER.info(
            f"\n{colorstr('PyTorch:')} starting from '{file}' with input shape {tuple(im.shape)} BCHW and "
            f'output shape(s) {self.output_shape} ({file_size(file):.1f} MB)'
        )

        # Exports
        f = [""] * len(fmts)  # exported filenames
        if jit or ncnn:  # TorchScript
            f[0], _ = self.export_torchscript()
        if engine:  # TensorRT required before ONNX
            f[1], _ = self.export_engine()
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
                f[7], _ = self.export_tflite(keras_model=keras_model, nms=False, agnostic_nms=self.args.agnostic_nms)
            if edgetpu:
                f[8], _ = self.export_edgetpu(tflite_model=Path(f[5]) / f"{self.file.stem}_full_integer_quant.tflite")
            if tfjs:
                f[9], _ = self.export_tfjs()
        if paddle:  # PaddlePaddle
            f[10], _ = self.export_paddle()
        if ncnn:  # NCNN
            f[11], _ = self.export_ncnn()
        if neuronx:  # NeuronX
            f[12], _ = self.export_neuronx()
        if neuron:  # Neuron
            f[13], _ = self.export_neuron()

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
            predict_data = f"data={data}" if model.task == "segment" and fmt == "pb" else ""
            q = "int8" if self.args.int8 else "half" if self.args.half else ""  # quantization
            LOGGER.info(
                f'\nExport complete ({time.time() - t:.1f}s)'
                f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                f'\nPredict:         yolo predict task={model.task} model={f} imgsz={imgsz} {q} {predict_data}'
                f'\nValidate:        yolo val task={model.task} model={f} imgsz={imgsz} data={data} {q} {s}'
                f'\nVisualize:       https://netron.app'
            )

        self.run_callbacks("on_export_end")
        return f  # return list of exported files/dirs

    @try_export
    def export_neuronx(self, prefix=colorstr("AWS NeuronX:")):
        import torch_neuronx

        """YOLOv8 NeuronX model export."""
        LOGGER.info(f"\n{prefix} starting export with torch {torch_neuronx.__version__}...")
        f = self.file.with_suffix(".neuronx")
        ts = torch_neuronx.trace(self.model, self.im, strict=False)
        extra_files = {"config.txt": json.dumps(self.metadata)}  # torch._C.ExtraFilesMap()
        ts.save(str(f), _extra_files=extra_files)
        return f, None

    @try_export
    def export_neuron(self, prefix=colorstr("AWS Neuron:")):
        import torch_neuron

        """YOLOv8 Neuron model export."""
        LOGGER.info(f"\n{prefix} starting export with torch {torch_neuron.__version__}...")
        f = self.file.with_suffix(".neuron")
        ts = torch_neuron.trace(self.model, self.im, strict=False)
        extra_files = {"config.txt": json.dumps(self.metadata)}
        ts.save(str(f), _extra_files=extra_files)
        return f, None
