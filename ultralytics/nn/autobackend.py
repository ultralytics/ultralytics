# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import ast
import json
import platform
import zipfile
from collections import OrderedDict, namedtuple
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ultralytics.utils import ARM64, IS_JETSON, IS_RASPBERRYPI, LINUX, LOGGER, PYTHON_VERSION, ROOT, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_version, check_yaml, is_rockchip
from ultralytics.utils.downloads import attempt_download_asset, is_url


def check_class_names(names):
    """
    Check class names.

    Map imagenet class codes to human-readable names if required. Convert lists to dicts.
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
            names_map = yaml_load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]  # human-readable names
            names = {k: names_map[v] for k, v in names.items()}
    return names


def default_class_names(data=None):
    """Applies default class names to an input YAML file or returns numerical class names."""
    if data:
        try:
            return yaml_load(check_yaml(data))["names"]
        except Exception:
            pass
    return {i: f"class{i}" for i in range(999)}  # return default if above errors


class AutoBackend(nn.Module):
    """
    Handles dynamic backend selection for running inference using Ultralytics YOLO models.

    The AutoBackend class is designed to provide an abstraction layer for various inference engines. It supports a wide
    range of formats, each with specific naming conventions as outlined below:

        Supported Formats and Naming Conventions:
            | Format                | File Suffix       |
            | --------------------- | ----------------- |
            | PyTorch               | *.pt              |
            | TorchScript           | *.torchscript     |
            | ONNX Runtime          | *.onnx            |
            | ONNX OpenCV DNN       | *.onnx (dnn=True) |
            | OpenVINO              | *openvino_model/  |
            | CoreML                | *.mlpackage       |
            | TensorRT              | *.engine          |
            | TensorFlow SavedModel | *_saved_model/    |
            | TensorFlow GraphDef   | *.pb              |
            | TensorFlow Lite       | *.tflite          |
            | TensorFlow Edge TPU   | *_edgetpu.tflite  |
            | PaddlePaddle          | *_paddle_model/   |
            | MNN                   | *.mnn             |
            | NCNN                  | *_ncnn_model/     |
            | IMX                   | *_imx_model/      |
            | RKNN                  | *_rknn_model/     |

    This class offers dynamic backend switching capabilities based on the input model format, making it easier to deploy
    models across various platforms.
    """

    @torch.no_grad()
    def __init__(
        self,
        weights="yolo11n.pt",
        device=torch.device("cpu"),
        dnn=False,
        data=None,
        fp16=False,
        batch=1,
        fuse=True,
        verbose=True,
    ):
        """
        Initialize the AutoBackend for inference.

        Args:
            weights (str | torch.nn.Module): Path to the model weights file or a module instance. Defaults to 'yolo11n.pt'.
            device (torch.device): Device to run the model on. Defaults to CPU.
            dnn (bool): Use OpenCV DNN module for ONNX inference. Defaults to False.
            data (str | Path | optional): Path to the additional data.yaml file containing class names. Optional.
            fp16 (bool): Enable half-precision inference. Supported only on specific backends. Defaults to False.
            batch (int): Batch-size to assume for inference.
            fuse (bool): Fuse Conv2D + BatchNorm layers for optimization. Defaults to True.
            verbose (bool): Enable verbose logging. Defaults to True.
        """
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        nn_module = isinstance(weights, torch.nn.Module)
        (
            pt,
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
            triton,
        ) = self._model_type(w)
        fp16 &= pt or jit or onnx or xml or engine or nn_module or triton  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu or rknn  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        end2end = False  # default end2end
        model, metadata, task = None, None, None

        # Set device
        cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA
        if cuda and not any([nn_module, pt, jit, engine, onnx, paddle]):  # GPU dataloader formats
            device = torch.device("cpu")
            cuda = False

        # Download if not local
        if not (pt or triton or nn_module):
            w = attempt_download_asset(w)

        # In-memory PyTorch model
        if nn_module:
            model = weights.to(device)
            if fuse:
                model = model.fuse(verbose=verbose)
            if hasattr(model, "kpt_shape"):
                kpt_shape = model.kpt_shape  # pose-only
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
            pt = True

        # PyTorch
        elif pt:
            from ultralytics.nn.tasks import attempt_load_weights

            model = attempt_load_weights(
                weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse
            )
            if hasattr(model, "kpt_shape"):
                kpt_shape = model.kpt_shape  # pose-only
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

        # TorchScript
        elif jit:
            LOGGER.info(f"Loading {w} for TorchScript inference...")
            extra_files = {"config.txt": ""}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files["config.txt"]:  # load metadata dict
                metadata = json.loads(extra_files["config.txt"], object_hook=lambda x: dict(x.items()))

        # ONNX OpenCV DNN
        elif dnn:
            LOGGER.info(f"Loading {w} for ONNX OpenCV DNN inference...")
            check_requirements("opencv-python>=4.5.4")
            net = cv2.dnn.readNetFromONNX(w)

        # ONNX Runtime and IMX
        elif onnx or imx:
            LOGGER.info(f"Loading {w} for ONNX Runtime inference...")
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
            if IS_RASPBERRYPI or IS_JETSON:
                # Fix 'numpy.linalg._umath_linalg' has no attribute '_ilp64' for TF SavedModel on RPi and Jetson
                check_requirements("numpy==1.23.5")
            import onnxruntime

            providers = ["CPUExecutionProvider"]
            if cuda and "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            elif cuda:  # Only log warning if CUDA was requested but unavailable
                LOGGER.warning("WARNING ⚠️ Failed to start ONNX Runtime with CUDA. Using CPU...")
                device = torch.device("cpu")
                cuda = False
            LOGGER.info(f"Using ONNX Runtime {providers[0]}")
            if onnx:
                session = onnxruntime.InferenceSession(w, providers=providers)
            else:
                check_requirements(
                    ["model-compression-toolkit==2.1.1", "sony-custom-layers[torch]==0.2.0", "onnxruntime-extensions"]
                )
                w = next(Path(w).glob("*.onnx"))
                LOGGER.info(f"Loading {w} for ONNX IMX inference...")
                import mct_quantizers as mctq
                from sony_custom_layers.pytorch.object_detection import nms_ort  # noqa

                session = onnxruntime.InferenceSession(
                    w, mctq.get_ort_session_options(), providers=["CPUExecutionProvider"]
                )
                task = "detect"

            output_names = [x.name for x in session.get_outputs()]
            metadata = session.get_modelmeta().custom_metadata_map
            dynamic = isinstance(session.get_outputs()[0].shape[0], str)
            fp16 = True if "float16" in session.get_inputs()[0].type else False
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

        # OpenVINO
        elif xml:
            LOGGER.info(f"Loading {w} for OpenVINO inference...")
            check_requirements("openvino>=2024.0.0")
            import openvino as ov

            core = ov.Core()
            w = Path(w)
            if not w.is_file():  # if not *.xml
                w = next(w.glob("*.xml"))  # get *.xml file from *_openvino_model dir
            ov_model = core.read_model(model=str(w), weights=w.with_suffix(".bin"))
            if ov_model.get_parameters()[0].get_layout().empty:
                ov_model.get_parameters()[0].set_layout(ov.Layout("NCHW"))

            # OpenVINO inference modes are 'LATENCY', 'THROUGHPUT' (not recommended), or 'CUMULATIVE_THROUGHPUT'
            inference_mode = "CUMULATIVE_THROUGHPUT" if batch > 1 else "LATENCY"
            LOGGER.info(f"Using OpenVINO {inference_mode} mode for batch={batch} inference...")
            ov_compiled_model = core.compile_model(
                ov_model,
                device_name="AUTO",  # AUTO selects best available device, do not modify
                config={"PERFORMANCE_HINT": inference_mode},
            )
            input_name = ov_compiled_model.input().get_any_name()
            metadata = w.parent / "metadata.yaml"

        # TensorRT
        elif engine:
            LOGGER.info(f"Loading {w} for TensorRT inference...")

            if IS_JETSON and PYTHON_VERSION <= "3.8.0":
                # fix error: `np.bool` was a deprecated alias for the builtin `bool` for JetPack 4 with Python <= 3.8.0
                check_requirements("numpy==1.23.5")

            try:
                import tensorrt as trt  # noqa https://developer.nvidia.com/nvidia-tensorrt-download
            except ImportError:
                if LINUX:
                    check_requirements("tensorrt>7.0.0,!=10.1.0")
                import tensorrt as trt  # noqa
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
                except UnicodeDecodeError:
                    f.seek(0)  # engine file may lack embedded Ultralytics metadata
                dla = metadata.get("dla", None)
                if dla is not None:
                    runtime.DLA_core = int(dla)
                model = runtime.deserialize_cuda_engine(f.read())  # read engine

            # Model context
            try:
                context = model.create_execution_context()
            except Exception as e:  # model is None
                LOGGER.error(f"ERROR: TensorRT model exported with a different version than {trt.__version__}\n")
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
            batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size

        # CoreML
        elif coreml:
            LOGGER.info(f"Loading {w} for CoreML inference...")
            import coremltools as ct

            model = ct.models.MLModel(w)
            metadata = dict(model.user_defined_metadata)

        # TF SavedModel
        elif saved_model:
            LOGGER.info(f"Loading {w} for TensorFlow SavedModel inference...")
            import tensorflow as tf

            keras = False  # assume TF1 saved_model
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
            metadata = Path(w) / "metadata.yaml"

        # TF GraphDef
        elif pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            LOGGER.info(f"Loading {w} for TensorFlow GraphDef inference...")
            import tensorflow as tf

            from ultralytics.engine.exporter import gd_outputs

            def wrap_frozen_graph(gd, inputs, outputs):
                """Wrap frozen graphs for deployment."""
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            gd = tf.Graph().as_graph_def()  # TF GraphDef
            with open(w, "rb") as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
            try:  # find metadata in SavedModel alongside GraphDef
                metadata = next(Path(w).resolve().parent.rglob(f"{Path(w).stem}_saved_model*/metadata.yaml"))
            except StopIteration:
                pass

        # TFLite or TFLite Edge TPU
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf

                Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                device = device[3:] if str(device).startswith("tpu") else ":0"
                LOGGER.info(f"Loading {w} on device {device[1:]} for TensorFlow Lite Edge TPU inference...")
                delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                    platform.system()
                ]
                interpreter = Interpreter(
                    model_path=w,
                    experimental_delegates=[load_delegate(delegate, options={"device": device})],
                )
                device = "cpu"  # Required, otherwise PyTorch will try to use the wrong device
            else:  # TFLite
                LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
                interpreter = Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # Load metadata
            try:
                with zipfile.ZipFile(w, "r") as model:
                    meta_file = model.namelist()[0]
                    metadata = ast.literal_eval(model.read(meta_file).decode("utf-8"))
            except zipfile.BadZipFile:
                pass

        # TF.js
        elif tfjs:
            raise NotImplementedError("YOLOv8 TF.js inference is not currently supported.")

        # PaddlePaddle
        elif paddle:
            LOGGER.info(f"Loading {w} for PaddlePaddle inference...")
            check_requirements("paddlepaddle-gpu" if cuda else "paddlepaddle")
            import paddle.inference as pdi  # noqa

            w = Path(w)
            if not w.is_file():  # if not *.pdmodel
                w = next(w.rglob("*.pdmodel"))  # get *.pdmodel file from *_paddle_model dir
            config = pdi.Config(str(w), str(w.with_suffix(".pdiparams")))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()
            metadata = w.parents[1] / "metadata.yaml"

        # MNN
        elif mnn:
            LOGGER.info(f"Loading {w} for MNN inference...")
            check_requirements("MNN")  # requires MNN
            import os

            import MNN

            config = {"precision": "low", "backend": "CPU", "numThread": (os.cpu_count() + 1) // 2}
            rt = MNN.nn.create_runtime_manager((config,))
            net = MNN.nn.load_module_from_file(w, [], [], runtime_manager=rt, rearrange=True)

            def torch_to_mnn(x):
                return MNN.expr.const(x.data_ptr(), x.shape)

            metadata = json.loads(net.get_info()["bizCode"])

        # NCNN
        elif ncnn:
            LOGGER.info(f"Loading {w} for NCNN inference...")
            check_requirements("git+https://github.com/Tencent/ncnn.git" if ARM64 else "ncnn")  # requires NCNN
            import ncnn as pyncnn

            net = pyncnn.Net()
            net.opt.use_vulkan_compute = cuda
            w = Path(w)
            if not w.is_file():  # if not *.param
                w = next(w.glob("*.param"))  # get *.param file from *_ncnn_model dir
            net.load_param(str(w))
            net.load_model(str(w.with_suffix(".bin")))
            metadata = w.parent / "metadata.yaml"

        # NVIDIA Triton Inference Server
        elif triton:
            check_requirements("tritonclient[all]")
            from ultralytics.utils.triton import TritonRemoteModel

            model = TritonRemoteModel(w)
            metadata = model.metadata

        # RKNN
        elif rknn:
            if not is_rockchip():
                raise OSError("RKNN inference is only supported on Rockchip devices.")
            LOGGER.info(f"Loading {w} for RKNN inference...")
            check_requirements("rknn-toolkit-lite2")
            from rknnlite.api import RKNNLite

            w = Path(w)
            if not w.is_file():  # if not *.rknn
                w = next(w.rglob("*.rknn"))  # get *.rknn file from *_rknn_model dir
            rknn_model = RKNNLite()
            rknn_model.load_rknn(w)
            rknn_model.init_runtime()
            metadata = Path(w).parent / "metadata.yaml"

        # Any other format (unsupported)
        else:
            from ultralytics.engine.exporter import export_formats

            raise TypeError(
                f"model='{w}' is not a supported model format. Ultralytics supports: {export_formats()['Format']}\n"
                f"See https://docs.ultralytics.com/modes/predict for help."
            )

        # Load external metadata YAML
        if isinstance(metadata, (str, Path)) and Path(metadata).exists():
            metadata = yaml_load(metadata)
        if metadata and isinstance(metadata, dict):
            for k, v in metadata.items():
                if k in {"stride", "batch"}:
                    metadata[k] = int(v)
                elif k in {"imgsz", "names", "kpt_shape", "args"} and isinstance(v, str):
                    metadata[k] = eval(v)
            stride = metadata["stride"]
            task = metadata["task"]
            batch = metadata["batch"]
            imgsz = metadata["imgsz"]
            names = metadata["names"]
            kpt_shape = metadata.get("kpt_shape")
            end2end = metadata.get("args", {}).get("nms", False)
        elif not (pt or triton or nn_module):
            LOGGER.warning(f"WARNING ⚠️ Metadata not found for 'model={weights}'")

        # Check names
        if "names" not in locals():  # names missing
            names = default_class_names(data)
        names = check_class_names(names)

        # Disable gradients
        if pt:
            for p in model.parameters():
                p.requires_grad = False

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, embed=None):
        """
        Runs inference on the YOLOv8 MultiBackend model.

        Args:
            im (torch.Tensor): The image tensor to perform inference on.
            augment (bool): whether to perform data augmentation during inference, defaults to False
            visualize (bool): whether to visualize the output predictions, defaults to False
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (tuple): Tuple containing the raw output tensor, and processed output for visualization (if visualize=True)
        """
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        # PyTorch
        if self.pt or self.nn_module:
            y = self.model(im, augment=augment, visualize=visualize, embed=embed)

        # TorchScript
        elif self.jit:
            y = self.model(im)

        # ONNX OpenCV DNN
        elif self.dnn:
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()

        # ONNX Runtime
        elif self.onnx or self.imx:
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
            if self.imx:
                # boxes, conf, cls
                y = np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None]], axis=-1)

        # OpenVINO
        elif self.xml:
            im = im.cpu().numpy()  # FP32

            if self.inference_mode in {"THROUGHPUT", "CUMULATIVE_THROUGHPUT"}:  # optimized for larger batch-sizes
                n = im.shape[0]  # number of images in batch
                results = [None] * n  # preallocate list with None to match the number of images

                def callback(request, userdata):
                    """Places result in preallocated list using userdata index."""
                    results[userdata] = request.results

                # Create AsyncInferQueue, set the callback and start asynchronous inference for each input image
                async_queue = self.ov.runtime.AsyncInferQueue(self.ov_compiled_model)
                async_queue.set_callback(callback)
                for i in range(n):
                    # Start async inference with userdata=i to specify the position in results list
                    async_queue.start_async(inputs={self.input_name: im[i : i + 1]}, userdata=i)  # keep image as BCHW
                async_queue.wait_all()  # wait for all inference requests to complete
                y = np.concatenate([list(r.values())[0] for r in results])

            else:  # inference_mode = "LATENCY", optimized for fastest first result at batch-size 1
                y = list(self.ov_compiled_model(im).values())

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

        # CoreML
        elif self.coreml:
            im = im[0].cpu().numpy()
            im_pil = Image.fromarray((im * 255).astype("uint8"))
            # im = im.resize((192, 320), Image.BILINEAR)
            y = self.model.predict({"image": im_pil})  # coordinates are xywh normalized
            if "confidence" in y:
                raise TypeError(
                    "Ultralytics only supports inference of non-pipelined CoreML models exported with "
                    f"'nms=False', but 'model={w}' has an NMS pipeline created by an 'nms=True' export."
                )
                # TODO: CoreML NMS inference handling
                # from ultralytics.utils.ops import xywh2xyxy
                # box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                # conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float32)
                # y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            y = list(y.values())
            if len(y) == 2 and len(y[1].shape) != 4:  # segmentation model
                y = list(reversed(y))  # reversed for segmentation models (pred, proto)

        # PaddlePaddle
        elif self.paddle:
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]

        # MNN
        elif self.mnn:
            input_var = self.torch_to_mnn(im)
            output_var = self.net.onForward([input_var])
            y = [x.read() for x in output_var]

        # NCNN
        elif self.ncnn:
            mat_in = self.pyncnn.Mat(im[0].cpu().numpy())
            with self.net.create_extractor() as ex:
                ex.input(self.net.input_names()[0], mat_in)
                # WARNING: 'output_names' sorted as a temporary fix for https://github.com/pnnx/pnnx/issues/130
                y = [np.array(ex.extract(x)[1])[None] for x in sorted(self.net.output_names())]

        # NVIDIA Triton Inference Server
        elif self.triton:
            im = im.cpu().numpy()  # torch to numpy
            y = self.model(im)

        # RKNN
        elif self.rknn:
            im = (im.cpu().numpy() * 255).astype("uint8")
            im = im if isinstance(im, (list, tuple)) else [im]
            y = self.rknn_model.inference(inputs=im)

        # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
        else:
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
                if not isinstance(y, list):
                    y = [y]
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite or Edge TPU
                details = self.input_details[0]
                is_int = details["dtype"] in {np.int8, np.int16}  # is TFLite quantized int8 or int16 model
                if is_int:
                    scale, zero_point = details["quantization"]
                    im = (im / scale + zero_point).astype(details["dtype"])  # de-scale
                self.interpreter.set_tensor(details["index"], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output["index"])
                    if is_int:
                        scale, zero_point = output["quantization"]
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    if x.ndim == 3:  # if task is not classification, excluding masks (ndim=4) as well
                        # Denormalize xywh by image size. See https://github.com/ultralytics/ultralytics/pull/1695
                        # xywh are normalized in TFLite/EdgeTPU to mitigate quantization error of integer models
                        if x.shape[-1] == 6 or self.end2end:  # end-to-end model
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
            # TF segment fixes: export is reversed vs ONNX export and protos are transposed
            if len(y) == 2:  # segment with (det, proto) output order reversed
                if len(y[1].shape) != 4:
                    y = list(reversed(y))  # should be y = (1, 116, 8400), (1, 160, 160, 32)
                if y[1].shape[-1] == 6:  # end-to-end model
                    y = [y[1]]
                else:
                    y[1] = np.transpose(y[1], (0, 3, 1, 2))  # should be y = (1, 116, 8400), (1, 32, 160, 160)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]

        # for x in y:
        #     print(type(x), len(x)) if isinstance(x, (list, tuple)) else print(type(x), x.shape)  # debug shapes
        if isinstance(y, (list, tuple)):
            if len(self.names) == 999 and (self.task == "segment" or len(y) == 2):  # segments and names not defined
                nc = y[0].shape[1] - y[1].shape[1] - 4  # y = (1, 32, 160, 160), (1, 116, 8400)
                self.names = {i: f"class{i}" for i in range(nc)}
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """
        Convert a numpy array to a tensor.

        Args:
            x (np.ndarray): The array to be converted.

        Returns:
            (torch.Tensor): The converted tensor
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        Warm up the model by running one forward pass with a dummy input.

        Args:
            imgsz (tuple): The shape of the dummy input tensor in the format (batch_size, channels, height, width)
        """
        import torchvision  # noqa (import here so torchvision import time not recorded in postprocess time)

        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton, self.nn_module
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p="path/to/model.pt"):
        """
        Takes a path to a model file and returns the model type. Possibles types are pt, jit, onnx, xml, engine, coreml,
        saved_model, pb, tflite, edgetpu, tfjs, ncnn or paddle.

        Args:
            p (str): path to the model file. Defaults to path/to/model.pt

        Examples:
            >>> model = AutoBackend(weights="path/to/model.onnx")
            >>> model_type = model._model_type()  # returns "onnx"
        """
        from ultralytics.engine.exporter import export_formats

        sf = export_formats()["Suffix"]  # export suffixes
        if not is_url(p) and not isinstance(p, str):
            check_suffix(p, sf)  # checks
        name = Path(p).name
        types = [s in name for s in sf]
        types[5] |= name.endswith(".mlmodel")  # retain support for older Apple CoreML *.mlmodel formats
        types[8] &= not types[9]  # tflite &= not edgetpu
        if any(types):
            triton = False
        else:
            from urllib.parse import urlsplit

            url = urlsplit(p)
            triton = bool(url.netloc) and bool(url.path) and url.scheme in {"http", "grpc"}

        return types + [triton]
