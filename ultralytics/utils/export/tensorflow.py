# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.nn.modules import Detect, Pose
from ultralytics.utils import LOGGER
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.files import spaces_in_path
from ultralytics.utils.tal import make_anchors


def tf_wrapper(model: torch.nn.Module) -> torch.nn.Module:
    """A wrapper to add TensorFlow compatible inference methods to Detect and Pose layers."""
    for m in model.modules():
        if not isinstance(m, Detect):
            continue
        import types

        m._inference = types.MethodType(_tf_inference, m)
        if type(m) is Pose:
            m.kpts_decode = types.MethodType(tf_kpts_decode, m)
    return model


def _tf_inference(self, x: list[torch.Tensor]) -> tuple[torch.Tensor]:
    """Decode boxes and cls scores for tf object detection."""
    shape = x[0].shape  # BCHW
    x_cat = torch.cat([xi.view(x[0].shape[0], self.no, -1) for xi in x], 2)
    box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    if self.dynamic or self.shape != shape:
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        self.shape = shape
    grid_h, grid_w = shape[2], shape[3]
    grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
    norm = self.strides / (self.stride[0] * grid_size)
    dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
    return torch.cat((dbox, cls.sigmoid()), 1)


def tf_kpts_decode(self, bs: int, kpts: torch.Tensor) -> torch.Tensor:
    """Decode keypoints for tf pose estimation."""
    ndim = self.kpt_shape[1]
    # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
    # Precompute normalization factor to increase numerical stability
    y = kpts.view(bs, *self.kpt_shape, -1)
    grid_h, grid_w = self.shape[2], self.shape[3]
    grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
    norm = self.strides / (self.stride[0] * grid_size)
    a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * norm
    if ndim == 3:
        a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
    return a.view(bs, self.nk, -1)


def onnx2saved_model(
    onnx_file: str,
    output_dir: Path,
    int8: bool = False,
    images: np.ndarray = None,
    disable_group_convolution: bool = False,
    prefix="",
):
    """Convert a ONNX model to TensorFlow SavedModel format via ONNX.

    Args:
        onnx_file (str): ONNX file path.
        output_dir (Path): Output directory path for the SavedModel.
        int8 (bool, optional): Enable INT8 quantization. Defaults to False.
        images (np.ndarray, optional): Calibration images for INT8 quantization in BHWC format.
        disable_group_convolution (bool, optional): Disable group convolution optimization. Defaults to False.
        prefix (str, optional): Logging prefix. Defaults to "".

    Returns:
        (keras.Model): Converted Keras model.

    Notes:
        - Requires onnx2tf package. Downloads calibration data if INT8 quantization is enabled.
        - Removes temporary files and renames quantized models after conversion.
    """
    # Pre-download calibration file to fix https://github.com/PINTO0309/onnx2tf/issues/545
    onnx2tf_file = Path("calibration_image_sample_data_20x128x128x3_float32.npy")
    if not onnx2tf_file.exists():
        attempt_download_asset(f"{onnx2tf_file}.zip", unzip=True, delete=True)
    np_data = None
    if int8:
        tmp_file = output_dir / "tmp_tflite_int8_calibration_images.npy"  # int8 calibration images file
        if images is not None:
            output_dir.mkdir()
            np.save(str(tmp_file), images)  # BHWC
            np_data = [["images", tmp_file, [[[[0, 0, 0]]]], [[[[255, 255, 255]]]]]]

    import onnx2tf  # scoped for after ONNX export for reduced conflict during import

    LOGGER.info(f"{prefix} starting TFLite export with onnx2tf {onnx2tf.__version__}...")
    keras_model = onnx2tf.convert(
        input_onnx_file_path=onnx_file,
        output_folder_path=str(output_dir),
        not_use_onnxsim=True,
        verbosity="error",  # note INT8-FP16 activation bug https://github.com/ultralytics/ultralytics/issues/15873
        output_integer_quantized_tflite=int8,
        custom_input_op_name_np_data_path=np_data,
        enable_batchmatmul_unfold=True and not int8,  # fix lower no. of detected objects on GPU delegate
        output_signaturedefs=True,  # fix error with Attention block group convolution
        disable_group_convolution=disable_group_convolution,  # fix error with group convolution
    )

    # Remove/rename TFLite models
    if int8:
        tmp_file.unlink(missing_ok=True)
        for file in output_dir.rglob("*_dynamic_range_quant.tflite"):
            file.rename(file.with_name(file.stem.replace("_dynamic_range_quant", "_int8") + file.suffix))
        for file in output_dir.rglob("*_integer_quant_with_int16_act.tflite"):
            file.unlink()  # delete extra fp16 activation TFLite files
    return keras_model


def keras2pb(keras_model, file: Path, prefix=""):
    """Convert a Keras model to TensorFlow GraphDef (.pb) format.

    Args:
        keras_model(tf_keras): Keras model to convert to frozen graph format.
        file (Path): Output file path (suffix will be changed to .pb).
        prefix (str, optional): Logging prefix. Defaults to "".

    Notes:
        Creates a frozen graph by converting variables to constants for inference optimization.
    """
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
    m = tf.function(lambda x: keras_model(x))  # full model
    m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(m)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(file.parent), name=file.name, as_text=False)


def tflite2edgetpu(tflite_file: str | Path, output_dir: str | Path, prefix: str = ""):
    """Convert a TensorFlow Lite model to Edge TPU format using the Edge TPU compiler.

    Args:
        tflite_file (str | Path): Path to the input TensorFlow Lite (.tflite) model file.
        output_dir (str | Path): Output directory path for the compiled Edge TPU model.
        prefix (str, optional): Logging prefix. Defaults to "".

    Notes:
        Requires the Edge TPU compiler to be installed. The function compiles the TFLite model
        for optimal performance on Google's Edge TPU hardware accelerator.
    """
    import subprocess

    cmd = (
        "edgetpu_compiler "
        f'--out_dir "{output_dir}" '
        "--show_operations "
        "--search_delegate "
        "--delegate_search_step 30 "
        "--timeout_sec 180 "
        f'"{tflite_file}"'
    )
    LOGGER.info(f"{prefix} running '{cmd}'")
    subprocess.run(cmd, shell=True)


def pb2tfjs(pb_file: str, output_dir: str, half: bool = False, int8: bool = False, prefix: str = ""):
    """Convert a TensorFlow GraphDef (.pb) model to TensorFlow.js format.

    Args:
        pb_file (str): Path to the input TensorFlow GraphDef (.pb) model file.
        output_dir (str): Output directory path for the converted TensorFlow.js model.
        half (bool, optional): Enable FP16 quantization. Defaults to False.
        int8 (bool, optional): Enable INT8 quantization. Defaults to False.
        prefix (str, optional): Logging prefix. Defaults to "".

    Notes:
        Requires tensorflowjs package. Uses tensorflowjs_converter command-line tool for conversion.
        Handles spaces in file paths and warns if output directory contains spaces.
    """
    import subprocess

    import tensorflow as tf
    import tensorflowjs as tfjs

    LOGGER.info(f"\n{prefix} starting export with tensorflowjs {tfjs.__version__}...")

    gd = tf.Graph().as_graph_def()  # TF GraphDef
    with open(pb_file, "rb") as file:
        gd.ParseFromString(file.read())
    outputs = ",".join(gd_outputs(gd))
    LOGGER.info(f"\n{prefix} output node names: {outputs}")

    quantization = "--quantize_float16" if half else "--quantize_uint8" if int8 else ""
    with spaces_in_path(pb_file) as fpb_, spaces_in_path(output_dir) as f_:  # exporter can not handle spaces in path
        cmd = (
            "tensorflowjs_converter "
            f'--input_format=tf_frozen_model {quantization} --output_node_names={outputs} "{fpb_}" "{f_}"'
        )
        LOGGER.info(f"{prefix} running '{cmd}'")
        subprocess.run(cmd, shell=True)

    if " " in output_dir:
        LOGGER.warning(f"{prefix} your model may not work correctly with spaces in path '{output_dir}'.")


def gd_outputs(gd):
    """Return TensorFlow GraphDef model output node names."""
    name_list, input_list = [], []
    for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
        name_list.append(node.name)
        input_list.extend(node.input)
    return sorted(f"{x}:0" for x in list(set(name_list) - set(input_list)) if not x.startswith("NoOp"))
