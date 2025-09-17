from pathlib import Path
from ultralytics.utils.export import torch2onnx
from ultralytics.utils import LOGGER
from ultralytics.utils.downloads import attempt_download_asset
import numpy as np
import torch


def torch2saved_model(
    model: torch.nn.Module,
    file: Path,
    im: torch.Tensor,
    opset: int = 14,
    int8: bool = False,
    images: np.ndarray = None,
    disable_group_convolution: bool = False,
    prefix="",
):
    """
    Convert a PyTorch model to TensorFlow SavedModel format via ONNX.

    Args:
        model (torch.nn.Module): PyTorch model to convert.
        file (Path): Output directory path for the SavedModel.
        im (torch.Tensor): Sample input tensor for model tracing.
        opset (int, optional): ONNX opset version. Defaults to 14.
        int8 (bool, optional): Enable INT8 quantization. Defaults to False.
        images (np.ndarray, optional): Calibration images for INT8 quantization in BHWC format.
        disable_group_convolution (bool, optional): Disable group convolution optimization. Defaults to False.
        prefix (str, optional): Logging prefix. Defaults to "".

    Returns:
        (keras.Model): Converted Keras model.

    Note:
        Requires onnx2tf package. Downloads calibration data if INT8 quantization is enabled.
        Removes temporary files and renames quantized models after conversion.
    """
    # Pre-download calibration file to fix https://github.com/PINTO0309/onnx2tf/issues/545
    onnx2tf_file = Path("calibration_image_sample_data_20x128x128x3_float32.npy")
    if not onnx2tf_file.exists():
        attempt_download_asset(f"{onnx2tf_file}.zip", unzip=True, delete=True)
    f_onnx = str(file.with_suffix(".onnx"))
    torch2onnx(
        model,
        im,
        f_onnx,
        opset=opset,
        input_names=["images"],
        output_names=["output0", "output1"] if model.task == "segment" else ["output0"],
        simplify=True,
    )
    np_data = None
    if int8:
        tmp_file = file / "tmp_tflite_int8_calibration_images.npy"  # int8 calibration images file
        if images is not None:
            file.mkdir()
            np.save(str(tmp_file), images)  # BHWC
            np_data = [["images", tmp_file, [[[[0, 0, 0]]]], [[[[255, 255, 255]]]]]]

    import onnx2tf  # scoped for after ONNX export for reduced conflict during import

    LOGGER.info(f"{prefix} starting TFLite export with onnx2tf {onnx2tf.__version__}...")
    keras_model = onnx2tf.convert(
        input_onnx_file_path=f_onnx,
        output_folder_path=str(file),
        not_use_onnxsim=True,
        verbosity="error",  # note INT8-FP16 activation bug https://github.com/ultralytics/ultralytics/issues/15873
        output_integer_quantized_tflite=int8,
        quant_type="per-tensor",  # "per-tensor" (faster) or "per-channel" (slower but more accurate)
        custom_input_op_name_np_data_path=np_data,
        enable_batchmatmul_unfold=True,  # fix lower no. of detected objects on GPU delegate
        output_signaturedefs=True,  # fix error with Attention block group convolution
        disable_group_convolution=disable_group_convolution,  # fix error with group convolution
    )

    # Remove/rename TFLite models
    if int8:
        tmp_file.unlink(missing_ok=True)
        for file in file.rglob("*_dynamic_range_quant.tflite"):
            file.rename(file.with_name(file.stem.replace("_dynamic_range_quant", "_int8") + file.suffix))
        for file in file.rglob("*_integer_quant_with_int16_act.tflite"):
            file.unlink()  # delete extra fp16 activation TFLite files
    return keras_model


def keras2pb(keras_model, file: Path, prefix=""):
    """
    Convert a Keras model to TensorFlow Protocol Buffer (.pb) format.

    Args:
        keras_model: Keras model to convert to frozen graph format.
        file (Path): Output file path (suffix will be changed to .pb).
        prefix (str, optional): Logging prefix. Defaults to "".

    Returns:
        Path: Path to the exported .pb file.

    Note:
        Creates a frozen graph by converting variables to constants for inference optimization.
    """
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2  # noqa

    LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
    m = tf.function(lambda x: keras_model(x))  # full model
    m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(m)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(file.parent), name=file.name, as_text=False)
