# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys

import tensorrt as trt

from trt_calibrator import FeatEntropyCalibrator

sys.path.append('.')

from supervision.tracker.utils.fast_reid.fastreid.utils.logger import setup_logger, PathManager

logger = setup_logger(name="trt_export")


def get_parser():
    parser = argparse.ArgumentParser(description="Convert ONNX to TRT model")

    parser.add_argument(
        '--name',
        default='baseline',
        help="name for converted model"
    )
    parser.add_argument(
        '--output',
        default='outputs/trt_model',
        help="path to save converted trt model"
    )
    parser.add_argument(
        '--mode',
        default='fp32',
        help="which mode is used in tensorRT engine, mode can be ['fp32', 'fp16' 'int8']"
    )
    parser.add_argument(
        '--batch-size',
        default=1,
        type=int,
        help="the maximum batch size of trt module"
    )
    parser.add_argument(
        '--height',
        default=256,
        type=int,
        help="input image height"
    )
    parser.add_argument(
        '--width',
        default=128,
        type=int,
        help="input image width"
    )
    parser.add_argument(
        '--channel',
        default=3,
        type=int,
        help="input image channel"
    )
    parser.add_argument(
        '--calib-data',
        default='Market1501',
        help="int8 calibrator dataset name"
    )
    parser.add_argument(
        "--onnx-model",
        default='outputs/onnx_model/baseline.onnx',
        help='path to onnx model'
    )
    return parser


def onnx2trt(
        onnx_file_path,
        save_path,
        mode,
        log_level='ERROR',
        max_workspace_size=1,
        strict_type_constraints=False,
        int8_calibrator=None,
):
    """build TensorRT model from onnx model.
    Args:
        onnx_file_path (string or io object): onnx model name
        save_path (string): tensortRT serialization save path
        mode (string): Whether or not FP16 or Int8 kernels are permitted during engine build.
        log_level (string, default is ERROR): tensorrt logger level, now
            INTERNAL_ERROR, ERROR, WARNING, INFO, VERBOSE are support.
        max_workspace_size (int, default is 1): The maximum GPU temporary memory which the ICudaEngine can use at
            execution time. default is 1GB.
        strict_type_constraints (bool, default is False): When strict type constraints is set, TensorRT will choose
            the type constraints that conforms to type constraints. If the flag is not enabled higher precision
            implementation may be chosen if it results in higher performance.
        int8_calibrator (volksdep.calibrators.base.BaseCalibrator, default is None): calibrator for int8 mode,
            if None, default calibrator will be used as calibration data.
    """
    mode = mode.lower()
    assert mode in ['fp32', 'fp16', 'int8'], "mode should be in ['fp32', 'fp16', 'int8'], " \
                                             "but got {}".format(mode)

    trt_logger = trt.Logger(getattr(trt.Logger, log_level))
    builder = trt.Builder(trt_logger)

    logger.info("Loading ONNX file from path {}...".format(onnx_file_path))
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, trt_logger)
    if isinstance(onnx_file_path, str):
        with open(onnx_file_path, 'rb') as f:
            logger.info("Beginning ONNX file parsing")
            flag = parser.parse(f.read())
    else:
        flag = parser.parse(onnx_file_path.read())
    if not flag:
        for error in range(parser.num_errors):
            logger.info(parser.get_error(error))

    logger.info("Completed parsing of ONNX file.")
    # re-order output tensor
    output_tensors = [network.get_output(i) for i in range(network.num_outputs)]
    [network.unmark_output(tensor) for tensor in output_tensors]
    for tensor in output_tensors:
        identity_out_tensor = network.add_identity(tensor).get_output(0)
        identity_out_tensor.name = 'identity_{}'.format(tensor.name)
        network.mark_output(tensor=identity_out_tensor)

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size * (1 << 25)
    if mode == 'fp16':
        assert builder.platform_has_fast_fp16, "not support fp16"
        builder.fp16_mode = True
    if mode == 'int8':
        assert builder.platform_has_fast_int8, "not support int8"
        builder.int8_mode = True
        builder.int8_calibrator = int8_calibrator

    if strict_type_constraints:
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)

    logger.info("Building an engine from file {}; this may take a while...".format(onnx_file_path))
    engine = builder.build_cuda_engine(network)
    logger.info("Create engine successfully!")

    logger.info("Saving TRT engine file to path {}".format(save_path))
    with open(save_path, 'wb') as f:
        f.write(engine.serialize())
    logger.info("Engine file has already saved to {}!".format(save_path))


if __name__ == '__main__':
    args = get_parser().parse_args()

    onnx_file_path = args.onnx_model
    engineFile = os.path.join(args.output, args.name + '.engine')

    if args.mode.lower() == 'int8':
        int8_calib = FeatEntropyCalibrator(args)
    else:
        int8_calib = None

    PathManager.mkdirs(args.output)
    onnx2trt(onnx_file_path, engineFile, args.mode, int8_calibrator=int8_calib)
