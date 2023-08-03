# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Export a YOLOv8 PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit

Format                  | `format=argument`         | Model
---                     | ---                       | ---
PyTorch                 | -                         | yolov8n.pt
TorchScript             | `torchscript`             | yolov8n.torchscript
ONNX                    | `onnx`                    | yolov8n.onnx
OpenVINO                | `openvino`                | yolov8n_openvino_model/
TensorRT                | `engine`                  | yolov8n.engine
CoreML                  | `coreml`                  | yolov8n.mlmodel
TensorFlow SavedModel   | `saved_model`             | yolov8n_saved_model/
TensorFlow GraphDef     | `pb`                      | yolov8n.pb
TensorFlow Lite         | `tflite`                  | yolov8n.tflite
TensorFlow Edge TPU     | `edgetpu`                 | yolov8n_edgetpu.tflite
TensorFlow.js           | `tfjs`                    | yolov8n_web_model/
PaddlePaddle            | `paddle`                  | yolov8n_paddle_model/
ncnn                    | `ncnn`                    | yolov8n_ncnn_model/

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
                         yolov8n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                         yolov8n_openvino_model     # OpenVINO
                         yolov8n.engine             # TensorRT
                         yolov8n.mlmodel            # CoreML (macOS-only)
                         yolov8n_saved_model        # TensorFlow SavedModel
                         yolov8n.pb                 # TensorFlow GraphDef
                         yolov8n.tflite             # TensorFlow Lite
                         yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                         yolov8n_paddle_model       # PaddlePaddle

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolov5/yolov8n_web_model public/yolov8n_web_model
    $ npm start
"""
import json
import os
import shutil
import subprocess
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch

from ultralytics.cfg import get_cfg
from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder
from ultralytics.nn.tasks import DetectionModel, SegmentationModel
from ultralytics.utils import (ARM64, DEFAULT_CFG, LINUX, LOGGER, MACOS, ROOT, WINDOWS, __version__, callbacks,
                               colorstr, get_default_args, yaml_save)
from ultralytics.utils.checks import check_imgsz, check_requirements, check_version
from ultralytics.utils.downloads import attempt_download_asset, get_github_assets
from ultralytics.utils.files import file_size, spaces_in_path
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import get_latest_opset, select_device, smart_inference_mode


def export_formats():
    """YOLOv8 export formats."""
    import pandas
    x = [
        ['PyTorch', '-', '.pt', True, True],
        ['TorchScript', 'torchscript', '.torchscript', True, True],
        ['ONNX', 'onnx', '.onnx', True, True],
        ['OpenVINO', 'openvino', '_openvino_model', True, False],
        ['TensorRT', 'engine', '.engine', False, True],
        ['CoreML', 'coreml', '.mlmodel', True, False],
        ['TensorFlow SavedModel', 'saved_model', '_saved_model', True, True],
        ['TensorFlow GraphDef', 'pb', '.pb', True, True],
        ['TensorFlow Lite', 'tflite', '.tflite', True, False],
        ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', True, False],
        ['TensorFlow.js', 'tfjs', '_web_model', True, False],
        ['PaddlePaddle', 'paddle', '_paddle_model', True, True],
        ['ncnn', 'ncnn', '_ncnn_model', True, True], ]
    return pandas.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])


def gd_outputs(gd):
    """TensorFlow GraphDef model output node names."""
    name_list, input_list = [], []
    for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
        name_list.append(node.name)
        input_list.extend(node.input)
    return sorted(f'{x}:0' for x in list(set(name_list) - set(input_list)) if not x.startswith('NoOp'))


def try_export(inner_func):
    """YOLOv8 export decorator, i..e @try_export."""
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        """Export a model."""
        prefix = inner_args['prefix']
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f"{prefix} export success ✅ {dt.t:.1f}s, saved as '{f}' ({file_size(f):.1f} MB)")
            return f, model
        except Exception as e:
            LOGGER.info(f'{prefix} export failure ❌ {dt.t:.1f}s: {e}')
            raise e

    return outer_func


class Exporter:
    """
    A class for exporting a model.

    Attributes:
        args (SimpleNamespace): Configuration for the exporter.
        save_dir (Path): Directory to save results.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the Exporter class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
            _callbacks (list, optional): List of callback functions. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    @smart_inference_mode()
    def __call__(self, model=None):
        """Returns list of exported files/dirs after running callbacks."""
        self.run_callbacks('on_export_start')
        t = time.time()
        format = self.args.format.lower()  # to lowercase
        if format in ('tensorrt', 'trt'):  # engine aliases
            format = 'engine'
        fmts = tuple(export_formats()['Argument'][1:])  # available export formats
        flags = [x == format for x in fmts]
        if sum(flags) != 1:
            raise ValueError(f"Invalid export format='{format}'. Valid formats are {fmts}")
        jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, ncnn = flags  # export booleans

        # Load PyTorch model
        self.device = select_device('cpu' if self.args.device is None else self.args.device)

        # Checks
        model.names = check_class_names(model.names)
        if self.args.half and onnx and self.device.type == 'cpu':
            LOGGER.warning('WARNING ⚠️ half=True only compatible with GPU export, i.e. use device=0')
            self.args.half = False
            assert not self.args.dynamic, 'half=True not compatible with dynamic=True, i.e. use only one.'
        self.imgsz = check_imgsz(self.args.imgsz, stride=model.stride, min_dim=2)  # check image size
        if self.args.optimize:
            assert not ncnn, "optimize=True not compatible with format='ncnn', i.e. use optimize=False"
            assert self.device.type == 'cpu', "optimize=True not compatible with cuda devices, i.e. use device='cpu'"
        if edgetpu and not LINUX:
            raise SystemError('Edge TPU export only supported on Linux. See https://coral.ai/docs/edgetpu/compiler/')

        # Input
        im = torch.zeros(self.args.batch, 3, *self.imgsz).to(self.device)
        file = Path(
            getattr(model, 'pt_path', None) or getattr(model, 'yaml_file', None) or model.yaml.get('yaml_file', ''))
        if file.suffix in ('.yaml', '.yml'):
            file = Path(file.name)

        # Update model
        model = deepcopy(model).to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()
        for k, m in model.named_modules():
            if isinstance(m, (Detect, RTDETRDecoder)):  # Segment and Pose use Detect base class
                m.dynamic = self.args.dynamic
                m.export = True
                m.format = self.args.format
            elif isinstance(m, C2f) and not any((saved_model, pb, tflite, edgetpu, tfjs)):
                # EdgeTPU does not support FlexSplitV while split provides cleaner ONNX graph
                m.forward = m.forward_split

        y = None
        for _ in range(2):
            y = model(im)  # dry runs
        if self.args.half and (engine or onnx) and self.device.type != 'cpu':
            im, model = im.half(), model.half()  # to FP16

        # Filter warnings
        warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
        warnings.filterwarnings('ignore', category=UserWarning)  # suppress shape prim::Constant missing ONNX warning
        warnings.filterwarnings('ignore', category=DeprecationWarning)  # suppress CoreML np.bool deprecation warning

        # Assign
        self.im = im
        self.model = model
        self.file = file
        self.output_shape = tuple(y.shape) if isinstance(y, torch.Tensor) else \
            tuple(tuple(x.shape if isinstance(x, torch.Tensor) else []) for x in y)
        self.pretty_name = Path(self.model.yaml.get('yaml_file', self.file)).stem.replace('yolo', 'YOLO')
        trained_on = f'trained on {Path(self.args.data).name}' if self.args.data else '(untrained)'
        description = f'Ultralytics {self.pretty_name} model {trained_on}'
        self.metadata = {
            'description': description,
            'author': 'Ultralytics',
            'license': 'AGPL-3.0 https://ultralytics.com/license',
            'date': datetime.now().isoformat(),
            'version': __version__,
            'stride': int(max(model.stride)),
            'task': model.task,
            'batch': self.args.batch,
            'imgsz': self.imgsz,
            'names': model.names}  # model metadata
        if model.task == 'pose':
            self.metadata['kpt_shape'] = model.model[-1].kpt_shape

        LOGGER.info(f"\n{colorstr('PyTorch:')} starting from '{file}' with input shape {tuple(im.shape)} BCHW and "
                    f'output shape(s) {self.output_shape} ({file_size(file):.1f} MB)')

        # Exports
        f = [''] * len(fmts)  # exported filenames
        if jit or ncnn:  # TorchScript
            f[0], _ = self.export_torchscript()
        if engine:  # TensorRT required before ONNX
            f[1], _ = self.export_engine()
        if onnx or xml:  # OpenVINO requires ONNX
            f[2], _ = self.export_onnx()
        if xml:  # OpenVINO
            f[3], _ = self.export_openvino()
        if coreml:  # CoreML
            f[4], _ = self.export_coreml()
        if any((saved_model, pb, tflite, edgetpu, tfjs)):  # TensorFlow formats
            self.args.int8 |= edgetpu
            f[5], s_model = self.export_saved_model()
            if pb or tfjs:  # pb prerequisite to tfjs
                f[6], _ = self.export_pb(s_model)
            if tflite:
                f[7], _ = self.export_tflite(s_model, nms=False, agnostic_nms=self.args.agnostic_nms)
            if edgetpu:
                f[8], _ = self.export_edgetpu(tflite_model=Path(f[5]) / f'{self.file.stem}_full_integer_quant.tflite')
            if tfjs:
                f[9], _ = self.export_tfjs()
        if paddle:  # PaddlePaddle
            f[10], _ = self.export_paddle()
        if ncnn:  # ncnn
            f[11], _ = self.export_ncnn()

        # Finish
        f = [str(x) for x in f if x]  # filter out '' and None
        if any(f):
            f = str(Path(f[-1]))
            square = self.imgsz[0] == self.imgsz[1]
            s = '' if square else f"WARNING ⚠️ non-PyTorch val requires square images, 'imgsz={self.imgsz}' will not " \
                                  f"work. Use export 'imgsz={max(self.imgsz)}' if val is required."
            imgsz = self.imgsz[0] if square else str(self.imgsz)[1:-1].replace(' ', '')
            data = f'data={self.args.data}' if model.task == 'segment' and format == 'pb' else ''
            LOGGER.info(
                f'\nExport complete ({time.time() - t:.1f}s)'
                f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                f'\nPredict:         yolo predict task={model.task} model={f} imgsz={imgsz} {data}'
                f'\nValidate:        yolo val task={model.task} model={f} imgsz={imgsz} data={self.args.data} {s}'
                f'\nVisualize:       https://netron.app')

        self.run_callbacks('on_export_end')
        return f  # return list of exported files/dirs

    @try_export
    def export_torchscript(self, prefix=colorstr('TorchScript:')):
        """YOLOv8 TorchScript model export."""
        LOGGER.info(f'\n{prefix} starting export with torch {torch.__version__}...')
        f = self.file.with_suffix('.torchscript')

        ts = torch.jit.trace(self.model, self.im, strict=False)
        extra_files = {'config.txt': json.dumps(self.metadata)}  # torch._C.ExtraFilesMap()
        if self.args.optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
            LOGGER.info(f'{prefix} optimizing for mobile...')
            from torch.utils.mobile_optimizer import optimize_for_mobile
            optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
        else:
            ts.save(str(f), _extra_files=extra_files)
        return f, None

    @try_export
    def export_onnx(self, prefix=colorstr('ONNX:')):
        """YOLOv8 ONNX export."""
        requirements = ['onnx>=1.12.0']
        if self.args.simplify:
            requirements += ['onnxsim>=0.4.17', 'onnxruntime-gpu' if torch.cuda.is_available() else 'onnxruntime']
        check_requirements(requirements)
        import onnx  # noqa

        opset_version = self.args.opset or get_latest_opset()
        LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__} opset {opset_version}...')
        f = str(self.file.with_suffix('.onnx'))

        output_names = ['output0', 'output1'] if isinstance(self.model, SegmentationModel) else ['output0']
        dynamic = self.args.dynamic
        if dynamic:
            dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
            if isinstance(self.model, SegmentationModel):
                dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 116, 8400)
                dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
            elif isinstance(self.model, DetectionModel):
                dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 84, 8400)

        torch.onnx.export(
            self.model.cpu() if dynamic else self.model,  # --dynamic only compatible with cpu
            self.im.cpu() if dynamic else self.im,
            f,
            verbose=False,
            opset_version=opset_version,
            do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
            input_names=['images'],
            output_names=output_names,
            dynamic_axes=dynamic or None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        # onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify
        if self.args.simplify:
            try:
                import onnxsim

                LOGGER.info(f'{prefix} simplifying with onnxsim {onnxsim.__version__}...')
                # subprocess.run(f'onnxsim "{f}" "{f}"', shell=True)
                model_onnx, check = onnxsim.simplify(model_onnx)
                assert check, 'Simplified ONNX model could not be validated'
            except Exception as e:
                LOGGER.info(f'{prefix} simplifier failure: {e}')

        # Metadata
        for k, v in self.metadata.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)

        onnx.save(model_onnx, f)
        return f, model_onnx

    @try_export
    def export_openvino(self, prefix=colorstr('OpenVINO:')):
        """YOLOv8 OpenVINO export."""
        check_requirements('openvino-dev>=2023.0')  # requires openvino-dev: https://pypi.org/project/openvino-dev/
        import openvino.runtime as ov  # noqa
        from openvino.tools import mo  # noqa

        LOGGER.info(f'\n{prefix} starting export with openvino {ov.__version__}...')
        f = str(self.file).replace(self.file.suffix, f'_openvino_model{os.sep}')
        f_onnx = self.file.with_suffix('.onnx')
        f_ov = str(Path(f) / self.file.with_suffix('.xml').name)

        ov_model = mo.convert_model(f_onnx,
                                    model_name=self.pretty_name,
                                    framework='onnx',
                                    compress_to_fp16=self.args.half)  # export

        # Set RT info
        ov_model.set_rt_info('YOLOv8', ['model_info', 'model_type'])
        ov_model.set_rt_info(True, ['model_info', 'reverse_input_channels'])
        ov_model.set_rt_info(114, ['model_info', 'pad_value'])
        ov_model.set_rt_info([255.0], ['model_info', 'scale_values'])
        ov_model.set_rt_info(self.args.iou, ['model_info', 'iou_threshold'])
        ov_model.set_rt_info([v.replace(' ', '_') for k, v in sorted(self.model.names.items())],
                             ['model_info', 'labels'])
        if self.model.task != 'classify':
            ov_model.set_rt_info('fit_to_window_letterbox', ['model_info', 'resize_type'])

        ov.serialize(ov_model, f_ov)  # save
        yaml_save(Path(f) / 'metadata.yaml', self.metadata)  # add metadata.yaml
        return f, None

    @try_export
    def export_paddle(self, prefix=colorstr('PaddlePaddle:')):
        """YOLOv8 Paddle export."""
        check_requirements(('paddlepaddle', 'x2paddle'))
        import x2paddle  # noqa
        from x2paddle.convert import pytorch2paddle  # noqa

        LOGGER.info(f'\n{prefix} starting export with X2Paddle {x2paddle.__version__}...')
        f = str(self.file).replace(self.file.suffix, f'_paddle_model{os.sep}')

        pytorch2paddle(module=self.model, save_dir=f, jit_type='trace', input_examples=[self.im])  # export
        yaml_save(Path(f) / 'metadata.yaml', self.metadata)  # add metadata.yaml
        return f, None

    @try_export
    def export_ncnn(self, prefix=colorstr('ncnn:')):
        """
        YOLOv8 ncnn export using PNNX https://github.com/pnnx/pnnx.
        """
        check_requirements('git+https://github.com/Tencent/ncnn.git' if ARM64 else 'ncnn')  # requires ncnn
        import ncnn  # noqa

        LOGGER.info(f'\n{prefix} starting export with ncnn {ncnn.__version__}...')
        f = Path(str(self.file).replace(self.file.suffix, f'_ncnn_model{os.sep}'))
        f_ts = self.file.with_suffix('.torchscript')

        pnnx_filename = 'pnnx.exe' if WINDOWS else 'pnnx'
        if Path(pnnx_filename).is_file():
            pnnx = pnnx_filename
        elif (ROOT / pnnx_filename).is_file():
            pnnx = ROOT / pnnx_filename
        else:
            LOGGER.warning(
                f'{prefix} WARNING ⚠️ PNNX not found. Attempting to download binary file from '
                'https://github.com/pnnx/pnnx/.\nNote PNNX Binary file must be placed in current working directory '
                f'or in {ROOT}. See PNNX repo for full installation instructions.')
            _, assets = get_github_assets(repo='pnnx/pnnx', retry=True)
            asset = [x for x in assets if ('macos' if MACOS else 'ubuntu' if LINUX else 'windows') in x][0]
            attempt_download_asset(asset, repo='pnnx/pnnx', release='latest')
            unzip_dir = Path(asset).with_suffix('')
            pnnx = ROOT / pnnx_filename  # new location
            (unzip_dir / pnnx_filename).rename(pnnx)  # move binary to ROOT
            shutil.rmtree(unzip_dir)  # delete unzip dir
            Path(asset).unlink()  # delete zip
            pnnx.chmod(0o777)  # set read, write, and execute permissions for everyone

        use_ncnn = True
        ncnn_args = [
            f'ncnnparam={f / "model.ncnn.param"}',
            f'ncnnbin={f / "model.ncnn.bin"}',
            f'ncnnpy={f / "model_ncnn.py"}', ] if use_ncnn else []

        use_pnnx = False
        pnnx_args = [
            f'pnnxparam={f / "model.pnnx.param"}',
            f'pnnxbin={f / "model.pnnx.bin"}',
            f'pnnxpy={f / "model_pnnx.py"}',
            f'pnnxonnx={f / "model.pnnx.onnx"}', ] if use_pnnx else []

        cmd = [
            str(pnnx),
            str(f_ts),
            *ncnn_args,
            *pnnx_args,
            f'fp16={int(self.args.half)}',
            f'device={self.device.type}',
            f'inputshape="{[self.args.batch, 3, *self.imgsz]}"', ]
        f.mkdir(exist_ok=True)  # make ncnn_model directory
        LOGGER.info(f"{prefix} running '{' '.join(cmd)}'")
        subprocess.run(cmd, check=True)
        for f_debug in 'debug.bin', 'debug.param', 'debug2.bin', 'debug2.param':  # remove debug files
            Path(f_debug).unlink(missing_ok=True)

        yaml_save(f / 'metadata.yaml', self.metadata)  # add metadata.yaml
        return str(f), None

    @try_export
    def export_coreml(self, prefix=colorstr('CoreML:')):
        """YOLOv8 CoreML export."""
        check_requirements('coremltools>=6.0,<=6.2')
        import coremltools as ct  # noqa

        LOGGER.info(f'\n{prefix} starting export with coremltools {ct.__version__}...')
        f = self.file.with_suffix('.mlmodel')

        bias = [0.0, 0.0, 0.0]
        scale = 1 / 255
        classifier_config = None
        if self.model.task == 'classify':
            classifier_config = ct.ClassifierConfig(list(self.model.names.values())) if self.args.nms else None
            model = self.model
        elif self.model.task == 'detect':
            model = iOSDetectModel(self.model, self.im) if self.args.nms else self.model
        else:
            # TODO CoreML Segment and Pose model pipelining
            model = self.model

        ts = torch.jit.trace(model.eval(), self.im, strict=False)  # TorchScript model
        ct_model = ct.convert(ts,
                              inputs=[ct.ImageType('image', shape=self.im.shape, scale=scale, bias=bias)],
                              classifier_config=classifier_config)
        bits, mode = (8, 'kmeans_lut') if self.args.int8 else (16, 'linear') if self.args.half else (32, None)
        if bits < 32:
            if 'kmeans' in mode:
                check_requirements('scikit-learn')  # scikit-learn package required for k-means quantization
            ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
        if self.args.nms and self.model.task == 'detect':
            ct_model = self._pipeline_coreml(ct_model)

        m = self.metadata  # metadata dict
        ct_model.short_description = m.pop('description')
        ct_model.author = m.pop('author')
        ct_model.license = m.pop('license')
        ct_model.version = m.pop('version')
        ct_model.user_defined_metadata.update({k: str(v) for k, v in m.items()})
        ct_model.save(str(f))
        return f, ct_model

    @try_export
    def export_engine(self, prefix=colorstr('TensorRT:')):
        """YOLOv8 TensorRT export https://developer.nvidia.com/tensorrt."""
        assert self.im.device.type != 'cpu', "export running on CPU but must be on GPU, i.e. use 'device=0'"
        try:
            import tensorrt as trt  # noqa
        except ImportError:
            if LINUX:
                check_requirements('nvidia-tensorrt', cmds='-U --index-url https://pypi.ngc.nvidia.com')
            import tensorrt as trt  # noqa

        check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
        self.args.simplify = True
        f_onnx, _ = self.export_onnx()

        LOGGER.info(f'\n{prefix} starting export with TensorRT {trt.__version__}...')
        assert Path(f_onnx).exists(), f'failed to export ONNX file: {f_onnx}'
        f = self.file.with_suffix('.engine')  # TensorRT engine file
        logger = trt.Logger(trt.Logger.INFO)
        if self.args.verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = self.args.workspace * 1 << 30
        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(f_onnx):
            raise RuntimeError(f'failed to load ONNX file: {f_onnx}')

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
            LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        for out in outputs:
            LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

        if self.args.dynamic:
            shape = self.im.shape
            if shape[0] <= 1:
                LOGGER.warning(f'{prefix} WARNING ⚠️ --dynamic model requires maximum --batch-size argument')
            profile = builder.create_optimization_profile()
            for inp in inputs:
                profile.set_shape(inp.name, (1, *shape[1:]), (max(1, shape[0] // 2), *shape[1:]), shape)
            config.add_optimization_profile(profile)

        LOGGER.info(
            f'{prefix} building FP{16 if builder.platform_has_fast_fp16 and self.args.half else 32} engine as {f}')
        if builder.platform_has_fast_fp16 and self.args.half:
            config.set_flag(trt.BuilderFlag.FP16)

        # Write file
        with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
            # Metadata
            meta = json.dumps(self.metadata)
            t.write(len(meta).to_bytes(4, byteorder='little', signed=True))
            t.write(meta.encode())
            # Model
            t.write(engine.serialize())

        return f, None

    @try_export
    def export_saved_model(self, prefix=colorstr('TensorFlow SavedModel:')):
        """YOLOv8 TensorFlow SavedModel export."""
        try:
            import tensorflow as tf  # noqa
        except ImportError:
            cuda = torch.cuda.is_available()
            check_requirements(f"tensorflow{'-macos' if MACOS else '-aarch64' if ARM64 else '' if cuda else '-cpu'}")
            import tensorflow as tf  # noqa
        check_requirements(('onnx', 'onnx2tf>=1.9.1', 'sng4onnx>=1.0.1', 'onnxsim>=0.4.17', 'onnx_graphsurgeon>=0.3.26',
                            'tflite_support', 'onnxruntime-gpu' if torch.cuda.is_available() else 'onnxruntime'),
                           cmds='--extra-index-url https://pypi.ngc.nvidia.com')

        LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        f = Path(str(self.file).replace(self.file.suffix, '_saved_model'))
        if f.is_dir():
            import shutil
            shutil.rmtree(f)  # delete output folder

        # Export to ONNX
        self.args.simplify = True
        f_onnx, _ = self.export_onnx()

        # Export to TF
        tmp_file = f / 'tmp_tflite_int8_calibration_images.npy'  # int8 calibration images file
        if self.args.int8:
            if self.args.data:
                import numpy as np

                from ultralytics.data.dataset import YOLODataset
                from ultralytics.data.utils import check_det_dataset

                # Generate calibration data for integer quantization
                LOGGER.info(f"{prefix} collecting INT8 calibration images from 'data={self.args.data}'")
                dataset = YOLODataset(check_det_dataset(self.args.data)['val'], imgsz=self.imgsz[0], augment=False)
                images = []
                n_images = 100  # maximum number of images
                for n, batch in enumerate(dataset):
                    if n >= n_images:
                        break
                    im = batch['img'].permute(1, 2, 0)[None]  # list to nparray, CHW to BHWC,
                    images.append(im)
                f.mkdir()
                images = torch.cat(images, 0).float()
                # mean = images.view(-1, 3).mean(0)  # imagenet mean [123.675, 116.28, 103.53]
                # std = images.view(-1, 3).std(0)  # imagenet std [58.395, 57.12, 57.375]
                np.save(str(tmp_file), images.numpy())  # BHWC
                int8 = f'-oiqt -qt per-tensor -cind images "{tmp_file}" "[[[[0, 0, 0]]]]" "[[[[255, 255, 255]]]]"'
            else:
                int8 = '-oiqt -qt per-tensor'
        else:
            int8 = ''

        cmd = f'onnx2tf -i "{f_onnx}" -o "{f}" -nuo --non_verbose {int8}'.strip()
        LOGGER.info(f"{prefix} running '{cmd}'")
        subprocess.run(cmd, shell=True)
        yaml_save(f / 'metadata.yaml', self.metadata)  # add metadata.yaml

        # Remove/rename TFLite models
        if self.args.int8:
            tmp_file.unlink(missing_ok=True)
            for file in f.rglob('*_dynamic_range_quant.tflite'):
                file.rename(file.with_name(file.stem.replace('_dynamic_range_quant', '_int8') + file.suffix))
            for file in f.rglob('*_integer_quant_with_int16_act.tflite'):
                file.unlink()  # delete extra fp16 activation TFLite files

        # Add TFLite metadata
        for file in f.rglob('*.tflite'):
            f.unlink() if 'quant_with_int16_act.tflite' in str(f) else self._add_tflite_metadata(file)

        # Load saved_model
        keras_model = tf.saved_model.load(f, tags=None, options=None)

        return str(f), keras_model

    @try_export
    def export_pb(self, keras_model, prefix=colorstr('TensorFlow GraphDef:')):
        """YOLOv8 TensorFlow GraphDef *.pb export https://github.com/leimao/Frozen_Graph_TensorFlow."""
        import tensorflow as tf  # noqa
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2  # noqa

        LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        f = self.file.with_suffix('.pb')

        m = tf.function(lambda x: keras_model(x))  # full model
        m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
        frozen_func = convert_variables_to_constants_v2(m)
        frozen_func.graph.as_graph_def()
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(f.parent), name=f.name, as_text=False)
        return f, None

    @try_export
    def export_tflite(self, keras_model, nms, agnostic_nms, prefix=colorstr('TensorFlow Lite:')):
        """YOLOv8 TensorFlow Lite export."""
        import tensorflow as tf  # noqa

        LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        saved_model = Path(str(self.file).replace(self.file.suffix, '_saved_model'))
        if self.args.int8:
            f = saved_model / f'{self.file.stem}_int8.tflite'  # fp32 in/out
        elif self.args.half:
            f = saved_model / f'{self.file.stem}_float16.tflite'  # fp32 in/out
        else:
            f = saved_model / f'{self.file.stem}_float32.tflite'
        return str(f), None

    @try_export
    def export_edgetpu(self, tflite_model='', prefix=colorstr('Edge TPU:')):
        """YOLOv8 Edge TPU export https://coral.ai/docs/edgetpu/models-intro/."""
        LOGGER.warning(f'{prefix} WARNING ⚠️ Edge TPU known bug https://github.com/ultralytics/ultralytics/issues/1185')

        cmd = 'edgetpu_compiler --version'
        help_url = 'https://coral.ai/docs/edgetpu/compiler/'
        assert LINUX, f'export only supported on Linux. See {help_url}'
        if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True).returncode != 0:
            LOGGER.info(f'\n{prefix} export requires Edge TPU compiler. Attempting install from {help_url}')
            sudo = subprocess.run('sudo --version >/dev/null', shell=True).returncode == 0  # sudo installed on system
            for c in (
                    'curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -',
                    'echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list',
                    'sudo apt-get update', 'sudo apt-get install edgetpu-compiler'):
                subprocess.run(c if sudo else c.replace('sudo ', ''), shell=True, check=True)
        ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1]

        LOGGER.info(f'\n{prefix} starting export with Edge TPU compiler {ver}...')
        f = str(tflite_model).replace('.tflite', '_edgetpu.tflite')  # Edge TPU model

        cmd = f'edgetpu_compiler -s -d -k 10 --out_dir "{Path(f).parent}" "{tflite_model}"'
        LOGGER.info(f"{prefix} running '{cmd}'")
        subprocess.run(cmd, shell=True)
        self._add_tflite_metadata(f)
        return f, None

    @try_export
    def export_tfjs(self, prefix=colorstr('TensorFlow.js:')):
        """YOLOv8 TensorFlow.js export."""
        check_requirements('tensorflowjs')
        import tensorflow as tf
        import tensorflowjs as tfjs  # noqa

        LOGGER.info(f'\n{prefix} starting export with tensorflowjs {tfjs.__version__}...')
        f = str(self.file).replace(self.file.suffix, '_web_model')  # js dir
        f_pb = str(self.file.with_suffix('.pb'))  # *.pb path

        gd = tf.Graph().as_graph_def()  # TF GraphDef
        with open(f_pb, 'rb') as file:
            gd.ParseFromString(file.read())
        outputs = ','.join(gd_outputs(gd))
        LOGGER.info(f'\n{prefix} output node names: {outputs}')

        with spaces_in_path(f_pb) as fpb_, spaces_in_path(f) as f_:  # exporter can not handle spaces in path
            cmd = f'tensorflowjs_converter --input_format=tf_frozen_model --output_node_names={outputs} "{fpb_}" "{f_}"'
            LOGGER.info(f"{prefix} running '{cmd}'")
            subprocess.run(cmd, shell=True)

        if ' ' in str(f):
            LOGGER.warning(f"{prefix} WARNING ⚠️ your model may not work correctly with spaces in path '{f}'.")

        # f_json = Path(f) / 'model.json'  # *.json path
        # with open(f_json, 'w') as j:  # sort JSON Identity_* in ascending order
        #     subst = re.sub(
        #         r'{"outputs": {"Identity.?.?": {"name": "Identity.?.?"}, '
        #         r'"Identity.?.?": {"name": "Identity.?.?"}, '
        #         r'"Identity.?.?": {"name": "Identity.?.?"}, '
        #         r'"Identity.?.?": {"name": "Identity.?.?"}}}',
        #         r'{"outputs": {"Identity": {"name": "Identity"}, '
        #         r'"Identity_1": {"name": "Identity_1"}, '
        #         r'"Identity_2": {"name": "Identity_2"}, '
        #         r'"Identity_3": {"name": "Identity_3"}}}',
        #         f_json.read_text(),
        #     )
        #     j.write(subst)
        yaml_save(Path(f) / 'metadata.yaml', self.metadata)  # add metadata.yaml
        return f, None

    def _add_tflite_metadata(self, file):
        """Add metadata to *.tflite models per https://www.tensorflow.org/lite/models/convert/metadata."""
        from tflite_support import flatbuffers  # noqa
        from tflite_support import metadata as _metadata  # noqa
        from tflite_support import metadata_schema_py_generated as _metadata_fb  # noqa

        # Create model info
        model_meta = _metadata_fb.ModelMetadataT()
        model_meta.name = self.metadata['description']
        model_meta.version = self.metadata['version']
        model_meta.author = self.metadata['author']
        model_meta.license = self.metadata['license']

        # Label file
        tmp_file = Path(file).parent / 'temp_meta.txt'
        with open(tmp_file, 'w') as f:
            f.write(str(self.metadata))

        label_file = _metadata_fb.AssociatedFileT()
        label_file.name = tmp_file.name
        label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS

        # Create input info
        input_meta = _metadata_fb.TensorMetadataT()
        input_meta.name = 'image'
        input_meta.description = 'Input image to be detected.'
        input_meta.content = _metadata_fb.ContentT()
        input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
        input_meta.content.contentProperties.colorSpace = _metadata_fb.ColorSpaceType.RGB
        input_meta.content.contentPropertiesType = _metadata_fb.ContentProperties.ImageProperties

        # Create output info
        output1 = _metadata_fb.TensorMetadataT()
        output1.name = 'output'
        output1.description = 'Coordinates of detected objects, class labels, and confidence score'
        output1.associatedFiles = [label_file]
        if self.model.task == 'segment':
            output2 = _metadata_fb.TensorMetadataT()
            output2.name = 'output'
            output2.description = 'Mask protos'
            output2.associatedFiles = [label_file]

        # Create subgraph info
        subgraph = _metadata_fb.SubGraphMetadataT()
        subgraph.inputTensorMetadata = [input_meta]
        subgraph.outputTensorMetadata = [output1, output2] if self.model.task == 'segment' else [output1]
        model_meta.subgraphMetadata = [subgraph]

        b = flatbuffers.Builder(0)
        b.Finish(model_meta.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
        metadata_buf = b.Output()

        populator = _metadata.MetadataPopulator.with_model_file(str(file))
        populator.load_metadata_buffer(metadata_buf)
        populator.load_associated_files([str(tmp_file)])
        populator.populate()
        tmp_file.unlink()

    def _pipeline_coreml(self, model, prefix=colorstr('CoreML Pipeline:')):
        """YOLOv8 CoreML pipeline."""
        import coremltools as ct  # noqa

        LOGGER.info(f'{prefix} starting pipeline with coremltools {ct.__version__}...')
        batch_size, ch, h, w = list(self.im.shape)  # BCHW

        # Output shapes
        spec = model.get_spec()
        out0, out1 = iter(spec.description.output)
        if MACOS:
            from PIL import Image
            img = Image.new('RGB', (w, h))  # img(192 width, 320 height)
            # img = torch.zeros((*opt.img_size, 3)).numpy()  # img size(320,192,3) iDetection
            out = model.predict({'image': img})
            out0_shape = out[out0.name].shape
            out1_shape = out[out1.name].shape
        else:  # linux and windows can not run model.predict(), get sizes from pytorch output y
            out0_shape = self.output_shape[2], self.output_shape[1] - 4  # (3780, 80)
            out1_shape = self.output_shape[2], 4  # (3780, 4)

        # Checks
        names = self.metadata['names']
        nx, ny = spec.description.input[0].type.imageType.width, spec.description.input[0].type.imageType.height
        na, nc = out0_shape
        # na, nc = out0.type.multiArrayType.shape  # number anchors, classes
        assert len(names) == nc, f'{len(names)} names found for nc={nc}'  # check

        # Define output shapes (missing)
        out0.type.multiArrayType.shape[:] = out0_shape  # (3780, 80)
        out1.type.multiArrayType.shape[:] = out1_shape  # (3780, 4)
        # spec.neuralNetwork.preprocessing[0].featureName = '0'

        # Flexible input shapes
        # from coremltools.models.neural_network import flexible_shape_utils
        # s = [] # shapes
        # s.append(flexible_shape_utils.NeuralNetworkImageSize(320, 192))
        # s.append(flexible_shape_utils.NeuralNetworkImageSize(640, 384))  # (height, width)
        # flexible_shape_utils.add_enumerated_image_sizes(spec, feature_name='image', sizes=s)
        # r = flexible_shape_utils.NeuralNetworkImageSizeRange()  # shape ranges
        # r.add_height_range((192, 640))
        # r.add_width_range((192, 640))
        # flexible_shape_utils.update_image_size_range(spec, feature_name='image', size_range=r)

        # Print
        # print(spec.description)

        # Model from spec
        model = ct.models.MLModel(spec)

        # 3. Create NMS protobuf
        nms_spec = ct.proto.Model_pb2.Model()
        nms_spec.specificationVersion = 5
        for i in range(2):
            decoder_output = model._spec.description.output[i].SerializeToString()
            nms_spec.description.input.add()
            nms_spec.description.input[i].ParseFromString(decoder_output)
            nms_spec.description.output.add()
            nms_spec.description.output[i].ParseFromString(decoder_output)

        nms_spec.description.output[0].name = 'confidence'
        nms_spec.description.output[1].name = 'coordinates'

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
        nms.confidenceOutputFeatureName = 'confidence'
        nms.coordinatesOutputFeatureName = 'coordinates'
        nms.iouThresholdInputFeatureName = 'iouThreshold'
        nms.confidenceThresholdInputFeatureName = 'confidenceThreshold'
        nms.iouThreshold = 0.45
        nms.confidenceThreshold = 0.25
        nms.pickTop.perClass = True
        nms.stringClassLabels.vector.extend(names.values())
        nms_model = ct.models.MLModel(nms_spec)

        # 4. Pipeline models together
        pipeline = ct.models.pipeline.Pipeline(input_features=[('image', ct.models.datatypes.Array(3, ny, nx)),
                                                               ('iouThreshold', ct.models.datatypes.Double()),
                                                               ('confidenceThreshold', ct.models.datatypes.Double())],
                                               output_features=['confidence', 'coordinates'])
        pipeline.add_model(model)
        pipeline.add_model(nms_model)

        # Correct datatypes
        pipeline.spec.description.input[0].ParseFromString(model._spec.description.input[0].SerializeToString())
        pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
        pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

        # Update metadata
        pipeline.spec.specificationVersion = 5
        pipeline.spec.description.metadata.userDefined.update({
            'IoU threshold': str(nms.iouThreshold),
            'Confidence threshold': str(nms.confidenceThreshold)})

        # Save the model
        model = ct.models.MLModel(pipeline.spec)
        model.input_description['image'] = 'Input image'
        model.input_description['iouThreshold'] = f'(optional) IOU threshold override (default: {nms.iouThreshold})'
        model.input_description['confidenceThreshold'] = \
            f'(optional) Confidence threshold override (default: {nms.confidenceThreshold})'
        model.output_description['confidence'] = 'Boxes × Class confidence (see user-defined metadata "classes")'
        model.output_description['coordinates'] = 'Boxes × [x, y, width, height] (relative to image size)'
        LOGGER.info(f'{prefix} pipeline success')
        return model

    def add_callback(self, event: str, callback):
        """
        Appends the given callback.
        """
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Execute all callbacks for a given event."""
        for callback in self.callbacks.get(event, []):
            callback(self)


class iOSDetectModel(torch.nn.Module):
    """Wrap an Ultralytics YOLO model for iOS export."""

    def __init__(self, model, im):
        """Initialize the iOSDetectModel class with a YOLO model and example image."""
        super().__init__()
        b, c, h, w = im.shape  # batch, channel, height, width
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


def export(cfg=DEFAULT_CFG):
    """Export a YOLOv model to a specific format."""
    cfg.model = cfg.model or 'yolov8n.yaml'
    cfg.format = cfg.format or 'torchscript'

    from ultralytics import YOLO
    model = YOLO(cfg.model)
    model.export(**vars(cfg))


if __name__ == '__main__':
    """
    CLI:
    yolo mode=export model=yolov8n.yaml format=onnx
    """
    export()
