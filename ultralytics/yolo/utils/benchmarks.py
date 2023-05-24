# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Benchmark a YOLO model formats for speed and accuracy

Usage:
    from ultralytics.yolo.utils.benchmarks import ProfileModels, benchmark
    ProfileModels(['yolov8n.yaml', 'yolov8s.yaml'])
    run_benchmarks(model='yolov8n.pt', imgsz=160)

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
"""

import glob
import platform
import time
from pathlib import Path

import numpy as np
import torch.cuda
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.yolo.engine.exporter import export_formats
from ultralytics.yolo.utils import LINUX, LOGGER, MACOS, ROOT, SETTINGS
from ultralytics.yolo.utils.checks import check_requirements, check_yolo
from ultralytics.yolo.utils.downloads import download
from ultralytics.yolo.utils.files import file_size
from ultralytics.yolo.utils.torch_utils import select_device


def benchmark(model=Path(SETTINGS['weights_dir']) / 'yolov8n.pt',
              imgsz=160,
              half=False,
              int8=False,
              device='cpu',
              hard_fail=False):
    """
    Benchmark a YOLO model across different formats for speed and accuracy.

    Args:
        model (Union[str, Path], optional): Path to the model file or directory. Default is
            Path(SETTINGS['weights_dir']) / 'yolov8n.pt'.
        imgsz (int, optional): Image size for the benchmark. Default is 160.
        half (bool, optional): Use half-precision for the model if True. Default is False.
        int8 (bool, optional): Use int8-precision for the model if True. Default is False.
        device (str, optional): Device to run the benchmark on, either 'cpu' or 'cuda'. Default is 'cpu'.
        hard_fail (Union[bool, float], optional): If True or a float, assert benchmarks pass with given metric.
            Default is False.

    Returns:
        df (pandas.DataFrame): A pandas DataFrame with benchmark results for each format, including file size,
            metric, and inference time.
    """

    import pandas as pd
    pd.options.display.max_columns = 10
    pd.options.display.width = 120
    device = select_device(device, verbose=False)
    if isinstance(model, (str, Path)):
        model = YOLO(model)

    y = []
    t0 = time.time()
    for i, (name, format, suffix, cpu, gpu) in export_formats().iterrows():  # index, (name, format, suffix, CPU, GPU)
        emoji, filename = '❌', None  # export defaults
        try:
            assert i != 9 or LINUX, 'Edge TPU export only supported on Linux'
            if i == 10:
                assert MACOS or LINUX, 'TF.js export only supported on macOS and Linux'
            if 'cpu' in device.type:
                assert cpu, 'inference not supported on CPU'
            if 'cuda' in device.type:
                assert gpu, 'inference not supported on GPU'

            # Export
            if format == '-':
                filename = model.ckpt_path or model.cfg
                export = model  # PyTorch format
            else:
                filename = model.export(imgsz=imgsz, format=format, half=half, int8=int8, device=device)  # all others
                export = YOLO(filename, task=model.task)
                assert suffix in str(filename), 'export failed'
            emoji = '❎'  # indicates export succeeded

            # Predict
            assert i not in (9, 10), 'inference not supported'  # Edge TPU and TF.js are unsupported
            assert i != 5 or platform.system() == 'Darwin', 'inference only supported on macOS>=10.13'  # CoreML
            if not (ROOT / 'assets/bus.jpg').exists():
                download(url='https://ultralytics.com/images/bus.jpg', dir=ROOT / 'assets')
            export.predict(ROOT / 'assets/bus.jpg', imgsz=imgsz, device=device, half=half)

            # Validate
            if model.task == 'detect':
                data, key = 'coco8.yaml', 'metrics/mAP50-95(B)'
            elif model.task == 'segment':
                data, key = 'coco8-seg.yaml', 'metrics/mAP50-95(M)'
            elif model.task == 'classify':
                data, key = 'imagenet100', 'metrics/accuracy_top5'
            elif model.task == 'pose':
                data, key = 'coco8-pose.yaml', 'metrics/mAP50-95(P)'

            results = export.val(data=data,
                                 batch=1,
                                 imgsz=imgsz,
                                 plots=False,
                                 device=device,
                                 half=half,
                                 int8=int8,
                                 verbose=False)
            metric, speed = results.results_dict[key], results.speed['inference']
            y.append([name, '✅', round(file_size(filename), 1), round(metric, 4), round(speed, 2)])
        except Exception as e:
            if hard_fail:
                assert type(e) is AssertionError, f'Benchmark hard_fail for {name}: {e}'
            LOGGER.warning(f'ERROR ❌️ Benchmark failure for {name}: {e}')
            y.append([name, emoji, round(file_size(filename), 1), None, None])  # mAP, t_inference

    # Print results
    check_yolo(device=device)  # print system info
    df = pd.DataFrame(y, columns=['Format', 'Status❔', 'Size (MB)', key, 'Inference time (ms/im)'])

    name = Path(model.ckpt_path).name
    s = f'\nBenchmarks complete for {name} on {data} at imgsz={imgsz} ({time.time() - t0:.2f}s)\n{df}\n'
    LOGGER.info(s)
    with open('benchmarks.log', 'a', errors='ignore', encoding='utf-8') as f:
        f.write(s)

    if hard_fail and isinstance(hard_fail, float):
        metrics = df[key].array  # values to compare to floor
        floor = hard_fail  # minimum metric floor to pass, i.e. = 0.29 mAP for YOLOv5n
        assert all(x > floor for x in metrics if pd.notna(x)), f'HARD FAIL: one or more metric(s) < floor {floor}'

    return df


class ProfileModels:
    """
    ProfileModels class for profiling different models on ONNX and TensorRT.

    This class profiles the performance of different models, provided their paths. The profiling includes parameters such as
    model speed and FLOPs.

    Attributes:
        paths (list): Paths of the models to profile.
        num_timed_runs (int): Number of timed runs for the profiling. Default is 100.
        num_warmup_runs (int): Number of warmup runs before profiling. Default is 3.
        imgsz (int): Image size used in the models. Default is 640.

    Methods:
        profile(): Profiles the models and prints the result.
    """

    def __init__(self, paths: list, num_timed_runs=100, num_warmup_runs=10, imgsz=640, trt=True):
        self.paths = paths
        self.num_timed_runs = num_timed_runs
        self.num_warmup_runs = num_warmup_runs
        self.imgsz = imgsz
        self.trt = trt  # run TensorRT profiling
        self.profile()  # run profiling

    def profile(self):
        files = self.get_files()

        if not files:
            print('No matching *.pt or *.onnx files found.')
            return

        table_rows = []
        device = 0 if torch.cuda.is_available() else 'cpu'
        for file in files:
            engine_file = file.with_suffix('.engine')
            if file.suffix in ('.pt', '.yaml'):
                model = YOLO(str(file))
                model_info = model.info()
                if self.trt and device == 0 and not engine_file.is_file():
                    engine_file = model.export(format='engine', half=True, imgsz=self.imgsz, device=device)
                onnx_file = model.export(format='onnx', half=True, imgsz=self.imgsz, simplify=True, device=device)
            elif file.suffix == '.onnx':
                model_info = self.get_onnx_model_info(file)
                onnx_file = file
            else:
                continue

            t_engine = self.profile_tensorrt_model(str(engine_file))
            t_onnx = self.profile_onnx_model(str(onnx_file))
            table_rows.append(self.generate_table_row(file.stem, t_onnx, t_engine, model_info))

        self.print_table(table_rows)

    def get_files(self):
        files = []
        for path in self.paths:
            path = Path(path)
            if path.is_dir():
                extensions = ['*.pt', '*.onnx', '*.yaml']
                files.extend([file for ext in extensions for file in glob.glob(str(path / ext))])
            elif path.suffix in {'.pt', '.yaml'}:  # add non-existing
                files.append(str(path))
            else:
                files.extend(glob.glob(str(path)))

        print(f'Profiling: {sorted(files)}')
        return [Path(file) for file in sorted(files)]

    def get_onnx_model_info(self, onnx_file: str):
        # return (num_layers, num_params, num_gradients, num_flops)
        return 0.0, 0.0, 0.0, 0.0

    def iterative_sigma_clipping(self, data, sigma=2, max_iters=5):
        data = np.array(data)
        for _ in range(max_iters):
            mean, std = np.mean(data), np.std(data)
            clipped_data = data[(data > mean - sigma * std) & (data < mean + sigma * std)]
            if len(clipped_data) == len(data):
                break
            data = clipped_data
        return data

    def profile_tensorrt_model(self, engine_file: str):
        if not self.trt or not Path(engine_file).is_file():
            return 0.0, 0.0

        # Warmup runs
        model = YOLO(engine_file)
        input_data = np.random.rand(self.imgsz, self.imgsz, 3).astype(np.float32)
        for _ in range(self.num_warmup_runs):
            model(input_data, verbose=False)

        # Timed runs
        run_times = []
        for _ in tqdm(range(self.num_timed_runs * 30), desc=engine_file):
            results = model(input_data, verbose=False)
            run_times.append(results[0].speed['inference'])  # Convert to milliseconds

        run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=3)  # sigma clipping
        return np.mean(run_times), np.std(run_times)

    def profile_onnx_model(self, onnx_file: str):
        check_requirements('onnxruntime')
        import onnxruntime as ort

        # Session with either 'TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(onnx_file, sess_options, providers=['CPUExecutionProvider'])

        input_tensor = sess.get_inputs()[0]
        input_type = input_tensor.type

        # Mapping ONNX datatype to numpy datatype
        if 'float16' in input_type:
            input_dtype = np.float16
        elif 'float' in input_type:
            input_dtype = np.float32
        elif 'double' in input_type:
            input_dtype = np.float64
        elif 'int64' in input_type:
            input_dtype = np.int64
        elif 'int32' in input_type:
            input_dtype = np.int32
        else:
            raise ValueError(f'Unsupported ONNX datatype {input_type}')

        input_data = np.random.rand(*input_tensor.shape).astype(input_dtype)
        input_name = input_tensor.name
        output_name = sess.get_outputs()[0].name

        # Warmup runs
        for _ in range(self.num_warmup_runs):
            sess.run([output_name], {input_name: input_data})

        # Timed runs
        run_times = []
        for _ in tqdm(range(self.num_timed_runs), desc=onnx_file):
            start_time = time.time()
            sess.run([output_name], {input_name: input_data})
            run_times.append((time.time() - start_time) * 1000)  # Convert to milliseconds

        run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=3)  # sigma clipping
        return np.mean(run_times), np.std(run_times)

    def generate_table_row(self, model_name, t_onnx, t_engine, model_info):
        layers, params, gradients, flops = model_info
        return f'| {model_name:18s} | {self.imgsz} | - | {t_onnx[0]:.2f} ± {t_onnx[1]:.2f} ms | {t_engine[0]:.2f} ± {t_engine[1]:.2f} ms | {params / 1e6:.1f} | {flops:.1f} |'

    def print_table(self, table_rows):
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU'
        header = f'| Model | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>{gpu} TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |'
        separator = '|-------------|---------------------|--------------------|------------------------------|-----------------------------------|------------------|-----------------|'

        print(f'\n\n{header}')
        print(separator)
        for row in table_rows:
            print(row)


if __name__ == '__main__':
    # Benchmark all export formats
    benchmark()

    # Profiling models on ONNX and TensorRT
    ProfileModels(['yolov8n.yaml', 'yolov8s.yaml'])
