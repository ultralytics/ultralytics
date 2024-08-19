# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Benchmark a YOLO model formats for speed and accuracy.

Usage:
    from ultralytics.utils.benchmarks import ProfileModels, benchmark
    ProfileModels(['yolov8n.yaml', 'yolov8s.yaml']).profile()
    benchmark(model='yolov8n.pt', imgsz=160)

Format                  | `format=argument`         | Model
---                     | ---                       | ---
PyTorch                 | -                         | yolov8n.pt
TorchScript             | `torchscript`             | yolov8n.torchscript
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
"""

import glob
import os
import platform
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch.cuda
import yaml

from ultralytics import YOLO, YOLOWorld
from ultralytics.cfg import TASK2DATA, TASK2METRIC
from ultralytics.engine.exporter import export_formats
from ultralytics.utils import ARM64, ASSETS, IS_JETSON, IS_RASPBERRYPI, LINUX, LOGGER, MACOS, TQDM, WEIGHTS_DIR
from ultralytics.utils.checks import IS_PYTHON_3_12, check_requirements, check_yolo
from ultralytics.utils.downloads import safe_download
from ultralytics.utils.files import file_size
from ultralytics.utils.torch_utils import select_device


def benchmark(
    model=WEIGHTS_DIR / "yolov8n.pt", data=None, imgsz=160, half=False, int8=False, device="cpu", verbose=False
):
    """
    Benchmark a YOLO model across different formats for speed and accuracy.

    Args:
        model (str | Path | optional): Path to the model file or directory. Default is
            Path(SETTINGS['weights_dir']) / 'yolov8n.pt'.
        data (str, optional): Dataset to evaluate on, inherited from TASK2DATA if not passed. Default is None.
        imgsz (int, optional): Image size for the benchmark. Default is 160.
        half (bool, optional): Use half-precision for the model if True. Default is False.
        int8 (bool, optional): Use int8-precision for the model if True. Default is False.
        device (str, optional): Device to run the benchmark on, either 'cpu' or 'cuda'. Default is 'cpu'.
        verbose (bool | float | optional): If True or a float, assert benchmarks pass with given metric.
            Default is False.

    Returns:
        df (pandas.DataFrame): A pandas DataFrame with benchmark results for each format, including file size,
            metric, and inference time.

    Example:
        ```python
        from ultralytics.utils.benchmarks import benchmark

        benchmark(model='yolov8n.pt', imgsz=640)
        ```
    """
    import pandas as pd  # scope for faster 'import ultralytics'

    pd.options.display.max_columns = 10
    pd.options.display.width = 120
    device = select_device(device, verbose=False)
    if isinstance(model, (str, Path)):
        model = YOLO(model)
    is_end2end = getattr(model.model.model[-1], "end2end", False)

    y = []
    t0 = time.time()
    for i, (name, format, suffix, cpu, gpu) in export_formats().iterrows():  # index, (name, format, suffix, CPU, GPU)
        emoji, filename = "❌", None  # export defaults
        try:
            # Checks
            if i == 7:  # TF GraphDef
                assert model.task != "obb", "TensorFlow GraphDef not supported for OBB task"
            elif i == 9:  # Edge TPU
                assert LINUX and not ARM64, "Edge TPU export only supported on non-aarch64 Linux"
            elif i in {5, 10}:  # CoreML and TF.js
                assert MACOS or LINUX, "CoreML and TF.js export only supported on macOS and Linux"
                assert not IS_RASPBERRYPI, "CoreML and TF.js export not supported on Raspberry Pi"
                assert not IS_JETSON, "CoreML and TF.js export not supported on NVIDIA Jetson"
                assert not is_end2end, "End-to-end models not supported by CoreML and TF.js yet"
            if i in {3, 5}:  # CoreML and OpenVINO
                assert not IS_PYTHON_3_12, "CoreML and OpenVINO not supported on Python 3.12"
            if i in {6, 7, 8}:  # TF SavedModel, TF GraphDef, and TFLite
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 TensorFlow exports not supported by onnx2tf yet"
            if i in {9, 10}:  # TF EdgeTPU and TF.js
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 TensorFlow exports not supported by onnx2tf yet"
                assert not is_end2end, "End-to-end models not supported by TF EdgeTPU and TF.js yet"
            if i in {11}:  # Paddle
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 Paddle exports not supported yet"
                assert not is_end2end, "End-to-end models not supported by PaddlePaddle yet"
            if i in {12}:  # NCNN
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 NCNN exports not supported yet"
                assert not is_end2end, "End-to-end models not supported by NCNN yet"
            if "cpu" in device.type:
                assert cpu, "inference not supported on CPU"
            if "cuda" in device.type:
                assert gpu, "inference not supported on GPU"

            # Export
            if format == "-":
                filename = model.ckpt_path or model.cfg
                exported_model = model  # PyTorch format
            else:
                filename = model.export(imgsz=imgsz, format=format, half=half, int8=int8, device=device, verbose=False)
                exported_model = YOLO(filename, task=model.task)
                assert suffix in str(filename), "export failed"
            emoji = "❎"  # indicates export succeeded

            # Predict
            assert model.task != "pose" or i != 7, "GraphDef Pose inference is not supported"
            assert i not in {9, 10}, "inference not supported"  # Edge TPU and TF.js are unsupported
            assert i != 5 or platform.system() == "Darwin", "inference only supported on macOS>=10.13"  # CoreML
            exported_model.predict(ASSETS / "bus.jpg", imgsz=imgsz, device=device, half=half)

            # Validate
            data = data or TASK2DATA[model.task]  # task to dataset, i.e. coco8.yaml for task=detect
            key = TASK2METRIC[model.task]  # task to metric, i.e. metrics/mAP50-95(B) for task=detect
            results = exported_model.val(
                data=data, batch=1, imgsz=imgsz, plots=False, device=device, half=half, int8=int8, verbose=False
            )
            metric, speed = results.results_dict[key], results.speed["inference"]
            fps = round((1000 / speed), 2)  # frames per second
            y.append([name, "✅", round(file_size(filename), 1), round(metric, 4), round(speed, 2), fps])
        except Exception as e:
            if verbose:
                assert type(e) is AssertionError, f"Benchmark failure for {name}: {e}"
            LOGGER.warning(f"ERROR ❌️ Benchmark failure for {name}: {e}")
            y.append([name, emoji, round(file_size(filename), 1), None, None, None])  # mAP, t_inference

    # Print results
    check_yolo(device=device)  # print system info
    df = pd.DataFrame(y, columns=["Format", "Status❔", "Size (MB)", key, "Inference time (ms/im)", "FPS"])

    name = Path(model.ckpt_path).name
    s = f"\nBenchmarks complete for {name} on {data} at imgsz={imgsz} ({time.time() - t0:.2f}s)\n{df}\n"
    LOGGER.info(s)
    with open("benchmarks.log", "a", errors="ignore", encoding="utf-8") as f:
        f.write(s)

    if verbose and isinstance(verbose, float):
        metrics = df[key].array  # values to compare to floor
        floor = verbose  # minimum metric floor to pass, i.e. = 0.29 mAP for YOLOv5n
        assert all(x > floor for x in metrics if pd.notna(x)), f"Benchmark failure: metric(s) < floor {floor}"

    return df


class RF100Benchmark:
    """Benchmark YOLO model performance across formats for speed and accuracy."""

    def __init__(self):
        """Function for initialization of RF100Benchmark."""
        self.ds_names = []
        self.ds_cfg_list = []
        self.rf = None
        self.val_metrics = ["class", "images", "targets", "precision", "recall", "map50", "map95"]

    def set_key(self, api_key):
        """
        Set Roboflow API key for processing.

        Args:
            api_key (str): The API key.
        """

        check_requirements("roboflow")
        from roboflow import Roboflow

        self.rf = Roboflow(api_key=api_key)

    def parse_dataset(self, ds_link_txt="datasets_links.txt"):
        """
        Parse dataset links and downloads datasets.

        Args:
            ds_link_txt (str): Path to dataset_links file.
        """

        (shutil.rmtree("rf-100"), os.mkdir("rf-100")) if os.path.exists("rf-100") else os.mkdir("rf-100")
        os.chdir("rf-100")
        os.mkdir("ultralytics-benchmarks")
        safe_download("https://github.com/ultralytics/assets/releases/download/v0.0.0/datasets_links.txt")

        with open(ds_link_txt, "r") as file:
            for line in file:
                try:
                    _, url, workspace, project, version = re.split("/+", line.strip())
                    self.ds_names.append(project)
                    proj_version = f"{project}-{version}"
                    if not Path(proj_version).exists():
                        self.rf.workspace(workspace).project(project).version(version).download("yolov8")
                    else:
                        print("Dataset already downloaded.")
                    self.ds_cfg_list.append(Path.cwd() / proj_version / "data.yaml")
                except Exception:
                    continue

        return self.ds_names, self.ds_cfg_list

    @staticmethod
    def fix_yaml(path):
        """
        Function to fix YAML train and val path.

        Args:
            path (str): YAML file path.
        """

        with open(path, "r") as file:
            yaml_data = yaml.safe_load(file)
        yaml_data["train"] = "train/images"
        yaml_data["val"] = "valid/images"
        with open(path, "w") as file:
            yaml.safe_dump(yaml_data, file)

    def evaluate(self, yaml_path, val_log_file, eval_log_file, list_ind):
        """
        Model evaluation on validation results.

        Args:
            yaml_path (str): YAML file path.
            val_log_file (str): val_log_file path.
            eval_log_file (str): eval_log_file path.
            list_ind (int): Index for current dataset.
        """
        skip_symbols = ["🚀", "⚠️", "💡", "❌"]
        with open(yaml_path) as stream:
            class_names = yaml.safe_load(stream)["names"]
        with open(val_log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            eval_lines = []
            for line in lines:
                if any(symbol in line for symbol in skip_symbols):
                    continue
                entries = line.split(" ")
                entries = list(filter(lambda val: val != "", entries))
                entries = [e.strip("\n") for e in entries]
                eval_lines.extend(
                    {
                        "class": entries[0],
                        "images": entries[1],
                        "targets": entries[2],
                        "precision": entries[3],
                        "recall": entries[4],
                        "map50": entries[5],
                        "map95": entries[6],
                    }
                    for e in entries
                    if e in class_names or (e == "all" and "(AP)" not in entries and "(AR)" not in entries)
                )
        map_val = 0.0
        if len(eval_lines) > 1:
            print("There's more dicts")
            for lst in eval_lines:
                if lst["class"] == "all":
                    map_val = lst["map50"]
        else:
            print("There's only one dict res")
            map_val = [res["map50"] for res in eval_lines][0]

        with open(eval_log_file, "a") as f:
            f.write(f"{self.ds_names[list_ind]}: {map_val}\n")


class ProfileModels:
    """
    ProfileModels class for profiling different models on ONNX and TensorRT.

    This class profiles the performance of different models, returning results such as model speed and FLOPs.

    Attributes:
        paths (list): Paths of the models to profile.
        num_timed_runs (int): Number of timed runs for the profiling. Default is 100.
        num_warmup_runs (int): Number of warmup runs before profiling. Default is 10.
        min_time (float): Minimum number of seconds to profile for. Default is 60.
        imgsz (int): Image size used in the models. Default is 640.

    Methods:
        profile(): Profiles the models and prints the result.

    Example:
        ```python
        from ultralytics.utils.benchmarks import ProfileModels

        ProfileModels(['yolov8n.yaml', 'yolov8s.yaml'], imgsz=640).profile()
        ```
    """

    def __init__(
        self,
        paths: list,
        num_timed_runs=100,
        num_warmup_runs=10,
        min_time=60,
        imgsz=640,
        half=True,
        trt=True,
        device=None,
    ):
        """
        Initialize the ProfileModels class for profiling models.

        Args:
            paths (list): List of paths of the models to be profiled.
            num_timed_runs (int, optional): Number of timed runs for the profiling. Default is 100.
            num_warmup_runs (int, optional): Number of warmup runs before the actual profiling starts. Default is 10.
            min_time (float, optional): Minimum time in seconds for profiling a model. Default is 60.
            imgsz (int, optional): Size of the image used during profiling. Default is 640.
            half (bool, optional): Flag to indicate whether to use half-precision floating point for profiling.
            trt (bool, optional): Flag to indicate whether to profile using TensorRT. Default is True.
            device (torch.device, optional): Device used for profiling. If None, it is determined automatically.
        """
        self.paths = paths
        self.num_timed_runs = num_timed_runs
        self.num_warmup_runs = num_warmup_runs
        self.min_time = min_time
        self.imgsz = imgsz
        self.half = half
        self.trt = trt  # run TensorRT profiling
        self.device = device or torch.device(0 if torch.cuda.is_available() else "cpu")

    def profile(self):
        """Logs the benchmarking results of a model, checks metrics against floor and returns the results."""
        files = self.get_files()

        if not files:
            print("No matching *.pt or *.onnx files found.")
            return

        table_rows = []
        output = []
        for file in files:
            engine_file = file.with_suffix(".engine")
            if file.suffix in {".pt", ".yaml", ".yml"}:
                model = YOLO(str(file))
                model.fuse()  # to report correct params and GFLOPs in model.info()
                model_info = model.info()
                if self.trt and self.device.type != "cpu" and not engine_file.is_file():
                    engine_file = model.export(
                        format="engine", half=self.half, imgsz=self.imgsz, device=self.device, verbose=False
                    )
                onnx_file = model.export(
                    format="onnx", half=self.half, imgsz=self.imgsz, simplify=True, device=self.device, verbose=False
                )
            elif file.suffix == ".onnx":
                model_info = self.get_onnx_model_info(file)
                onnx_file = file
            else:
                continue

            t_engine = self.profile_tensorrt_model(str(engine_file))
            t_onnx = self.profile_onnx_model(str(onnx_file))
            table_rows.append(self.generate_table_row(file.stem, t_onnx, t_engine, model_info))
            output.append(self.generate_results_dict(file.stem, t_onnx, t_engine, model_info))

        self.print_table(table_rows)
        return output

    def get_files(self):
        """Returns a list of paths for all relevant model files given by the user."""
        files = []
        for path in self.paths:
            path = Path(path)
            if path.is_dir():
                extensions = ["*.pt", "*.onnx", "*.yaml"]
                files.extend([file for ext in extensions for file in glob.glob(str(path / ext))])
            elif path.suffix in {".pt", ".yaml", ".yml"}:  # add non-existing
                files.append(str(path))
            else:
                files.extend(glob.glob(str(path)))

        print(f"Profiling: {sorted(files)}")
        return [Path(file) for file in sorted(files)]

    def get_onnx_model_info(self, onnx_file: str):
        """Retrieves the information including number of layers, parameters, gradients and FLOPs for an ONNX model
        file.
        """
        return 0.0, 0.0, 0.0, 0.0  # return (num_layers, num_params, num_gradients, num_flops)

    @staticmethod
    def iterative_sigma_clipping(data, sigma=2, max_iters=3):
        """Applies an iterative sigma clipping algorithm to the given data times number of iterations."""
        data = np.array(data)
        for _ in range(max_iters):
            mean, std = np.mean(data), np.std(data)
            clipped_data = data[(data > mean - sigma * std) & (data < mean + sigma * std)]
            if len(clipped_data) == len(data):
                break
            data = clipped_data
        return data

    def profile_tensorrt_model(self, engine_file: str, eps: float = 1e-3):
        """Profiles the TensorRT model, measuring average run time and standard deviation among runs."""
        if not self.trt or not Path(engine_file).is_file():
            return 0.0, 0.0

        # Model and input
        model = YOLO(engine_file)
        input_data = np.random.rand(self.imgsz, self.imgsz, 3).astype(np.float32)  # must be FP32

        # Warmup runs
        elapsed = 0.0
        for _ in range(3):
            start_time = time.time()
            for _ in range(self.num_warmup_runs):
                model(input_data, imgsz=self.imgsz, verbose=False)
            elapsed = time.time() - start_time

        # Compute number of runs as higher of min_time or num_timed_runs
        num_runs = max(round(self.min_time / (elapsed + eps) * self.num_warmup_runs), self.num_timed_runs * 50)

        # Timed runs
        run_times = []
        for _ in TQDM(range(num_runs), desc=engine_file):
            results = model(input_data, imgsz=self.imgsz, verbose=False)
            run_times.append(results[0].speed["inference"])  # Convert to milliseconds

        run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=3)  # sigma clipping
        return np.mean(run_times), np.std(run_times)

    def profile_onnx_model(self, onnx_file: str, eps: float = 1e-3):
        """Profiles an ONNX model by executing it multiple times and returns the mean and standard deviation of run
        times.
        """
        check_requirements("onnxruntime")
        import onnxruntime as ort

        # Session with either 'TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 8  # Limit the number of threads
        sess = ort.InferenceSession(onnx_file, sess_options, providers=["CPUExecutionProvider"])

        input_tensor = sess.get_inputs()[0]
        input_type = input_tensor.type
        dynamic = not all(isinstance(dim, int) and dim >= 0 for dim in input_tensor.shape)  # dynamic input shape
        input_shape = (1, 3, self.imgsz, self.imgsz) if dynamic else input_tensor.shape

        # Mapping ONNX datatype to numpy datatype
        if "float16" in input_type:
            input_dtype = np.float16
        elif "float" in input_type:
            input_dtype = np.float32
        elif "double" in input_type:
            input_dtype = np.float64
        elif "int64" in input_type:
            input_dtype = np.int64
        elif "int32" in input_type:
            input_dtype = np.int32
        else:
            raise ValueError(f"Unsupported ONNX datatype {input_type}")

        input_data = np.random.rand(*input_shape).astype(input_dtype)
        input_name = input_tensor.name
        output_name = sess.get_outputs()[0].name

        # Warmup runs
        elapsed = 0.0
        for _ in range(3):
            start_time = time.time()
            for _ in range(self.num_warmup_runs):
                sess.run([output_name], {input_name: input_data})
            elapsed = time.time() - start_time

        # Compute number of runs as higher of min_time or num_timed_runs
        num_runs = max(round(self.min_time / (elapsed + eps) * self.num_warmup_runs), self.num_timed_runs)

        # Timed runs
        run_times = []
        for _ in TQDM(range(num_runs), desc=onnx_file):
            start_time = time.time()
            sess.run([output_name], {input_name: input_data})
            run_times.append((time.time() - start_time) * 1000)  # Convert to milliseconds

        run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=5)  # sigma clipping
        return np.mean(run_times), np.std(run_times)

    def generate_table_row(self, model_name, t_onnx, t_engine, model_info):
        """Generates a formatted string for a table row that includes model performance and metric details."""
        layers, params, gradients, flops = model_info
        return (
            f"| {model_name:18s} | {self.imgsz} | - | {t_onnx[0]:.2f} ± {t_onnx[1]:.2f} ms | {t_engine[0]:.2f} ± "
            f"{t_engine[1]:.2f} ms | {params / 1e6:.1f} | {flops:.1f} |"
        )

    @staticmethod
    def generate_results_dict(model_name, t_onnx, t_engine, model_info):
        """Generates a dictionary of model details including name, parameters, GFLOPS and speed metrics."""
        layers, params, gradients, flops = model_info
        return {
            "model/name": model_name,
            "model/parameters": params,
            "model/GFLOPs": round(flops, 3),
            "model/speed_ONNX(ms)": round(t_onnx[0], 3),
            "model/speed_TensorRT(ms)": round(t_engine[0], 3),
        }

    @staticmethod
    def print_table(table_rows):
        """Formats and prints a comparison table for different models with given statistics and performance data."""
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
        header = (
            f"| Model | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | "
            f"Speed<br><sup>{gpu} TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |"
        )
        separator = (
            "|-------------|---------------------|--------------------|------------------------------|"
            "-----------------------------------|------------------|-----------------|"
        )

        print(f"\n\n{header}")
        print(separator)
        for row in table_rows:
            print(row)
