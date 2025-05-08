# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Benchmark a YOLO model formats for speed and accuracy.

Usage:
    from ultralytics.utils.benchmarks import ProfileModels, benchmark
    ProfileModels(['yolo11n.yaml', 'yolov8s.yaml']).run()
    benchmark(model='yolo11n.pt', imgsz=160)

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
RKNN                    | `rknn`                    | yolo11n_rknn_model/
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

from ultralytics import YOLO, YOLOWorld
from ultralytics.cfg import TASK2DATA, TASK2METRIC
from ultralytics.engine.exporter import export_formats
from ultralytics.utils import ARM64, ASSETS, IS_JETSON, LINUX, LOGGER, MACOS, TQDM, WEIGHTS_DIR, YAML
from ultralytics.utils.checks import IS_PYTHON_3_13, check_imgsz, check_requirements, check_yolo, is_rockchip
from ultralytics.utils.downloads import safe_download
from ultralytics.utils.files import file_size
from ultralytics.utils.torch_utils import get_cpu_info, select_device


def benchmark(
    model=WEIGHTS_DIR / "yolo11n.pt",
    data=None,
    imgsz=160,
    half=False,
    int8=False,
    device="cpu",
    verbose=False,
    eps=1e-3,
    format="",
):
    """
    Benchmark a YOLO model across different formats for speed and accuracy.

    Args:
        model (str | Path): Path to the model file or directory.
        data (str | None): Dataset to evaluate on, inherited from TASK2DATA if not passed.
        imgsz (int): Image size for the benchmark.
        half (bool): Use half-precision for the model if True.
        int8 (bool): Use int8-precision for the model if True.
        device (str): Device to run the benchmark on, either 'cpu' or 'cuda'.
        verbose (bool | float): If True or a float, assert benchmarks pass with given metric.
        eps (float): Epsilon value for divide by zero prevention.
        format (str): Export format for benchmarking. If not supplied all formats are benchmarked.

    Returns:
        (pandas.DataFrame): A pandas DataFrame with benchmark results for each format, including file size, metric,
            and inference time.

    Examples:
        Benchmark a YOLO model with default settings:
        >>> from ultralytics.utils.benchmarks import benchmark
        >>> benchmark(model="yolo11n.pt", imgsz=640)
    """
    imgsz = check_imgsz(imgsz)
    assert imgsz[0] == imgsz[1] if isinstance(imgsz, list) else True, "benchmark() only supports square imgsz."

    import pandas as pd  # scope for faster 'import ultralytics'

    pd.options.display.max_columns = 10
    pd.options.display.width = 120
    device = select_device(device, verbose=False)
    if isinstance(model, (str, Path)):
        model = YOLO(model)
    is_end2end = getattr(model.model.model[-1], "end2end", False)
    data = data or TASK2DATA[model.task]  # task to dataset, i.e. coco8.yaml for task=detect
    key = TASK2METRIC[model.task]  # task to metric, i.e. metrics/mAP50-95(B) for task=detect

    y = []
    t0 = time.time()

    format_arg = format.lower()
    if format_arg:
        formats = frozenset(export_formats()["Argument"])
        assert format in formats, f"Expected format to be one of {formats}, but got '{format_arg}'."
    for i, (name, format, suffix, cpu, gpu, _) in enumerate(zip(*export_formats().values())):
        emoji, filename = "❌", None  # export defaults
        try:
            if format_arg and format_arg != format:
                continue

            # Checks
            if i == 7:  # TF GraphDef
                assert model.task != "obb", "TensorFlow GraphDef not supported for OBB task"
            elif i == 9:  # Edge TPU
                assert LINUX and not ARM64, "Edge TPU export only supported on non-aarch64 Linux"
            elif i in {5, 10}:  # CoreML and TF.js
                assert MACOS or (LINUX and not ARM64), (
                    "CoreML and TF.js export only supported on macOS and non-aarch64 Linux"
                )
            if i in {5}:  # CoreML
                assert not IS_PYTHON_3_13, "CoreML not supported on Python 3.13"
            if i in {6, 7, 8, 9, 10}:  # TF SavedModel, TF GraphDef, and TFLite, TF EdgeTPU and TF.js
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 TensorFlow exports not supported by onnx2tf yet"
                # assert not IS_PYTHON_MINIMUM_3_12, "TFLite exports not supported on Python>=3.12 yet"
            if i == 11:  # Paddle
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 Paddle exports not supported yet"
                assert model.task != "obb", "Paddle OBB bug https://github.com/PaddlePaddle/Paddle/issues/72024"
                assert not is_end2end, "End-to-end models not supported by PaddlePaddle yet"
                assert (LINUX and not IS_JETSON) or MACOS, "Windows and Jetson Paddle exports not supported yet"
            if i == 12:  # MNN
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 MNN exports not supported yet"
            if i == 13:  # NCNN
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 NCNN exports not supported yet"
            if i == 14:  # IMX
                assert not is_end2end
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 IMX exports not supported"
                assert model.task == "detect", "IMX only supported for detection task"
                assert "C2f" in model.__str__(), "IMX only supported for YOLOv8"  # TODO: enable for YOLO11
            if i == 15:  # RKNN
                assert not isinstance(model, YOLOWorld), "YOLOWorldv2 RKNN exports not supported yet"
                assert not is_end2end, "End-to-end models not supported by RKNN yet"
                assert LINUX, "RKNN only supported on Linux"
                assert not is_rockchip(), "RKNN Inference only supported on Rockchip devices"
            if "cpu" in device.type:
                assert cpu, "inference not supported on CPU"
            if "cuda" in device.type:
                assert gpu, "inference not supported on GPU"

            # Export
            if format == "-":
                filename = model.pt_path or model.ckpt_path or model.model_name
                exported_model = model  # PyTorch format
            else:
                filename = model.export(
                    imgsz=imgsz, format=format, half=half, int8=int8, data=data, device=device, verbose=False
                )
                exported_model = YOLO(filename, task=model.task)
                assert suffix in str(filename), "export failed"
            emoji = "❎"  # indicates export succeeded

            # Predict
            assert model.task != "pose" or i != 7, "GraphDef Pose inference is not supported"
            assert i not in {9, 10}, "inference not supported"  # Edge TPU and TF.js are unsupported
            assert i != 5 or platform.system() == "Darwin", "inference only supported on macOS>=10.13"  # CoreML
            if i in {13}:
                assert not is_end2end, "End-to-end torch.topk operation is not supported for NCNN prediction yet"
            exported_model.predict(ASSETS / "bus.jpg", imgsz=imgsz, device=device, half=half, verbose=False)

            # Validate
            results = exported_model.val(
                data=data, batch=1, imgsz=imgsz, plots=False, device=device, half=half, int8=int8, verbose=False
            )
            metric, speed = results.results_dict[key], results.speed["inference"]
            fps = round(1000 / (speed + eps), 2)  # frames per second
            y.append([name, "✅", round(file_size(filename), 1), round(metric, 4), round(speed, 2), fps])
        except Exception as e:
            if verbose:
                assert type(e) is AssertionError, f"Benchmark failure for {name}: {e}"
            LOGGER.error(f"Benchmark failure for {name}: {e}")
            y.append([name, emoji, round(file_size(filename), 1), None, None, None])  # mAP, t_inference

    # Print results
    check_yolo(device=device)  # print system info
    df = pd.DataFrame(y, columns=["Format", "Status❔", "Size (MB)", key, "Inference time (ms/im)", "FPS"])

    name = model.model_name
    dt = time.time() - t0
    legend = "Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed"
    s = f"\nBenchmarks complete for {name} on {data} at imgsz={imgsz} ({dt:.2f}s)\n{legend}\n{df.fillna('-')}\n"
    LOGGER.info(s)
    with open("benchmarks.log", "a", errors="ignore", encoding="utf-8") as f:
        f.write(s)

    if verbose and isinstance(verbose, float):
        metrics = df[key].array  # values to compare to floor
        floor = verbose  # minimum metric floor to pass, i.e. = 0.29 mAP for YOLOv5n
        assert all(x > floor for x in metrics if pd.notna(x)), f"Benchmark failure: metric(s) < floor {floor}"

    return df


class RF100Benchmark:
    """
    Benchmark YOLO model performance across various formats for speed and accuracy.

    This class provides functionality to benchmark YOLO models on the RF100 dataset collection.

    Attributes:
        ds_names (List[str]): Names of datasets used for benchmarking.
        ds_cfg_list (List[Path]): List of paths to dataset configuration files.
        rf (Roboflow): Roboflow instance for accessing datasets.
        val_metrics (List[str]): Metrics used for validation.

    Methods:
        set_key: Set Roboflow API key for accessing datasets.
        parse_dataset: Parse dataset links and download datasets.
        fix_yaml: Fix train and validation paths in YAML files.
        evaluate: Evaluate model performance on validation results.
    """

    def __init__(self):
        """Initialize the RF100Benchmark class for benchmarking YOLO model performance across various formats."""
        self.ds_names = []
        self.ds_cfg_list = []
        self.rf = None
        self.val_metrics = ["class", "images", "targets", "precision", "recall", "map50", "map95"]

    def set_key(self, api_key):
        """
        Set Roboflow API key for processing.

        Args:
            api_key (str): The API key.

        Examples:
            Set the Roboflow API key for accessing datasets:
            >>> benchmark = RF100Benchmark()
            >>> benchmark.set_key("your_roboflow_api_key")
        """
        check_requirements("roboflow")
        from roboflow import Roboflow

        self.rf = Roboflow(api_key=api_key)

    def parse_dataset(self, ds_link_txt="datasets_links.txt"):
        """
        Parse dataset links and download datasets.

        Args:
            ds_link_txt (str): Path to the file containing dataset links.

        Returns:
            ds_names (List[str]): List of dataset names.
            ds_cfg_list (List[Path]): List of paths to dataset configuration files.

        Examples:
            >>> benchmark = RF100Benchmark()
            >>> benchmark.set_key("api_key")
            >>> benchmark.parse_dataset("datasets_links.txt")
        """
        (shutil.rmtree("rf-100"), os.mkdir("rf-100")) if os.path.exists("rf-100") else os.mkdir("rf-100")
        os.chdir("rf-100")
        os.mkdir("ultralytics-benchmarks")
        safe_download("https://github.com/ultralytics/assets/releases/download/v0.0.0/datasets_links.txt")

        with open(ds_link_txt, encoding="utf-8") as file:
            for line in file:
                try:
                    _, url, workspace, project, version = re.split("/+", line.strip())
                    self.ds_names.append(project)
                    proj_version = f"{project}-{version}"
                    if not Path(proj_version).exists():
                        self.rf.workspace(workspace).project(project).version(version).download("yolov8")
                    else:
                        LOGGER.info("Dataset already downloaded.")
                    self.ds_cfg_list.append(Path.cwd() / proj_version / "data.yaml")
                except Exception:
                    continue

        return self.ds_names, self.ds_cfg_list

    @staticmethod
    def fix_yaml(path):
        """Fix the train and validation paths in a given YAML file."""
        yaml_data = YAML.load(path)
        yaml_data["train"] = "train/images"
        yaml_data["val"] = "valid/images"
        YAML.dump(yaml_data, path)

    def evaluate(self, yaml_path, val_log_file, eval_log_file, list_ind):
        """
        Evaluate model performance on validation results.

        Args:
            yaml_path (str): Path to the YAML configuration file.
            val_log_file (str): Path to the validation log file.
            eval_log_file (str): Path to the evaluation log file.
            list_ind (int): Index of the current dataset in the list.

        Returns:
            (float): The mean average precision (mAP) value for the evaluated model.

        Examples:
            Evaluate a model on a specific dataset
            >>> benchmark = RF100Benchmark()
            >>> benchmark.evaluate("path/to/data.yaml", "path/to/val_log.txt", "path/to/eval_log.txt", 0)
        """
        skip_symbols = ["🚀", "⚠️", "💡", "❌"]
        class_names = YAML.load(yaml_path)["names"]
        with open(val_log_file, encoding="utf-8") as f:
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
            LOGGER.info("Multiple dicts found")
            for lst in eval_lines:
                if lst["class"] == "all":
                    map_val = lst["map50"]
        else:
            LOGGER.info("Single dict found")
            map_val = [res["map50"] for res in eval_lines][0]

        with open(eval_log_file, "a", encoding="utf-8") as f:
            f.write(f"{self.ds_names[list_ind]}: {map_val}\n")


class ProfileModels:
    """
    ProfileModels class for profiling different models on ONNX and TensorRT.

    This class profiles the performance of different models, returning results such as model speed and FLOPs.

    Attributes:
        paths (List[str]): Paths of the models to profile.
        num_timed_runs (int): Number of timed runs for the profiling.
        num_warmup_runs (int): Number of warmup runs before profiling.
        min_time (float): Minimum number of seconds to profile for.
        imgsz (int): Image size used in the models.
        half (bool): Flag to indicate whether to use FP16 half-precision for TensorRT profiling.
        trt (bool): Flag to indicate whether to profile using TensorRT.
        device (torch.device): Device used for profiling.

    Methods:
        profile: Profiles the models and prints the result.
        get_files: Gets all relevant model files.
        get_onnx_model_info: Extracts metadata from an ONNX model.
        iterative_sigma_clipping: Applies sigma clipping to remove outliers.
        profile_tensorrt_model: Profiles a TensorRT model.
        profile_onnx_model: Profiles an ONNX model.
        generate_table_row: Generates a table row with model metrics.
        generate_results_dict: Generates a dictionary of profiling results.
        print_table: Prints a formatted table of results.

    Examples:
        Profile models and print results
        >>> from ultralytics.utils.benchmarks import ProfileModels
        >>> profiler = ProfileModels(["yolo11n.yaml", "yolov8s.yaml"], imgsz=640)
        >>> profiler.run()
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
            paths (List[str]): List of paths of the models to be profiled.
            num_timed_runs (int): Number of timed runs for the profiling.
            num_warmup_runs (int): Number of warmup runs before the actual profiling starts.
            min_time (float): Minimum time in seconds for profiling a model.
            imgsz (int): Size of the image used during profiling.
            half (bool): Flag to indicate whether to use FP16 half-precision for TensorRT profiling.
            trt (bool): Flag to indicate whether to profile using TensorRT.
            device (torch.device | str | None): Device used for profiling. If None, it is determined automatically.

        Notes:
            FP16 'half' argument option removed for ONNX as slower on CPU than FP32.

        Examples:
            Initialize and profile models
            >>> from ultralytics.utils.benchmarks import ProfileModels
            >>> profiler = ProfileModels(["yolo11n.yaml", "yolov8s.yaml"], imgsz=640)
            >>> profiler.run()
        """
        self.paths = paths
        self.num_timed_runs = num_timed_runs
        self.num_warmup_runs = num_warmup_runs
        self.min_time = min_time
        self.imgsz = imgsz
        self.half = half
        self.trt = trt  # run TensorRT profiling
        self.device = device if isinstance(device, torch.device) else select_device(device)

    def run(self):
        """
        Profile YOLO models for speed and accuracy across various formats including ONNX and TensorRT.

        Returns:
            (List[Dict]): List of dictionaries containing profiling results for each model.

        Examples:
            Profile models and print results
            >>> from ultralytics.utils.benchmarks import ProfileModels
            >>> profiler = ProfileModels(["yolo11n.yaml", "yolov8s.yaml"])
            >>> results = profiler.run()
        """
        files = self.get_files()

        if not files:
            LOGGER.warning("No matching *.pt or *.onnx files found.")
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
                        format="engine",
                        half=self.half,
                        imgsz=self.imgsz,
                        device=self.device,
                        verbose=False,
                    )
                onnx_file = model.export(
                    format="onnx",
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
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
        """
        Return a list of paths for all relevant model files given by the user.

        Returns:
            (List[Path]): List of Path objects for the model files.
        """
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

        LOGGER.info(f"Profiling: {sorted(files)}")
        return [Path(file) for file in sorted(files)]

    @staticmethod
    def get_onnx_model_info(onnx_file: str):
        """Extracts metadata from an ONNX model file including parameters, GFLOPs, and input shape."""
        return 0.0, 0.0, 0.0, 0.0  # return (num_layers, num_params, num_gradients, num_flops)

    @staticmethod
    def iterative_sigma_clipping(data, sigma=2, max_iters=3):
        """
        Apply iterative sigma clipping to data to remove outliers.

        Args:
            data (numpy.ndarray): Input data array.
            sigma (float): Number of standard deviations to use for clipping.
            max_iters (int): Maximum number of iterations for the clipping process.

        Returns:
            (numpy.ndarray): Clipped data array with outliers removed.
        """
        data = np.array(data)
        for _ in range(max_iters):
            mean, std = np.mean(data), np.std(data)
            clipped_data = data[(data > mean - sigma * std) & (data < mean + sigma * std)]
            if len(clipped_data) == len(data):
                break
            data = clipped_data
        return data

    def profile_tensorrt_model(self, engine_file: str, eps: float = 1e-3):
        """
        Profile YOLO model performance with TensorRT, measuring average run time and standard deviation.

        Args:
            engine_file (str): Path to the TensorRT engine file.
            eps (float): Small epsilon value to prevent division by zero.

        Returns:
            mean_time (float): Mean inference time in milliseconds.
            std_time (float): Standard deviation of inference time in milliseconds.
        """
        if not self.trt or not Path(engine_file).is_file():
            return 0.0, 0.0

        # Model and input
        model = YOLO(engine_file)
        input_data = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)  # use uint8 for Classify

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
        """
        Profile an ONNX model, measuring average inference time and standard deviation across multiple runs.

        Args:
            onnx_file (str): Path to the ONNX model file.
            eps (float): Small epsilon value to prevent division by zero.

        Returns:
            mean_time (float): Mean inference time in milliseconds.
            std_time (float): Standard deviation of inference time in milliseconds.
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
        """
        Generate a table row string with model performance metrics.

        Args:
            model_name (str): Name of the model.
            t_onnx (tuple): ONNX model inference time statistics (mean, std).
            t_engine (tuple): TensorRT engine inference time statistics (mean, std).
            model_info (tuple): Model information (layers, params, gradients, flops).

        Returns:
            (str): Formatted table row string with model metrics.
        """
        layers, params, gradients, flops = model_info
        return (
            f"| {model_name:18s} | {self.imgsz} | - | {t_onnx[0]:.1f}±{t_onnx[1]:.1f} ms | {t_engine[0]:.1f}±"
            f"{t_engine[1]:.1f} ms | {params / 1e6:.1f} | {flops:.1f} |"
        )

    @staticmethod
    def generate_results_dict(model_name, t_onnx, t_engine, model_info):
        """
        Generate a dictionary of profiling results.

        Args:
            model_name (str): Name of the model.
            t_onnx (tuple): ONNX model inference time statistics (mean, std).
            t_engine (tuple): TensorRT engine inference time statistics (mean, std).
            model_info (tuple): Model information (layers, params, gradients, flops).

        Returns:
            (dict): Dictionary containing profiling results.
        """
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
        """
        Print a formatted table of model profiling results.

        Args:
            table_rows (List[str]): List of formatted table row strings.
        """
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
        headers = [
            "Model",
            "size<br><sup>(pixels)",
            "mAP<sup>val<br>50-95",
            f"Speed<br><sup>CPU ({get_cpu_info()}) ONNX<br>(ms)",
            f"Speed<br><sup>{gpu} TensorRT<br>(ms)",
            "params<br><sup>(M)",
            "FLOPs<br><sup>(B)",
        ]
        header = "|" + "|".join(f" {h} " for h in headers) + "|"
        separator = "|" + "|".join("-" * (len(h) + 2) for h in headers) + "|"

        LOGGER.info(f"\n\n{header}")
        LOGGER.info(separator)
        for row in table_rows:
            LOGGER.info(row)
