# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import contextlib
import importlib.metadata
import inspect
import json
import logging
import os
import platform
import re
import socket
import sys
import threading
import time
import warnings
from functools import lru_cache
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from urllib.parse import unquote

import cv2
import numpy as np
import torch

from ultralytics import __version__
from ultralytics.utils.git import GitRepo
from ultralytics.utils.patches import imread, imshow, imwrite, torch_save  # for patches
from ultralytics.utils.tqdm import TQDM  # noqa

# PyTorch Multi-GPU DDP Constants
RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html

# Other Constants
ARGV = sys.argv or ["", ""]  # sometimes sys.argv = []
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO
ASSETS = ROOT / "assets"  # default images
ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"  # assets GitHub URL
DEFAULT_CFG_PATH = ROOT / "cfg/default.yaml"
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLO multiprocessing threads
AUTOINSTALL = str(os.getenv("YOLO_AUTOINSTALL", True)).lower() == "true"  # global auto-install mode
VERBOSE = str(os.getenv("YOLO_VERBOSE", True)).lower() == "true"  # global verbose mode
LOGGING_NAME = "ultralytics"
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans
MACOS_VERSION = platform.mac_ver()[0] if MACOS else None
NOT_MACOS14 = not (MACOS and MACOS_VERSION.startswith("14."))
ARM64 = platform.machine() in {"arm64", "aarch64"}  # ARM64 booleans
PYTHON_VERSION = platform.python_version()
TORCH_VERSION = str(torch.__version__)  # Normalize torch.__version__ (PyTorch>1.9 returns TorchVersion objects)
TORCHVISION_VERSION = importlib.metadata.version("torchvision")  # faster than importing torchvision
IS_VSCODE = os.environ.get("TERM_PROGRAM", False) == "vscode"
RKNN_CHIPS = frozenset(
    {
        "rk3588",
        "rk3576",
        "rk3566",
        "rk3568",
        "rk3562",
        "rv1103",
        "rv1106",
        "rv1103b",
        "rv1106b",
        "rk2118",
        "rv1126b",
    }
)  # Rockchip processors available for export
HELP_MSG = """
    Examples for running Ultralytics:

    1. Install the ultralytics package:

        pip install ultralytics

    2. Use the Python SDK:

        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.yaml")  # build a new model from scratch
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Use the model
        results = model.train(data="coco8.yaml", epochs=3)  # train the model
        results = model.val()  # evaluate model performance on the validation set
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
        success = model.export(format="onnx")  # export the model to ONNX format

    3. Use the command line interface (CLI):

        Ultralytics 'yolo' CLI commands use the following syntax:

            yolo TASK MODE ARGS

            Where   TASK (optional) is one of [detect, segment, classify, pose, obb]
                    MODE (required) is one of [train, val, predict, export, track, benchmark]
                    ARGS (optional) are any number of custom "arg=value" pairs like "imgsz=320" that override defaults.
                        See all ARGS at https://docs.ultralytics.com/usage/cfg or with "yolo cfg"

        - Train a detection model for 10 epochs with an initial learning_rate of 0.01
            yolo detect train data=coco8.yaml model=yolo26n.pt epochs=10 lr0=0.01

        - Predict a YouTube video using a pretrained segmentation model at image size 320:
            yolo segment predict model=yolo26n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320

        - Val a pretrained detection model at batch-size 1 and image size 640:
            yolo detect val model=yolo26n.pt data=coco8.yaml batch=1 imgsz=640

        - Export a YOLO26n classification model to ONNX format at image size 224 by 128 (no TASK required)
            yolo export model=yolo26n-cls.pt format=onnx imgsz=224,128

        - Run special commands:
            yolo help
            yolo checks
            yolo version
            yolo settings
            yolo copy-cfg
            yolo cfg

    Docs: https://docs.ultralytics.com
    Community: https://community.ultralytics.com
    GitHub: https://github.com/ultralytics/ultralytics
    """

# Settings and Environment Variables
torch.set_printoptions(linewidth=320, precision=4, profile="default")
np.set_printoptions(linewidth=320, formatter=dict(float_kind="{:11.5g}".format))  # format short g, %precision=5
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_THREADS)  # NumExpr max threads
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress verbose TF compiler warnings in Colab
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # suppress "NNPACK.cpp could not initialize NNPACK" warnings
os.environ["KINETO_LOG_LEVEL"] = "5"  # suppress verbose PyTorch profiler output when computing FLOPs

# Centralized warning suppression
warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated")  # PyTorch deprecation
warnings.filterwarnings("ignore", message="The figure layout has changed to tight")  # matplotlib>=3.7.2
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")  # mobileclip timm.layers deprecation
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)  # ONNX/TorchScript export tracer warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*prim::Constant.*")  # ONNX shape warning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="coremltools")  # CoreML np.bool deprecation

# Precompiled type tuples for faster isinstance() checks
FLOAT_OR_INT = (float, int)
STR_OR_PATH = (str, Path)


class DataExportMixin:
    """Mixin class for exporting validation metrics or prediction results in various formats.

    This class provides utilities to export performance metrics (e.g., mAP, precision, recall) or prediction results
    from classification, object detection, segmentation, or pose estimation tasks into various formats: Polars
    DataFrame, CSV, and JSON.

    Methods:
        to_df: Convert summary to a Polars DataFrame.
        to_csv: Export results as a CSV string.
        to_json: Export results as a JSON string.
        tojson: Deprecated alias for `to_json()`.

    Examples:
        >>> model = YOLO("yolo26n.pt")
        >>> results = model("image.jpg")
        >>> df = results.to_df()
        >>> print(df)
        >>> csv_data = results.to_csv()
    """

    def to_df(self, normalize=False, decimals=5):
        """Create a Polars DataFrame from the prediction results summary or validation metrics.

        Args:
            normalize (bool, optional): Normalize numerical values for easier comparison.
            decimals (int, optional): Decimal places to round floats.

        Returns:
            (polars.DataFrame): Polars DataFrame containing the summary data.
        """
        import polars as pl  # scope for faster 'import ultralytics'

        return pl.DataFrame(self.summary(normalize=normalize, decimals=decimals))

    def to_csv(self, normalize=False, decimals=5):
        """Export results or metrics to CSV string format.

        Args:
            normalize (bool, optional): Normalize numeric values.
            decimals (int, optional): Decimal precision.

        Returns:
            (str): CSV content as string.
        """
        import polars as pl

        df = self.to_df(normalize=normalize, decimals=decimals)

        try:
            return df.write_csv()
        except Exception:
            # Minimal string conversion for any remaining complex types
            def _to_str_simple(v):
                if v is None:
                    return ""
                elif isinstance(v, (dict, list, tuple, set)):
                    return repr(v)
                else:
                    return str(v)

            df_str = df.select(
                [pl.col(c).map_elements(_to_str_simple, return_dtype=pl.String).alias(c) for c in df.columns]
            )
            return df_str.write_csv()

    def to_json(self, normalize=False, decimals=5):
        """Export results to JSON format.

        Args:
            normalize (bool, optional): Normalize numeric values.
            decimals (int, optional): Decimal precision.

        Returns:
            (str): JSON-formatted string of the results.
        """
        return self.to_df(normalize=normalize, decimals=decimals).write_json()


class SimpleClass:
    """A simple base class for creating objects with string representations of their attributes.

    This class provides a foundation for creating objects that can be easily printed or represented as strings, showing
    all their non-callable attributes. It's useful for debugging and introspection of object states.

    Methods:
        __str__: Return a human-readable string representation of the object.
        __repr__: Return a machine-readable string representation of the object.
        __getattr__: Provide a custom attribute access error message with helpful information.

    Examples:
        >>> class MyClass(SimpleClass):
        ...     def __init__(self):
        ...         self.x = 10
        ...         self.y = "hello"
        >>> obj = MyClass()
        >>> print(obj)
        __main__.MyClass object with attributes:

        x: 10
        y: 'hello'

    Notes:
        - This class is designed to be subclassed. It provides a convenient way to inspect object attributes.
        - The string representation includes the module and class name of the object.
        - Callable attributes and attributes starting with an underscore are excluded from the string representation.
    """

    def __str__(self):
        """Return a human-readable string representation of the object."""
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith("_"):
                if isinstance(v, SimpleClass):
                    # Display only the module and class name for subclasses
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
                else:
                    s = f"{a}: {v!r}"
                attr.append(s)
        return f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n" + "\n".join(attr)

    def __repr__(self):
        """Return a machine-readable string representation of the object."""
        return self.__str__()

    def __getattr__(self, attr):
        """Provide a custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


class IterableSimpleNamespace(SimpleNamespace):
    """An iterable SimpleNamespace class that provides enhanced functionality for attribute access and iteration.

    This class extends the SimpleNamespace class with additional methods for iteration, string representation, and
    attribute access. It is designed to be used as a convenient container for storing and accessing configuration
    parameters.

    Methods:
        __iter__: Return an iterator of key-value pairs from the namespace's attributes.
        __str__: Return a human-readable string representation of the object.
        __getattr__: Provide a custom attribute access error message with helpful information.
        get: Retrieve the value of a specified key, or a default value if the key doesn't exist.

    Examples:
        >>> cfg = IterableSimpleNamespace(a=1, b=2, c=3)
        >>> for k, v in cfg:
        ...     print(f"{k}: {v}")
        a: 1
        b: 2
        c: 3
        >>> print(cfg)
        a=1
        b=2
        c=3
        >>> cfg.get("b")
        2
        >>> cfg.get("d", "default")
        'default'

    Notes:
        This class is particularly useful for storing configuration parameters in a more accessible
        and iterable format compared to a standard dictionary.
    """

    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr):
        """Provide a custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(
            f"""
            '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date ultralytics
            'default.yaml' file.\nPlease update your code with 'pip install -U ultralytics' and if necessary replace
            {DEFAULT_CFG_PATH} with the latest version from
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
            """
        )

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)


def plt_settings(rcparams=None, backend="Agg"):
    """Decorator to temporarily set rc parameters and the backend for a plotting function.

    Args:
        rcparams (dict, optional): Dictionary of rc parameters to set.
        backend (str, optional): Name of the backend to use.

    Returns:
        (Callable): Decorated function with temporarily set rc parameters and backend.

    Examples:
        >>> @plt_settings({"font.size": 12})
        ... def plot_function():
        ...     plt.figure()
        ...     plt.plot([1, 2, 3])
        ...     plt.show()

        >>> with plt_settings({"font.size": 12}):
        ...     plt.figure()
        ...     plt.plot([1, 2, 3])
        ...     plt.show()
    """
    if rcparams is None:
        rcparams = {"font.size": 11}

    def decorator(func):
        """Decorator to apply temporary rc parameters and backend to a function."""

        def wrapper(*args, **kwargs):
            """Set rc parameters and backend, call the original function, and restore the settings."""
            import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

            original_backend = plt.get_backend()
            switch = backend.lower() != original_backend.lower()
            if switch:
                plt.close("all")  # auto-close()ing of figures upon backend switching is deprecated since 3.8
                plt.switch_backend(backend)

            # Plot with backend and always revert to original backend
            try:
                with plt.rc_context(rcparams):
                    result = func(*args, **kwargs)
            finally:
                if switch:
                    plt.close("all")
                    plt.switch_backend(original_backend)
            return result

        return wrapper

    return decorator


def set_logging(name="LOGGING_NAME", verbose=True):
    """Set up logging with UTF-8 encoding and configurable verbosity.

    This function configures logging for the Ultralytics library, setting the appropriate logging level and formatter
    based on the verbosity flag and the current process rank. It handles special cases for Windows environments where
    UTF-8 encoding might not be the default.

    Args:
        name (str): Name of the logger.
        verbose (bool): Flag to set logging level to INFO if True, ERROR otherwise.

    Returns:
        (logging.Logger): Configured logger object.

    Examples:
        >>> set_logging(name="ultralytics", verbose=True)
        >>> logger = logging.getLogger("ultralytics")
        >>> logger.info("This is an info message")

    Notes:
        - On Windows, this function attempts to reconfigure stdout to use UTF-8 encoding if possible.
        - If reconfiguration is not possible, it falls back to a custom formatter that handles non-UTF-8 environments.
        - The function sets up a StreamHandler with the appropriate formatter and level.
        - The logger's propagate flag is set to False to prevent duplicate logging in parent loggers.
    """
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR  # rank in world for Multi-GPU trainings

    class PrefixFormatter(logging.Formatter):
        def format(self, record):
            """Format log records with prefixes based on level."""
            # Apply prefixes based on log level
            if record.levelno == logging.WARNING:
                prefix = "WARNING" if WINDOWS else "WARNING ⚠️"
                record.msg = f"{prefix} {record.msg}"
            elif record.levelno == logging.ERROR:
                prefix = "ERROR" if WINDOWS else "ERROR ❌"
                record.msg = f"{prefix} {record.msg}"

            # Handle emojis in message based on platform
            formatted_message = super().format(record)
            return emojis(formatted_message)

    formatter = PrefixFormatter("%(message)s")

    # Handle Windows UTF-8 encoding issues
    if WINDOWS and hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8":
        with contextlib.suppress(Exception):
            # Attempt to reconfigure stdout to use UTF-8 encoding if possible
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            # For environments where reconfigure is not available, wrap stdout in a TextIOWrapper
            elif hasattr(sys.stdout, "buffer"):
                import io

                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    # Create and configure the StreamHandler with the appropriate formatter and level
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    # Set up the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


# Set logger
LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)  # define globally (used in train.py, val.py, predict.py, etc.)
logging.getLogger("sentry_sdk").setLevel(logging.CRITICAL + 1)


def emojis(string=""):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode("ascii", "ignore") if WINDOWS else string


class ThreadingLocked:
    """A decorator class for ensuring thread-safe execution of a function or method.

    This class can be used as a decorator to make sure that if the decorated function is called from multiple threads,
    only one thread at a time will be able to execute the function.

    Attributes:
        lock (threading.Lock): A lock object used to manage access to the decorated function.

    Examples:
        >>> from ultralytics.utils import ThreadingLocked
        >>> @ThreadingLocked()
        ... def my_function():
        ...    # Your code here
    """

    def __init__(self):
        """Initialize the decorator class with a threading lock."""
        self.lock = threading.Lock()

    def __call__(self, f):
        """Run thread-safe execution of function or method."""
        from functools import wraps

        @wraps(f)
        def decorated(*args, **kwargs):
            """Apply thread-safety to the decorated function or method."""
            with self.lock:
                return f(*args, **kwargs)

        return decorated


class YAML:
    """YAML utility class for efficient file operations with automatic C-implementation detection.

    This class provides optimized YAML loading and saving operations using PyYAML's fastest available implementation
    (C-based when possible). It implements a singleton pattern with lazy initialization, allowing direct class method
    usage without explicit instantiation. The class handles file path creation, validation, and character encoding
    issues automatically.

    The implementation prioritizes performance through:
        - Automatic C-based loader/dumper selection when available
        - Singleton pattern to reuse the same instance
        - Lazy initialization to defer import costs until needed
        - Fallback mechanisms for handling problematic YAML content

    Attributes:
        _instance: Internal singleton instance storage.
        yaml: Reference to the PyYAML module.
        SafeLoader: Best available YAML loader (CSafeLoader if available).
        SafeDumper: Best available YAML dumper (CSafeDumper if available).

    Examples:
        >>> data = YAML.load("config.yaml")
        >>> data["new_value"] = 123
        >>> YAML.save("updated_config.yaml", data)
        >>> YAML.print(data)
    """

    _instance = None

    @classmethod
    def _get_instance(cls):
        """Initialize singleton instance on first use."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize with optimal YAML implementation (C-based when available)."""
        import yaml

        self.yaml = yaml
        # Use C-based implementation if available for better performance
        try:
            self.SafeLoader = yaml.CSafeLoader
            self.SafeDumper = yaml.CSafeDumper
        except (AttributeError, ImportError):
            self.SafeLoader = yaml.SafeLoader
            self.SafeDumper = yaml.SafeDumper

    @classmethod
    def save(cls, file="data.yaml", data=None, header=""):
        """Save Python object as YAML file.

        Args:
            file (str | Path): Path to save YAML file.
            data (dict | None): Dict or compatible object to save.
            header (str): Optional string to add at file beginning.
        """
        instance = cls._get_instance()
        if data is None:
            data = {}

        # Create parent directories if needed
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)

        # Convert non-serializable objects to strings
        valid_types = int, float, str, bool, list, tuple, dict, type(None)
        for k, v in data.items():
            if not isinstance(v, valid_types):
                data[k] = str(v)

        # Write YAML file
        with open(file, "w", errors="ignore", encoding="utf-8") as f:
            if header:
                f.write(header)
            instance.yaml.dump(data, f, sort_keys=False, allow_unicode=True, Dumper=instance.SafeDumper)

    @classmethod
    def load(cls, file="data.yaml", append_filename=False):
        """Load YAML file to Python object with robust error handling.

        Args:
            file (str | Path): Path to YAML file.
            append_filename (bool): Whether to add filename to returned dict.

        Returns:
            (dict): Loaded YAML content.
        """
        instance = cls._get_instance()
        assert str(file).endswith((".yaml", ".yml")), f"Not a YAML file: {file}"

        # Read file content
        with open(file, errors="ignore", encoding="utf-8") as f:
            s = f.read()

        # Try loading YAML with fallback for problematic characters
        try:
            data = instance.yaml.load(s, Loader=instance.SafeLoader) or {}
        except Exception:
            # Remove problematic characters and retry
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
            data = instance.yaml.load(s, Loader=instance.SafeLoader) or {}

        # Check for accidental user-error None strings (should be 'null' in YAML)
        if "None" in data.values():
            data = {k: None if v == "None" else v for k, v in data.items()}

        if append_filename:
            data["yaml_file"] = str(file)
        return data

    @classmethod
    def print(cls, yaml_file):
        """Pretty print YAML file or object to console.

        Args:
            yaml_file (str | Path | dict): Path to YAML file or dict to print.
        """
        instance = cls._get_instance()

        # Load file if path provided
        yaml_dict = cls.load(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file

        # Use -1 for unlimited width in C implementation
        dump = instance.yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True, width=-1, Dumper=instance.SafeDumper)

        LOGGER.info(f"Printing '{colorstr('bold', 'black', yaml_file)}'\n\n{dump}")


# Default configuration
DEFAULT_CFG_DICT = YAML.load(DEFAULT_CFG_PATH)
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)


def read_device_model() -> str:
    """Read the device model information from the system.

    Returns:
        (str): Platform release string in lowercase, used to identify device models like Jetson or Raspberry Pi.
    """
    return platform.release().lower()


def is_ubuntu() -> bool:
    """Check if the OS is Ubuntu.

    Returns:
        (bool): True if OS is Ubuntu, False otherwise.
    """
    try:
        with open("/etc/os-release") as f:
            return "ID=ubuntu" in f.read()
    except FileNotFoundError:
        return False


def is_debian(codenames: list[str] | None | str = None) -> list[bool] | bool:
    """Check if the OS is Debian.

    Args:
        codenames (list[str] | None | str): Specific Debian codename to check for (e.g., 'buster', 'bullseye'). If None,
            only checks for Debian.

    Returns:
        (list[bool] | bool): List of booleans indicating if OS matches each Debian codename, or a single boolean if no
            codenames provided.
    """
    try:
        with open("/etc/os-release") as f:
            content = f.read()
            if codenames is None:
                return "ID=debian" in content
            if isinstance(codenames, str):
                codenames = [codenames]
            return [
                f"VERSION_CODENAME={codename}" in content if codename else "ID=debian" in content
                for codename in codenames
            ]
    except FileNotFoundError:
        return [False] * len(codenames) if codenames else False


def is_colab():
    """Check if the current script is running inside a Google Colab notebook.

    Returns:
        (bool): True if running inside a Colab notebook, False otherwise.
    """
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ


def is_kaggle():
    """Check if the current script is running inside a Kaggle kernel.

    Returns:
        (bool): True if running inside a Kaggle kernel, False otherwise.
    """
    return os.environ.get("PWD") == "/kaggle/working" and os.environ.get("KAGGLE_URL_BASE") == "https://www.kaggle.com"


def is_jupyter():
    """Check if the current script is running inside a Jupyter Notebook.

    Returns:
        (bool): True if running inside a Jupyter Notebook, False otherwise.

    Notes:
        - Only works on Colab and Kaggle, other environments like Jupyterlab and Paperspace are not reliably detectable.
        - "get_ipython" in globals() method suffers false positives when IPython package installed manually.
    """
    return IS_COLAB or IS_KAGGLE


def is_runpod():
    """Check if the current script is running inside a RunPod container.

    Returns:
        (bool): True if running in RunPod, False otherwise.
    """
    return "RUNPOD_POD_ID" in os.environ


def is_docker() -> bool:
    """Determine if the script is running inside a Docker container.

    Returns:
        (bool): True if the script is running inside a Docker container, False otherwise.
    """
    try:
        return os.path.exists("/.dockerenv")
    except Exception:
        return False


def is_raspberrypi() -> bool:
    """Determine if the Python environment is running on a Raspberry Pi.

    Returns:
        (bool): True if running on a Raspberry Pi, False otherwise.
    """
    return "rpi" in DEVICE_MODEL


@lru_cache(maxsize=3)
def is_jetson(jetpack=None) -> bool:
    """Determine if the Python environment is running on an NVIDIA Jetson device.

    Args:
        jetpack (int | None): If specified, check for specific JetPack version (4, 5, 6).

    Returns:
        (bool): True if running on an NVIDIA Jetson device, False otherwise.
    """
    jetson = "tegra" in DEVICE_MODEL
    if jetson and jetpack:
        try:
            content = open("/etc/nv_tegra_release").read()
            version_map = {4: "R32", 5: "R35", 6: "R36", 7: "R38"}  # JetPack to L4T major version mapping
            return jetpack in version_map and version_map[jetpack] in content
        except Exception:
            return False
    return jetson


def is_dgx() -> bool:
    """Check if the current script is running inside a DGX (NVIDIA Data Center GPU), DGX-Ready or DGX Spark system.

    Returns:
        (bool): True if running in a DGX or DGX-Ready or DGX Spark system, False otherwise.
    """
    try:
        with open("/etc/dgx-release") as f:
            return "DGX" in f.read()
    except FileNotFoundError:
        return False


def is_online() -> bool:
    """Fast online check using DNS (v4/v6) resolution (Cloudflare + Google).

    Returns:
        (bool): True if connection is successful, False otherwise.
    """
    if str(os.getenv("YOLO_OFFLINE", "")).lower() == "true":
        return False

    for host in ("one.one.one.one", "dns.google"):
        try:
            socket.getaddrinfo(host, 0, socket.AF_UNSPEC, 0, 0, socket.AI_ADDRCONFIG)
            return True
        except OSError:
            continue
    return False


def is_pip_package(filepath: str = __name__) -> bool:
    """Determine if the file at the given filepath is part of a pip package.

    Args:
        filepath (str): The filepath to check.

    Returns:
        (bool): True if the file is part of a pip package, False otherwise.
    """
    import importlib.util

    # Get the spec for the module
    spec = importlib.util.find_spec(filepath)

    # Return whether the spec is not None and the origin is not None (indicating it is a package)
    return spec is not None and spec.origin is not None


def is_dir_writeable(dir_path: str | Path) -> bool:
    """Check if a directory is writable.

    Args:
        dir_path (str | Path): The path to the directory.

    Returns:
        (bool): True if the directory is writable, False otherwise.
    """
    return os.access(str(dir_path), os.W_OK)


def is_pytest_running():
    """Determine whether pytest is currently running or not.

    Returns:
        (bool): True if pytest is running, False otherwise.
    """
    return ("PYTEST_CURRENT_TEST" in os.environ) or ("pytest" in sys.modules) or ("pytest" in Path(ARGV[0]).stem)


def is_github_action_running() -> bool:
    """Determine if the current environment is a GitHub Actions runner.

    Returns:
        (bool): True if the current environment is a GitHub Actions runner, False otherwise.
    """
    return "GITHUB_ACTIONS" in os.environ and "GITHUB_WORKFLOW" in os.environ and "RUNNER_OS" in os.environ


def get_default_args(func):
    """Return a dictionary of default arguments for a function.

    Args:
        func (callable): The function to inspect.

    Returns:
        (dict): A dictionary where each key is a parameter name, and each value is the default value of that parameter.
    """
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_ubuntu_version():
    """Retrieve the Ubuntu version if the OS is Ubuntu.

    Returns:
        (str): Ubuntu version or None if not an Ubuntu OS.
    """
    if is_ubuntu():
        try:
            with open("/etc/os-release") as f:
                return re.search(r'VERSION_ID="(\d+\.\d+)"', f.read())[1]
        except (FileNotFoundError, AttributeError):
            return None


def get_user_config_dir(sub_dir="Ultralytics"):
    """Return a writable config dir, preferring YOLO_CONFIG_DIR and being OS-aware.

    Args:
        sub_dir (str): The name of the subdirectory to create.

    Returns:
        (Path): The path to the user config directory.
    """
    if env_dir := os.getenv("YOLO_CONFIG_DIR"):
        p = Path(env_dir).expanduser() / sub_dir
    elif LINUX:
        p = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / sub_dir
    elif WINDOWS:
        p = Path.home() / "AppData" / "Roaming" / sub_dir
    elif MACOS:
        p = Path.home() / "Library" / "Application Support" / sub_dir
    else:
        raise ValueError(f"Unsupported operating system: {platform.system()}")

    if p.exists():  # already created → trust it
        return p
    if is_dir_writeable(p.parent):  # create if possible
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Fallbacks for Docker, GCP/AWS functions where only /tmp is writable
    for alt in [Path("/tmp") / sub_dir, Path.cwd() / sub_dir]:
        if alt.exists():
            return alt
        if is_dir_writeable(alt.parent):
            alt.mkdir(parents=True, exist_ok=True)
            LOGGER.warning(
                f"user config directory '{p}' is not writable, using '{alt}'. Set YOLO_CONFIG_DIR to override."
            )
            return alt

    # Last fallback → CWD
    p = Path.cwd() / sub_dir
    p.mkdir(parents=True, exist_ok=True)
    return p


# Define constants (required below)
DEVICE_MODEL = read_device_model()  # is_jetson() and is_raspberrypi() depend on this constant
ONLINE = is_online()
IS_COLAB = is_colab()
IS_KAGGLE = is_kaggle()
IS_DOCKER = is_docker()
IS_JETSON = is_jetson()
IS_JUPYTER = is_jupyter()
IS_PIP_PACKAGE = is_pip_package()
IS_RASPBERRYPI = is_raspberrypi()
IS_DEBIAN, IS_DEBIAN_BOOKWORM, IS_DEBIAN_TRIXIE = is_debian([None, "bookworm", "trixie"])
IS_UBUNTU = is_ubuntu()
GIT = GitRepo()
USER_CONFIG_DIR = get_user_config_dir()  # Ultralytics settings dir
SETTINGS_FILE = USER_CONFIG_DIR / "settings.json"


def colorstr(*input):
    r"""Color a string based on the provided color and style arguments using ANSI escape codes.

    This function can be called in two ways:
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')

    In the second form, 'blue' and 'bold' will be applied by default.

    Args:
        *input (str | Path): A sequence of strings where the first n-1 strings are color and style arguments, and the
            last string is the one to be colored.

    Returns:
        (str): The input string wrapped with ANSI escape codes for the specified color and style.

    Examples:
        >>> colorstr("blue", "bold", "hello world")
        "\033[34m\033[1mhello world\033[0m"

    Notes:
        Supported Colors and Styles:
        - Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        - Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        - Misc: 'end', 'bold', 'underline'

    References:
        https://en.wikipedia.org/wiki/ANSI_escape_code
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def remove_colorstr(input_string):
    """Remove ANSI escape codes from a string, effectively un-coloring it.

    Args:
        input_string (str): The string to remove color and style from.

    Returns:
        (str): A new string with all ANSI escape codes removed.

    Examples:
        >>> remove_colorstr(colorstr("blue", "bold", "hello world"))
        "hello world"
    """
    ansi_escape = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
    return ansi_escape.sub("", input_string)


class TryExcept(contextlib.ContextDecorator):
    """Ultralytics TryExcept class for handling exceptions gracefully.

    This class can be used as a decorator or context manager to catch exceptions and optionally print warning messages.
    It allows code to continue execution even when exceptions occur, which is useful for non-critical operations.

    Attributes:
        msg (str): Optional message to display when an exception occurs.
        verbose (bool): Whether to print the exception message.

    Examples:
        As a decorator:
        >>> @TryExcept(msg="Error occurred in func", verbose=True)
        ... def func():
        ...     # Function logic here
        ...     pass

        As a context manager:
        >>> with TryExcept(msg="Error occurred in block", verbose=True):
        ...     # Code block here
        ...     pass
    """

    def __init__(self, msg="", verbose=True):
        """Initialize TryExcept class with optional message and verbosity settings."""
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        """Execute when entering TryExcept context, initialize instance."""
        pass

    def __exit__(self, exc_type, value, traceback):
        """Define behavior when exiting a 'with' block, print error message if necessary."""
        if self.verbose and value:
            LOGGER.warning(f"{self.msg}{': ' if self.msg else ''}{value}")
        return True


class Retry(contextlib.ContextDecorator):
    """Retry class for function execution with exponential backoff.

    This decorator can be used to retry a function on exceptions, up to a specified number of times with an
    exponentially increasing delay between retries. It's useful for handling transient failures in network operations or
    other unreliable processes.

    Attributes:
        times (int): Maximum number of retry attempts.
        delay (int): Initial delay between retries in seconds.

    Examples:
        Example usage as a decorator:
        >>> @Retry(times=3, delay=2)
        ... def test_func():
        ...     # Replace with function logic that may raise exceptions
        ...     return True
    """

    def __init__(self, times=3, delay=2):
        """Initialize Retry class with specified number of retries and delay."""
        self.times = times
        self.delay = delay
        self._attempts = 0

    def __call__(self, func):
        """Decorator implementation for Retry with exponential backoff."""

        def wrapped_func(*args, **kwargs):
            """Apply retries to the decorated function or method."""
            self._attempts = 0
            while self._attempts < self.times:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self._attempts += 1
                    LOGGER.warning(f"Retry {self._attempts}/{self.times} failed: {e}")
                    if self._attempts >= self.times:
                        raise e
                    time.sleep(self.delay * (2**self._attempts))  # exponential backoff delay

        return wrapped_func


def threaded(func):
    """Multi-thread a target function by default and return the thread or function result.

    This decorator provides flexible execution of the target function, either in a separate thread or synchronously. By
    default, the function runs in a thread, but this can be controlled via the 'threaded=False' keyword argument which
    is removed from kwargs before calling the function.

    Args:
        func (callable): The function to be potentially executed in a separate thread.

    Returns:
        (callable): A wrapper function that either returns a daemon thread or the direct function result.

    Examples:
        >>> @threaded
        ... def process_data(data):
        ...     return data
        >>>
        >>> thread = process_data(my_data)  # Runs in background thread
        >>> result = process_data(my_data, threaded=False)  # Runs synchronously, returns function result
    """

    def wrapper(*args, **kwargs):
        """Multi-thread a given function based on 'threaded' kwarg and return the thread or function result."""
        if kwargs.pop("threaded", True):  # run in thread
            thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
            thread.start()
            return thread
        else:
            return func(*args, **kwargs)

    return wrapper


def set_sentry():
    """Initialize the Sentry SDK for error tracking and reporting.

    Only used if sentry_sdk package is installed and sync=True in settings. Run 'yolo settings' to see and update
    settings.

    Conditions required to send errors (ALL conditions must be met or no errors will be reported):
        - sentry_sdk package is installed
        - sync=True in YOLO settings
        - pytest is not running
        - running in a pip package installation
        - running in a non-git directory
        - running with rank -1 or 0
        - online environment
        - CLI used to run package (checked with 'yolo' as the name of the main CLI command)
    """
    if (
        not SETTINGS["sync"]
        or RANK not in {-1, 0}
        or Path(ARGV[0]).name != "yolo"
        or TESTS_RUNNING
        or not ONLINE
        or not IS_PIP_PACKAGE
        or GIT.is_repo
    ):
        return
    # If sentry_sdk package is not installed then return and do not use Sentry
    try:
        import sentry_sdk
    except ImportError:
        return

    def before_send(event, hint):
        """Modify the event before sending it to Sentry based on specific exception types and messages.

        Args:
            event (dict): The event dictionary containing information about the error.
            hint (dict): A dictionary containing additional information about the error.

        Returns:
            (dict | None): The modified event or None if the event should not be sent to Sentry.
        """
        if "exc_info" in hint:
            exc_type, exc_value, _ = hint["exc_info"]
            if exc_type in {KeyboardInterrupt, FileNotFoundError} or "out of memory" in str(exc_value):
                return None  # do not send event

        event["tags"] = {
            "sys_argv": ARGV[0],
            "sys_argv_name": Path(ARGV[0]).name,
            "install": "git" if GIT.is_repo else "pip" if IS_PIP_PACKAGE else "other",
            "os": ENVIRONMENT,
        }
        return event

    sentry_sdk.init(
        dsn="https://888e5a0778212e1d0314c37d4b9aae5d@o4504521589325824.ingest.us.sentry.io/4504521592406016",
        debug=False,
        auto_enabling_integrations=False,
        traces_sample_rate=1.0,
        release=__version__,
        environment="runpod" if is_runpod() else "production",
        before_send=before_send,
        ignore_errors=[KeyboardInterrupt, FileNotFoundError],
    )
    sentry_sdk.set_user({"id": SETTINGS["uuid"]})  # SHA-256 anonymized UUID hash


class JSONDict(dict):
    """A dictionary-like class that provides JSON persistence for its contents.

    This class extends the built-in dictionary to automatically save its contents to a JSON file whenever they are
    modified. It ensures thread-safe operations using a lock and handles JSON serialization of Path objects.

    Attributes:
        file_path (Path): The path to the JSON file used for persistence.
        lock (threading.Lock): A lock object to ensure thread-safe operations.

    Methods:
        _load: Load the data from the JSON file into the dictionary.
        _save: Save the current state of the dictionary to the JSON file.
        __setitem__: Store a key-value pair and persist it to disk.
        __delitem__: Remove an item and update the persistent storage.
        update: Update the dictionary and persist changes.
        clear: Clear all entries and update the persistent storage.

    Examples:
        >>> json_dict = JSONDict("data.json")
        >>> json_dict["key"] = "value"
        >>> print(json_dict["key"])
        value
        >>> del json_dict["key"]
        >>> json_dict.update({"new_key": "new_value"})
        >>> json_dict.clear()
    """

    def __init__(self, file_path: str | Path = "data.json"):
        """Initialize a JSONDict object with a specified file path for JSON persistence."""
        super().__init__()
        self.file_path = Path(file_path)
        self.lock = Lock()
        self._load()

    def _load(self):
        """Load the data from the JSON file into the dictionary."""
        try:
            if self.file_path.exists():
                with open(self.file_path) as f:
                    # Use the base dict update to avoid persisting during reads
                    super().update(json.load(f))
        except json.JSONDecodeError:
            LOGGER.warning(f"Error decoding JSON from {self.file_path}. Starting with an empty dictionary.")
        except Exception as e:
            LOGGER.error(f"Error reading from {self.file_path}: {e}")

    def _save(self):
        """Save the current state of the dictionary to the JSON file."""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(dict(self), f, indent=2, default=self._json_default)
        except Exception as e:
            LOGGER.error(f"Error writing to {self.file_path}: {e}")

    @staticmethod
    def _json_default(obj):
        """Handle JSON serialization of Path objects."""
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def __setitem__(self, key, value):
        """Store a key-value pair and persist to disk."""
        with self.lock:
            super().__setitem__(key, value)
            self._save()

    def __delitem__(self, key):
        """Remove an item and update the persistent storage."""
        with self.lock:
            super().__delitem__(key)
            self._save()

    def __str__(self):
        """Return a pretty-printed JSON string representation of the dictionary."""
        contents = json.dumps(dict(self), indent=2, ensure_ascii=False, default=self._json_default)
        return f'JSONDict("{self.file_path}"):\n{contents}'

    def update(self, *args, **kwargs):
        """Update the dictionary and persist changes."""
        with self.lock:
            super().update(*args, **kwargs)
            self._save()

    def clear(self):
        """Clear all entries and update the persistent storage."""
        with self.lock:
            super().clear()
            self._save()


class SettingsManager(JSONDict):
    """SettingsManager class for managing and persisting Ultralytics settings.

    This class extends JSONDict to provide JSON persistence for settings, ensuring thread-safe operations and default
    values. It validates settings on initialization and provides methods to update or reset settings. The settings
    include directories for datasets, weights, and runs, as well as various integration flags.

    Attributes:
        file (Path): The path to the JSON file used for persistence.
        version (str): The version of the settings schema.
        defaults (dict): A dictionary containing default settings.
        help_msg (str): A help message for users on how to view and update settings.

    Methods:
        _validate_settings: Validate the current settings and reset if necessary.
        update: Update settings, validating keys and types.
        reset: Reset the settings to default and save them.

    Examples:
        Initialize and update settings:
        >>> settings = SettingsManager()
        >>> settings.update(runs_dir="/new/runs/dir")
        >>> print(settings["runs_dir"])
        /new/runs/dir
    """

    def __init__(self, file=SETTINGS_FILE, version="0.0.6"):
        """Initialize the SettingsManager with default settings and load user settings."""
        import hashlib
        import uuid

        from ultralytics.utils.torch_utils import torch_distributed_zero_first

        root = GIT.root or Path()
        datasets_root = (root.parent if GIT.root and is_dir_writeable(root.parent) else root).resolve()

        self.file = Path(file)
        self.version = version
        self.defaults = {
            "settings_version": version,  # Settings schema version
            "datasets_dir": str(datasets_root / "datasets"),  # Datasets directory
            "weights_dir": str(root / "weights"),  # Model weights directory
            "runs_dir": str(root / "runs"),  # Experiment runs directory
            "uuid": hashlib.sha256(str(uuid.getnode()).encode()).hexdigest(),  # SHA-256 anonymized UUID hash
            "sync": True,  # Enable synchronization
            "api_key": "",  # Ultralytics API Key
            "openai_api_key": "",  # OpenAI API Key
            "clearml": True,  # ClearML integration
            "comet": True,  # Comet integration
            "dvc": True,  # DVC integration
            "hub": True,  # Ultralytics HUB integration
            "mlflow": True,  # MLflow integration
            "neptune": True,  # Neptune integration
            "raytune": True,  # Ray Tune integration
            "tensorboard": False,  # TensorBoard logging
            "wandb": False,  # Weights & Biases logging
            "vscode_msg": True,  # VSCode message
            "openvino_msg": True,  # OpenVINO export on Intel CPU message
        }

        self.help_msg = (
            f"\nView Ultralytics Settings with 'yolo settings' or at '{self.file}'"
            "\nUpdate Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. "
            "For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings."
        )

        with torch_distributed_zero_first(LOCAL_RANK):
            super().__init__(self.file)

            if not self.file.exists() or not self:  # Check if file doesn't exist or is empty
                LOGGER.info(f"Creating new Ultralytics Settings v{version} file ✅ {self.help_msg}")
                self.reset()

            self._validate_settings()

    def _validate_settings(self):
        """Validate the current settings and reset if necessary."""
        correct_keys = frozenset(self.keys()) == frozenset(self.defaults.keys())
        correct_types = all(isinstance(self.get(k), type(v)) for k, v in self.defaults.items())
        correct_version = self.get("settings_version", "") == self.version

        if not (correct_keys and correct_types and correct_version):
            LOGGER.warning(
                "Ultralytics settings reset to default values. This may be due to a possible problem "
                f"with your settings or a recent ultralytics package update. {self.help_msg}"
            )
            self.reset()

        if self.get("datasets_dir") == self.get("runs_dir"):
            LOGGER.warning(
                f"Ultralytics setting 'datasets_dir: {self.get('datasets_dir')}' "
                f"must be different than 'runs_dir: {self.get('runs_dir')}'. "
                f"Please change one to avoid possible issues during training. {self.help_msg}"
            )

    def __setitem__(self, key, value):
        """Update one key: value pair."""
        self.update({key: value})

    def update(self, *args, **kwargs):
        """Update settings, validating keys and types."""
        for arg in args:
            if isinstance(arg, dict):
                kwargs.update(arg)
        for k, v in kwargs.items():
            if k not in self.defaults:
                raise KeyError(f"No Ultralytics setting '{k}'. {self.help_msg}")
            t = type(self.defaults[k])
            if not isinstance(v, t):
                raise TypeError(
                    f"Ultralytics setting '{k}' must be '{t.__name__}' type, not '{type(v).__name__}'. {self.help_msg}"
                )
        super().update(*args, **kwargs)

    def reset(self):
        """Reset the settings to default and save them."""
        self.clear()
        self.update(self.defaults)


def deprecation_warn(arg, new_arg=None):
    """Issue a deprecation warning when a deprecated argument is used, suggesting an updated argument."""
    msg = f"'{arg}' is deprecated and will be removed in the future."
    if new_arg is not None:
        msg += f" Use '{new_arg}' instead."
    LOGGER.warning(msg)


def clean_url(url):
    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    url = Path(url).as_posix().replace(":/", "://")  # Pathlib turns :// -> :/, as_posix() for Windows
    return unquote(url).split("?", 1)[0]  # '%2F' to '/', split https://url.com/file.txt?auth


def url2file(url):
    """Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt."""
    return Path(clean_url(url)).name


def vscode_msg(ext="ultralytics.ultralytics-snippets") -> str:
    """Display a message to install Ultralytics-Snippets for VS Code if not already installed."""
    path = (USER_CONFIG_DIR.parents[2] if WINDOWS else USER_CONFIG_DIR.parents[1]) / ".vscode/extensions"
    obs_file = path / ".obsolete"  # file tracks uninstalled extensions, while source directory remains
    installed = any(path.glob(f"{ext}*")) and ext not in (obs_file.read_text("utf-8") if obs_file.exists() else "")
    url = "https://docs.ultralytics.com/integrations/vscode"
    return "" if installed else f"{colorstr('VS Code:')} view Ultralytics VS Code Extension ⚡ at {url}"


# Run below code on utils init ------------------------------------------------------------------------------------

# Check first-install steps
PREFIX = colorstr("Ultralytics: ")
SETTINGS = SettingsManager()  # initialize settings
PERSISTENT_CACHE = JSONDict(USER_CONFIG_DIR / "persistent_cache.json")  # initialize persistent cache
DATASETS_DIR = Path(SETTINGS["datasets_dir"])  # global datasets directory
WEIGHTS_DIR = Path(SETTINGS["weights_dir"])  # global weights directory
RUNS_DIR = Path(SETTINGS["runs_dir"])  # global runs directory
ENVIRONMENT = (
    "Colab"
    if IS_COLAB
    else "Kaggle"
    if IS_KAGGLE
    else "Jupyter"
    if IS_JUPYTER
    else "Docker"
    if IS_DOCKER
    else platform.system()
)
TESTS_RUNNING = is_pytest_running() or is_github_action_running()
set_sentry()

# Apply monkey patches
torch.save = torch_save
if WINDOWS:
    # Apply cv2 patches for non-ASCII and non-UTF characters in image paths
    cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow
