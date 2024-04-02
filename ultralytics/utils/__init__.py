# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import inspect
import logging.config
import os
import platform
import re
import subprocess
import sys
import threading
import time
import urllib
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm as tqdm_original

from ultralytics import __version__

# PyTorch Multi-GPU DDP Constants
RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html

# Other Constants
ARGV = sys.argv or ["", ""]  # sometimes sys.argv = []
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO
ASSETS = ROOT / "assets"  # default images
DEFAULT_CFG_PATH = ROOT / "cfg/default.yaml"
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
AUTOINSTALL = str(os.getenv("YOLO_AUTOINSTALL", True)).lower() == "true"  # global auto-install mode
VERBOSE = str(os.getenv("YOLO_VERBOSE", True)).lower() == "true"  # global verbose mode
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}" if VERBOSE else None  # tqdm bar format
LOGGING_NAME = "ultralytics"
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans
ARM64 = platform.machine() in {"arm64", "aarch64"}  # ARM64 booleans
HELP_MSG = """
    Usage examples for running YOLOv8:

    1. Install the ultralytics package:

        pip install ultralytics

    2. Use the Python SDK:

        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n.yaml')  # build a new model from scratch
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Use the model
        results = model.train(data="coco128.yaml", epochs=3)  # train the model
        results = model.val()  # evaluate model performance on the validation set
        results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
        success = model.export(format='onnx')  # export the model to ONNX format

    3. Use the command line interface (CLI):

        YOLOv8 'yolo' CLI commands use the following syntax:

            yolo TASK MODE ARGS

            Where   TASK (optional) is one of [detect, segment, classify]
                    MODE (required) is one of [train, val, predict, export]
                    ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.
                        See all ARGS at https://docs.ultralytics.com/usage/cfg or with 'yolo cfg'

        - Train a detection model for 10 epochs with an initial learning_rate of 0.01
            yolo detect train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01

        - Predict a YouTube video using a pretrained segmentation model at image size 320:
            yolo segment predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320

        - Val a pretrained detection model at batch-size 1 and image size 640:
            yolo detect val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640

        - Export a YOLOv8n classification model to ONNX format at image size 224 by 128 (no TASK required)
            yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128

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

# Settings
torch.set_printoptions(linewidth=320, precision=4, profile="default")
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})  # format short g, %precision=5
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_THREADS)  # NumExpr max threads
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # for deterministic training
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress verbose TF compiler warnings in Colab


class TQDM(tqdm_original):
    """
    Custom Ultralytics tqdm class with different default arguments.

    Args:
        *args (list): Positional arguments passed to original tqdm.
        **kwargs (any): Keyword arguments, with custom defaults applied.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize custom Ultralytics tqdm class with different default arguments.

        Note these can still be overridden when calling TQDM.
        """
        kwargs["disable"] = not VERBOSE or kwargs.get("disable", False)  # logical 'and' with default value if passed
        kwargs.setdefault("bar_format", TQDM_BAR_FORMAT)  # override default value if passed
        super().__init__(*args, **kwargs)


class SimpleClass:
    """Ultralytics SimpleClass is a base class providing helpful string representation, error reporting, and attribute
    access methods for easier debugging and usage.
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
                    s = f"{a}: {repr(v)}"
                attr.append(s)
        return f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n" + "\n".join(attr)

    def __repr__(self):
        """Return a machine-readable string representation of the object."""
        return self.__str__()

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


class IterableSimpleNamespace(SimpleNamespace):
    """Ultralytics IterableSimpleNamespace is an extension class of SimpleNamespace that adds iterable functionality and
    enables usage with dict() and for loops.
    """

    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
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
    """
    Decorator to temporarily set rc parameters and the backend for a plotting function.

    Example:
        decorator: @plt_settings({"font.size": 12})
        context manager: with plt_settings({"font.size": 12}):

    Args:
        rcparams (dict): Dictionary of rc parameters to set.
        backend (str, optional): Name of the backend to use. Defaults to 'Agg'.

    Returns:
        (Callable): Decorated function with temporarily set rc parameters and backend. This decorator can be
            applied to any function that needs to have specific matplotlib rc parameters and backend for its execution.
    """

    if rcparams is None:
        rcparams = {"font.size": 11}

    def decorator(func):
        """Decorator to apply temporary rc parameters and backend to a function."""

        def wrapper(*args, **kwargs):
            """Sets rc parameters and backend, calls the original function, and restores the settings."""
            original_backend = plt.get_backend()
            if backend.lower() != original_backend.lower():
                plt.close("all")  # auto-close()ing of figures upon backend switching is deprecated since 3.8
                plt.switch_backend(backend)

            with plt.rc_context(rcparams):
                result = func(*args, **kwargs)

            if backend != original_backend:
                plt.close("all")
                plt.switch_backend(original_backend)
            return result

        return wrapper

    return decorator


def set_logging(name="LOGGING_NAME", verbose=True):
    """Sets up logging for the given name with UTF-8 encoding support, ensuring compatibility across different
    environments.
    """
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR  # rank in world for Multi-GPU trainings

    # Configure the console (stdout) encoding to UTF-8, with checks for compatibility
    formatter = logging.Formatter("%(message)s")  # Default formatter
    if WINDOWS and hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8":

        class CustomFormatter(logging.Formatter):
            def format(self, record):
                """Sets up logging with UTF-8 encoding and configurable verbosity."""
                return emojis(super().format(record))

        try:
            # Attempt to reconfigure stdout to use UTF-8 encoding if possible
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            # For environments where reconfigure is not available, wrap stdout in a TextIOWrapper
            elif hasattr(sys.stdout, "buffer"):
                import io

                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
            else:
                formatter = CustomFormatter("%(message)s")
        except Exception as e:
            print(f"Creating custom formatter for non UTF-8 environments due to {e}")
            formatter = CustomFormatter("%(message)s")

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
for logger in "sentry_sdk", "urllib3.connectionpool":
    logging.getLogger(logger).setLevel(logging.CRITICAL + 1)


def emojis(string=""):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode("ascii", "ignore") if WINDOWS else string


class ThreadingLocked:
    """
    A decorator class for ensuring thread-safe execution of a function or method. This class can be used as a decorator
    to make sure that if the decorated function is called from multiple threads, only one thread at a time will be able
    to execute the function.

    Attributes:
        lock (threading.Lock): A lock object used to manage access to the decorated function.

    Example:
        ```python
        from ultralytics.utils import ThreadingLocked

        @ThreadingLocked()
        def my_function():
            # Your code here
            pass
        ```
    """

    def __init__(self):
        """Initializes the decorator class for thread-safe execution of a function or method."""
        self.lock = threading.Lock()

    def __call__(self, f):
        """Run thread-safe execution of function or method."""
        from functools import wraps

        @wraps(f)
        def decorated(*args, **kwargs):
            """Applies thread-safety to the decorated function or method."""
            with self.lock:
                return f(*args, **kwargs)

        return decorated


def yaml_save(file="data.yaml", data=None, header=""):
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict): Data to save in YAML format.
        header (str, optional): YAML header to add.

    Returns:
        (None): Data is saved to the specified file.
    """
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        # Create parent directories if they don't exist
        file.parent.mkdir(parents=True, exist_ok=True)

    # Convert Path objects to strings
    valid_types = int, float, str, bool, list, tuple, dict, type(None)
    for k, v in data.items():
        if not isinstance(v, valid_types):
            data[k] = str(v)

    # Dump data to file in YAML format
    with open(file, "w", errors="ignore", encoding="utf-8") as f:
        if header:
            f.write(header)
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def yaml_load(file="data.yaml", append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["yaml_file"] = str(file)
        return data


def yaml_print(yaml_file: Union[str, Path, dict]) -> None:
    """
    Pretty prints a YAML file or a YAML-formatted dictionary.

    Args:
        yaml_file: The file path of the YAML file or a YAML-formatted dictionary.

    Returns:
        (None)
    """
    yaml_dict = yaml_load(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file
    dump = yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True)
    LOGGER.info(f"Printing '{colorstr('bold', 'black', yaml_file)}'\n\n{dump}")


# Default configuration
DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
for k, v in DEFAULT_CFG_DICT.items():
    if isinstance(v, str) and v.lower() == "none":
        DEFAULT_CFG_DICT[k] = None
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)


def is_ubuntu() -> bool:
    """
    Check if the OS is Ubuntu.

    Returns:
        (bool): True if OS is Ubuntu, False otherwise.
    """
    with contextlib.suppress(FileNotFoundError):
        with open("/etc/os-release") as f:
            return "ID=ubuntu" in f.read()
    return False


def is_colab():
    """
    Check if the current script is running inside a Google Colab notebook.

    Returns:
        (bool): True if running inside a Colab notebook, False otherwise.
    """
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ


def is_kaggle():
    """
    Check if the current script is running inside a Kaggle kernel.

    Returns:
        (bool): True if running inside a Kaggle kernel, False otherwise.
    """
    return os.environ.get("PWD") == "/kaggle/working" and os.environ.get("KAGGLE_URL_BASE") == "https://www.kaggle.com"


def is_jupyter():
    """
    Check if the current script is running inside a Jupyter Notebook. Verified on Colab, Jupyterlab, Kaggle, Paperspace.

    Returns:
        (bool): True if running inside a Jupyter Notebook, False otherwise.
    """
    with contextlib.suppress(Exception):
        from IPython import get_ipython

        return get_ipython() is not None
    return False


def is_docker() -> bool:
    """
    Determine if the script is running inside a Docker container.

    Returns:
        (bool): True if the script is running inside a Docker container, False otherwise.
    """
    file = Path("/proc/self/cgroup")
    if file.exists():
        with open(file) as f:
            return "docker" in f.read()
    else:
        return False


def is_online() -> bool:
    """
    Check internet connectivity by attempting to connect to a known online host.

    Returns:
        (bool): True if connection is successful, False otherwise.
    """
    import socket

    for host in "1.1.1.1", "8.8.8.8", "223.5.5.5":  # Cloudflare, Google, AliDNS:
        try:
            test_connection = socket.create_connection(address=(host, 53), timeout=2)
        except (socket.timeout, socket.gaierror, OSError):
            continue
        else:
            # If the connection was successful, close it to avoid a ResourceWarning
            test_connection.close()
            return True
    return False


ONLINE = is_online()


def is_pip_package(filepath: str = __name__) -> bool:
    """
    Determines if the file at the given filepath is part of a pip package.

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


def is_dir_writeable(dir_path: Union[str, Path]) -> bool:
    """
    Check if a directory is writeable.

    Args:
        dir_path (str | Path): The path to the directory.

    Returns:
        (bool): True if the directory is writeable, False otherwise.
    """
    return os.access(str(dir_path), os.W_OK)


def is_pytest_running():
    """
    Determines whether pytest is currently running or not.

    Returns:
        (bool): True if pytest is running, False otherwise.
    """
    return ("PYTEST_CURRENT_TEST" in os.environ) or ("pytest" in sys.modules) or ("pytest" in Path(ARGV[0]).stem)


def is_github_action_running() -> bool:
    """
    Determine if the current environment is a GitHub Actions runner.

    Returns:
        (bool): True if the current environment is a GitHub Actions runner, False otherwise.
    """
    return "GITHUB_ACTIONS" in os.environ and "GITHUB_WORKFLOW" in os.environ and "RUNNER_OS" in os.environ


def is_git_dir():
    """
    Determines whether the current file is part of a git repository. If the current file is not part of a git
    repository, returns None.

    Returns:
        (bool): True if current file is part of a git repository.
    """
    return get_git_dir() is not None


def get_git_dir():
    """
    Determines whether the current file is part of a git repository and if so, returns the repository root directory. If
    the current file is not part of a git repository, returns None.

    Returns:
        (Path | None): Git root directory if found or None if not found.
    """
    for d in Path(__file__).parents:
        if (d / ".git").is_dir():
            return d


def get_git_origin_url():
    """
    Retrieves the origin URL of a git repository.

    Returns:
        (str | None): The origin URL of the git repository or None if not git directory.
    """
    if is_git_dir():
        with contextlib.suppress(subprocess.CalledProcessError):
            origin = subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
            return origin.decode().strip()


def get_git_branch():
    """
    Returns the current git branch name. If not in a git repository, returns None.

    Returns:
        (str | None): The current git branch name or None if not a git directory.
    """
    if is_git_dir():
        with contextlib.suppress(subprocess.CalledProcessError):
            origin = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            return origin.decode().strip()


def get_default_args(func):
    """
    Returns a dictionary of default arguments for a function.

    Args:
        func (callable): The function to inspect.

    Returns:
        (dict): A dictionary where each key is a parameter name, and each value is the default value of that parameter.
    """
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_ubuntu_version():
    """
    Retrieve the Ubuntu version if the OS is Ubuntu.

    Returns:
        (str): Ubuntu version or None if not an Ubuntu OS.
    """
    if is_ubuntu():
        with contextlib.suppress(FileNotFoundError, AttributeError):
            with open("/etc/os-release") as f:
                return re.search(r'VERSION_ID="(\d+\.\d+)"', f.read())[1]


def get_user_config_dir(sub_dir="Ultralytics"):
    """
    Return the appropriate config directory based on the environment operating system.

    Args:
        sub_dir (str): The name of the subdirectory to create.

    Returns:
        (Path): The path to the user config directory.
    """
    if WINDOWS:
        path = Path.home() / "AppData" / "Roaming" / sub_dir
    elif MACOS:  # macOS
        path = Path.home() / "Library" / "Application Support" / sub_dir
    elif LINUX:
        path = Path.home() / ".config" / sub_dir
    else:
        raise ValueError(f"Unsupported operating system: {platform.system()}")

    # GCP and AWS lambda fix, only /tmp is writeable
    if not is_dir_writeable(path.parent):
        LOGGER.warning(
            f"WARNING ⚠️ user config directory '{path}' is not writeable, defaulting to '/tmp' or CWD."
            "Alternatively you can define a YOLO_CONFIG_DIR environment variable for this path."
        )
        path = Path("/tmp") / sub_dir if is_dir_writeable("/tmp") else Path().cwd() / sub_dir

    # Create the subdirectory if it does not exist
    path.mkdir(parents=True, exist_ok=True)

    return path


USER_CONFIG_DIR = Path(os.getenv("YOLO_CONFIG_DIR") or get_user_config_dir())  # Ultralytics settings dir
SETTINGS_YAML = USER_CONFIG_DIR / "settings.yaml"


def colorstr(*input):
    """
    Colors a string based on the provided color and style arguments. Utilizes ANSI escape codes.
    See https://en.wikipedia.org/wiki/ANSI_escape_code for more details.

    This function can be called in two ways:
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')

    In the second form, 'blue' and 'bold' will be applied by default.

    Args:
        *input (str): A sequence of strings where the first n-1 strings are color and style arguments,
                      and the last string is the one to be colored.

    Supported Colors and Styles:
        Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        Misc: 'end', 'bold', 'underline'

    Returns:
        (str): The input string wrapped with ANSI escape codes for the specified color and style.

    Examples:
        >>> colorstr('blue', 'bold', 'hello world')
        >>> '\033[34m\033[1mhello world\033[0m'
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
    """
    Removes ANSI escape codes from a string, effectively un-coloring it.

    Args:
        input_string (str): The string to remove color and style from.

    Returns:
        (str): A new string with all ANSI escape codes removed.

    Examples:
        >>> remove_colorstr(colorstr('blue', 'bold', 'hello world'))
        >>> 'hello world'
    """
    ansi_escape = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
    return ansi_escape.sub("", input_string)


class TryExcept(contextlib.ContextDecorator):
    """
    Ultralytics TryExcept class. Use as @TryExcept() decorator or 'with TryExcept():' context manager.

    Examples:
        As a decorator:
        >>> @TryExcept(msg="Error occurred in func", verbose=True)
        >>> def func():
        >>>    # Function logic here
        >>>     pass

        As a context manager:
        >>> with TryExcept(msg="Error occurred in block", verbose=True):
        >>>     # Code block here
        >>>     pass
    """

    def __init__(self, msg="", verbose=True):
        """Initialize TryExcept class with optional message and verbosity settings."""
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        """Executes when entering TryExcept context, initializes instance."""
        pass

    def __exit__(self, exc_type, value, traceback):
        """Defines behavior when exiting a 'with' block, prints error message if necessary."""
        if self.verbose and value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True


class Retry(contextlib.ContextDecorator):
    """
    Retry class for function execution with exponential backoff.

    Can be used as a decorator or a context manager to retry a function or block of code on exceptions, up to a
    specified number of times with an exponentially increasing delay between retries.

    Examples:
        Example usage as a decorator:
        >>> @Retry(times=3, delay=2)
        >>> def test_func():
        >>>     # Replace with function logic that may raise exceptions
        >>>     return True

        Example usage as a context manager:
        >>> with Retry(times=3, delay=2):
        >>>     # Replace with code block that may raise exceptions
        >>>     pass
    """

    def __init__(self, times=3, delay=2):
        """Initialize Retry class with specified number of retries and delay."""
        self.times = times
        self.delay = delay
        self._attempts = 0

    def __call__(self, func):
        """Decorator implementation for Retry with exponential backoff."""

        def wrapped_func(*args, **kwargs):
            """Applies retries to the decorated function or method."""
            self._attempts = 0
            while self._attempts < self.times:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self._attempts += 1
                    print(f"Retry {self._attempts}/{self.times} failed: {e}")
                    if self._attempts >= self.times:
                        raise e
                    time.sleep(self.delay * (2**self._attempts))  # exponential backoff delay

        return wrapped_func

    def __enter__(self):
        """Enter the runtime context related to this object."""
        self._attempts = 0

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context related to this object with exponential backoff."""
        if exc_type is not None:
            self._attempts += 1
            if self._attempts < self.times:
                print(f"Retry {self._attempts}/{self.times} failed: {exc_value}")
                time.sleep(self.delay * (2**self._attempts))  # exponential backoff delay
                return True  # Suppresses the exception and retries
        return False  # Re-raises the exception if retries are exhausted


def threaded(func):
    """
    Multi-threads a target function by default and returns the thread or function result.

    Use as @threaded decorator. The function runs in a separate thread unless 'threaded=False' is passed.
    """

    def wrapper(*args, **kwargs):
        """Multi-threads a given function based on 'threaded' kwarg and returns the thread or function result."""
        if kwargs.pop("threaded", True):  # run in thread
            thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
            thread.start()
            return thread
        else:
            return func(*args, **kwargs)

    return wrapper


def set_sentry():
    """
    Initialize the Sentry SDK for error tracking and reporting. Only used if sentry_sdk package is installed and
    sync=True in settings. Run 'yolo settings' to see and update settings YAML file.

    Conditions required to send errors (ALL conditions must be met or no errors will be reported):
        - sentry_sdk package is installed
        - sync=True in YOLO settings
        - pytest is not running
        - running in a pip package installation
        - running in a non-git directory
        - running with rank -1 or 0
        - online environment
        - CLI used to run package (checked with 'yolo' as the name of the main CLI command)

    The function also configures Sentry SDK to ignore KeyboardInterrupt and FileNotFoundError
    exceptions and to exclude events with 'out of memory' in their exception message.

    Additionally, the function sets custom tags and user information for Sentry events.
    """

    def before_send(event, hint):
        """
        Modify the event before sending it to Sentry based on specific exception types and messages.

        Args:
            event (dict): The event dictionary containing information about the error.
            hint (dict): A dictionary containing additional information about the error.

        Returns:
            dict: The modified event or None if the event should not be sent to Sentry.
        """
        if "exc_info" in hint:
            exc_type, exc_value, tb = hint["exc_info"]
            if exc_type in {KeyboardInterrupt, FileNotFoundError} or "out of memory" in str(exc_value):
                return None  # do not send event

        event["tags"] = {
            "sys_argv": ARGV[0],
            "sys_argv_name": Path(ARGV[0]).name,
            "install": "git" if is_git_dir() else "pip" if is_pip_package() else "other",
            "os": ENVIRONMENT,
        }
        return event

    if (
        SETTINGS["sync"]
        and RANK in {-1, 0}
        and Path(ARGV[0]).name == "yolo"
        and not TESTS_RUNNING
        and ONLINE
        and is_pip_package()
        and not is_git_dir()
    ):
        # If sentry_sdk package is not installed then return and do not use Sentry
        try:
            import sentry_sdk  # noqa
        except ImportError:
            return

        sentry_sdk.init(
            dsn="https://5ff1556b71594bfea135ff0203a0d290@o4504521589325824.ingest.sentry.io/4504521592406016",
            debug=False,
            traces_sample_rate=1.0,
            release=__version__,
            environment="production",  # 'dev' or 'production'
            before_send=before_send,
            ignore_errors=[KeyboardInterrupt, FileNotFoundError],
        )
        sentry_sdk.set_user({"id": SETTINGS["uuid"]})  # SHA-256 anonymized UUID hash


class SettingsManager(dict):
    """
    Manages Ultralytics settings stored in a YAML file.

    Args:
        file (str | Path): Path to the Ultralytics settings YAML file. Default is USER_CONFIG_DIR / 'settings.yaml'.
        version (str): Settings version. In case of local version mismatch, new default settings will be saved.
    """

    def __init__(self, file=SETTINGS_YAML, version="0.0.4"):
        """Initialize the SettingsManager with default settings, load and validate current settings from the YAML
        file.
        """
        import copy
        import hashlib

        from ultralytics.utils.checks import check_version
        from ultralytics.utils.torch_utils import torch_distributed_zero_first

        git_dir = get_git_dir()
        root = git_dir or Path()
        datasets_root = (root.parent if git_dir and is_dir_writeable(root.parent) else root).resolve()

        self.file = Path(file)
        self.version = version
        self.defaults = {
            "settings_version": version,
            "datasets_dir": str(datasets_root / "datasets"),
            "weights_dir": str(root / "weights"),
            "runs_dir": str(root / "runs"),
            "uuid": hashlib.sha256(str(uuid.getnode()).encode()).hexdigest(),
            "sync": True,
            "api_key": "",
            "openai_api_key": "",
            "clearml": True,  # integrations
            "comet": True,
            "dvc": True,
            "hub": True,
            "mlflow": True,
            "neptune": True,
            "raytune": True,
            "tensorboard": True,
            "wandb": True,
        }

        super().__init__(copy.deepcopy(self.defaults))

        with torch_distributed_zero_first(RANK):
            if not self.file.exists():
                self.save()

            self.load()
            correct_keys = self.keys() == self.defaults.keys()
            correct_types = all(type(a) is type(b) for a, b in zip(self.values(), self.defaults.values()))
            correct_version = check_version(self["settings_version"], self.version)
            help_msg = (
                f"\nView settings with 'yolo settings' or at '{self.file}'"
                "\nUpdate settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. "
                "For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings."
            )
            if not (correct_keys and correct_types and correct_version):
                LOGGER.warning(
                    "WARNING ⚠️ Ultralytics settings reset to default values. This may be due to a possible problem "
                    f"with your settings or a recent ultralytics package update. {help_msg}"
                )
                self.reset()

            if self.get("datasets_dir") == self.get("runs_dir"):
                LOGGER.warning(
                    f"WARNING ⚠️ Ultralytics setting 'datasets_dir: {self.get('datasets_dir')}' "
                    f"must be different than 'runs_dir: {self.get('runs_dir')}'. "
                    f"Please change one to avoid possible issues during training. {help_msg}"
                )

    def load(self):
        """Loads settings from the YAML file."""
        super().update(yaml_load(self.file))

    def save(self):
        """Saves the current settings to the YAML file."""
        yaml_save(self.file, dict(self))

    def update(self, *args, **kwargs):
        """Updates a setting value in the current settings."""
        super().update(*args, **kwargs)
        self.save()

    def reset(self):
        """Resets the settings to default and saves them."""
        self.clear()
        self.update(self.defaults)
        self.save()


def deprecation_warn(arg, new_arg, version=None):
    """Issue a deprecation warning when a deprecated argument is used, suggesting an updated argument."""
    if not version:
        version = float(__version__[:3]) + 0.2  # deprecate after 2nd major release
    LOGGER.warning(
        f"WARNING ⚠️ '{arg}' is deprecated and will be removed in 'ultralytics {version}' in the future. "
        f"Please use '{new_arg}' instead."
    )


def clean_url(url):
    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    url = Path(url).as_posix().replace(":/", "://")  # Pathlib turns :// -> :/, as_posix() for Windows
    return urllib.parse.unquote(url).split("?")[0]  # '%2F' to '/', split https://url.com/file.txt?auth


def url2file(url):
    """Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt."""
    return Path(clean_url(url)).name


# Run below code on utils init ------------------------------------------------------------------------------------

# Check first-install steps
PREFIX = colorstr("Ultralytics: ")
SETTINGS = SettingsManager()  # initialize settings
DATASETS_DIR = Path(SETTINGS["datasets_dir"])  # global datasets directory
WEIGHTS_DIR = Path(SETTINGS["weights_dir"])  # global weights directory
RUNS_DIR = Path(SETTINGS["runs_dir"])  # global runs directory
ENVIRONMENT = (
    "Colab"
    if is_colab()
    else "Kaggle"
    if is_kaggle()
    else "Jupyter"
    if is_jupyter()
    else "Docker"
    if is_docker()
    else platform.system()
)
TESTS_RUNNING = is_pytest_running() or is_github_action_running()
set_sentry()

# Apply monkey patches
from .patches import imread, imshow, imwrite, torch_save

torch.save = torch_save
if WINDOWS:
    # Apply cv2 patches for non-ASCII and non-UTF characters in image paths
    cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow
