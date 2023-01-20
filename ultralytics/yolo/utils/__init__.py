# Ultralytics YOLO 🚀, GPL-3.0 license

import contextlib
import inspect
import logging.config
import os
import platform
import sys
import tempfile
import threading
import types
import uuid
from pathlib import Path
from typing import Union

import cv2
import git
import numpy as np
import pandas as pd
import torch
import yaml

# Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLO
DEFAULT_CFG_PATH = ROOT / "yolo/configs/default.yaml"
RANK = int(os.getenv('RANK', -1))
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
AUTOINSTALL = str(os.getenv('YOLO_AUTOINSTALL', True)).lower() == 'true'  # global auto-install mode
FONT = 'Arial.ttf'  # https://ultralytics.com/assets/Arial.ttf
VERBOSE = str(os.getenv('YOLO_VERBOSE', True)).lower() == 'true'  # global verbose mode
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format
LOGGING_NAME = 'ultralytics'
HELP_MSG = \
    """
    Usage examples for running YOLOv8:

    1. Install the ultralytics package:

        pip install ultralytics

    2. Use the Python SDK:

        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.yaml")  # build a new model from scratch
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Use the model
        results = model.train(data="coco128.yaml", epochs=3)  # train the model
        results = model.val()  # evaluate model performance on the validation set
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
        success = model.export(format="onnx")  # export the model to ONNX format

    3. Use the command line interface (CLI):

        yolo task=detect    mode=train    model=yolov8n.yaml      args...
                  classify       predict        yolov8n-cls.yaml  args...
                  segment        val            yolov8n-seg.yaml  args...
                                 export         yolov8n.pt        format=onnx  args...

    Docs: https://docs.ultralytics.com
    Community: https://community.ultralytics.com
    GitHub: https://github.com/ultralytics/ultralytics
    """

# Settings
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)  # NumExpr max threads
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # for deterministic training

# Default config dictionary
with open(DEFAULT_CFG_PATH, errors='ignore') as f:
    DEFAULT_CFG_DICT = yaml.safe_load(f)
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
DEFAULT_CFG = types.SimpleNamespace(**DEFAULT_CFG_DICT)


def is_colab():
    """
    Check if the current script is running inside a Google Colab notebook.

    Returns:
        bool: True if running inside a Colab notebook, False otherwise.
    """
    # Check if the google.colab module is present in sys.modules
    return 'google.colab' in sys.modules


def is_kaggle():
    """
    Check if the current script is running inside a Kaggle kernel.

    Returns:
        bool: True if running inside a Kaggle kernel, False otherwise.
    """
    return os.environ.get('PWD') == '/kaggle/working' and os.environ.get('KAGGLE_URL_BASE') == 'https://www.kaggle.com'


def is_jupyter_notebook():
    """
    Check if the current script is running inside a Jupyter Notebook.
    Verified on Colab, Jupyterlab, Kaggle, Paperspace.

    Returns:
        bool: True if running inside a Jupyter Notebook, False otherwise.
    """
    # Check if the get_ipython function exists
    # (it does not exist when running as a standalone script)
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def is_docker() -> bool:
    """
    Determine if the script is running inside a Docker container.

    Returns:
        bool: True if the script is running inside a Docker container, False otherwise.
    """
    file = Path('/proc/self/cgroup')
    if file.exists():
        with open(file) as f:
            return 'docker' in f.read()
    else:
        return False


def is_git_directory() -> bool:
    """
    Check if the current working directory is inside a git repository.

    Returns:
        bool: True if the current working directory is inside a git repository, False otherwise.
    """
    try:
        git.Repo(search_parent_directories=True)
        # subprocess.run(["git", "rev-parse", "--git-dir"], capture_output=True, check=True)  # CLI alternative
        return True
    except git.exc.InvalidGitRepositoryError:  # subprocess.CalledProcessError:
        return False


def is_pip_package(filepath: str = __name__) -> bool:
    """
    Determines if the file at the given filepath is part of a pip package.

    Args:
        filepath (str): The filepath to check.

    Returns:
        bool: True if the file is part of a pip package, False otherwise.
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
        dir_path (str) or (Path): The path to the directory.

    Returns:
        bool: True if the directory is writeable, False otherwise.
    """
    try:
        with tempfile.TemporaryFile(dir=dir_path):
            pass
        return True
    except OSError:
        return False


def is_pytest_running():
    """
    Returns a boolean indicating if pytest is currently running or not
    :return: True if pytest is running, False otherwise
    """
    try:
        import sys
        return "pytest" in sys.modules
    except ImportError:
        return False


def get_git_root_dir():
    """
    Determines whether the current file is part of a git repository and if so, returns the repository root directory.
    If the current file is not part of a git repository, returns None.
    """
    try:
        # output = subprocess.run(["git", "rev-parse", "--git-dir"], capture_output=True, check=True)
        # return Path(output.stdout.strip().decode('utf-8')).parent.resolve()  # CLI alternative
        return Path(git.Repo(search_parent_directories=True).working_tree_dir)
    except git.exc.InvalidGitRepositoryError:  # (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_default_args(func):
    # Get func() default arguments
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_user_config_dir(sub_dir='Ultralytics'):
    """
    Get the user config directory.

    Args:
        sub_dir (str): The name of the subdirectory to create.

    Returns:
        Path: The path to the user config directory.
    """
    # Get the operating system name
    os_name = platform.system()

    # Return the appropriate config directory for each operating system
    if os_name == 'Windows':
        path = Path.home() / 'AppData' / 'Roaming' / sub_dir
    elif os_name == 'Darwin':  # macOS
        path = Path.home() / 'Library' / 'Application Support' / sub_dir
    elif os_name == 'Linux':
        path = Path.home() / '.config' / sub_dir
    else:
        raise ValueError(f'Unsupported operating system: {os_name}')

    # GCP and AWS lambda fix, only /tmp is writeable
    if not is_dir_writeable(str(path.parent)):
        path = Path('/tmp') / sub_dir

    # Create the subdirectory if it does not exist
    path.mkdir(parents=True, exist_ok=True)

    return path


USER_CONFIG_DIR = get_user_config_dir()  # Ultralytics settings dir


def emojis(string=''):
    # Return platform-dependent emoji-safe version of string
    return string.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else string


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
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
        "underline": "\033[4m",}
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            name: {
                "format": "%(message)s"}},
        "handlers": {
            name: {
                "class": "logging.StreamHandler",
                "formatter": name,
                "level": level,}},
        "loggers": {
            name: {
                "level": level,
                "handlers": [name],
                "propagate": False,}}})


class TryExcept(contextlib.ContextDecorator):
    # YOLOv8 TryExcept class. Usage: @TryExcept() decorator or 'with TryExcept():' context manager
    def __init__(self, msg=''):
        self.msg = msg

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True


def threaded(func):
    # Multi-threads a target function and returns thread. Usage: @threaded decorator
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper


def yaml_save(file='data.yaml', data=None):
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict, optional): Data to save in YAML format. Default is None.

    Returns:
        None: Data is saved to the specified file.
    """
    file = Path(file)
    if not file.parent.exists():
        # Create parent directories if they don't exist
        file.parent.mkdir(parents=True, exist_ok=True)

    with open(file, 'w') as f:
        # Dump data to file in YAML format, converting Path objects to strings
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)


def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        dict: YAML data and file name.
    """
    with open(file, errors='ignore') as f:
        # Add YAML filename to dict and return
        return {**yaml.safe_load(f), 'yaml_file': str(file)} if append_filename else yaml.safe_load(f)


def set_sentry(dsn=None):
    """
    Initialize the Sentry SDK for error tracking and reporting if pytest is not currently running.
    """
    if dsn and not is_pytest_running():
        import sentry_sdk  # noqa

        import ultralytics
        sentry_sdk.init(
            dsn=dsn,
            debug=False,
            traces_sample_rate=1.0,
            release=ultralytics.__version__,
            send_default_pii=True,
            environment='production',  # 'dev' or 'production'
            ignore_errors=[KeyboardInterrupt])


def get_settings(file=USER_CONFIG_DIR / 'settings.yaml', version='0.0.1'):
    """
    Loads a global Ultralytics settings YAML file or creates one with default values if it does not exist.

    Args:
        file (Path): Path to the Ultralytics settings YAML file. Defaults to 'settings.yaml' in the USER_CONFIG_DIR.
        version (str): Settings version. If min settings version not met, new default settings will be saved.

    Returns:
        dict: Dictionary of settings key-value pairs.
    """
    from ultralytics.yolo.utils.checks import check_version
    from ultralytics.yolo.utils.torch_utils import torch_distributed_zero_first

    is_git = is_git_directory()  # True if ultralytics installed via git
    root = get_git_root_dir() if is_git else Path()
    datasets_root = (root.parent if (is_git and is_dir_writeable(root.parent)) else root).resolve()
    defaults = {
        'datasets_dir': str(datasets_root / 'datasets'),  # default datasets directory.
        'weights_dir': str(root / 'weights'),  # default weights directory.
        'runs_dir': str(root / 'runs'),  # default runs directory.
        'sync': True,  # sync analytics to help with YOLO development
        'uuid': uuid.getnode(),  # device UUID to align analytics
        'settings_version': version}  # Ultralytics settings version

    with torch_distributed_zero_first(RANK):
        if not file.exists():
            yaml_save(file, defaults)

        settings = yaml_load(file)

        # Check that settings keys and types match defaults
        correct = settings.keys() == defaults.keys() \
                  and all(type(a) == type(b) for a, b in zip(settings.values(), defaults.values())) \
                  and check_version(settings['settings_version'], version)
        if not correct:
            LOGGER.warning('WARNING ⚠️ Ultralytics settings reset to defaults. '
                           '\nThis is normal and may be due to a recent ultralytics package update, '
                           'but may have overwritten previous settings. '
                           f"\nYou may view and update settings directly in '{file}'")
            settings = defaults  # merge **defaults with **settings (prefer **settings)
            yaml_save(file, settings)  # save updated defaults

        return settings


def set_settings(kwargs, file=USER_CONFIG_DIR / 'settings.yaml'):
    """
    Function that runs on a first-time ultralytics package installation to set up global settings and create necessary
    directories.
    """
    SETTINGS.update(kwargs)
    yaml_save(file, SETTINGS)


def print_settings():
    """
    Function that prints Ultralytics settings
    """
    import json
    s = f'\n{PREFIX}Settings:\n'
    s += json.dumps(SETTINGS, indent=2)
    s += f"\n\nUpdate settings at {USER_CONFIG_DIR / 'settings.yaml'}"
    LOGGER.info(s)


# Run below code on utils init -----------------------------------------------------------------------------------------

# Set logger
set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)
if platform.system() == 'Windows':
    for fn in LOGGER.info, LOGGER.warning:
        setattr(LOGGER, fn.__name__, lambda x: fn(emojis(x)))  # emoji safe logging

# Check first-install steps
PREFIX = colorstr("Ultralytics: ")
SETTINGS = get_settings()
DATASETS_DIR = Path(SETTINGS['datasets_dir'])  # global datasets directory
set_sentry()
