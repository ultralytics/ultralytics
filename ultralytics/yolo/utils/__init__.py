# Ultralytics YOLO 🚀, GPL-3.0 license

import contextlib
import inspect
import logging.config
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import yaml

# Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLO
DEFAULT_CFG_PATH = ROOT / "yolo/cfg/default.yaml"
RANK = int(os.getenv('RANK', -1))
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
AUTOINSTALL = str(os.getenv('YOLO_AUTOINSTALL', True)).lower() == 'true'  # global auto-install mode
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

        YOLOv8 'yolo' CLI commands use the following syntax:

            yolo TASK MODE ARGS

            Where   TASK (optional) is one of [detect, segment, classify]
                    MODE (required) is one of [train, val, predict, export]
                    ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.
                        See all ARGS at https://docs.ultralytics.com/cfg or with 'yolo cfg'

        - Train a detection model for 10 epochs with an initial learning_rate of 0.01
            yolo detect train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01

        - Predict a YouTube video using a pretrained segmentation model at image size 320:
            yolo segment predict model=yolov8n-seg.pt source='https://youtu.be/Zgi9g1ksQHc' imgsz=320

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
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)  # NumExpr max threads
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # for deterministic training


class IterableSimpleNamespace(SimpleNamespace):
    """
    Iterable SimpleNamespace class to allow SimpleNamespace to be used with dict() and in for loops
    """

    def __iter__(self):
        return iter(vars(self).items())

    def __str__(self):
        return '\n'.join(f"{k}={v}" for k, v in vars(self).items())

    def get(self, key, default=None):
        return getattr(self, key, default)


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
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v
                        for k, v in data.items()},
                       f,
                       sort_keys=False,
                       allow_unicode=True)


def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        dict: YAML data and file name.
    """
    with open(file, errors='ignore', encoding='utf-8') as f:
        # Add YAML filename to dict and return
        s = f.read()  # string
        if not s.isprintable():  # remove special characters
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)


def yaml_print(yaml_file: Union[str, Path, dict]) -> None:
    """
    Pretty prints a yaml file or a yaml-formatted dictionary.

    Args:
        yaml_file: The file path of the yaml file or a yaml-formatted dictionary.

    Returns:
        None
    """
    yaml_dict = yaml_load(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file
    dump = yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True)
    LOGGER.info(f"Printing '{colorstr('bold', 'black', yaml_file)}'\n\n{dump}")


# Default configuration
DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
for k, v in DEFAULT_CFG_DICT.items():
    if isinstance(v, str) and v.lower() == 'none':
        DEFAULT_CFG_DICT[k] = None
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)


def is_colab():
    """
    Check if the current script is running inside a Google Colab notebook.

    Returns:
        bool: True if running inside a Colab notebook, False otherwise.
    """
    return 'COLAB_RELEASE_TAG' in os.environ or 'COLAB_BACKEND_VERSION' in os.environ


def is_kaggle():
    """
    Check if the current script is running inside a Kaggle kernel.

    Returns:
        bool: True if running inside a Kaggle kernel, False otherwise.
    """
    return os.environ.get('PWD') == '/kaggle/working' and os.environ.get('KAGGLE_URL_BASE') == 'https://www.kaggle.com'


def is_jupyter():
    """
    Check if the current script is running inside a Jupyter Notebook.
    Verified on Colab, Jupyterlab, Kaggle, Paperspace.

    Returns:
        bool: True if running inside a Jupyter Notebook, False otherwise.
    """
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
    Determines whether pytest is currently running or not.

    Returns:
        (bool): True if pytest is running, False otherwise.
    """
    try:
        import sys
        return "pytest" in sys.modules
    except ImportError:
        return False


def is_github_actions_ci() -> bool:
    """
    Determine if the current environment is a GitHub Actions CI Python runner.

    Returns:
        (bool): True if the current environment is a GitHub Actions CI Python runner, False otherwise.
    """
    return 'GITHUB_ACTIONS' in os.environ and 'RUNNER_OS' in os.environ and 'RUNNER_TOOL_CACHE' in os.environ


def is_git_dir():
    """
    Determines whether the current file is part of a git repository.
    If the current file is not part of a git repository, returns None.

    Returns:
        (bool): True if current file is part of a git repository.
    """
    return get_git_dir() is not None


def get_git_dir():
    """
    Determines whether the current file is part of a git repository and if so, returns the repository root directory.
    If the current file is not part of a git repository, returns None.

    Returns:
        (Path) or (None): Git root directory if found or None if not found.
    """
    for d in Path(__file__).parents:
        if (d / '.git').is_dir():
            return d
    return None  # no .git dir found


def get_git_origin_url():
    """
    Retrieves the origin URL of a git repository.

    Returns:
        (str) or (None): The origin URL of the git repository.
    """
    if is_git_dir():
        with contextlib.suppress(subprocess.CalledProcessError):
            origin = subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
            return origin.decode().strip()
    return None  # if not git dir or on error


def get_git_branch():
    """
    Returns the current git branch name. If not in a git repository, returns None.

    Returns:
        (str) or (None): The current git branch name.
    """
    if is_git_dir():
        with contextlib.suppress(subprocess.CalledProcessError):
            origin = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            return origin.decode().strip()
    return None  # if not git dir or on error


def get_latest_pypi_version(package_name='ultralytics'):
    """
    Returns the latest version of a PyPI package without downloading or installing it.

    Parameters:
        package_name (str): The name of the package to find the latest version for.

    Returns:
        str: The latest version of the package.
    """
    response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
    if response.status_code == 200:
        return response.json()["info"]["version"]
    return None


def get_default_args(func):
    """Returns a dictionary of default arguments for a function.

    Args:
        func (callable): The function to inspect.

    Returns:
        dict: A dictionary where each key is a parameter name, and each value is the default value of that parameter.
    """
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
        "underline": "\033[4m"}
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
                "level": level}},
        "loggers": {
            name: {
                "level": level,
                "handlers": [name],
                "propagate": False}}})


class TryExcept(contextlib.ContextDecorator):
    # YOLOv8 TryExcept class. Usage: @TryExcept() decorator or 'with TryExcept():' context manager
    def __init__(self, msg='', verbose=True):
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if self.verbose and value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True


def threaded(func):
    # Multi-threads a target function and returns thread. Usage: @threaded decorator
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper


def set_sentry():
    """
    Initialize the Sentry SDK for error tracking and reporting if pytest is not currently running.
    """

    def before_send(event, hint):
        if 'exc_info' in hint:
            exc_type, exc_value, tb = hint['exc_info']
            if exc_type in (KeyboardInterrupt, FileNotFoundError) \
                    or 'out of memory' in str(exc_value) \
                    or not sys.argv[0].endswith('yolo'):
                return None  # do not send event

        env = 'Colab' if is_colab() else 'Kaggle' if is_kaggle() else 'Jupyter' if is_jupyter() else \
            'Docker' if is_docker() else platform.system()
        event['tags'] = {
            "sys_argv": sys.argv[0],
            "sys_argv_name": Path(sys.argv[0]).name,
            "install": 'git' if is_git_dir() else 'pip' if is_pip_package() else 'other',
            "os": env}
        return event

    if SETTINGS['sync'] and \
            RANK in {-1, 0} and \
            sys.argv[0].endswith('yolo') and \
            not is_pytest_running() and \
            not is_github_actions_ci() and \
            ((is_pip_package() and not is_git_dir()) or
             (get_git_origin_url() == "https://github.com/ultralytics/ultralytics.git" and get_git_branch() == "main")):

        import hashlib
        import sentry_sdk  # noqa
        from ultralytics import __version__

        sentry_sdk.init(
            dsn="https://f805855f03bb4363bc1e16cb7d87b654@o4504521589325824.ingest.sentry.io/4504521592406016",
            debug=False,
            traces_sample_rate=1.0,
            release=__version__,
            environment='production',  # 'dev' or 'production'
            before_send=before_send,
            ignore_errors=[KeyboardInterrupt, FileNotFoundError])
        sentry_sdk.set_user({"id": SETTINGS['uuid']})

        # Disable all sentry logging
        for logger in "sentry_sdk", "sentry_sdk.errors":
            logging.getLogger(logger).setLevel(logging.CRITICAL)


def get_settings(file=USER_CONFIG_DIR / 'settings.yaml', version='0.0.2'):
    """
    Loads a global Ultralytics settings YAML file or creates one with default values if it does not exist.

    Args:
        file (Path): Path to the Ultralytics settings YAML file. Defaults to 'settings.yaml' in the USER_CONFIG_DIR.
        version (str): Settings version. If min settings version not met, new default settings will be saved.

    Returns:
        dict: Dictionary of settings key-value pairs.
    """
    import hashlib
    from ultralytics.yolo.utils.checks import check_version
    from ultralytics.yolo.utils.torch_utils import torch_distributed_zero_first

    git_dir = get_git_dir()
    root = git_dir or Path()
    datasets_root = (root.parent if git_dir and is_dir_writeable(root.parent) else root).resolve()
    defaults = {
        'datasets_dir': str(datasets_root / 'datasets'),  # default datasets directory.
        'weights_dir': str(root / 'weights'),  # default weights directory.
        'runs_dir': str(root / 'runs'),  # default runs directory.
        'sync': True,  # sync analytics to help with YOLO development
        'uuid': hashlib.sha256(str(uuid.getnode()).encode()).hexdigest(),  # anonymized uuid hash
        'settings_version': version}  # Ultralytics settings version

    with torch_distributed_zero_first(RANK):
        if not file.exists():
            yaml_save(file, defaults)
        settings = yaml_load(file)

        # Check that settings keys and types match defaults
        correct = \
            settings.keys() == defaults.keys() \
            and all(type(a) == type(b) for a, b in zip(settings.values(), defaults.values())) \
            and check_version(settings['settings_version'], version)
        if not correct:
            LOGGER.warning('WARNING ⚠️ Ultralytics settings reset to defaults. This is normal and may be due to a '
                           'recent ultralytics package update, but may have overwritten previous settings. '
                           f"\nView and update settings with 'yolo settings' or at '{file}'")
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


# Run below code on yolo/utils init ------------------------------------------------------------------------------------

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
