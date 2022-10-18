import contextlib
import logging
import os
import platform
from pathlib import Path

from .files import user_config_dir
from .loggers import emojis, set_logging

# Constants

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLOv5 root directory
RANK = int(os.getenv('RANK', -1))
DATASETS_DIR = ROOT.parent / 'datasets'  # YOLOv5 datasets directory
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
AUTOINSTALL = str(os.getenv('YOLOv5_AUTOINSTALL', True)).lower() == 'true'  # global auto-install mode
FONT = 'Arial.ttf'  # https://ultralytics.com/assets/Arial.ttf

CONFIG_DIR = user_config_dir()  # Ultralytics settings dir

set_logging()  # run before defining LOGGER
LOGGER = logging.getLogger("yolov5")  # define globally
if platform.system() == "Windows":
    for fn in LOGGER.info, LOGGER.warning:
        setattr(LOGGER, fn.__name__, lambda x: fn(emojis(x)))  # emoji safe logging


class TryExcept(contextlib.ContextDecorator):
    # YOLOv5 TryExcept class. Usage: @TryExcept() decorator or 'with TryExcept():' context manager
    def __init__(self, msg=''):
        self.msg = msg

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if value:
            print(emojis(f'{self.msg}{value}'))
        return True
