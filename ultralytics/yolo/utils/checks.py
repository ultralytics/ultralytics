# Ultralytics YOLO 🚀, GPL-3.0 license
import contextlib
import glob
import inspect
import math
import os
import platform
import re
import shutil
import subprocess
import urllib
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pkg_resources as pkg
import psutil
import requests
import torch
from matplotlib import font_manager

from ultralytics.yolo.utils import (AUTOINSTALL, LOGGER, ONLINE, ROOT, USER_CONFIG_DIR, TryExcept, colorstr, downloads,
                                    emojis, is_colab, is_docker, is_kaggle, is_online, is_pip_package)


def is_ascii(s) -> bool:
    """
    Check if a string is composed of only ASCII characters.

    Args:
        s (str): String to be checked.

    Returns:
        bool: True if the string is composed only of ASCII characters, False otherwise.
    """
    # Convert list, tuple, None, etc. to string
    s = str(s)

    # Check if the string is composed of only ASCII characters
    return all(ord(c) < 128 for c in s)


def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    """
    Verify image size is a multiple of the given stride in each dimension. If the image size is not a multiple of the
    stride, update it to the nearest multiple of the stride that is greater than or equal to the given floor value.

    Args:
        imgsz (int or List[int]): Image size.
        stride (int): Stride value.
        min_dim (int): Minimum number of dimensions.
        floor (int): Minimum allowed value for image size.

    Returns:
        List[int]: Updated image size.
    """
    # Convert stride to integer if it is a tensor
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    # Convert image size to list if it is an integer
    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    else:
        raise TypeError(f"'imgsz={imgsz}' is of invalid type {type(imgsz).__name__}. "
                        f"Valid imgsz types are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640,640]'")

    # Apply max_dim
    if len(imgsz) > max_dim:
        msg = "'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list " \
              "or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'"
        if max_dim != 1:
            raise ValueError(f'imgsz={imgsz} is not a valid image size. {msg}')
        LOGGER.warning(f"WARNING ⚠️ updating to 'imgsz={max(imgsz)}'. {msg}")
        imgsz = [max(imgsz)]
    # Make image size a multiple of the stride
    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]

    # Print warning message if image size was updated
    if sz != imgsz:
        LOGGER.warning(f'WARNING ⚠️ imgsz={imgsz} must be multiple of max stride {stride}, updating to {sz}')

    # Add missing dimensions if necessary
    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz

    return sz


def check_version(current: str = '0.0.0',
                  minimum: str = '0.0.0',
                  name: str = 'version ',
                  pinned: bool = False,
                  hard: bool = False,
                  verbose: bool = False) -> bool:
    """
    Check current version against the required minimum version.

    Args:
        current (str): Current version.
        minimum (str): Required minimum version.
        name (str): Name to be used in warning message.
        pinned (bool): If True, versions must match exactly. If False, minimum version must be satisfied.
        hard (bool): If True, raise an AssertionError if the minimum version is not met.
        verbose (bool): If True, print warning message if minimum version is not met.

    Returns:
        bool: True if minimum version is met, False otherwise.
    """
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    warning_message = f'WARNING ⚠️ {name}{minimum} is required by YOLOv8, but {name}{current} is currently installed'
    if hard:
        assert result, emojis(warning_message)  # assert min requirements met
    if verbose and not result:
        LOGGER.warning(warning_message)
    return result


def check_latest_pypi_version(package_name='ultralytics'):
    """
    Returns the latest version of a PyPI package without downloading or installing it.

    Parameters:
        package_name (str): The name of the package to find the latest version for.

    Returns:
        str: The latest version of the package.
    """
    response = requests.get(f'https://pypi.org/pypi/{package_name}/json')
    if response.status_code == 200:
        return response.json()['info']['version']
    return None


def check_pip_update_available():
    """
    Checks if a new version of the ultralytics package is available on PyPI.

    Returns:
        bool: True if an update is available, False otherwise.
    """
    if ONLINE and is_pip_package():
        with contextlib.suppress(Exception):
            from ultralytics import __version__
            latest = check_latest_pypi_version()
            if pkg.parse_version(__version__) < pkg.parse_version(latest):  # update is available
                LOGGER.info(f'New https://pypi.org/project/ultralytics/{latest} available 😃 '
                            f"Update with 'pip install -U ultralytics'")
                return True
    return False


def check_font(font='Arial.ttf'):
    """
    Find font locally or download to user's configuration directory if it does not already exist.

    Args:
        font (str): Path or name of font.

    Returns:
        file (Path): Resolved font file path.
    """
    name = Path(font).name

    # Check USER_CONFIG_DIR
    file = USER_CONFIG_DIR / name
    if file.exists():
        return file

    # Check system fonts
    matches = [s for s in font_manager.findSystemFonts() if font in s]
    if any(matches):
        return matches[0]

    # Download to USER_CONFIG_DIR if missing
    url = f'https://ultralytics.com/assets/{name}'
    if downloads.is_url(url):
        downloads.safe_download(url=url, file=file)
        return file


def check_python(minimum: str = '3.7.0') -> bool:
    """
    Check current python version against the required minimum version.

    Args:
        minimum (str): Required minimum version of python.

    Returns:
        None
    """
    return check_version(platform.python_version(), minimum, name='Python ', hard=True)


@TryExcept()
def check_requirements(requirements=ROOT.parent / 'requirements.txt', exclude=(), install=True, cmds=''):
    # Check installed dependencies meet YOLOv5 requirements (pass *.txt file or list of packages or single package str)
    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()  # check python version
    file = None
    if isinstance(requirements, Path):  # requirements.txt file
        file = requirements.resolve()
        assert file.exists(), f'{prefix} {file} not found, check failed.'
        with file.open() as f:
            requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(f) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]

    s = ''
    n = 0
    for r in requirements:
        try:
            pkg.require(r)
        except (pkg.VersionConflict, pkg.DistributionNotFound):  # exception if requirements not met
            try:  # attempt to import (slower but more accurate)
                import importlib
                importlib.import_module(next(pkg.parse_requirements(r)).name)
            except ImportError:
                s += f'"{r}" '
                n += 1

    if s and install and AUTOINSTALL:  # check environment variable
        LOGGER.info(f"{prefix} YOLOv8 requirement{'s' * (n > 1)} {s}not found, attempting AutoUpdate...")
        try:
            assert is_online(), 'AutoUpdate skipped (offline)'
            LOGGER.info(subprocess.check_output(f'pip install {s} {cmds}', shell=True).decode())
            s = f"{prefix} {n} package{'s' * (n > 1)} updated per {file or requirements}\n" \
                f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
            LOGGER.info(s)
        except Exception as e:
            LOGGER.warning(f'{prefix} ❌ {e}')


def check_suffix(file='yolov8n.pt', suffix='.pt', msg=''):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix, )
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower().strip()  # file suffix
            if len(s):
                assert s in suffix, f'{msg}{f} acceptable suffix is {suffix}, not {s}'


def check_yolov5u_filename(file: str, verbose: bool = True):
    # Replace legacy YOLOv5 filenames with updated YOLOv5u filenames
    if ('yolov3' in file or 'yolov5' in file) and 'u' not in file:
        original_file = file
        file = re.sub(r'(.*yolov5([nsmlx]))\.pt', '\\1u.pt', file)  # i.e. yolov5n.pt -> yolov5nu.pt
        file = re.sub(r'(.*yolov5([nsmlx])6)\.pt', '\\1u.pt', file)  # i.e. yolov5n6.pt -> yolov5n6u.pt
        file = re.sub(r'(.*yolov3(|-tiny|-spp))\.pt', '\\1u.pt', file)  # i.e. yolov3-spp.pt -> yolov3-sppu.pt
        if file != original_file and verbose:
            LOGGER.info(f"PRO TIP 💡 Replace 'model={original_file}' with new 'model={file}'.\nYOLOv5 'u' models are "
                        f'trained with https://github.com/ultralytics/ultralytics and feature improved performance vs '
                        f'standard YOLOv5 models trained with https://github.com/ultralytics/yolov5.\n')
    return file


def check_file(file, suffix='', download=True, hard=True):
    # Search/download file (if necessary) and return path
    check_suffix(file, suffix)  # optional
    file = str(file).strip()  # convert to string and strip spaces
    file = check_yolov5u_filename(file)  # yolov5n -> yolov5nu
    if not file or ('://' not in file and Path(file).exists()):  # exists ('://' check required in Windows Python<3.10)
        return file
    elif download and file.lower().startswith(('https://', 'http://', 'rtsp://', 'rtmp://')):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = Path(urllib.parse.unquote(file).split('?')[0]).name  # '%2F' to '/', split https://url.com/file.txt?auth
        if Path(file).exists():
            LOGGER.info(f'Found {url} locally at {file}')  # file already exists
        else:
            downloads.safe_download(url=url, file=file, unzip=False)
        return file
    else:  # search
        files = []
        for d in 'models', 'datasets', 'tracker/cfg', 'yolo/cfg':  # search directories
            files.extend(glob.glob(str(ROOT / d / '**' / file), recursive=True))  # find file
        if not files and hard:
            raise FileNotFoundError(f"'{file}' does not exist")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
        return files[0] if len(files) else []  # return file


def check_yaml(file, suffix=('.yaml', '.yml'), hard=True):
    # Search/download YAML file (if necessary) and return path, checking suffix
    return check_file(file, suffix, hard=hard)


def check_imshow(warn=False):
    # Check if environment supports image displays
    try:
        assert not any((is_colab(), is_kaggle(), is_docker()))
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f'WARNING ⚠️ Environment does not support cv2.imshow() or PIL Image.show()\n{e}')
        return False


def check_yolo(verbose=True, device=''):
    from ultralytics.yolo.utils.torch_utils import select_device

    if is_colab():
        shutil.rmtree('sample_data', ignore_errors=True)  # remove colab /sample_data directory

    if verbose:
        # System info
        gib = 1 << 30  # bytes per GiB
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage('/')
        s = f'({os.cpu_count()} CPUs, {ram / gib:.1f} GB RAM, {(total - free) / gib:.1f}/{total / gib:.1f} GB disk)'
        with contextlib.suppress(Exception):  # clear display if ipython is installed
            from IPython import display
            display.clear_output()
    else:
        s = ''

    select_device(device=device, newline=False)
    LOGGER.info(f'Setup complete ✅ {s}')


def git_describe(path=ROOT):  # path must be a directory
    # Return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    try:
        assert (Path(path) / '.git').is_dir()
        return subprocess.check_output(f'git -C {path} describe --tags --long --always', shell=True).decode()[:-1]
    except AssertionError:
        return ''


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    # Print function arguments (optional args dict)
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    LOGGER.info(colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))
