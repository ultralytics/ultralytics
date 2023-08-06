# Ultralytics YOLO 🚀, AGPL-3.0 license
import contextlib
import glob
import inspect
import math
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pkg_resources as pkg
import psutil
import requests
import torch
from matplotlib import font_manager

from ultralytics.utils import (AUTOINSTALL, LOGGER, ONLINE, ROOT, USER_CONFIG_DIR, ThreadingLocked, TryExcept,
                               clean_url, colorstr, downloads, emojis, is_colab, is_docker, is_jupyter, is_kaggle,
                               is_online, is_pip_package, url2file)


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
        imgsz (int | cList[int]): Image size.
        stride (int): Stride value.
        min_dim (int): Minimum number of dimensions.
        max_dim (int): Maximum number of dimensions.
        floor (int): Minimum allowed value for image size.

    Returns:
        (List[int]): Updated image size.
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
                  required: str = '0.0.0',
                  name: str = 'version ',
                  hard: bool = False,
                  verbose: bool = False) -> bool:
    """
    Check current version against the required version or range.

    Args:
        current (str): Current version.
        required (str): Required version or range (in pip-style format).
        name (str): Name to be used in warning message.
        hard (bool): If True, raise an AssertionError if the requirement is not met.
        verbose (bool): If True, print warning message if requirement is not met.

    Returns:
        (bool): True if requirement is met, False otherwise.

    Example:
        # check if current version is exactly 22.04
        check_version(current='22.04', required='==22.04')

        # check if current version is greater than or equal to 22.04
        check_version(current='22.10', required='22.04')  # assumes '>=' inequality if none passed

        # check if current version is less than or equal to 22.04
        check_version(current='22.04', required='<=22.04')

        # check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        check_version(current='21.10', required='>20.04,<22.04')
    """
    current = pkg.parse_version(current)
    constraints = re.findall(r'([<>!=]{1,2}\s*\d+\.\d+)', required) or [f'>={required}']

    result = True
    for constraint in constraints:
        op, version = re.match(r'([<>!=]{1,2})\s*(\d+\.\d+)', constraint).groups()
        version = pkg.parse_version(version)
        if op == '==' and current != version:
            result = False
        elif op == '!=' and current == version:
            result = False
        elif op == '>=' and not (current >= version):
            result = False
        elif op == '<=' and not (current <= version):
            result = False
        elif op == '>' and not (current > version):
            result = False
        elif op == '<' and not (current < version):
            result = False
    if not result:
        warning_message = f'WARNING ⚠️ {name}{required} is required, but {name}{current} is currently installed'
        if hard:
            raise ModuleNotFoundError(emojis(warning_message))  # assert version requirements met
        if verbose:
            LOGGER.warning(warning_message)
    return result


def check_latest_pypi_version(package_name='ultralytics'):
    """
    Returns the latest version of a PyPI package without downloading or installing it.

    Parameters:
        package_name (str): The name of the package to find the latest version for.

    Returns:
        (str): The latest version of the package.
    """
    with contextlib.suppress(Exception):
        requests.packages.urllib3.disable_warnings()  # Disable the InsecureRequestWarning
        response = requests.get(f'https://pypi.org/pypi/{package_name}/json', timeout=3)
        if response.status_code == 200:
            return response.json()['info']['version']
    return None


def check_pip_update_available():
    """
    Checks if a new version of the ultralytics package is available on PyPI.

    Returns:
        (bool): True if an update is available, False otherwise.
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


@ThreadingLocked()
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
    """
    Check if installed dependencies meet YOLOv8 requirements and attempt to auto-update if needed.

    Args:
        requirements (Union[Path, str, List[str]]): Path to a requirements.txt file, a single package requirement as a
            string, or a list of package requirements as strings.
        exclude (Tuple[str]): Tuple of package names to exclude from checking.
        install (bool): If True, attempt to auto-update packages that don't meet requirements.
        cmds (str): Additional commands to pass to the pip install command when auto-updating.

    Example:
        ```python
        from ultralytics.utils.checks import check_requirements

        # Check a requirements.txt file
        check_requirements('path/to/requirements.txt')

        # Check a single package
        check_requirements('ultralytics>=8.0.0')

        # Check multiple packages
        check_requirements(['numpy', 'ultralytics>=8.0.0'])
        ```
    """
    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()  # check python version
    check_torchvision()  # check torch-torchvision compatibility
    if isinstance(requirements, Path):  # requirements.txt file
        file = requirements.resolve()
        assert file.exists(), f'{prefix} {file} not found, check failed.'
        with file.open() as f:
            requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(f) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]

    s = ''  # console string
    pkgs = []
    for r in requirements:
        r_stripped = r.split('/')[-1].replace('.git', '')  # replace git+https://org/repo.git -> 'repo'
        try:
            pkg.require(r_stripped)
        except (pkg.VersionConflict, pkg.DistributionNotFound):  # exception if requirements not met
            try:  # attempt to import (slower but more accurate)
                import importlib
                importlib.import_module(next(pkg.parse_requirements(r_stripped)).name)
            except ImportError:
                s += f'"{r}" '
                pkgs.append(r)

    if s:
        if install and AUTOINSTALL:  # check environment variable
            n = len(pkgs)  # number of packages updates
            LOGGER.info(f"{prefix} Ultralytics requirement{'s' * (n > 1)} {pkgs} not found, attempting AutoUpdate...")
            try:
                t = time.time()
                assert is_online(), 'AutoUpdate skipped (offline)'
                LOGGER.info(subprocess.check_output(f'pip install --no-cache {s} {cmds}', shell=True).decode())
                dt = time.time() - t
                LOGGER.info(
                    f"{prefix} AutoUpdate success ✅ {dt:.1f}s, installed {n} package{'s' * (n > 1)}: {pkgs}\n"
                    f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n")
            except Exception as e:
                LOGGER.warning(f'{prefix} ❌ {e}')
                return False
        else:
            return False

    return True


def check_torchvision():
    """
    Checks the installed versions of PyTorch and Torchvision to ensure they're compatible.

    This function checks the installed versions of PyTorch and Torchvision, and warns if they're incompatible according
    to the provided compatibility table based on https://github.com/pytorch/vision#installation. The
    compatibility table is a dictionary where the keys are PyTorch versions and the values are lists of compatible
    Torchvision versions.
    """

    import torchvision

    # Compatibility table
    compatibility_table = {'2.0': ['0.15'], '1.13': ['0.14'], '1.12': ['0.13']}

    # Extract only the major and minor versions
    v_torch = '.'.join(torch.__version__.split('+')[0].split('.')[:2])
    v_torchvision = '.'.join(torchvision.__version__.split('+')[0].split('.')[:2])

    if v_torch in compatibility_table:
        compatible_versions = compatibility_table[v_torch]
        if all(pkg.parse_version(v_torchvision) != pkg.parse_version(v) for v in compatible_versions):
            print(f'WARNING ⚠️ torchvision=={v_torchvision} is incompatible with torch=={v_torch}.\n'
                  f"Run 'pip install torchvision=={compatible_versions[0]}' to fix torchvision or "
                  "'pip install -U torch torchvision' to update both.\n"
                  'For a full compatibility table see https://github.com/pytorch/vision#installation')


def check_suffix(file='yolov8n.pt', suffix='.pt', msg=''):
    """Check file(s) for acceptable suffix."""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix, )
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower().strip()  # file suffix
            if len(s):
                assert s in suffix, f'{msg}{f} acceptable suffix is {suffix}, not {s}'


def check_yolov5u_filename(file: str, verbose: bool = True):
    """Replace legacy YOLOv5 filenames with updated YOLOv5u filenames."""
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
    """Search/download file (if necessary) and return path."""
    check_suffix(file, suffix)  # optional
    file = str(file).strip()  # convert to string and strip spaces
    file = check_yolov5u_filename(file)  # yolov5n -> yolov5nu
    if not file or ('://' not in file and Path(file).exists()):  # exists ('://' check required in Windows Python<3.10)
        return file
    elif download and file.lower().startswith(('https://', 'http://', 'rtsp://', 'rtmp://')):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = url2file(file)  # '%2F' to '/', split https://url.com/file.txt?auth
        if Path(file).exists():
            LOGGER.info(f'Found {clean_url(url)} locally at {file}')  # file already exists
        else:
            downloads.safe_download(url=url, file=file, unzip=False)
        return file
    else:  # search
        files = glob.glob(str(ROOT / 'cfg' / '**' / file), recursive=True)  # find file
        if not files and hard:
            raise FileNotFoundError(f"'{file}' does not exist")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
        return files[0] if len(files) else []  # return file


def check_yaml(file, suffix=('.yaml', '.yml'), hard=True):
    """Search/download YAML file (if necessary) and return path, checking suffix."""
    return check_file(file, suffix, hard=hard)


def check_imshow(warn=False):
    """Check if environment supports image displays."""
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
    """Return a human-readable YOLO software and hardware summary."""
    from ultralytics.utils.torch_utils import select_device

    if is_jupyter():
        if check_requirements('wandb', install=False):
            os.system('pip uninstall -y wandb')  # uninstall wandb: unwanted account creation prompt with infinite hang
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


def check_amp(model):
    """
    This function checks the PyTorch Automatic Mixed Precision (AMP) functionality of a YOLOv8 model.
    If the checks fail, it means there are anomalies with AMP on the system that may cause NaN losses or zero-mAP
    results, so AMP will be disabled during training.

    Args:
        model (nn.Module): A YOLOv8 model instance.

    Returns:
        (bool): Returns True if the AMP functionality works correctly with YOLOv8 model, else False.

    Raises:
        AssertionError: If the AMP checks fail, indicating anomalies with the AMP functionality on the system.
    """
    device = next(model.parameters()).device  # get model device
    if device.type in ('cpu', 'mps'):
        return False  # AMP only used on CUDA devices

    def amp_allclose(m, im):
        """All close FP32 vs AMP results."""
        a = m(im, device=device, verbose=False)[0].boxes.data  # FP32 inference
        with torch.cuda.amp.autocast(True):
            b = m(im, device=device, verbose=False)[0].boxes.data  # AMP inference
        del m
        return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)  # close to 0.5 absolute tolerance

    f = ROOT / 'assets/bus.jpg'  # image to check
    im = f if f.exists() else 'https://ultralytics.com/images/bus.jpg' if ONLINE else np.ones((640, 640, 3))
    prefix = colorstr('AMP: ')
    LOGGER.info(f'{prefix}running Automatic Mixed Precision (AMP) checks with YOLOv8n...')
    warning_msg = "Setting 'amp=True'. If you experience zero-mAP or NaN losses you can disable AMP with amp=False."
    try:
        from ultralytics import YOLO
        assert amp_allclose(YOLO('yolov8n.pt'), im)
        LOGGER.info(f'{prefix}checks passed ✅')
    except ConnectionError:
        LOGGER.warning(f'{prefix}checks skipped ⚠️, offline and unable to download YOLOv8n. {warning_msg}')
    except (AttributeError, ModuleNotFoundError):
        LOGGER.warning(
            f'{prefix}checks skipped ⚠️. Unable to load YOLOv8n due to possible Ultralytics package modifications. {warning_msg}'
        )
    except AssertionError:
        LOGGER.warning(f'{prefix}checks failed ❌. Anomalies were detected with AMP on your system that may lead to '
                       f'NaN losses or zero-mAP results, so AMP will be disabled during training.')
        return False
    return True


def git_describe(path=ROOT):  # path must be a directory
    """Return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe."""
    try:
        assert (Path(path) / '.git').is_dir()
        return subprocess.check_output(f'git -C {path} describe --tags --long --always', shell=True).decode()[:-1]
    except AssertionError:
        return ''


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    """Print function arguments (optional args dict)."""

    def strip_auth(v):
        """Clean longer Ultralytics HUB URLs by stripping potential authentication information."""
        return clean_url(v) if (isinstance(v, str) and v.startswith('http') and len(v) > 100) else v

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
    LOGGER.info(colorstr(s) + ', '.join(f'{k}={strip_auth(v)}' for k, v in args.items()))
