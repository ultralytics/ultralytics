# Ultralytics YOLO 🚀, GPL-3.0 license
import contextlib
import re
import shutil
import sys
from difflib import get_close_matches
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Union

from ultralytics import __version__
from ultralytics.yolo.utils import (DEFAULT_CFG, DEFAULT_CFG_DICT, DEFAULT_CFG_PATH, LOGGER, PREFIX, ROOT,
                                    USER_CONFIG_DIR, IterableSimpleNamespace, colorstr, emojis, yaml_load, yaml_print)
from ultralytics.yolo.utils.checks import check_yolo

CLI_HELP_MSG = \
    """
    YOLOv8 'yolo' CLI commands use the following syntax:

        yolo TASK MODE ARGS

        Where   TASK (optional) is one of [detect, segment, classify]
                MODE (required) is one of [train, val, predict, export]
                ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.
                    See all ARGS at https://docs.ultralytics.com/cfg or with 'yolo cfg'

    1. Train a detection model for 10 epochs with an initial learning_rate of 0.01
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01

    2. Predict a YouTube video using a pretrained segmentation model at image size 320:
        yolo segment predict model=yolov8n-seg.pt source='https://youtu.be/Zgi9g1ksQHc' imgsz=320

    3. Val a pretrained detection model at batch-size 1 and image size 640:
        yolo detect val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640

    4. Export a YOLOv8n classification model to ONNX format at image size 224 by 128 (no TASK required)
        yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128

    5. Run special commands:
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg

    Docs: https://docs.ultralytics.com/cli
    Community: https://community.ultralytics.com
    GitHub: https://github.com/ultralytics/ultralytics
    """


def cfg2dict(cfg):
    """
    Convert a configuration object to a dictionary.

    This function converts a configuration object to a dictionary, whether it is a file path, a string, or a SimpleNamespace object.

    Inputs:
        cfg (str) or (Path) or (SimpleNamespace): Configuration object to be converted to a dictionary.

    Returns:
        cfg (dict): Configuration object in dictionary format.
    """
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)  # load dict
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # convert to dict
    return cfg


def get_cfg(cfg: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_CFG, overrides: Dict = None):
    """
    Load and merge configuration data from a file or dictionary.

    Args:
        cfg (str) or (Path) or (Dict) or (SimpleNamespace): Configuration data.
        overrides (str) or (Dict), optional: Overrides in the form of a file name or a dictionary. Default is None.

    Returns:
        (SimpleNamespace): Training arguments namespace.
    """
    cfg = cfg2dict(cfg)

    # Merge overrides
    if overrides:
        overrides = cfg2dict(overrides)
        check_cfg_mismatch(cfg, overrides)
        cfg = {**cfg, **overrides}  # merge cfg and overrides dicts (prefer overrides)

    # Type checks
    for k in 'project', 'name':
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])

    # Return instance
    return IterableSimpleNamespace(**cfg)


def check_cfg_mismatch(base: Dict, custom: Dict):
    """
    This function checks for any mismatched keys between a custom configuration list and a base configuration list.
    If any mismatched keys are found, the function prints out similar keys from the base list and exits the program.

    Inputs:
        - custom (Dict): a dictionary of custom configuration options
        - base (Dict): a dictionary of base configuration options
    """
    base, custom = (set(x.keys()) for x in (base, custom))
    mismatched = [x for x in custom if x not in base]
    if mismatched:
        for x in mismatched:
            matches = get_close_matches(x, base, 3, 0.6)
            match_str = f"Similar arguments are {matches}." if matches else 'There are no similar arguments.'
            LOGGER.warning(f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}")
        LOGGER.warning(CLI_HELP_MSG)
        sys.exit()


def merge_equals_args(args: List[str]) -> List[str]:
    """
    Merges arguments around isolated '=' args in a list of strings.
    The function considers cases where the first argument ends with '=' or the second starts with '=',
    as well as when the middle one is an equals sign.

    Args:
        args (List[str]): A list of strings where each element is an argument.

    Returns:
        List[str]: A list of strings where the arguments around isolated '=' are merged.
    """
    new_args = []
    for i, arg in enumerate(args):
        if arg == '=' and 0 < i < len(args) - 1:  # merge ['arg', '=', 'val']
            new_args[-1] += f"={args[i + 1]}"
            del args[i + 1]
        elif arg.endswith('=') and i < len(args) - 1 and '=' not in args[i + 1]:  # merge ['arg=', 'val']
            new_args.append(f"{arg}{args[i + 1]}")
            del args[i + 1]
        elif arg.startswith('=') and i > 0:  # merge ['arg', '=val']
            new_args[-1] += arg
        else:
            new_args.append(arg)
    return new_args


def argument_error(arg):
    return SyntaxError(f"'{arg}' is not a valid YOLO argument.\n{CLI_HELP_MSG}")


def entrypoint(debug=False):
    """
    This function is the ultralytics package entrypoint, it's responsible for parsing the command line arguments passed
    to the package.

    This function allows for:
    - passing mandatory YOLO args as a list of strings
    - specifying the task to be performed, either 'detect', 'segment' or 'classify'
    - specifying the mode, either 'train', 'val', 'test', or 'predict'
    - running special modes like 'checks'
    - passing overrides to the package's configuration

    It uses the package's default cfg and initializes it using the passed overrides.
    Then it calls the CLI function with the composed cfg
    """
    args = ['train', 'model=yolov8n.pt', 'data=coco128.yaml', 'imgsz=32', 'epochs=1'] if debug else sys.argv[1:]
    if not args:  # no arguments passed
        LOGGER.info(CLI_HELP_MSG)
        return

    tasks = 'detect', 'segment', 'classify'
    modes = 'train', 'val', 'predict', 'export'
    special = {
        'help': lambda: LOGGER.info(CLI_HELP_MSG),
        'checks': check_yolo,
        'version': lambda: LOGGER.info(__version__),
        'settings': lambda: yaml_print(USER_CONFIG_DIR / 'settings.yaml'),
        'cfg': lambda: yaml_print(DEFAULT_CFG_PATH),
        'copy-cfg': copy_default_cfg}

    overrides = {}  # basic overrides, i.e. imgsz=320
    for a in merge_equals_args(args):  # merge spaces around '=' sign
        if '=' in a:
            try:
                re.sub(r' *= *', '=', a)  # remove spaces around equals sign
                k, v = a.split('=', 1)  # split on first '=' sign
                if k == 'cfg':  # custom.yaml passed
                    LOGGER.info(f"{PREFIX}Overriding {DEFAULT_CFG_PATH} with {v}")
                    overrides = {k: val for k, val in yaml_load(v).items() if k != 'cfg'}
                else:
                    if v.lower() == 'none':
                        v = None
                    elif v.lower() == 'true':
                        v = True
                    elif v.lower() == 'false':
                        v = False
                    else:
                        with contextlib.suppress(Exception):
                            v = eval(v)
                    overrides[k] = v
            except (NameError, SyntaxError, ValueError) as e:
                raise argument_error(a) from e

        elif a in tasks:
            overrides['task'] = a
        elif a in modes:
            overrides['mode'] = a
        elif a in special:
            special[a]()
            return
        elif a in DEFAULT_CFG_DICT and isinstance(DEFAULT_CFG_DICT[a], bool):
            overrides[a] = True  # auto-True for default bool args, i.e. 'yolo show' sets show=True
        elif a in DEFAULT_CFG_DICT:
            raise SyntaxError(f"'{colorstr('red', 'bold', a)}' is a valid YOLO argument but is missing an '=' sign "
                              f"to set its value, i.e. try '{a}={DEFAULT_CFG_DICT[a]}'\n{CLI_HELP_MSG}")
        else:
            raise argument_error(a)

    # Defaults
    task2model = dict(detect='yolov8n.pt', segment='yolov8n-seg.pt', classify='yolov8n-cls.pt')
    task2data = dict(detect='coco128.yaml', segment='coco128-seg.yaml', classify='mnist160')

    # Mode
    mode = overrides.get('mode', None)
    if mode is None:
        mode = DEFAULT_CFG.mode or 'predict'
        LOGGER.warning(f"WARNING ⚠️ 'mode=' is missing. Valid modes are {modes}. Using default 'mode={mode}'.")
    elif mode not in modes:
        if mode != 'checks':
            raise ValueError(emojis(f"ERROR ❌ Invalid 'mode={mode}'. Valid modes are {modes}."))
        LOGGER.warning("WARNING ⚠️ 'yolo mode=checks' is deprecated. Use 'yolo checks' instead.")
        check_yolo()
        return

    # Model
    model = overrides.pop('model', DEFAULT_CFG.model)
    task = overrides.pop('task', None)
    if model is None:
        model = task2model.get(task, 'yolov8n.pt')
        LOGGER.warning(f"WARNING ⚠️ 'model=' is missing. Using default 'model={model}'.")
    from ultralytics.yolo.engine.model import YOLO
    overrides['model'] = model
    model = YOLO(model)

    # Task
    if task and task != model.task:
        LOGGER.warning(f"WARNING ⚠️ 'task={task}' conflicts with {model.task} model {overrides['model']}. "
                       f"Inheriting 'task={model.task}' from {overrides['model']} and ignoring 'task={task}'.")
    task = model.task
    overrides['task'] = task
    if mode == 'predict' and 'source' not in overrides:
        overrides['source'] = DEFAULT_CFG.source or ROOT / "assets" if (ROOT / "assets").exists() \
            else "https://ultralytics.com/images/bus.jpg"
        LOGGER.warning(f"WARNING ⚠️ 'source=' is missing. Using default 'source={overrides['source']}'.")
    elif mode in ('train', 'val'):
        if 'data' not in overrides:
            overrides['data'] = task2data.get(task, DEFAULT_CFG.data)
            LOGGER.warning(f"WARNING ⚠️ 'data=' is missing. Using {model.task} default 'data={overrides['data']}'.")
    elif mode == 'export':
        if 'format' not in overrides:
            overrides['format'] = DEFAULT_CFG.format or 'torchscript'
            LOGGER.warning(f"WARNING ⚠️ 'format=' is missing. Using default 'format={overrides['format']}'.")

    # Run command in python
    cfg = get_cfg(overrides=overrides)
    getattr(model, mode)(**vars(cfg))


# Special modes --------------------------------------------------------------------------------------------------------
def copy_default_cfg():
    new_file = Path.cwd() / DEFAULT_CFG_PATH.name.replace('.yaml', '_copy.yaml')
    shutil.copy2(DEFAULT_CFG_PATH, new_file)
    LOGGER.info(f"{PREFIX}{DEFAULT_CFG_PATH} copied to {new_file}\n"
                f"Example YOLO command with this new custom cfg:\n    yolo cfg='{new_file}' imgsz=320 batch=8")


if __name__ == '__main__':
    entrypoint(debug=True)
