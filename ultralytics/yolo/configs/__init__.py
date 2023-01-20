# Ultralytics YOLO 🚀, GPL-3.0 license
import argparse
import re
import shutil
import sys
from difflib import get_close_matches
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Union

from ultralytics import __version__, yolo
from ultralytics.yolo.utils import DEFAULT_CFG_PATH, LOGGER, PREFIX, checks, colorstr, print_settings, yaml_load

DIR = Path(__file__).parent

CLI_HELP_MSG = \
    """
    YOLOv8 CLI Usage examples:

    1. Install the ultralytics package:

        pip install ultralytics

    2. Train, Val, Predict and Export using 'yolo' commands:

            yolo TASK MODE ARGS

            Where   TASK (optional) is one of [detect, segment, classify]
                    MODE (required) is one of [train, val, predict, export]
                    ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.
                        For a full list of available ARGS see https://docs.ultralytics.com/config.

        Train a detection model for 10 epochs with an initial learning_rate of 0.01
            yolo detect train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01

        Predict a YouTube video using a pretrained segmentation model at image size 320:
            yolo segment predict model=yolov8n-seg.pt source=https://youtu.be/Zgi9g1ksQHc imgsz=320

        Validate a pretrained detection model at batch-size 1 and image size 640:
            yolo detect val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640

        Export a YOLOv8n classification model to ONNX format at image size 224 by 128 (no TASK required)
            yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128

    3. Run special commands:

        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-config

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


def get_config(config: Union[str, Path, Dict, SimpleNamespace], overrides: Dict = None):
    """
    Load and merge configuration data from a file or dictionary.

    Args:
        config (str) or (Path) or (Dict) or (SimpleNamespace): Configuration data.
        overrides (str) or (Dict), optional: Overrides in the form of a file name or a dictionary. Default is None.

    Returns:
        (SimpleNamespace): Training arguments namespace.
    """
    config = cfg2dict(config)

    # Merge overrides
    if overrides:
        overrides = cfg2dict(overrides)
        check_config_mismatch(config, overrides)
        config = {**config, **overrides}  # merge config and overrides dicts (prefer overrides)

    # Return instance
    return SimpleNamespace(**config)


def check_config_mismatch(base: Dict, custom: Dict):
    """
    This function checks for any mismatched keys between a custom configuration list and a base configuration list.
    If any mismatched keys are found, the function prints out similar keys from the base list and exits the program.

    Inputs:
        - custom (Dict): a dictionary of custom configuration options
        - base (Dict): a dictionary of base configuration options
    """
    base, custom = (set(x.keys()) for x in (base, custom))
    mismatched = [x for x in custom if x not in base]
    for option in mismatched:
        LOGGER.info(f"{colorstr(option)} is not a valid key. Similar keys: {get_close_matches(option, base, 3, 0.6)}")
    if mismatched:
        sys.exit()


def entrypoint(debug=True):
    """
    This function is the ultralytics package entrypoint, it's responsible for parsing the command line arguments passed
    to the package.

    This function allows for:
    - passing mandatory YOLO args as a list of strings
    - specifying the task to be performed, either 'detect', 'segment' or 'classify'
    - specifying the mode, either 'train', 'val', 'test', or 'predict'
    - running special modes like 'checks'
    - passing overrides to the package's configuration

    It uses the package's default config and initializes it using the passed overrides.
    Then it calls the CLI function with the composed config
    """
    if debug:
        args = ['train', 'predict', 'model=yolov8n.pt']  # for testing
    else:
        if len(sys.argv) == 1:  # no arguments passed
            LOGGER.info(CLI_HELP_MSG)
            return

        parser = argparse.ArgumentParser(description='YOLO parser')
        parser.add_argument('args', type=str, nargs='+', help='YOLO args')
        args = parser.parse_args().args
        args = re.sub(r'\s*=\s*', '=', ' '.join(args)).split(' ')  # remove whitespaces around = sign

    tasks = 'detect', 'segment', 'classify'
    modes = 'train', 'val', 'predict', 'export'
    special_modes = {
        'help': lambda: LOGGER.info(CLI_HELP_MSG),
        'checks': checks.check_yolo,
        'version': lambda: LOGGER.info(__version__),
        'settings': print_settings,
        'copy-config': copy_default_config}

    overrides = {}  # basic overrides, i.e. imgsz=320
    defaults = yaml_load(DEFAULT_CFG_PATH)
    for a in args:
        if '=' in a:
            if a.startswith('cfg='):  # custom.yaml passed
                custom_config = Path(a.split('=')[-1])
                LOGGER.info(f"{PREFIX}Overriding {DEFAULT_CFG_PATH} with {custom_config}")
                overrides = {k: v for k, v in yaml_load(custom_config).items() if k not in {'cfg'}}
            else:
                k, v = a.split('=')
                try:
                    if k == 'device':  # special DDP handling, i.e. device='0,1,2,3'
                        v = v.replace('[', '').replace(']', '')  # handle device=[0,1,2,3]
                        v = v.replace(" ", "").replace('')  # handle device=[0, 1, 2, 3]
                        v = v.replace('\\', '')  # handle device=\'0,1,2,3\'
                        overrides[k] = v
                    else:
                        overrides[k] = eval(v)  # convert strings to integers, floats, bools, etc.
                except (NameError, SyntaxError):
                    overrides[k] = v
        elif a in tasks:
            overrides['task'] = a
        elif a in modes:
            overrides['mode'] = a
        elif a in special_modes:
            special_modes[a]()
            return
        elif a in defaults and defaults[a] is False:
            overrides[a] = True  # auto-True for default False args, i.e. 'yolo show' sets show=True
        elif a in defaults:
            raise SyntaxError(f"'{a}' is a valid YOLO argument but is missing an '=' sign to set its value, "
                              f"i.e. try '{a}={defaults[a]}'"
                              f"\n{CLI_HELP_MSG}")
        else:
            raise SyntaxError(
                f"'{a}' is not a valid YOLO argument. For a full list of valid arguments see "
                f"https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/configs/default.yaml"
                f"\n{CLI_HELP_MSG}")

    cfg = get_config(defaults, overrides)  # create CFG instance

    # Mapping from task to module
    module = {"detect": yolo.v8.detect, "segment": yolo.v8.segment, "classify": yolo.v8.classify}.get(cfg.task)
    if not module:
        raise SyntaxError(f"yolo task={cfg.task} is invalid. Valid tasks are: {', '.join(tasks)}\n{CLI_HELP_MSG}")

    # Mapping from mode to function
    func = {
        "train": module.train,
        "val": module.val,
        "predict": module.predict,
        "export": yolo.engine.exporter.export}.get(cfg.mode)
    if not func:
        raise SyntaxError(f"yolo mode={cfg.mode} is invalid. Valid modes are: {', '.join(modes)}\n{CLI_HELP_MSG}")

    func(cfg)


# Special modes --------------------------------------------------------------------------------------------------------
def copy_default_config():
    new_file = Path.cwd() / DEFAULT_CFG_PATH.name.replace('.yaml', '_copy.yaml')
    shutil.copy2(DEFAULT_CFG_PATH, new_file)
    LOGGER.info(f"{PREFIX}{DEFAULT_CFG_PATH} copied to {new_file}\n"
                f"Usage for running YOLO with this new custom config:\nyolo cfg={new_file} args...")


if __name__ == '__main__':
    entrypoint()
