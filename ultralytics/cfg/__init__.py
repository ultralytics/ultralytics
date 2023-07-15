# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import re
import shutil
import sys
from difflib import get_close_matches
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Union

from ultralytics.utils import (DEFAULT_CFG, DEFAULT_CFG_DICT, DEFAULT_CFG_PATH, LOGGER, ROOT, USER_CONFIG_DIR,
                               IterableSimpleNamespace, __version__, checks, colorstr, deprecation_warn, get_settings,
                               yaml_load, yaml_print)

# Define valid tasks and modes
MODES = 'train', 'val', 'predict', 'export', 'track', 'benchmark'
TASKS = 'detect', 'segment', 'classify', 'pose'
TASK2DATA = {'detect': 'coco8.yaml', 'segment': 'coco8-seg.yaml', 'classify': 'imagenet100', 'pose': 'coco8-pose.yaml'}
TASK2MODEL = {
    'detect': 'yolov8n.pt',
    'segment': 'yolov8n-seg.pt',
    'classify': 'yolov8n-cls.pt',
    'pose': 'yolov8n-pose.pt'}
TASK2METRIC = {
    'detect': 'metrics/mAP50-95(B)',
    'segment': 'metrics/mAP50-95(M)',
    'classify': 'metrics/accuracy_top1',
    'pose': 'metrics/mAP50-95(P)'}


CLI_HELP_MSG = \
    f"""
    Arguments received: {str(['yolo'] + sys.argv[1:])}. Ultralytics 'yolo' commands use the following syntax:

        yolo TASK MODE ARGS

        Where   TASK (optional) is one of {TASKS}
                MODE (required) is one of {MODES}
                ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.
                    See all ARGS at https://docs.ultralytics.com/usage/cfg or with 'yolo cfg'

    1. Train a detection model for 10 epochs with an initial learning_rate of 0.01
        yolo train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01

    2. Predict a YouTube video using a pretrained segmentation model at image size 320:
        yolo predict model=yolov8n-seg.pt source='https://youtu.be/Zgi9g1ksQHc' imgsz=320

    3. Val a pretrained detection model at batch-size 1 and image size 640:
        yolo val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640

    4. Export a YOLOv8n classification model to ONNX format at image size 224 by 128 (no TASK required)
        yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128

    5. Run special commands:
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

# Define keys for arg type checks
CFG_FLOAT_KEYS = 'warmup_epochs', 'box', 'cls', 'dfl', 'degrees', 'shear'
CFG_FRACTION_KEYS = ('dropout', 'iou', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_momentum', 'warmup_bias_lr',
                     'label_smoothing', 'hsv_h', 'hsv_s', 'hsv_v', 'translate', 'scale', 'perspective', 'flipud',
                     'fliplr', 'mosaic', 'mixup', 'copy_paste', 'conf', 'iou', 'fraction')  # fraction floats 0.0 - 1.0
CFG_INT_KEYS = ('epochs', 'patience', 'batch', 'workers', 'seed', 'close_mosaic', 'mask_ratio', 'max_det', 'vid_stride',
                'line_width', 'workspace', 'nbs', 'save_period')
CFG_BOOL_KEYS = ('save', 'exist_ok', 'verbose', 'deterministic', 'single_cls', 'rect', 'cos_lr', 'overlap_mask', 'val',
                 'save_json', 'save_hybrid', 'half', 'dnn', 'plots', 'show', 'save_txt', 'save_conf', 'save_crop',
                 'show_labels', 'show_conf', 'visualize', 'augment', 'agnostic_nms', 'retina_masks', 'boxes', 'keras',
                 'optimize', 'int8', 'dynamic', 'simplify', 'nms', 'profile')


def cfg2dict(cfg):
    """
    Convert a configuration object to a dictionary, whether it is a file path, a string, or a SimpleNamespace object.

    Args:
        cfg (str | Path | SimpleNamespace): Configuration object to be converted to a dictionary.

    Returns:
        cfg (dict): Configuration object in dictionary format.
    """
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)  # load dict
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # convert to dict
    return cfg


def get_cfg(cfg: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_CFG_DICT, overrides: Dict = None):
    """
    Load and merge configuration data from a file or dictionary.

    Args:
        cfg (str | Path | Dict | SimpleNamespace): Configuration data.
        overrides (str | Dict | optional): Overrides in the form of a file name or a dictionary. Default is None.

    Returns:
        (SimpleNamespace): Training arguments namespace.
    """
    cfg = cfg2dict(cfg)

    # Merge overrides
    if overrides:
        overrides = cfg2dict(overrides)
        check_cfg_mismatch(cfg, overrides)
        cfg = {**cfg, **overrides}  # merge cfg and overrides dicts (prefer overrides)

    # Special handling for numeric project/name
    for k in 'project', 'name':
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])
    if cfg.get('name') == 'model':  # assign model to 'name' arg
        cfg['name'] = cfg.get('model', '').split('.')[0]
        LOGGER.warning(f"WARNING ⚠️ 'name=model' automatically updated to 'name={cfg['name']}'.")

    # Type and Value checks
    for k, v in cfg.items():
        if v is not None:  # None values may be from optional args
            if k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                    f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' is an invalid value. "
                                     f"Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                f"'{k}' must be an int (i.e. '{k}=8')")
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                f"'{k}' must be a bool (i.e. '{k}=True' or '{k}=False')")

    # Return instance
    return IterableSimpleNamespace(**cfg)


def _handle_deprecation(custom):
    """
    Hardcoded function to handle deprecated config keys
    """

    for key in custom.copy().keys():
        if key == 'hide_labels':
            deprecation_warn(key, 'show_labels')
            custom['show_labels'] = custom.pop('hide_labels') == 'False'
        if key == 'hide_conf':
            deprecation_warn(key, 'show_conf')
            custom['show_conf'] = custom.pop('hide_conf') == 'False'
        if key == 'line_thickness':
            deprecation_warn(key, 'line_width')
            custom['line_width'] = custom.pop('line_thickness')

    return custom


def check_cfg_mismatch(base: Dict, custom: Dict, e=None):
    """
    This function checks for any mismatched keys between a custom configuration list and a base configuration list.
    If any mismatched keys are found, the function prints out similar keys from the base list and exits the program.

    Args:
        custom (dict): a dictionary of custom configuration options
        base (dict): a dictionary of base configuration options
    """
    custom = _handle_deprecation(custom)
    base, custom = (set(x.keys()) for x in (base, custom))
    mismatched = [x for x in custom if x not in base]
    if mismatched:
        string = ''
        for x in mismatched:
            matches = get_close_matches(x, base)  # key list
            matches = [f'{k}={DEFAULT_CFG_DICT[k]}' if DEFAULT_CFG_DICT.get(k) is not None else k for k in matches]
            match_str = f'Similar arguments are i.e. {matches}.' if matches else ''
            string += f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}\n"
        raise SyntaxError(string + CLI_HELP_MSG) from e


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
            new_args[-1] += f'={args[i + 1]}'
            del args[i + 1]
        elif arg.endswith('=') and i < len(args) - 1 and '=' not in args[i + 1]:  # merge ['arg=', 'val']
            new_args.append(f'{arg}{args[i + 1]}')
            del args[i + 1]
        elif arg.startswith('=') and i > 0:  # merge ['arg', '=val']
            new_args[-1] += arg
        else:
            new_args.append(arg)
    return new_args


def handle_yolo_hub(args: List[str]) -> None:
    """
    Handle Ultralytics HUB command-line interface (CLI) commands.

    This function processes Ultralytics HUB CLI commands such as login and logout.
    It should be called when executing a script with arguments related to HUB authentication.

    Args:
        args (List[str]): A list of command line arguments

    Example:
        python my_script.py hub login your_api_key
    """
    from ultralytics import hub

    if args[0] == 'login':
        key = args[1] if len(args) > 1 else ''
        # Log in to Ultralytics HUB using the provided API key
        hub.login(key)
    elif args[0] == 'logout':
        # Log out from Ultralytics HUB
        hub.logout()


def handle_yolo_settings(args: List[str]) -> None:
    """
    Handle YOLO settings command-line interface (CLI) commands.

    This function processes YOLO settings CLI commands such as reset.
    It should be called when executing a script with arguments related to YOLO settings management.

    Args:
        args (List[str]): A list of command line arguments for YOLO settings management.

    Example:
        python my_script.py yolo settings reset
    """
    path = USER_CONFIG_DIR / 'settings.yaml'  # get SETTINGS YAML file path
    if any(args) and args[0] == 'reset':
        path.unlink()  # delete the settings file
        get_settings()  # create new settings
        LOGGER.info('Settings reset successfully')  # inform the user that settings have been reset
    yaml_print(path)  # print the current settings


def entrypoint(debug=''):
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
    args = (debug.split(' ') if debug else sys.argv)[1:]
    if not args:  # no arguments passed
        LOGGER.info(CLI_HELP_MSG)
        return

    special = {
        'help': lambda: LOGGER.info(CLI_HELP_MSG),
        'checks': checks.check_yolo,
        'version': lambda: LOGGER.info(__version__),
        'settings': lambda: handle_yolo_settings(args[1:]),
        'cfg': lambda: yaml_print(DEFAULT_CFG_PATH),
        'hub': lambda: handle_yolo_hub(args[1:]),
        'login': lambda: handle_yolo_hub(args),
        'copy-cfg': copy_default_cfg}
    full_args_dict = {**DEFAULT_CFG_DICT, **{k: None for k in TASKS}, **{k: None for k in MODES}, **special}

    # Define common mis-uses of special commands, i.e. -h, -help, --help
    special.update({k[0]: v for k, v in special.items()})  # singular
    special.update({k[:-1]: v for k, v in special.items() if len(k) > 1 and k.endswith('s')})  # singular
    special = {**special, **{f'-{k}': v for k, v in special.items()}, **{f'--{k}': v for k, v in special.items()}}

    overrides = {}  # basic overrides, i.e. imgsz=320
    for a in merge_equals_args(args):  # merge spaces around '=' sign
        if a.startswith('--'):
            LOGGER.warning(f"WARNING ⚠️ '{a}' does not require leading dashes '--', updating to '{a[2:]}'.")
            a = a[2:]
        if a.endswith(','):
            LOGGER.warning(f"WARNING ⚠️ '{a}' does not require trailing comma ',', updating to '{a[:-1]}'.")
            a = a[:-1]
        if '=' in a:
            try:
                re.sub(r' *= *', '=', a)  # remove spaces around equals sign
                k, v = a.split('=', 1)  # split on first '=' sign
                assert v, f"missing '{k}' value"
                if k == 'cfg':  # custom.yaml passed
                    LOGGER.info(f'Overriding {DEFAULT_CFG_PATH} with {v}')
                    overrides = {k: val for k, val in yaml_load(checks.check_yaml(v)).items() if k != 'cfg'}
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
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_cfg_mismatch(full_args_dict, {a: ''}, e)

        elif a in TASKS:
            overrides['task'] = a
        elif a in MODES:
            overrides['mode'] = a
        elif a.lower() in special:
            special[a.lower()]()
            return
        elif a in DEFAULT_CFG_DICT and isinstance(DEFAULT_CFG_DICT[a], bool):
            overrides[a] = True  # auto-True for default bool args, i.e. 'yolo show' sets show=True
        elif a in DEFAULT_CFG_DICT:
            raise SyntaxError(f"'{colorstr('red', 'bold', a)}' is a valid YOLO argument but is missing an '=' sign "
                              f"to set its value, i.e. try '{a}={DEFAULT_CFG_DICT[a]}'\n{CLI_HELP_MSG}")
        else:
            check_cfg_mismatch(full_args_dict, {a: ''})

    # Check keys
    check_cfg_mismatch(full_args_dict, overrides)

    # Mode
    mode = overrides.get('mode', None)
    if mode is None:
        mode = DEFAULT_CFG.mode or 'predict'
        LOGGER.warning(f"WARNING ⚠️ 'mode' is missing. Valid modes are {MODES}. Using default 'mode={mode}'.")
    elif mode not in MODES:
        if mode not in ('checks', checks):
            raise ValueError(f"Invalid 'mode={mode}'. Valid modes are {MODES}.\n{CLI_HELP_MSG}")
        LOGGER.warning("WARNING ⚠️ 'yolo mode=checks' is deprecated. Use 'yolo checks' instead.")
        checks.check_yolo()
        return

    # Task
    task = overrides.pop('task', None)
    if task:
        if task not in TASKS:
            raise ValueError(f"Invalid 'task={task}'. Valid tasks are {TASKS}.\n{CLI_HELP_MSG}")
        if 'model' not in overrides:
            overrides['model'] = TASK2MODEL[task]

    # Model
    model = overrides.pop('model', DEFAULT_CFG.model)
    if model is None:
        model = 'yolov8n.pt'
        LOGGER.warning(f"WARNING ⚠️ 'model' is missing. Using default 'model={model}'.")
    overrides['model'] = model
    if 'rtdetr' in model.lower():  # guess architecture
        from ultralytics import RTDETR
        model = RTDETR(model)  # no task argument
    elif 'sam' in model.lower():
        from ultralytics import SAM
        model = SAM(model)
    else:
        from ultralytics import YOLO
        model = YOLO(model, task=task)
    if isinstance(overrides.get('pretrained'), str):
        model.load(overrides['pretrained'])

    # Task Update
    if task != model.task:
        if task:
            LOGGER.warning(f"WARNING ⚠️ conflicting 'task={task}' passed with 'task={model.task}' model. "
                           f"Ignoring 'task={task}' and updating to 'task={model.task}' to match model.")
        task = model.task

    # Mode
    if mode in ('predict', 'track') and 'source' not in overrides:
        overrides['source'] = DEFAULT_CFG.source or ROOT / 'assets' if (ROOT / 'assets').exists() \
            else 'https://ultralytics.com/images/bus.jpg'
        LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using default 'source={overrides['source']}'.")
    elif mode in ('train', 'val'):
        if 'data' not in overrides:
            overrides['data'] = TASK2DATA.get(task or DEFAULT_CFG.task, DEFAULT_CFG.data)
            LOGGER.warning(f"WARNING ⚠️ 'data' is missing. Using default 'data={overrides['data']}'.")
    elif mode == 'export':
        if 'format' not in overrides:
            overrides['format'] = DEFAULT_CFG.format or 'torchscript'
            LOGGER.warning(f"WARNING ⚠️ 'format' is missing. Using default 'format={overrides['format']}'.")

    # Run command in python
    # getattr(model, mode)(**vars(get_cfg(overrides=overrides)))  # default args using default.yaml
    getattr(model, mode)(**overrides)  # default args from model


# Special modes --------------------------------------------------------------------------------------------------------
def copy_default_cfg():
    """Copy and create a new default configuration file with '_copy' appended to its name."""
    new_file = Path.cwd() / DEFAULT_CFG_PATH.name.replace('.yaml', '_copy.yaml')
    shutil.copy2(DEFAULT_CFG_PATH, new_file)
    LOGGER.info(f'{DEFAULT_CFG_PATH} copied to {new_file}\n'
                f"Example YOLO command with this new custom cfg:\n    yolo cfg='{new_file}' imgsz=320 batch=8")


if __name__ == '__main__':
    # Example Usage: entrypoint(debug='yolo predict model=yolov8n.pt')
    entrypoint(debug='')
