# Ultralytics YOLO 🚀, GPL-3.0 license

import argparse
import shutil
from pathlib import Path

from ultralytics import __version__, yolo
from ultralytics.yolo.utils import DEFAULT_CONFIG, LOGGER, PREFIX, checks, print_settings, yaml_load

DIR = Path(__file__).parent

CLI_HELP_MSG = \
    """
    YOLOv8 CLI Usage examples:

    1. Install the ultralytics package:

        pip install ultralytics

    2. Train, Val, Predict and Export using 'yolo' commands of the form:

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


def cli(cfg):
    """
    Run a specified task and mode with the given configuration.

    Args:
        cfg (DictConfig): Configuration for the task and mode.
    """
    # LOGGER.info(f"{colorstr(f'Ultralytics YOLO v{ultralytics.__version__}')}")
    from ultralytics.yolo.configs import get_config

    if cfg.cfg:
        LOGGER.info(f"{PREFIX}Overriding default config with {cfg.cfg}")
        cfg = get_config(cfg.cfg)
    task, mode = cfg.task.lower(), cfg.mode.lower()

    # Mapping from task to module
    tasks = {"detect": yolo.v8.detect, "segment": yolo.v8.segment, "classify": yolo.v8.classify}
    module = tasks.get(task)
    if not module:
        raise SyntaxError(f"yolo task={task} is invalid. Valid tasks are: {', '.join(tasks.keys())}\n{CLI_HELP_MSG}")

    # Mapping from mode to function
    modes = {"train": module.train, "val": module.val, "predict": module.predict, "export": yolo.engine.exporter.export}
    func = modes.get(mode)
    if not func:
        raise SyntaxError(f"yolo mode={mode} is invalid. Valid modes are: {', '.join(modes.keys())}\n{CLI_HELP_MSG}")

    func(cfg)


def entrypoint():
    """
    This function is the ultralytics package entrypoint, it's responsible for parsing the command line arguments passed
    to the package. It's a combination of argparse and hydra.

    This function allows for:
    - passing mandatory YOLO args as a list of strings
    - specifying the task to be performed, either 'detect', 'segment' or 'classify'
    - specifying the mode, either 'train', 'val', 'test', or 'predict'
    - running special modes like 'checks'
    - passing overrides to the package's configuration

    It uses the package's default config and initializes it using the passed overrides.
    Then it calls the CLI function with the composed config
    """
    parser = argparse.ArgumentParser(description='YOLO parser')
    parser.add_argument('args', type=str, nargs='+', help='YOLO args')
    args = parser.parse_args().args

    tasks = 'detect', 'segment', 'classify'
    modes = 'train', 'val', 'predict', 'export'
    special_modes = {
        'help': lambda: LOGGER.info(CLI_HELP_MSG),
        'checks': checks.check_yolo,
        'version': lambda: LOGGER.info(__version__),
        'settings': print_settings,
        'copy-config': copy_default_config}

    overrides = []  # basic overrides, i.e. imgsz=320
    defaults = yaml_load(DEFAULT_CONFIG)
    for a in args:
        if '=' in a:
            overrides.append(a)
        elif a in tasks:
            overrides.append(f'task={a}')
        elif a in modes:
            overrides.append(f'mode={a}')
        elif a in special_modes:
            special_modes[a]()
            return
        elif a in defaults and defaults[a] is False:
            overrides.append(f'{a}=True')  # auto-True for default False args, i.e. yolo show
        elif a in defaults:
            raise SyntaxError(f"'{a}' is a valid YOLO argument but is missing an '=' sign to set its value, "
                              f"i.e. try '{a}={defaults[a]}'"
                              f"\n{CLI_HELP_MSG}")
        else:
            raise SyntaxError(
                f"'{a}' is not a valid YOLO argument. For a full list of valid arguments see "
                f"https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/configs/default.yaml"
                f"\n{CLI_HELP_MSG}")

    from hydra import compose, initialize

    with initialize(version_base=None, config_path=str(DEFAULT_CONFIG.parent.relative_to(DIR)), job_name="YOLO"):
        cfg = compose(config_name=DEFAULT_CONFIG.name, overrides=overrides)
        cli(cfg)


# Special modes --------------------------------------------------------------------------------------------------------
def copy_default_config():
    new_file = Path.cwd() / DEFAULT_CONFIG.name.replace('.yaml', '_copy.yaml')
    shutil.copy2(DEFAULT_CONFIG, new_file)
    LOGGER.info(f"{PREFIX}{DEFAULT_CONFIG} copied to {new_file}\n"
                f"Usage for running YOLO with this new custom config:\nyolo cfg={new_file} args...")
