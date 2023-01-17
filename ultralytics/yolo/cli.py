# Ultralytics YOLO 🚀, GPL-3.0 license

import argparse
import shutil
from pathlib import Path

from hydra import compose, initialize

from ultralytics import hub, yolo
from ultralytics.yolo.utils import DEFAULT_CONFIG, HELP_MSG, LOGGER, PREFIX, print_settings, yaml_load

DIR = Path(__file__).parent


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
    task_module_map = {"detect": yolo.v8.detect, "segment": yolo.v8.segment, "classify": yolo.v8.classify}
    module = task_module_map.get(task)
    if not module:
        raise SyntaxError(f"task not recognized. Choices are {', '.join(task_module_map.keys())}")

    # Mapping from mode to function
    mode_func_map = {
        "train": module.train,
        "val": module.val,
        "predict": module.predict,
        "export": yolo.engine.exporter.export}
    func = mode_func_map.get(mode)
    if not func:
        raise SyntaxError(f"mode not recognized. Choices are {', '.join(mode_func_map.keys())}")

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
        'checks': hub.checks,
        'help': lambda: LOGGER.info(HELP_MSG),
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
        else:
            raise (SyntaxError(f"'{a}' is not a valid yolo argument\n{HELP_MSG}"))

    with initialize(version_base=None, config_path=str(DEFAULT_CONFIG.parent.relative_to(DIR)), job_name="YOLO"):
        cfg = compose(config_name=DEFAULT_CONFIG.name, overrides=overrides)
        cli(cfg)


# Special modes --------------------------------------------------------------------------------------------------------
def copy_default_config():
    new_file = Path.cwd() / DEFAULT_CONFIG.name.replace('.yaml', '_copy.yaml')
    shutil.copy2(DEFAULT_CONFIG, new_file)
    LOGGER.info(f"{PREFIX}{DEFAULT_CONFIG} copied to {new_file}\n"
                f"Usage for running YOLO with this new custom config:\nyolo cfg={new_file} args...")
