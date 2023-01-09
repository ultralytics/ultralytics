# Ultralytics YOLO 🚀, GPL-3.0 license

import shutil
from pathlib import Path

import hydra

from ultralytics import hub, yolo
from ultralytics.yolo.utils import DEFAULT_CONFIG, LOGGER, colorstr

DIR = Path(__file__).parent


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent.relative_to(DIR)), config_name=DEFAULT_CONFIG.name)
def cli(cfg):
    """
    Run a specified task and mode with the given configuration.

    Args:
        cfg (DictConfig): Configuration for the task and mode.
    """
    # LOGGER.info(f"{colorstr(f'Ultralytics YOLO v{ultralytics.__version__}')}")
    task, mode = cfg.task.lower(), cfg.mode.lower()

    # Special case for initializing the configuration
    if task == "init":
        shutil.copy2(DEFAULT_CONFIG, Path.cwd())
        LOGGER.info(f"""
        {colorstr("YOLO:")} configuration saved to {Path.cwd() / DEFAULT_CONFIG.name}.
        To run experiments using custom configuration:
        yolo task='task' mode='mode' --config-name config_file.yaml
                    """)
        return

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
        "export": yolo.engine.exporter.export,
        "checks": hub.checks}
    func = mode_func_map.get(mode)
    if not func:
        raise SyntaxError(f"mode not recognized. Choices are {', '.join(mode_func_map.keys())}")

    func(cfg)
