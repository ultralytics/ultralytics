import shutil
from pathlib import Path

import hydra

import ultralytics
from ultralytics import yolo
from ultralytics.yolo.utils import DEFAULT_CONFIG, LOGGER, colorstr

DIR = Path(__file__).parent


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent.relative_to(DIR)), config_name=DEFAULT_CONFIG.name)
def cli(cfg):
    """
    Run a specified task and mode with the given configuration.

    Args:
        cfg (DictConfig): Configuration for the task and mode.
    """
    LOGGER.info(f"{colorstr(f'Ultralytics YOLO v{ultralytics.__version__}')}")
    task, mode = cfg.task.lower(), cfg.mode.lower()

    # Special case for initializing the configuration
    if task == "init":
        shutil.copy2(DEFAULT_CONFIG, Path().cwd())
        LOGGER.info(f"""
        {colorstr("YOLO:")} configuration saved to {Path().cwd() / DEFAULT_CONFIG.name}.
        To run experiments using custom configuration:
        yolo task='task' mode='mode' --config-name config_file.yaml
                    """)
        return

    elif task == "detect":
        module = yolo.v8.detect
    elif task == "segment":
        module = yolo.v8.segment
    elif task == "classify":
        module = yolo.v8.classify
    elif task == "export":
        func = yolo.engine.exporter.export
    else:
        raise SyntaxError("task not recognized. Choices are `'detect', 'segment', 'classify'`")

    if mode == "train":
        func = module.train
    elif mode == "val":
        func = module.val
    elif mode == "predict":
        func = module.predict
    elif mode == "export":
        func = yolo.engine.exporter.export
    else:
        raise SyntaxError("mode not recognized. Choices are `'train', 'val', 'predict', 'export'`")
    func(cfg)
