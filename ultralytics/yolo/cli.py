import os
import shutil

import hydra

import ultralytics
import ultralytics.yolo.v8 as yolo
from ultralytics.yolo.engine.trainer import DEFAULT_CONFIG

from .utils import LOGGER, colorstr


@hydra.main(version_base=None, config_path="utils/configs", config_name="default")
def cli(cfg):
    LOGGER.info(f"{colorstr(f'Ultralytics YOLO v{ultralytics.__version__}')}")
    task, mode = cfg.task.lower(), cfg.mode.lower()

    if task == "init":  # special case
        shutil.copy2(DEFAULT_CONFIG, os.getcwd())
        LOGGER.info(f"""
        {colorstr("YOLO :")} configuration saved to {os.getcwd()}/{DEFAULT_CONFIG.name}.
        To run experiments using custom configuration:
        yolo task='task' mode='mode' --config-name config_file.yaml
                    """)
        return
    elif task == "detect":
        module_file = yolo.detect
    elif task == "segment":
        module_file = yolo.segment
    elif task == "classify":
        module_file = yolo.classify
    else:
        raise SyntaxError("task not recognized. Choices are `'detect', 'segment', 'classify'`")

    if mode == "train":
        module_function = module_file.train
    elif mode == "val":
        module_function = module_file.val
    elif mode == "predict":
        module_function = module_file.predict
    else:
        raise SyntaxError("mode not recognized. Choices are `'train', 'val', 'predict'`")
    module_function(cfg)
