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

    module_file = None
    if cfg.task.lower() == "init":  # special case
        shutil.copy2(DEFAULT_CONFIG, os.getcwd())
        LOGGER.info(f"""
        {colorstr("YOLO :")} configuration saved to {os.getcwd()}/{DEFAULT_CONFIG.name}.
        To run experiments using custom configuration:
        yolo task='task' mode='mode' --config-name config_file.yaml
                    """)
        return
    elif cfg.task.lower() == "detect":
        module_file = yolo.detect
    elif cfg.task.lower() == "segment":
        module_file = yolo.segment
    elif cfg.task.lower() == "classify":
        module_file = yolo.classify

    if not module_file:
        raise Exception("task not recognized. Choices are `'detect', 'segment', 'classify'`")

    module_function = None

    if cfg.mode.lower() == "train":
        module_function = module_file.train
    elif cfg.mode.lower() == "val":
        module_function = module_file.val
    elif cfg.mode.lower() == "infer":
        module_function = module_file.infer

    if not module_function:
        raise Exception("mode not recognized. Choices are `'train', 'val', 'infer'`")
    module_function(cfg)
