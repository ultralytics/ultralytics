import shutil
from pathlib import Path

import hydra

import ultralytics
from ultralytics import yolo

from .utils import DEFAULT_CONFIG, LOGGER, colorstr


@hydra.main(version_base=None, config_path="configs", config_name="default")
def cli(cfg):
    cwd = Path().cwd()
    LOGGER.info(f"{colorstr(f'Ultralytics YOLO v{ultralytics.__version__}')}")
    task, mode = cfg.task.lower(), cfg.mode.lower()

    if task == "init":  # special case
        shutil.copy2(DEFAULT_CONFIG, cwd)
        LOGGER.info(f"""
        {colorstr("YOLO:")} configuration saved to {cwd / DEFAULT_CONFIG.name}.
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
        func = yolo.trainer.exporter.export_model
    else:
        raise SyntaxError("task not recognized. Choices are `'detect', 'segment', 'classify'`")

    if mode == "train":
        func = module.train
    elif mode == "val":
        func = module.val
    elif mode == "predict":
        func = module.predict
    elif mode == "export":
        func = yolo.trainer.exporter.export_model
    else:
        raise SyntaxError("mode not recognized. Choices are `'train', 'val', 'predict', 'export'`")
    func(cfg)
