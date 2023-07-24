# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
RT-DETR model interface
"""

from pathlib import Path

import torch.nn as nn

from ultralytics.cfg import get_cfg
from ultralytics.engine.exporter import Exporter
from ultralytics.nn.tasks import RTDETRDetectionModel, attempt_load_one_weight, yaml_model_load
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, LOGGER, RANK, ROOT, is_git_dir
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import model_info, smart_inference_mode
from ultralytics.engine.model import Model

from .predict import RTDETRPredictor
from .train import RTDETRTrainer
from .val import RTDETRValidator


class RTDETR(Model):
    """
    RTDETR model interface.
    """
    def __init__(self, model='rtdetr-l.pt') -> None:
        if model and not model.endswith('.pt') and not model.endswith('.yaml'):
            raise NotImplementedError('RT-DETR only supports creating from pt file or yaml file.')
        super().__init__(model=model, task="detect")

    # def info(self, verbose=True):
    #     """Get model info"""
    #     return model_info(self.model, verbose=verbose)

    @property
    def task_map(self):
        return {"detect": {"predictor": RTDETRPredictor, "validator": RTDETRValidator, "trainer": RTDETRTrainer}}
