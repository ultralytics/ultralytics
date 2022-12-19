from pathlib import Path
from typing import Dict, Union

from omegaconf import DictConfig, OmegaConf

from ultralytics.yolo.utils.configs.hydra_patch import check_config_mismatch


def get_config(config: Union[str, DictConfig], overrides: Union[str, Dict] = {}):
    """
    Accepts yaml file name or DictConfig containing experiment configuration.
    Returns training args namespace
    :param config: Optional file name or DictConfig object
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)
    elif isinstance(config, Dict):
        config = OmegaConf.create(config)
    # override
    if isinstance(overrides, str):
        overrides = OmegaConf.load(overrides)
    elif isinstance(overrides, Dict):
        overrides = OmegaConf.create(overrides)

    check_config_mismatch(dict(overrides).keys(), dict(config).keys())

    return OmegaConf.merge(config, overrides)
