from pathlib import Path
from typing import Dict, Union

from omegaconf import DictConfig, OmegaConf

from ultralytics.yolo.configs.hydra_patch import check_config_mismatch


def get_config(config: Union[str, DictConfig], overrides: Union[str, Dict] = None, force_match=True):
    """
    Load and merge configuration data from a file or dictionary.

    Args:
        config (Union[str, DictConfig]): Configuration data in the form of a file name or a DictConfig object.
        overrides (Union[str, Dict], optional): Overrides in the form of a file name or a dictionary. Default is None.

    Returns:
        OmegaConf.Namespace: Training arguments namespace.
    """
    if overrides is None:
        overrides = {}
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)
    elif isinstance(config, Dict):
        config = OmegaConf.create(config)
    # override
    if isinstance(overrides, str):
        overrides = OmegaConf.load(overrides)
    elif isinstance(overrides, Dict):
        overrides = OmegaConf.create(overrides)

    # Patch. Force update the configs.
    if force_match:
        if overrides.get("nosave") is not None:
            overrides.save = not overrides.pop("nosave")
        if overrides.get("noval") is not None:
            overrides.val = not overrides.pop("noval")
        if overrides.get("view_img") is not None:
            overrides.show = overrides.pop("view_img")

    check_config_mismatch(dict(overrides).keys(), dict(config).keys())

    return OmegaConf.merge(config, overrides)
