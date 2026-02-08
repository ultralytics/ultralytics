# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Stereo 3D Object Detection model configuration parser."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.checks import check_yaml


def load_stereo_config(cfg_path: str | Path) -> dict[str, Any]:
    """Load and validate stereo model configuration from YAML file.

    Args:
        cfg_path: Path to stereo model configuration YAML file.

    Returns:
        dict: Configuration dictionary with stereo-specific parameters validated.

    Raises:
        FileNotFoundError: If configuration file does not exist.
        ValueError: If required stereo parameters are missing or invalid.

    Examples:
        >>> config = load_stereo_config("ultralytics/cfg/models/11/yolo11-stereo3ddet.yaml")
        >>> assert config["stereo"] is True
        >>> assert config["input_channels"] == 6
    """
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Stereo configuration file not found: {cfg_path}")

    # Load YAML file
    yaml_file = check_yaml(cfg_path)
    config = YAML.load(yaml_file)

    # Validate required stereo-specific parameters
    required_params = ["nc", "stereo", "input_channels"]
    for param in required_params:
        if param not in config:
            raise ValueError(f"Required parameter '{param}' not found in stereo config: {cfg_path}")

    # Validate stereo flag
    if not isinstance(config["stereo"], bool) or not config["stereo"]:
        raise ValueError(f"Stereo flag must be True in stereo config: {cfg_path}")

    # Validate input channels (must be 6 for stereo: RGB left + RGB right)
    if config["input_channels"] != 6:
        raise ValueError(
            f"Stereo model must have 6 input channels (got {config['input_channels']}), "
            f"expected RGB left (3) + RGB right (3)"
        )

    # Validate number of classes
    if not isinstance(config["nc"], int) or config["nc"] < 1:
        raise ValueError(f"Number of classes 'nc' must be a positive integer, got {config['nc']}")

    # Validate mean_dims if present (optional but recommended)
    if "mean_dims" in config:
        mean_dims = config["mean_dims"]
        if not isinstance(mean_dims, dict):
            raise ValueError(f"mean_dims must be a dictionary, got {type(mean_dims)}")

        # Validate mean_dims structure (should have entries for each class)
        for class_name, dims in mean_dims.items():
            if not isinstance(dims, (list, tuple)) or len(dims) != 3:
                raise ValueError(
                    f"mean_dims['{class_name}'] must be a list/tuple of 3 floats [L, W, H], got {dims}"
                )
            if not all(isinstance(d, (int, float)) and d > 0 for d in dims):
                raise ValueError(f"All dimensions in mean_dims['{class_name}'] must be positive numbers, got {dims}")

    # Add yaml_file path for reference
    config["yaml_file"] = str(cfg_path)

    if LOGGER:
        LOGGER.info(f"Loaded stereo configuration from {cfg_path}")
        LOGGER.info(f"  - Classes: {config['nc']}")
        LOGGER.info(f"  - Input channels: {config['input_channels']}")
        LOGGER.info(f"  - Stereo mode: {config['stereo']}")

    return config


def validate_stereo_config(config: dict[str, Any]) -> bool:
    """Validate stereo configuration dictionary.

    Args:
        config: Configuration dictionary to validate.

    Returns:
        bool: True if configuration is valid.

    Raises:
        ValueError: If configuration is invalid.
    """
    # Check required parameters
    required = ["nc", "stereo", "input_channels"]
    for param in required:
        if param not in config:
            raise ValueError(f"Missing required parameter: {param}")

    # Validate types and values
    if not isinstance(config["stereo"], bool) or not config["stereo"]:
        raise ValueError("stereo must be True")

    if config["input_channels"] != 6:
        raise ValueError(f"input_channels must be 6 for stereo, got {config['input_channels']}")

    if not isinstance(config["nc"], int) or config["nc"] < 1:
        raise ValueError(f"nc must be a positive integer, got {config['nc']}")

    return True


__all__ = ["load_stereo_config", "validate_stereo_config"]

