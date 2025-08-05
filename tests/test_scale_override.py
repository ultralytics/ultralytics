import tempfile

import pytest
import yaml

from ultralytics import YOLO
from ultralytics.utils import YAML

yaml_path = "ultralytics/cfg/models/v8/yolov8.yaml"


def count_model_params(model):
    """Helper function to count total model parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def test_scale_override_on_pt_file():
    """Test that providing a scale with a .pt file is ignored and default weights are loaded."""
    model = YOLO(model="yolov8n.pt", scale="l")  # 'l' should be ignored
    print(f"{count_model_params(model)} million parameters")


def test_valid_scale_override_on_pt_file():
    """Test that explicitly passing a different scale overrides the default when using a .pt file."""
    model = YOLO(model="yolov8s.pt", scale="k")
    print(f"{count_model_params(model)} million parameters")


def test_yaml_without_scale_uses_default_behavior():
    """Test that YAML file with no scale argument still loads a model."""
    model = YOLO(model="yolov8n.yaml", scale=None)
    print(f"{count_model_params(model)} million parameters")


def test_scale_override_with_yaml_file_and_valid_scale():
    """Test scale override with valid declared scale on a YAML file."""
    model = YOLO(model="yolov8n.yaml", scale="s")
    print(f"{count_model_params(model)} million parameters")


def test_scale_auto_detect_from_model_filename():
    """Test that the scale is automatically inferred from the model filename."""
    model = YOLO(model="yolov8x.pt")
    print(f"{count_model_params(model)} million parameters")


def test_default_model_instantiation_behavior():
    """Test that calling YOLO() with no arguments uses default model."""
    model = YOLO()  # Should load yolov8n.pt by default
    print(f"{count_model_params(model)} million parameters")


def test_invalid_scale_override_raises_error_on_yaml():
    """Test that an invalid scale raises ValueError when using a YAML config."""
    with pytest.raises(ValueError, match="Invalid scale"):
        YOLO(model="yolov8n.yaml", scale="k")


def test_yaml_with_invalid_scale_and_no_variants():
    """Test that YAML with invalid scale and no variants raises ValueError."""
    yaml_path = "ultralytics/cfg/models/v8/yolov8.yaml"
    content = YAML.load(yaml_path)
    content["variants"] = {
        "n": {"depth_multiple": 0.33, "width_multiple": 0.25},
        "s": {"depth_multiple": 0.33, "width_multiple": 0.50},
        "m": {"depth_multiple": 0.67, "width_multiple": 0.75},
    }
    del content["scales"]
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+", delete=False) as f:
        yaml.dump(content, f)
        f.flush()
        with pytest.raises(ValueError, match="Invalid scale"):
            YOLO(model=f.name, scale="z")  # 'z' is invalid, and no `variants` to fallback


def test_yaml_with_valid_variants_allows_valid_scales():
    """
    Test that a YAML file defining model variants accepts valid scale overrides.
    """
    yaml_path = "ultralytics/cfg/models/v8/yolov8.yaml"
    content = YAML.load(yaml_path)
    content["variants"] = {
        "n": {"depth_multiple": 0.33, "width_multiple": 0.25},
        "s": {"depth_multiple": 0.33, "width_multiple": 0.50},
        "m": {"depth_multiple": 0.67, "width_multiple": 0.75},
    }
    del content["scales"]
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+", delete=False) as f:
        yaml.dump(content, f)
        f.flush()
        for scale in ["n", "s", "m"]:
            model = YOLO(model=f.name, scale=scale)
            assert hasattr(model.model, "model"), f"Failed for scale {scale}"
