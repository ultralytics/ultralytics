"""Test YOLO model export to ONNX format with 'names' property as None type."""

import pytest
import onnxruntime as ort
from ultralytics import YOLO
from tests import SOURCE


def save_wrong_names_model(model, tmp_path):
    """Save a model with 'names' property as None type.

    Args:
        model (YOLO): The YOLO model instance to modify and save.
        tmp_path (Path): Temporary directory path provided by pytest fixture.

    Returns:
        file: Saved model path
    """
    file = tmp_path / "model_none_names_prop.pt"
    model.model.names = None
    model.save(file)
    return file


@pytest.mark.parametrize("end2end", [False, True])
def test_export_onnx(end2end, isolated_model, tmp_path):
    """Test YOLO model export to ONNX format with dynamic axes.

    Args:
        end2end (bool): Whether to test end2end export mode.
        isolated_model (Path): Path to isolated model fixture provided by pytest.
        tmp_path (Path): Temporary directory path provided by pytest fixture.
    """
    model = YOLO(isolated_model)

    # Save a None type 'names' model
    none_names_model_path = save_wrong_names_model(model, tmp_path)
    # Init it
    none_names_model = YOLO(none_names_model_path)
    # Check if 'names' is None
    assert getattr(none_names_model, "names") is None
    # Export it
    file = none_names_model.export(format="onnx", dynamic=True, imgsz=32, end2end=end2end)

    # Test correct YOLO type 'names'
    session = ort.InferenceSession(str(file))
    meta = session.get_modelmeta().custom_metadata_map
    assert not any(v is None for v in meta.values())

    YOLO(file)(SOURCE, imgsz=32)  # exported model inference
