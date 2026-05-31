"""Test YOLO model export to ONNX format with ABSENT 'names' property."""

import pytest
import onnxruntime as ort
from ultralytics import YOLO
from tests import SOURCE


def save_wrong_names_model(model, tmp_path):
    """Save a model with ABSENT 'names' property.

    Args:
        model (YOLO): The YOLO model instance to modify and save.
        tmp_path (Path): Temporary directory path provided by pytest fixture.

    Returns:
        file: Saved model path
    """
    file = tmp_path / "model_empty_names_prop.pt"
    delattr(model.model, "names")
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

    # Save an absent 'names' model
    empty_names_model_path = save_wrong_names_model(model, tmp_path)
    # Init it
    empty_names_model = YOLO(empty_names_model_path)

    # Export it
    file = empty_names_model.export(format="onnx", dynamic=True, imgsz=32)

    # Test correct YOLO type 'names'
    session = ort.InferenceSession(str(file))
    meta = session.get_modelmeta().custom_metadata_map
    assert "names" in meta, "ONNX missing 'names' property"

    YOLO(file)(SOURCE, imgsz=32)  # exported model inference
