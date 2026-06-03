"""Test YOLO model export to ONNX format where class ID are not int."""

import pytest
import ast
import onnxruntime as ort
from ultralytics import YOLO
from tests import SOURCE


def save_wrong_names_model(model, tmp_path):
    """Save a model where class ID are not int.

    Args:
        model (YOLO): The YOLO model instance to modify and save.
        tmp_path (Path): Temporary directory path provided by pytest fixture.

    Returns:
        file: Saved model path
    """
    file = tmp_path / "model_list_names_prop.pt"

    #  class ID is not int
    model.model.names = {"0": "cat", "1": "dog"}
    model.save(file)
    return file


@pytest.mark.parametrize("end2end", [True, False])
def test_export_onnx(end2end, isolated_model, tmp_path):
    """Test YOLO model export to ONNX format with dynamic axes.

    Args:
        end2end (bool): Whether to test end2end export mode.
        isolated_model (Path): Path to isolated model fixture provided by pytest.
        tmp_path (Path): Temporary directory path provided by pytest fixture.
    """
    model = YOLO(isolated_model)

    # Save a malformed 'names' model
    list_names_model_path = save_wrong_names_model(model, tmp_path)
    # Init it
    list_names_model = YOLO(list_names_model_path)

    # Export it
    file = list_names_model.export(format="onnx", dynamic=True, imgsz=32, end2end=end2end)

    # Test correct YOLO type 'names'
    session = ort.InferenceSession(str(file))
    meta = session.get_modelmeta().custom_metadata_map

    # conv = ast.literal_eval(meta["names"])
    # print(meta["names"], type(meta["names"]),conv)

    check_names_type = ast.literal_eval(meta["names"])
    assert all(isinstance(key, int) for key in check_names_type.keys())

    YOLO(file)(SOURCE, imgsz=32)  # exported model inference
