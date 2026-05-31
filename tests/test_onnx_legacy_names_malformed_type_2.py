import pytest
import ast
import onnxruntime as ort
from ultralytics import YOLO
from tests import SOURCE


def save_wrong_names_model(model, tmp_path):
    """
    Save a model where class names are not str
    Args:
        model:
        tmp_path:

    Returns:
        file: Saved model path
    """
    file = tmp_path / "model_list_names_prop.pt"

    #  class names are not str
    model.model.names = {0: 0, 1: 1}
    model.save(file)
    return file


@pytest.mark.parametrize("end2end", [False, True])
def test_export_onnx(end2end, isolated_model, tmp_path):
    model = YOLO(isolated_model)

    # Save a malformed 'names' model
    list_names_model_path = save_wrong_names_model(model, tmp_path)
    # Init it
    list_names_model = YOLO(list_names_model_path)

    # Export it
    file = list_names_model.export(format="onnx", dynamic=True, imgsz=32)

    # Test correct YOLO type 'names'
    session = ort.InferenceSession(str(file))
    meta = session.get_modelmeta().custom_metadata_map

    # conv = ast.literal_eval(meta["names"])
    # print(meta["names"], type(meta["names"]),conv)

    check_names_type = ast.literal_eval(meta["names"])
    assert all(isinstance(val, str) for val in check_names_type.values())

    YOLO(file)(SOURCE, imgsz=32)  # exported model inference
