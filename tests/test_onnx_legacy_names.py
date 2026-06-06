# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Tests for legacy fallback for older checkpoints export where `names` attribute might be missing, None, or malformed."""

import ast

import pytest


from tests import SOURCE
from ultralytics import YOLO

ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed; skipping ONNX metadata tests")


def _save_model_with_names(model: YOLO, names, tmp_path, filename: str) -> str:
    """Overwrite `model.model.names` with `names` and save to `tmp_path/filename`.

    Args:
        model (YOLO): A loaded YOLO wrapper whose inner nn.Module will be mutated.
        names: The value to assign to `model.model.names`. Pass the string `"__delete__"` to remove the attribute entirely.
        tmp_path (Path): Pytest-provided per-test temporary directory.
        filename (str): Filename for the saved checkpoint.

    Returns:
        str: Absolute path to the saved checkpoint.
    """
    path = tmp_path / filename
    if names == "__delete__":
        if hasattr(model.model, "names"):
            delattr(model.model, "names")
    else:
        model.model.names = names
    model.save(str(path))
    return str(path)


def _assert_names_valid(meta: dict) -> dict:
    """Verify if names is according to YOLO standards.

    Args:
        meta (dict): The `custom_metadata_map` returned by an ORT session.

    Returns:
        dict: `names` from model in dictionary fomat'
    """
    assert "names" in meta, "ONNX metadata is missing the 'names' key"
    assert meta["names"] is not None, "'names' value in ONNX metadata is None"

    names = ast.literal_eval(meta["names"])

    assert isinstance(names, dict), f"Expected 'names' to deserialize as a dict, got {type(names).__name__}"
    assert all(isinstance(k, int) for k in names.keys()), (
        f"All 'names' keys must be int after export, got: {set(type(k).__name__ for k in names.keys())}"
    )
    assert all(isinstance(v, str) for v in names.values()), (
        f"All 'names' values must be str after export, got: {set(type(v).__name__ for v in names.values())}"
    )
    return names


@pytest.mark.parametrize("end2end", [False, True])
def test_export_onnx_legacy_names_sparse_keys(end2end, isolated_model, tmp_path):
    """Export must not crash when `names` has sparse/non-contiguous integer keys.

    Args:
        end2end (bool): Whether to test end-to-end export mode.
        isolated_model (str): Path to isolated model fixture provided by pytest.
        tmp_path (Path): Temporary directory path provided by pytest fixture.
    """
    model = YOLO(isolated_model)
    expected_nc = model.model.yaml["nc"]

    sparse_names = {1: "cat", 3: "dog", 5: "elephant", 6: "lion"}
    corrupt_path = _save_model_with_names(model, sparse_names, tmp_path, "sparse_key_names.pt")
    corrupt_model = YOLO(corrupt_path)

    file = corrupt_model.export(format="onnx", dynamic=True, imgsz=32, end2end=end2end)
    meta = ort.InferenceSession(str(file)).get_modelmeta().custom_metadata_map
    names = _assert_names_valid(meta)

    assert len(names) == expected_nc, (
        f"Expected {expected_nc} classes (from model.yaml['nc'] fallback after KeyError), got {len(names)}"
    )
    assert len(names) != 999, (
        "Got 999 classes — KeyError fallback resolved to default_class_names() instead of yaml['nc']"
    )

    YOLO(file)(SOURCE, imgsz=32)  # exported model inference


@pytest.mark.parametrize("end2end", [False, True])
def test_export_onnx_legacy_names_absent(end2end, isolated_model, tmp_path):
    """Export model where `names` is completely empty.

    Args:
        end2end (bool): Whether to test end-to-end export mode.
        isolated_model (str): Path to isolated model fixture provided by pytest.
        tmp_path (Path): Temporary directory path provided by pytest fixture.
    """
    model = YOLO(isolated_model)
    expected_nc = model.model.yaml["nc"]  # 80 for yolo26n

    corrupt_path = _save_model_with_names(model, "__delete__", tmp_path, "absent_names.pt")
    corrupt_model = YOLO(corrupt_path)

    # check if `names` is completely deleted
    assert not hasattr(corrupt_model.model, "names"), (
        "Fixture setup failed: 'names' attribute should be absent after reload"
    )

    file = corrupt_model.export(format="onnx", dynamic=True, imgsz=32, end2end=end2end)

    import onnxruntime as ort

    meta = ort.InferenceSession(str(file)).get_modelmeta().custom_metadata_map
    names = _assert_names_valid(meta)

    assert len(names) == expected_nc, (
        f"Expected {expected_nc} classes (from model.yaml['nc']), "
        f"got {len(names)} — possible silent fallback to default_class_names()"
    )
    assert len(names) != 999, "Got 999 classes — the fix did not apply and default_class_names() was used"

    YOLO(file)(SOURCE, imgsz=32)  # exported model inference


@pytest.mark.parametrize("end2end", [False, True])
def test_export_onnx_legacy_names_none(end2end, isolated_model, tmp_path):
    """Export model with None type `names` param.

    Args:
        end2end (bool): Whether to test end-to-end export mode.
        isolated_model (str): Path to isolated model fixture provided by pytest.
        tmp_path (Path): Temporary directory path provided by pytest fixture.
    """
    model = YOLO(isolated_model)
    expected_nc = model.model.yaml["nc"]

    corrupt_path = _save_model_with_names(model, None, tmp_path, "none_names.pt")
    corrupt_model = YOLO(corrupt_path)

    assert corrupt_model.model.names is None, "Fixture setup failed: 'names' should be None after reload"

    file = corrupt_model.export(format="onnx", dynamic=True, imgsz=32, end2end=end2end)

    import onnxruntime as ort

    meta = ort.InferenceSession(str(file)).get_modelmeta().custom_metadata_map
    names = _assert_names_valid(meta)

    assert len(names) == expected_nc, (
        f"Expected {expected_nc} classes (from model.yaml['nc']), "
        f"got {len(names)} — possible silent fallback to default_class_names()"
    )
    assert len(names) != 999, "Got 999 classes — the fix did not apply and default_class_names() was used"

    YOLO(file)(SOURCE, imgsz=32)  # exported model inference


@pytest.mark.parametrize("end2end", [False, True])
def test_export_onnx_legacy_names_list(end2end, isolated_model, tmp_path):
    """Export model where `names` is a list instead of dict`.

    Args:
        end2end (bool): Whether to test end-to-end export mode.
        isolated_model (str): Path to isolated model fixture provided by pytest.
        tmp_path (Path): Temporary directory path provided by pytest fixture.
    """
    list_names = ["human", "cat", "dog"]

    model = YOLO(isolated_model)
    corrupt_path = _save_model_with_names(model, list_names, tmp_path, "list_names.pt")
    corrupt_model = YOLO(corrupt_path)

    file = corrupt_model.export(format="onnx", dynamic=True, imgsz=32, end2end=end2end)

    import onnxruntime as ort

    meta = ort.InferenceSession(str(file)).get_modelmeta().custom_metadata_map
    names = _assert_names_valid(meta)  # ensures dict with int keys and str values

    assert len(names) == len(list_names), f"Expected {len(list_names)} classes (length of input list), got {len(names)}"

    YOLO(file)(SOURCE, imgsz=32)  # exported model inference


@pytest.mark.parametrize("end2end", [False, True])
def test_export_onnx_legacy_names_string_keys(end2end, isolated_model, tmp_path):
    """Export mmodel where key is string instead of int.

    Args:
        end2end (bool): Whether to test end-to-end export mode.
        isolated_model (str): Path to isolated model fixture provided by pytest.
        tmp_path (Path): Temporary directory path provided by pytest fixture.
    """
    string_key_names = {"0": "cat", "1": "dog"}

    model = YOLO(isolated_model)
    corrupt_path = _save_model_with_names(model, string_key_names, tmp_path, "string_key_names.pt")
    corrupt_model = YOLO(corrupt_path)

    file = corrupt_model.export(format="onnx", dynamic=True, imgsz=32, end2end=end2end)

    import onnxruntime as ort

    meta = ort.InferenceSession(str(file)).get_modelmeta().custom_metadata_map
    names = _assert_names_valid(meta)  # verifies int keys specifically

    assert len(names) == len(string_key_names), f"Expected {len(string_key_names)} classes, got {len(names)}"

    YOLO(file)(SOURCE, imgsz=32)  # exported model inference


@pytest.mark.parametrize("end2end", [False, True])
def test_export_onnx_legacy_names_int_values(end2end, isolated_model, tmp_path):
    """Export model with int value instead of str.

    Args:
        end2end (bool): Whether to test end-to-end export mode.
        isolated_model (str): Path to isolated model fixture provided by pytest.
        tmp_path (Path): Temporary directory path provided by pytest fixture.
    """
    int_value_names = {0: 0, 1: 1}

    model = YOLO(isolated_model)
    corrupt_path = _save_model_with_names(model, int_value_names, tmp_path, "int_value_names.pt")
    corrupt_model = YOLO(corrupt_path)

    file = corrupt_model.export(format="onnx", dynamic=True, imgsz=32, end2end=end2end)

    import onnxruntime as ort

    meta = ort.InferenceSession(str(file)).get_modelmeta().custom_metadata_map
    names = _assert_names_valid(meta)  # verifies str values specifically

    assert len(names) == len(int_value_names), f"Expected {len(int_value_names)} classes, got {len(names)}"

    YOLO(file)(SOURCE, imgsz=32)  # exported model inference


@pytest.mark.parametrize("end2end", [False, True])
def test_export_onnx_legacy_names_mixed_types(end2end, isolated_model, tmp_path):
    """Export model where key and values are mixed types.

    Args:
        end2end (bool): Whether to test end-to-end export mode.
        isolated_model (str): Path to isolated model fixture provided by pytest.
        tmp_path (Path): Temporary directory path provided by pytest fixture.
    """
    mixed_names = {0: "cat", "1": 4.5, 2: [2, 3], "3": [5, 6]}

    model = YOLO(isolated_model)
    corrupt_path = _save_model_with_names(model, mixed_names, tmp_path, "mixed_type_names.pt")
    corrupt_model = YOLO(corrupt_path)

    file = corrupt_model.export(format="onnx", dynamic=True, imgsz=32, end2end=end2end)

    import onnxruntime as ort

    meta = ort.InferenceSession(str(file)).get_modelmeta().custom_metadata_map
    names = _assert_names_valid(meta)  # full dict/int-key/str-value check

    assert len(names) == len(mixed_names), f"Expected {len(mixed_names)} classes, got {len(names)}"
    # No None values anywhere in metadata
    assert not any(v is None for v in meta.values()), (
        "ONNX metadata contains None values — names fallback may have produced None"
    )

    YOLO(file)(SOURCE, imgsz=32)  # exported model inference
