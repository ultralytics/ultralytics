# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("cv2", reason="opencv-python not installed")

_spec = importlib.util.spec_from_file_location("main", Path(__file__).parent / "main.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
main = _mod.main


def test_imread_missing_file_raises():
    """main() should raise FileNotFoundError, not AttributeError, when the image path is invalid."""
    with pytest.raises(FileNotFoundError, match="Image Not Found"):
        main("yolov8n.onnx", "/nonexistent_image_path.jpg")
