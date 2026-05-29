from unittest.mock import patch

from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.dataset import DepthDataset
from ultralytics.utils import DEFAULT_CFG


def test_build_yolo_dataset_routes_depth(tmp_path):
    cfg = get_cfg(DEFAULT_CFG)
    cfg.task = "depth"
    data = {"nc": 1, "channels": 3, "names": {0: "depth"}, "path": str(tmp_path)}
    (tmp_path / "images").mkdir()

    # Patch the constructor so we can check the class without needing real image files.
    with patch.object(DepthDataset, "__init__", return_value=None):
        ds = build_yolo_dataset(cfg, str(tmp_path / "images"), batch=1, data=data, mode="val")

    assert isinstance(ds, DepthDataset)


def test_build_yolo_dataset_routes_depth_multisource(tmp_path):
    cfg = get_cfg(DEFAULT_CFG)
    cfg.task = "depth"
    data = {"nc": 1, "channels": 3, "names": {0: "depth"}, "path": str(tmp_path)}
    p1 = tmp_path / "images1"
    p1.mkdir()
    p2 = tmp_path / "images2"
    p2.mkdir()

    # Patch the constructor so we can check the class without needing real image files.
    with patch.object(DepthDataset, "__init__", return_value=None):
        ds = build_yolo_dataset(cfg, [str(p1), str(p2)], batch=1, data=data, mode="train")

    assert isinstance(ds, DepthDataset)
