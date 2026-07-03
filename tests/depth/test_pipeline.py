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


def test_streamlit_tord_has_depth_suffix():
    import ultralytics.solutions.streamlit_inference as si

    src = open(si.__file__).read()
    assert "-depth" in src  # depth model suffix registered in the task-ordering map


def test_depth_dataset_load_resize_does_not_blend_sparse_gt(tmp_path):
    """get_image_and_label resizes sparse depth to the image shape without blending (no bilinear)."""
    import numpy as np

    depth = np.zeros((8, 8), dtype=np.float32)
    depth[2:4, 2:4] = 10.0                      # sparse valid block on a zero (invalid) background
    npy = tmp_path / "d.npy"
    np.save(npy, depth)

    ds = DepthDataset.__new__(DepthDataset)     # bypass __init__
    ds._depth_stack = None
    ds.labels = [{"im_file": "x.png"}]
    ds._depth_path_for = lambda f: str(npy)

    base = {"resized_shape": (32, 32), "im_file": "x.png", "img": np.zeros((32, 32, 3), np.uint8)}
    with patch("ultralytics.data.dataset.YOLODataset.get_image_and_label", return_value=base):
        out = ds.get_image_and_label(0)

    d = out["depth"]
    blended = ((d > 1e-6) & (d < 9.0)).sum()    # values between background(0) and valid(10)
    assert blended == 0, f"{blended} spurious blended depth pixels from interpolation at load"


def _transform_names(augment):
    ds = DepthDataset.__new__(DepthDataset)
    ds.augment = augment
    ds.imgsz = 384
    return [type(t).__name__ for t in ds.build_transforms().transforms]


def test_depth_val_transforms_have_no_augmentation():
    """Validation (augment=False) must be deterministic: no random warp, flip, or jitter."""
    names = _transform_names(augment=False)
    assert "RandomPerspective" not in names
    assert "DepthRandomFlip" not in names
    assert "DepthColorJitter" not in names
    assert "DepthFormat" in names               # still formats img/depth to tensors


def test_depth_train_transforms_include_augmentation():
    """Training (augment=True) keeps the augmentation pipeline (perspective-safe affine warp)."""
    names = _transform_names(augment=True)
    assert "RandomPerspective" in names
    assert "DepthRandomFlip" in names
    assert "DepthFormat" in names
