# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Unit tests for the depth dataset: routing, GT loading/path mapping, and the augmentation pipeline."""

import os
from unittest.mock import patch

import numpy as np

from ultralytics.cfg import get_cfg
from ultralytics.data.augment import DepthFormat, RandomFlip, RandomHSV, RandomPerspective
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.dataset import DepthDataset
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.instance import Instances


def test_build_yolo_dataset_routes_depth(tmp_path):
    """Test build yolo dataset routes depth."""
    cfg = get_cfg(DEFAULT_CFG)
    cfg.task = "depth"
    data = {"nc": 1, "channels": 3, "names": {0: "depth"}, "path": str(tmp_path)}
    (tmp_path / "images").mkdir()

    # Patch the constructor so we can check the class without needing real image files.
    with patch.object(DepthDataset, "__init__", return_value=None):
        ds = build_yolo_dataset(cfg, str(tmp_path / "images"), batch=1, data=data, mode="val")

    assert isinstance(ds, DepthDataset)


def test_build_yolo_dataset_routes_depth_multisource(tmp_path):
    """Test build yolo dataset routes depth multisource."""
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


def test_depth_dataset_load_resize_does_not_blend_sparse_gt(tmp_path):
    """get_image_and_label resizes sparse depth to the image shape without blending (no bilinear)."""
    depth = np.zeros((8, 8), dtype=np.float32)
    depth[2:4, 2:4] = 10.0  # sparse valid block on a zero (invalid) background
    npy = tmp_path / "d.npy"
    np.save(npy, depth)

    ds = DepthDataset.__new__(DepthDataset)  # bypass __init__
    ds.im_files = ["x.png"]
    ds.labels = [{"im_file": "x.png"}]
    ds._depth_path_for = lambda f: str(npy)

    base = {"resized_shape": (32, 32), "im_file": "x.png", "img": np.zeros((32, 32, 3), np.uint8)}
    with patch("ultralytics.data.dataset.YOLODataset.get_image_and_label", return_value=base):
        out = ds.get_image_and_label(0)

    d = out["depth"]
    blended = ((d > 1e-6) & (d < 9.0)).sum()  # values between background(0) and valid(10)
    assert blended == 0, f"{blended} spurious blended depth pixels from interpolation at load"


def test_depth_path_mapping_handles_relative_and_absolute():
    """The images→depth rewrite works on relative, absolute, and nested paths (last 'images' wins)."""
    ds = DepthDataset.__new__(DepthDataset)
    assert ds._depth_path_for(os.path.join("images", "train", "x.jpg")) == os.path.join("depth", "train", "x.npy")
    absolute = os.path.join(os.sep, "data", "images", "val", "y.png")
    assert ds._depth_path_for(absolute) == os.path.join(os.sep, "data", "depth", "val", "y.npy")
    nested = os.path.join("images", "sub", "images", "z.jpg")
    assert ds._depth_path_for(nested) == os.path.join("images", "sub", "depth", "z.npy")


def _build_transforms(augment, **overrides):
    """Build the DepthDataset transform pipeline for the given augment flag and hyp overrides."""
    ds = DepthDataset.__new__(DepthDataset)
    ds.augment = augment
    ds.imgsz = 384
    hyp = get_cfg(DEFAULT_CFG)
    for k, v in overrides.items():
        setattr(hyp, k, v)
    return ds.build_transforms(hyp).transforms


def test_depth_val_transforms_have_no_augmentation():
    """Validation (augment=False) must be deterministic: no random warp, flip, or jitter."""
    names = [type(t).__name__ for t in _build_transforms(augment=False)]
    assert "RandomPerspective" not in names
    assert "RandomFlip" not in names
    assert "RandomHSV" not in names
    assert "DepthFormat" in names  # still formats img/depth to tensors


def test_depth_train_transforms_include_augmentation():
    """Training (augment=True) keeps the augmentation pipeline (perspective-safe affine warp)."""
    names = [type(t).__name__ for t in _build_transforms(augment=True)]
    assert "RandomPerspective" in names
    assert "RandomFlip" in names
    assert "RandomHSV" in names
    assert "DepthFormat" in names


def test_depth_train_transforms_read_cfg_args():
    """The augmentation pipeline is driven by the cfg args, not hardcoded values."""
    warp, vflip, hflip, hsv = _build_transforms(
        augment=True, degrees=33.0, translate=0.25, scale=0.75, shear=4.5, flipud=0.1, fliplr=0.9, hsv_h=0.2
    )[:4]
    assert warp.degrees == 33.0 and warp.translate == 0.25 and warp.scale == 0.75 and warp.shear == 4.5
    assert vflip.p == 0.1 and hflip.p == 0.9
    assert hsv.hgain == 0.2


def _empty_instances():
    """Empty cls/instances keys, as DepthDataset provides (depth label files are backgrounds)."""
    return {
        "cls": np.zeros((0, 1), dtype=np.float32),
        "instances": Instances(np.zeros((0, 4), dtype=np.float32), segments=np.zeros((0, 1000, 2), dtype=np.float32)),
    }


def test_depth_format_converts_to_tensors():
    """Test depth format converts to tensors."""
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    depth = np.ones((32, 32), dtype=np.float32)
    out = DepthFormat()({"img": img, "depth": depth})
    assert out["img"].shape == (3, 32, 32)  # CHW
    assert tuple(out["depth"].shape) == (1, 32, 32)  # (1,H,W)
    assert out["depth"].dtype.is_floating_point


def test_random_flip_flips_depth_with_image():
    """Test random flip flips depth with image."""
    img = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    depth = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
    out = RandomFlip(p=1.0, direction="horizontal")({"img": img.copy(), "depth": depth.copy(), **_empty_instances()})
    assert np.array_equal(out["img"], np.ascontiguousarray(np.fliplr(img)))
    assert np.array_equal(out["depth"], np.ascontiguousarray(np.fliplr(depth)))
    out = RandomFlip(p=1.0, direction="vertical")({"img": img.copy(), "depth": depth.copy(), **_empty_instances()})
    assert np.array_equal(out["depth"], np.ascontiguousarray(np.flipud(depth)))


def test_random_hsv_preserves_depth():
    """Test random hsv preserves depth."""
    img = np.full((16, 16, 3), 100, dtype=np.uint8)
    depth = np.ones((16, 16), dtype=np.float32)
    out = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)({"img": img, "depth": depth})
    assert out["img"].shape == (16, 16, 3) and out["img"].dtype == np.uint8
    assert np.array_equal(out["depth"], depth)  # depth unchanged by color jitter


def _sparse_depth(h, w, val=10.0):
    """Sparse depth map: mostly zero (invalid) with a block of valid metric depth."""
    d = np.zeros((h, w), dtype=np.float32)
    d[h // 4 : h // 2, w // 4 : w // 2] = val
    return d


def test_depth_format_resize_does_not_blend_sparse_depth():
    """Resizing sparse depth must not create intermediate near-zero values (no bilinear blend)."""
    img = np.zeros((32, 32, 3), dtype=np.uint8)  # forces depth resize 16->32
    depth = _sparse_depth(16, 16, val=10.0)
    out = DepthFormat()({"img": img, "depth": depth})["depth"].numpy()
    blended = ((out > 1e-6) & (out < 9.0)).sum()  # values between background(0) and valid(10)
    assert blended == 0, f"{blended} spurious blended depth pixels from interpolation"


def test_random_perspective_warps_depth_with_nearest_interpolation():
    """RandomPerspective should warp depth without inventing intermediate sparse values."""
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    depth = _sparse_depth(16, 16, val=10.0)
    labels = {"img": img.copy(), "depth": depth.copy(), **_empty_instances()}
    transform = RandomPerspective(
        degrees=0.0,
        translate=0.0,
        scale=(1.0, 1.0),
        shear=0.0,
        perspective=0.0,
        size=(32, 32),
    )
    out = transform(labels)
    assert out["img"].shape[:2] == out["depth"].shape[:2] == (32, 32)
    blended = ((out["depth"] > 1e-6) & (out["depth"] < 9.0)).sum()
    assert blended == 0, f"{blended} spurious blended depth pixels from interpolation"
