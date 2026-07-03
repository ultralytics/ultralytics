from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from ultralytics import YOLO
from ultralytics.models.yolo import depth
from ultralytics.utils import IS_JETSON, IS_RASPBERRYPI


def _make_depth_dataset(root):
    """Build a tiny hermetic depth dataset under *root* and return the YAML path.

    Layout mirrors what DepthDataset expects:
        <root>/images/<split>/*.jpg   — RGB images
        <root>/depth/<split>/*.npy    — float32 (H, W) depth maps

    The path mapping used by DepthDataset._depth_path_for() is a plain
    str.replace("/images/", "/depth/") followed by .with_suffix(".npy"),
    so basenames must match exactly.
    """
    root = Path(root)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "depth" / split).mkdir(parents=True, exist_ok=True)
        for i in range(4):
            img = (np.random.rand(64, 64, 3) * 255).astype("uint8")
            cv2.imwrite(str(root / "images" / split / f"{i}.jpg"), img)
            np.save(root / "depth" / split / f"{i}.npy", (np.random.rand(64, 64) * 10).astype("float32"))

    yaml_path = root / "depth8.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {"path": str(root), "train": "images/train", "val": "images/val", "nc": 1, "names": {0: "depth"}}
        )
    )
    return str(yaml_path)


@pytest.mark.skipif(IS_JETSON or IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_depth_engine_cycle(tmp_path):
    """Test depth estimation train / val / predict cycle using a hermetic synthetic dataset."""
    data = _make_depth_dataset(tmp_path)
    m = YOLO("ultralytics/cfg/models/26/yolo26-depth.yaml")
    m.train(data=data, epochs=1, imgsz=32, batch=2, device="cpu", cache=False, plots=False, workers=0)
    m.val(data=data, imgsz=32, batch=2, device="cpu", plots=False)
    r = m.predict(str(tmp_path / "images" / "val" / "0.jpg"), imgsz=32, device="cpu")
    assert r[0].depth is not None, "predict() should return a result with a depth map"


def test_depth_task_trainers_importable():
    """Smoke-check that DepthTrainer / DepthValidator / DepthPredictor are importable via the depth module."""
    assert hasattr(depth, "DepthTrainer"), "DepthTrainer not exported from depth module"
    assert hasattr(depth, "DepthValidator"), "DepthValidator not exported from depth module"
    assert hasattr(depth, "DepthPredictor"), "DepthPredictor not exported from depth module"
