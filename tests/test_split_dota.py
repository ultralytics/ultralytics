# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2
import numpy as np

from ultralytics.data.split_dota import crop_and_save


def test_crop_and_save_clips_labels(tmp_path):
    """Ensure crop_and_save clamps normalized OBB coords into [0, 1]."""
    # Create a dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_path = tmp_path / "img.jpg"
    cv2.imwrite(str(img_path), img)

    # Fake anno: one polygon that extends outside the window
    # class_id, x1, y1, x2, y2, x3, y3, x4, y4 (absolute pixels)
    label = np.array([[0, -10.0, 50.0, 50.0, 50.0, 120.0, 60.0, -10.0, 60.0]], dtype=np.float32)

    anno = {"filepath": str(img_path), "label": label.copy(), "ori_size": (100, 100)}

    # Single full-image window
    windows = np.array([[0, 0, 100, 100]], dtype=np.int64)
    window_objs = [label.copy()]

    im_dir = tmp_path / "images"
    lb_dir = tmp_path / "labels"
    im_dir.mkdir()
    lb_dir.mkdir()

    crop_and_save(anno, windows, window_objs, str(im_dir), str(lb_dir))

    txt_files = list(lb_dir.glob("*.txt"))
    assert txt_files, "Expected a label file to be written"

    coords = np.loadtxt(txt_files[0], ndmin=2)[:, 1:]  # drop class id
    eps = 1e-8
    assert coords.min() >= 0.0 - eps
    assert coords.max() <= 1.0 + eps
