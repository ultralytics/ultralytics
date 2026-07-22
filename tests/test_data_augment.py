# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np

from ultralytics.cfg import get_cfg
from ultralytics.data.augment import RandomPerspective
from ultralytics.utils.instance import Instances


def _labels(boxes: np.ndarray, normalized: bool = False) -> dict:
    """Build minimal detection labels for augmentation tests."""
    return {
        "img": np.zeros((100, 100, 3), dtype=np.uint8),
        "cls": np.zeros((len(boxes), 1), dtype=np.float32),
        "instances": Instances(
            boxes.astype(np.float32),
            segments=np.zeros((0, 1000, 2), dtype=np.float32),
            bbox_format="xyxy",
            normalized=normalized,
        ),
    }


def test_small_object_crop_centers_and_scales_target():
    """Small-object crop should zoom a selected target to the configured size and keep it centered."""
    transform = RandomPerspective(
        scale=0.0,
        translate=0.0,
        size=(100, 100),
        small_object_crop=1.0,
        small_object_crop_size=20.0,
        small_object_crop_max_scale=4.0,
    )
    labels = transform(_labels(np.array([[10, 20, 15, 25]])))
    box = labels["instances"].bboxes[0]

    np.testing.assert_allclose((box[:2] + box[2:]) / 2, (50, 50), atol=1e-5)
    np.testing.assert_allclose(box[2:] - box[:2], (20, 20), atol=1e-5)


def test_small_object_crop_mosaic_canvas_keeps_pixel_scale():
    """Cropping a 2x mosaic canvas to the output size should not halve target pixel dimensions."""
    transform = RandomPerspective(
        scale=0.0,
        translate=0.0,
        size=(640, 640),
        small_object_crop=1.0,
        small_object_crop_size=32.0,
        small_object_crop_max_scale=2.0,
    )
    labels = _labels(np.array([[100, 200, 116, 216]]))
    labels["img"] = np.zeros((1280, 1280, 3), dtype=np.uint8)
    labels = transform(labels)
    box = labels["instances"].bboxes[0]

    assert labels["img"].shape[:2] == (640, 640)
    np.testing.assert_allclose((box[:2] + box[2:]) / 2, (320, 320), atol=1e-5)
    np.testing.assert_allclose(box[2:] - box[:2], (32, 32), atol=1e-5)


def test_small_object_crop_defaults_off_and_configurable():
    """The feature should default off while accepting explicit training overrides."""
    cfg = get_cfg()
    assert cfg.small_object_crop == 0.0

    cfg = get_cfg(overrides={"small_object_crop": 0.5, "small_object_crop_size": 24, "small_object_crop_max_scale": 3})
    assert (cfg.small_object_crop, cfg.small_object_crop_size, cfg.small_object_crop_max_scale) == (0.5, 24, 3)
